
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW = Path("data/raw")
OUT = Path("models/outputs")
OUT.mkdir(parents=True, exist_ok=True)

SALES_FILE = RAW / "sales_train_validation.csv"
CALENDAR_FILE = RAW / "calendar.csv"

SERIES_LIMIT = 100
HORIZON = 28
RANDOM_STATE = 42
LAGS = (1, 7, 14, 28)


def load_data():
    sales = pd.read_csv(SALES_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)

    day_columns = [c for c in sales.columns if c.startswith("d_")]
    day_columns = sorted(day_columns, key=lambda x: int(x[2:]))

    sales[day_columns] = sales[day_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).astype("float32")

    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar.drop_duplicates("d").set_index("d")

    if len(day_columns) < 2 * HORIZON + max(LAGS):
        raise ValueError("Not enough historical data for validation.")

    # Sample across the full catalogue, not just the first store.
    sales = sales.sample(
        n=min(SERIES_LIMIT, len(sales)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return sales, calendar, day_columns


def calendar_features(day_id, row, calendar):
    cal = calendar.loc[day_id]
    date = cal["date"]

    state = str(row["state_id"])
    snap_column = f"snap_{state}"

    snap = (
        float(cal[snap_column])
        if snap_column in calendar.columns
        and pd.notna(cal[snap_column])
        else 0.0
    )

    return {
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "year": date.year,
        "week_of_year": int(date.isocalendar().week),
        "is_weekend": int(date.dayofweek >= 5),
        "is_event": int(
            pd.notna(cal.get("event_name_1", np.nan))
            or pd.notna(cal.get("event_name_2", np.nan))
        ),
        "snap": snap,
    }


def make_features(history, t, row, calendar, day_columns):
    # Only values before t are used to predict sales at t.
    history = np.asarray(history, dtype=float)

    features = {
        f"lag_{lag}": float(history[-lag])
        for lag in LAGS
    }

    features["rolling_mean_7"] = float(history[-7:].mean())
    features["rolling_mean_28"] = float(history[-28:].mean())
    features["rolling_std_28"] = float(history[-28:].std())
    features["zero_share_28"] = float((history[-28:] == 0).mean())

    features.update(
        calendar_features(day_columns[t], row, calendar)
    )

    return features


def build_training_rows(sales, calendar, day_columns, end_index):
    rows = []
    targets = []

    for _, series in sales.iterrows():
        history = series[day_columns].to_numpy(dtype=float)

        for t in range(max(LAGS), end_index):
            rows.append(
                make_features(
                    history[:t], t, series, calendar, day_columns
                )
            )
            targets.append(history[t])

    X = pd.DataFrame(rows).astype("float32")
    y = np.asarray(targets, dtype="float32")

    return X, y


def recursive_forecast(model, series, history, start, horizon,
                       calendar, day_columns):
    values = list(np.asarray(history, dtype=float))
    output = []

    for t in range(start, start + horizon):
        if t >= len(day_columns):
            raise ValueError(
                "Calendar does not contain all forecast dates. "
                "Check the M5 calendar file."
            )

        X_one = pd.DataFrame([
            make_features(values, t, series, calendar, day_columns)
        ])

        prediction = max(0.0, float(model.predict(X_one)[0]))

        output.append({
            "id": series["id"],
            "item_id": series["item_id"],
            "store_id": series["store_id"],
            "date": calendar.loc[day_columns[t], "date"],
            "forecast": prediction,
        })

        # Recursive forecasting: future predictions become subsequent lags.
        values.append(prediction)

    return output


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = actual.sum()
    wape = (
        float(np.abs(actual - predicted).sum() / denominator)
        if denominator > 0 else None
    )

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": wape,
        "bias": float((predicted - actual).sum()),
    }


def main():
    sales, calendar, day_columns = load_data()

    # Last 28 observed days are used for validation.
    n_days = len(day_columns)
    split = n_days - HORIZON

    # Train only on observations strictly before the holdout.
    X_train, y_train = build_training_rows(
        sales, calendar, day_columns, split
    )

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    validation_rows = []
    baseline_rows = []

    for _, series in sales.iterrows():
        history = series[day_columns].to_numpy(dtype=float)

        predictions = recursive_forecast(
            model=model,
            series=series,
            history=history[:split],
            start=split,
            horizon=HORIZON,
            calendar=calendar,
            day_columns=day_columns,
        )

        for j, prediction in enumerate(predictions):
            actual = float(history[split + j])

            validation_rows.append({
                **prediction,
                "actual": actual,
                "error": actual - prediction["forecast"],
            })

            # Seasonal-naive baseline: repeat the last 7-day pattern.
            baseline_rows.append({
                "id": series["id"],
                "actual": actual,
                "forecast": float(history[split + j - 7]),
            })

    validation = pd.DataFrame(validation_rows)
    baseline = pd.DataFrame(baseline_rows)

    model_metrics = metrics(
        validation["actual"], validation["forecast"]
    )
    baseline_metrics = metrics(
        baseline["actual"], baseline["forecast"]
    )

    # Final refit on all observed historical targets.
    X_full, y_full = build_training_rows(
        sales, calendar, day_columns, n_days
    )
    model.fit(X_full, y_full)

    future_rows = []

    for _, series in sales.iterrows():
        history = series[day_columns].to_numpy(dtype=float)

        future_rows.extend(
            recursive_forecast(
                model=model,
                series=series,
                history=history,
                start=n_days,
                horizon=HORIZON,
                calendar=calendar,
                day_columns=day_columns,
            )
        )

    validation.to_csv(
        OUT / "validation_predictions.csv", index=False
    )
    pd.DataFrame(future_rows).to_csv(
        OUT / "future_forecasts.csv", index=False
    )

    joblib.dump(model, OUT / "demand_model.joblib")

    report = {
        "series_count": int(len(sales)),
        "forecast_horizon_days": HORIZON,
        "validation_days": HORIZON,
        "model": model_metrics,
        "seasonal_naive": baseline_metrics,
        "model_improves_wape": (
            model_metrics["wape"] is not None
            and baseline_metrics["wape"] is not None
            and model_metrics["wape"] < baseline_metrics["wape"]
        ),
        "features": X_full.columns.tolist(),
    }

    with open(OUT / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nOutputs saved in {OUT.resolve()}")


if __name__ == "__main__":
    main()

