
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RAW = Path("data/raw")
OUT = Path("models/outputs")
OUT.mkdir(parents=True, exist_ok=True)

SALES_FILE = RAW / "sales_train_validation.csv"
CALENDAR_FILE = RAW / "calendar.csv"

SERIES_LIMIT = 100
HORIZON = 28
RANDOM_STATE = 42
LAGS = (1, 7, 14, 28)


# --------------------------------------------------
# 2. Load and validate the data
# --------------------------------------------------

def load_data():
    if not SALES_FILE.exists():
        raise FileNotFoundError(
            f"Sales file not found: {SALES_FILE}"
        )

    if not CALENDAR_FILE.exists():
        raise FileNotFoundError(
            f"Calendar file not found: {CALENDAR_FILE}"
        )

    sales = pd.read_csv(SALES_FILE)
    calendar_df = pd.read_csv(CALENDAR_FILE)

    required_sales_columns = {
        "id", "item_id", "store_id", "state_id"
    }

    missing_columns = required_sales_columns - set(sales.columns)

    if missing_columns:
        raise ValueError(
            f"Sales dataset is missing columns: {missing_columns}"
        )

    if not {"d", "date"}.issubset(calendar_df.columns):
        raise ValueError(
            "Calendar must contain 'd' and 'date' columns."
        )

    day_columns = [
        c for c in sales.columns if c.startswith("d_")
    ]

    day_columns = sorted(
        day_columns,
        key=lambda x: int(x.split("_")[1])
    )

    if not day_columns:
        raise ValueError("No daily sales columns found.")

    # Convert sales values to numeric.
    sales[day_columns] = sales[day_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).astype("float32")

    calendar_df["date"] = pd.to_datetime(
        calendar_df["date"], errors="raise"
    )

    if calendar_df["d"].duplicated().any():
        raise ValueError("Duplicate day IDs found in calendar.")

    calendar = calendar_df.set_index("d").sort_values("date")

    # Keep calendar IDs in chronological order.
    calendar_days = calendar.index.tolist()

    calendar_days = sorted(
        calendar_days,
        key=lambda x: int(x.split("_")[1])
    )

    missing_days = set(day_columns) - set(calendar_days)

    if missing_days:
        raise ValueError(
            f"Calendar is missing {len(missing_days)} historical day IDs."
        )

    # Ensure the calendar includes the future forecast horizon.
    last_observed_day = day_columns[-1]
    last_index = calendar_days.index(last_observed_day)

    if last_index + HORIZON >= len(calendar_days):
        raise ValueError(
            "Calendar does not contain enough future dates for "
            f"a {HORIZON}-day forecast. Obtain a complete calendar."
        )

    # Select a reproducible sample across the catalogue.
    sales = sales.sample(
        n=min(SERIES_LIMIT, len(sales)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"Selected {len(sales)} item-store series.")
    print(f"Historical sales days: {len(day_columns)}")
    print(f"Calendar days: {len(calendar_days)}")
    print(
        "Last observed date:",
        calendar.loc[last_observed_day, "date"].date()
    )

    return sales, calendar, day_columns, calendar_days


# --------------------------------------------------
# 3. Calendar feature engineering
# --------------------------------------------------

def calendar_features(day_id, series, calendar):
    cal = calendar.loc[day_id]
    date = cal["date"]

    state = str(series["state_id"])
    snap_column = f"snap_{state}"

    snap_value = (
        float(cal[snap_column])
        if snap_column in calendar.columns
        and pd.notna(cal[snap_column])
        else 0.0
    )

    event_1 = cal.get("event_name_1", np.nan)
    event_2 = cal.get("event_name_2", np.nan)

    return {
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "year": date.year,
        "week_of_year": int(date.isocalendar().week),
        "is_weekend": int(date.dayofweek >= 5),
        "is_event": int(
            pd.notna(event_1) or pd.notna(event_2)
        ),
        "snap": snap_value,
    }


# --------------------------------------------------
# 4. Demand feature engineering
# --------------------------------------------------

def make_features(history, day_id, series, calendar):
    """
    Create features using only demand observed before the
    date being predicted.

    During recursive forecasting, prior predictions are
    appended to history and used for subsequent predictions.
    """
    history = np.asarray(history, dtype=float)

    if len(history) < max(LAGS):
        raise ValueError(
            f"At least {max(LAGS)} observations are required."
        )

    features = {
        f"lag_{lag}": float(history[-lag])
        for lag in LAGS
    }

    features["rolling_mean_7"] = float(
        history[-7:].mean()
    )

    features["rolling_mean_28"] = float(
        history[-28:].mean()
    )

    features["rolling_std_28"] = float(
        history[-28:].std()
    )

    features["zero_share_28"] = float(
        (history[-28:] == 0).mean()
    )

    features.update(
        calendar_features(day_id, series, calendar)
    )

    return features


# --------------------------------------------------
# 5. Build supervised training data
# --------------------------------------------------

def build_training_rows(
    sales,
    calendar,
    day_columns,
    calendar_days,
    end_index
):
    """
    Train on historical observations strictly before end_index.
    """
    rows = []
    targets = []

    for _, series in sales.iterrows():
        history = series[day_columns].to_numpy(dtype=float)

        for t in range(max(LAGS), end_index):
            day_id = day_columns[t]

            features = make_features(
                history=history[:t],
                day_id=day_id,
                series=series,
                calendar=calendar
            )

            rows.append(features)
            targets.append(history[t])

    X = pd.DataFrame(rows).astype("float32")
    y = np.asarray(targets, dtype="float32")

    return X, y


# --------------------------------------------------
# 6. Recursive forecasting
# --------------------------------------------------

def recursive_forecast(
    model,
    series,
    history,
    start_index,
    horizon,
    calendar,
    calendar_days
):
    """
    Forecast one day at a time.

    The calendar_days list includes historical and future dates.
    After each prediction, append that prediction to the history
    so future predictions do not use actual future sales.
    """
    values = list(np.asarray(history, dtype=float))
    output = []

    for t in range(start_index, start_index + horizon):

        if t >= len(calendar_days):
            raise ValueError(
                "Forecast index exceeds available calendar days."
            )

        day_id = calendar_days[t]

        X_one = pd.DataFrame([
            make_features(
                history=values,
                day_id=day_id,
                series=series,
                calendar=calendar
            )
        ])

        prediction = max(
            0.0,
            float(model.predict(X_one)[0])
        )

        output.append({
            "id": series["id"],
            "item_id": series["item_id"],
            "store_id": series["store_id"],
            "date": calendar.loc[day_id, "date"],
            "forecast": prediction,
        })

        # Use the prediction as the next day's lag.
        values.append(prediction)

    return output


# --------------------------------------------------
# 7. Evaluation metrics
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = actual.sum()

    wape = (
        float(np.abs(actual - predicted).sum() / denominator)
        if denominator > 0
        else None
    )

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "wape": wape,
        # Positive bias means overforecasting.
        "bias": float(
            (predicted - actual).sum()
        ),
    }


# --------------------------------------------------
# 8. Main training and forecasting pipeline
# --------------------------------------------------

def main():
    sales, calendar, day_columns, calendar_days = load_data()

    n_days = len(day_columns)

    # Last 28 observed days form the validation window.
    split = n_days - HORIZON

    if split <= max(LAGS):
        raise ValueError(
            "Insufficient historical data for the selected lags."
        )

    print("\nBuilding training features...")

    X_train, y_train = build_training_rows(
        sales=sales,
        calendar=calendar,
        day_columns=day_columns,
        calendar_days=calendar_days,
        end_index=split
    )

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )

    print("Training gradient boosting model...")
    model.fit(X_train, y_train)

    # ----------------------------------------------
    # Chronological validation
    # ----------------------------------------------

    print("Generating validation forecasts...")

    validation_rows = []
    baseline_rows = []

    for _, series in sales.iterrows():
        full_history = series[day_columns].to_numpy(
            dtype=float
        )

        train_history = full_history[:split]

        predictions = recursive_forecast(
            model=model,
            series=series,
            history=train_history,
            start_index=split,
            horizon=HORIZON,
            calendar=calendar,
            calendar_days=calendar_days
        )

        for j, prediction in enumerate(predictions):
            actual = float(full_history[split + j])

            validation_rows.append({
                **prediction,
                "actual": actual,
                "error": actual - prediction["forecast"]
            })

            # Repeat the last observed seven-day pattern.
            # No actual validation-period sales are used.
            baseline_prediction = float(
                full_history[split - 7 + (j % 7)]
            )

            baseline_rows.append({
                "id": series["id"],
                "actual": actual,
                "forecast": baseline_prediction
            })

    validation_df = pd.DataFrame(validation_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    model_metrics = calculate_metrics(
        validation_df["actual"],
        validation_df["forecast"]
    )

    baseline_metrics = calculate_metrics(
        baseline_df["actual"],
        baseline_df["forecast"]
    )

    # ----------------------------------------------
    # Final refit using all observed sales
    # ----------------------------------------------

    print("Refitting model on all historical observations...")

    X_full, y_full = build_training_rows(
        sales=sales,
        calendar=calendar,
        day_columns=day_columns,
        calendar_days=calendar_days,
        end_index=n_days
    )

    model.fit(X_full, y_full)

    # ----------------------------------------------
    # Forecast the next 28 calendar days
    # ----------------------------------------------

    print("Generating future forecasts...")

    future_rows = []

    for _, series in sales.iterrows():
        full_history = series[day_columns].to_numpy(
            dtype=float
        )

        future_rows.extend(
            recursive_forecast(
                model=model,
                series=series,
                history=full_history,
                start_index=n_days,
                horizon=HORIZON,
                calendar=calendar,
                calendar_days=calendar_days
            )
        )

    # ----------------------------------------------
    # Save output files
    # ----------------------------------------------

    validation_path = OUT / "validation_predictions.csv"
    forecast_path = OUT / "future_forecasts.csv"
    model_path = OUT / "demand_model.joblib"
    metrics_path = OUT / "metrics.json"

    validation_df.to_csv(
        validation_path, index=False
    )

    pd.DataFrame(future_rows).to_csv(
        forecast_path, index=False
    )

    joblib.dump(model, model_path)

    report = {
        "series_count": int(len(sales)),
        "forecast_horizon_days": HORIZON,
        "validation_days": HORIZON,
        "validation_start": str(
            validation_df["date"].min().date()
        ),
        "validation_end": str(
            validation_df["date"].max().date()
        ),
        "future_forecast_start": str(
            pd.DataFrame(future_rows)["date"].min().date()
        ),
        "future_forecast_end": str(
            pd.DataFrame(future_rows)["date"].max().date()
        ),
        "model": model_metrics,
        "seasonal_naive": baseline_metrics,
        "model_improves_wape": (
            model_metrics["wape"] is not None
            and baseline_metrics["wape"] is not None
            and model_metrics["wape"] < baseline_metrics["wape"]
        ),
        "features": X_full.columns.tolist(),
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 55)
    print("DEMAND FORECASTING RESULTS")
    print("=" * 55)

    print(json.dumps(report, indent=2))

    print("\nSaved files:")
    print(f"- {validation_path}")
    print(f"- {forecast_path}")
    print(f"- {model_path}")
    print(f"- {metrics_path}")


if __name__ == "__main__":
    main()
