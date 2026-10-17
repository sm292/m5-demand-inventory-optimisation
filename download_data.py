
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")

REQUIRED = {
    "sales_train_validation.csv",
    "calendar.csv",
}

OPTIONAL = {"sell_prices.csv"}


def main():
    missing = [name for name in REQUIRED if not (RAW / name).exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing files in {RAW.resolve()}: {missing}\n"
            "Download the M5 files from Kaggle or the linked archive."
        )

    sales_path = RAW / "sales_train_validation.csv"
    calendar_path = RAW / "calendar.csv"

    # Read only headers first to avoid loading large files unnecessarily.
    sales_columns = pd.read_csv(sales_path, nrows=0).columns.tolist()
    calendar = pd.read_csv(calendar_path)

    day_columns = [c for c in sales_columns if c.startswith("d_")]

    if not day_columns:
        raise ValueError("No daily sales columns (d_1, d_2, ...) found.")

    if not {"d", "date"}.issubset(calendar.columns):
        raise ValueError("Calendar must contain 'd' and 'date' columns.")

    missing_days = sorted(set(day_columns) - set(calendar["d"]))

    if missing_days:
        raise ValueError(
            f"Calendar is missing {len(missing_days)} sales day IDs."
        )

    print("M5 files validated.")
    print(f"Sales day columns: {len(day_columns)}")
    print(f"Calendar rows: {len(calendar):,}")

    for name in REQUIRED | OPTIONAL:
        path = RAW / name
        print(f"{name}: {'found' if path.exists() else 'not present'}")

    print("\nNext: python train_model.py")


if __name__ == "__main__":
    main()

