"""Feature engineering and chronological train/test preparation for stock OHLCV data."""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "high_low_range",
    "return",
    "ma5",
    "ma20",
    "close_lag1",
    "volume_change",
]
TARGET_COLUMN = "target_next_return"


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create causal features independently for each stock symbol."""
    required = {"StockCode", "Date", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {sorted(missing)}")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["StockCode", "Date", "High", "Low", "Close", "Volume"])
    df = df.sort_values(["StockCode", "Date"]).drop_duplicates(
        subset=["StockCode", "Date"], keep="last"
    ).reset_index(drop=True)

    grouped = df.groupby("StockCode", sort=False)
    previous_close = grouped["Close"].shift(1)
    previous_volume = grouped["Volume"].shift(1)

    df["high_low_range"] = df["High"] - df["Low"]
    df["return"] = df["Close"].div(previous_close).sub(1)
    df["ma5"] = grouped["Close"].transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )
    df["ma20"] = grouped["Close"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )
    df["close_lag1"] = previous_close
    df["volume_change"] = df["Volume"].div(previous_volume).sub(1)

    # Label for next trading session; this is never included among input features.
    df[TARGET_COLUMN] = grouped["return"].shift(-1)
    df["target_date"] = grouped["Date"].shift(-1)

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, "target_date"])
    return df.reset_index(drop=True)


def prepare_train_test(data: pd.DataFrame, test_size: float = 0.2):
    """Chronological split by unique trading dates, with one-row label purge."""
    if not 0 < test_size < 1:
        raise ValueError("test_size phải nằm trong khoảng (0, 1)")

    df = build_features(data)
    dates = sorted(df["Date"].dropna().unique())
    if len(dates) < 2:
        raise ValueError("Không đủ ngày giao dịch để chia Train/Test.")

    split_pos = int(len(dates) * (1 - test_size))
    split_pos = min(max(split_pos, 1), len(dates) - 1)
    test_start_date = pd.Timestamp(dates[split_pos])

    # Purge the final training date: its next-day target would fall in Test.
    train_dates = dates[:split_pos]
    if len(train_dates) < 2:
        raise ValueError("Không đủ ngày cho tập Train sau khi loại bỏ biên.")
    train_end_feature_date = pd.Timestamp(train_dates[-2])

    train_raw = df[df["Date"] <= train_end_feature_date].copy()
    test_raw = df[df["Date"] >= test_start_date].copy()

    if train_raw.empty or test_raw.empty:
        raise ValueError("Train hoặc Test rỗng. Hãy kiểm tra dữ liệu đầu vào.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_raw[FEATURE_COLUMNS])
    X_test = scaler.transform(test_raw[FEATURE_COLUMNS])

    train = train_raw[["StockCode", "Date", "target_date", TARGET_COLUMN]].copy()
    test = test_raw[["StockCode", "Date", "target_date", TARGET_COLUMN]].copy()
    for i, col in enumerate(FEATURE_COLUMNS):
        train[col] = X_train[:, i]
        test[col] = X_test[:, i]

    train = train[["StockCode", "Date", "target_date"] + FEATURE_COLUMNS + [TARGET_COLUMN]]
    test = test[["StockCode", "Date", "target_date"] + FEATURE_COLUMNS + [TARGET_COLUMN]]

    return {
        "engineered_data": df,
        "train": train,
        "test": test,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "split_date": test_start_date,
        "train_last_date": train_raw["Date"].max(),
        "test_first_date": test_raw["Date"].min(),
    }


def run_pipeline(input_csv, output_dir="data/processed", test_size=0.2):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)
    result = prepare_train_test(raw, test_size=test_size)

    result["engineered_data"].to_csv(
        output_dir / "all_stocks_features.csv", index=False, encoding="utf-8-sig"
    )
    result["train"].to_csv(
        output_dir / "train_scaled.csv", index=False, encoding="utf-8-sig"
    )
    result["test"].to_csv(
        output_dir / "test_scaled.csv", index=False, encoding="utf-8-sig"
    )

    return result
