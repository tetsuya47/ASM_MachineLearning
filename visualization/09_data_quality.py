from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "all_stocks_20241113_20261113.csv"
OUTPUT_DIR = ROOT_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original data
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

required = [
    "StockCode", "Date", "Open",
    "High", "Low", "Close", "Volume"
]

missing_columns = set(required) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

original_count = len(df)

# Data type conversion
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 1. Missing values
missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": [int(df[col].isna().sum()) for col in df.columns],
    "Missing_Percentage": [
        round(df[col].isna().mean() * 100, 4)
        for col in df.columns
    ]
})

missing_report.to_csv(
    OUTPUT_DIR / "09_missing_values.csv",
    index=False
)

# 2. Duplicate records
duplicate_mask = df.duplicated(
    subset=["StockCode", "Date"],
    keep=False
)

duplicates = df.loc[duplicate_mask].sort_values(
    ["StockCode", "Date"]
)

duplicates.to_csv(
    OUTPUT_DIR / "09_duplicates.csv",
    index=False
)

duplicate_count = int(
    df.duplicated(subset=["StockCode", "Date"]).sum()
)

# 3. Ticker session gaps
valid_dates = df.dropna(subset=["StockCode", "Date"])
market_dates = set(valid_dates["Date"].unique())

gap_rows = []

for symbol in sorted(valid_dates["StockCode"].dropna().unique()):
    stock_dates = set(
        valid_dates.loc[
            valid_dates["StockCode"] == symbol, "Date"
        ].unique()
    )

    for date in sorted(market_dates - stock_dates):
        gap_rows.append({
            "StockCode": symbol,
            "Date": date,
            "Status": "No record on an observed market date",
            "Note": "Cause cannot be determined from this CSV alone"
        })

gaps = pd.DataFrame(
    gap_rows,
    columns=["StockCode", "Date", "Status", "Note"]
)

gaps.to_csv(
    OUTPUT_DIR / "09_ticker_session_gaps.csv",
    index=False
)

# 4. Prepare daily returns
valid = df.dropna(
    subset=["StockCode", "Date", "Close", "Volume"]
).copy()

valid = valid.sort_values(["StockCode", "Date"])

valid["DailyReturn"] = (
    valid.groupby("StockCode")["Close"]
    .pct_change(fill_method=None) * 100
)

# 5. Outlier detection
# DailyReturn: IQR on percentage returns.
# Volume: IQR on log1p(Volume), then convert bounds back to shares.
outlier_rows = []

for symbol, group in valid.groupby("StockCode"):

    # Daily return outliers
    returns = group["DailyReturn"].dropna()

    if len(returns) >= 4:
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flagged = group[
            (group["DailyReturn"] < lower) |
            (group["DailyReturn"] > upper)
        ]

        for _, row in flagged.iterrows():
            outlier_rows.append({
                "StockCode": symbol,
                "Date": row["Date"],
                "Metric": "DailyReturn",
                "Value": row["DailyReturn"],
                "Lower_Bound": lower,
                "Upper_Bound": upper,
                "Method": "IQR on daily percentage return",
                "Action": "Flagged for review - not deleted"
            })

    # Trading volume outliers on log scale
    volumes = group["Volume"].dropna()

    if len(volumes) >= 4 and (volumes >= 0).all():
        log_volume = np.log1p(volumes)

        q1 = log_volume.quantile(0.25)
        q3 = log_volume.quantile(0.75)
        iqr = q3 - q1

        lower_log = q1 - 1.5 * iqr
        upper_log = q3 + 1.5 * iqr

        lower_volume = max(0, np.expm1(lower_log))
        upper_volume = np.expm1(upper_log)

        flagged = group[
            (group["Volume"] < lower_volume) |
            (group["Volume"] > upper_volume)
        ]

        for _, row in flagged.iterrows():
            outlier_rows.append({
                "StockCode": symbol,
                "Date": row["Date"],
                "Metric": "Volume",
                "Value": row["Volume"],
                "Lower_Bound": lower_volume,
                "Upper_Bound": upper_volume,
                "Method": "IQR on log1p(Volume)",
                "Action": "Flagged for review - not deleted"
            })

outliers = pd.DataFrame(
    outlier_rows,
    columns=[
        "StockCode", "Date", "Metric", "Value",
        "Lower_Bound", "Upper_Bound", "Method", "Action"
    ]
)

outliers.to_csv(
    OUTPUT_DIR / "09_outliers.csv",
    index=False
)

# 6. OHLC validation
price_columns = ["Open", "High", "Low", "Close"]

complete_ohlc = df[price_columns].notna().all(axis=1)

ohlc_invalid = complete_ohlc & (
    (df["Open"] <= 0) |
    (df["High"] <= 0) |
    (df["Low"] <= 0) |
    (df["Close"] <= 0) |
    (df["High"] < df[["Open", "Close"]].max(axis=1)) |
    (df["Low"] > df[["Open", "Close"]].min(axis=1))
)

negative_volume = df["Volume"].notna() & (df["Volume"] < 0)

invalid_count = int((ohlc_invalid | negative_volume).sum())

# 7. Summary
outlier_counts = (
    outliers.groupby("Metric").size().to_dict()
    if not outliers.empty else {}
)

summary = pd.DataFrame([
    {"Metric": "Total original records", "Value": original_count},
    {"Metric": "Number of stock symbols", "Value": df["StockCode"].nunique()},
    {"Metric": "Missing cells", "Value": int(df.isna().sum().sum())},
    {"Metric": "Duplicate rows by StockCode and Date", "Value": duplicate_count},
    {"Metric": "Ticker missing observed market sessions", "Value": len(gaps)},
    {"Metric": "DailyReturn outliers (IQR)", "Value": outlier_counts.get("DailyReturn", 0)},
    {"Metric": "Volume outliers (log1p IQR)", "Value": outlier_counts.get("Volume", 0)},
    {"Metric": "Total flagged outlier observations", "Value": len(outliers)},
    {"Metric": "Invalid OHLC or negative volume rows", "Value": invalid_count},
    {"Metric": "Invalid dates", "Value": int(df["Date"].isna().sum())}
])

summary.to_csv(
    OUTPUT_DIR / "09_data_quality_summary.csv",
    index=False
)

# 8. Console report
print("\n========== DATA QUALITY REPORT ==========")
print(summary.to_string(index=False))

print("\nOUTLIERS BY METRIC:")
if not outliers.empty:
    print(outliers.groupby("Metric").size().to_string())
else:
    print("No outliers detected.")

print("\nMISSING VALUES:")
print(missing_report.to_string(index=False))

print("\nDuplicate records:", duplicate_count)
print("Ticker session gaps:", len(gaps))
print("Invalid OHLC / volume rows:", invalid_count)

print("\nOriginal data was not modified.")
print("Reports saved to:", OUTPUT_DIR)

print("\nMETRIC GUIDE:")
print("- Missing cells: null values after parsing; percentage is calculated per column.")
print("- Duplicate records: repeated StockCode + Date keys.")
print("- Ticker session gaps: a stock has no row on a date observed for another stock.")
print("- DailyReturn outliers: IQR rule applied to daily percentage returns per stock.")
print("- Volume outliers: IQR rule applied to log1p(Volume), with bounds converted back to shares.")
print("- IQR flags values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.")
print("- Outliers are flagged for review, not deleted.")
print("- A ticker gap alone cannot determine whether the cause is suspension or source coverage.")
