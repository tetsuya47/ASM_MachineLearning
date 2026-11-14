from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "all_stocks_20241113_20261113.csv"
OUT = ROOT / "visualizations"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "StockCode", "Date", "Open", "High",
    "Low", "Close", "Volume"
]
NUMERIC = ["Open", "High", "Low", "Close", "Volume"]

# 1. Read original data
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

missing_columns = set(REQUIRED) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df["SourceRow"] = np.arange(2, len(df) + 2)
df["StockCode"] = df["StockCode"].astype("string").str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in NUMERIC:
    df[col] = pd.to_numeric(df[col], errors="coerce")

original_count = len(df)

# 2. Missing-value report and quarantine
missing_report = pd.DataFrame({
    "Column": REQUIRED,
    "Missing_Count": [int(df[c].isna().sum()) for c in REQUIRED],
    "Missing_Percentage": [
        round(df[c].isna().mean() * 100, 4) for c in REQUIRED
    ]
})
missing_report.to_csv(OUT / "10_missing_values.csv", index=False)

missing_mask = df[REQUIRED].isna().any(axis=1)
missing_rows = df.loc[missing_mask].copy()
missing_rows["Treatment"] = (
    "Quarantined: required information missing; no artificial values inserted"
)
missing_rows.to_csv(OUT / "10_quarantined_missing.csv", index=False)

work = df.loc[~missing_mask].copy()

# 3. Duplicate detection
duplicate_mask = work.duplicated(
    subset=["StockCode", "Date"], keep=False
)
duplicates = work.loc[duplicate_mask].copy()

if not duplicates.empty:
    duplicates["Treatment"] = np.where(
        duplicates.duplicated(
            subset=["StockCode", "Date"], keep="first"
        ),
        "Removed: duplicate key",
        "Retained: first occurrence"
    )

duplicates.to_csv(OUT / "10_duplicate_records.csv", index=False)

duplicate_removed = int(
    work.duplicated(subset=["StockCode", "Date"], keep="first").sum()
)
work = work.drop_duplicates(
    subset=["StockCode", "Date"], keep="first"
).copy()

# 4. OHLC and volume validation
ohlc_complete = work[NUMERIC[:4]].notna().all(axis=1)

invalid_ohlc = ohlc_complete & (
    (work["Open"] <= 0)
    | (work["High"] <= 0)
    | (work["Low"] <= 0)
    | (work["Close"] <= 0)
    | (work["High"] < work[["Open", "Close"]].max(axis=1))
    | (work["Low"] > work[["Open", "Close"]].min(axis=1))
    | (work["High"] < work["Low"])
)

negative_volume = work["Volume"] < 0
invalid_mask = invalid_ohlc | negative_volume

invalid_rows = work.loc[invalid_mask].copy()
invalid_rows["Treatment"] = (
    "Quarantined: invalid OHLC relationship or negative volume"
)
invalid_rows.to_csv(OUT / "10_invalid_records.csv", index=False)

clean = work.loc[~invalid_mask].copy()
clean = clean.sort_values(["StockCode", "Date"]).reset_index(drop=True)

# 5. Detect ticker-specific missing observed sessions
market_dates = set(clean["Date"].dropna().unique())
gap_rows = []

for symbol in sorted(clean["StockCode"].unique()):
    stock_dates = set(
        clean.loc[clean["StockCode"] == symbol, "Date"].unique()
    )
    for date in sorted(market_dates - stock_dates):
        gap_rows.append({
            "StockCode": symbol,
            "Date": date,
            "Status": "Missing on a date observed for other stocks",
            "Note": "Check suspension, source coverage, or data ingestion"
        })

gaps = pd.DataFrame(
    gap_rows,
    columns=["StockCode", "Date", "Status", "Note"]
)
gaps.to_csv(OUT / "10_ticker_session_gaps.csv", index=False)

# 6. Verify exchange holidays and detect unexplained market-wide gaps.
# Dates are based on the holiday verification supplied for this dataset.

VERIFIED_HOLIDAY_DATES = [
    "2025-01-01",
    "2025-01-27",
    "2025-01-28",
    "2025-01-29",
    "2025-01-30",
    "2025-01-31",
    "2025-04-07",
    "2025-04-30",
    "2025-05-01",
    "2025-05-02",
    "2025-09-01",
    "2025-09-02",
    "2026-01-01",
    "2026-01-02",
    "2026-02-16",
    "2026-02-17",
    "2026-02-18",
    "2026-02-19",
    "2026-02-20",
    "2026-04-27",
    "2026-04-30",
    "2026-05-01",
    "2026-08-31",
    "2026-09-01",
    "2026-09-02",
]

verified_holidays = pd.DataFrame({
    "Date": pd.to_datetime(VERIFIED_HOLIDAY_DATES),
    "Status": "Verified: official HOSE/HNX exchange holiday",
    "Action": (
        "Exclude from missing trading-session alerts; "
        "retain original dataset unchanged"
    )
})

verified_holidays.to_csv(
    OUT / "10_verified_exchange_holidays.csv",
    index=False,
    date_format="%Y-%m-%d"
)

holiday_dates = set(
    verified_holidays["Date"].dt.normalize()
)

if not clean.empty:
    weekdays = pd.bdate_range(
        clean["Date"].min(),
        clean["Date"].max()
    ).normalize()

    observed_market_dates = set(
        pd.to_datetime(clean["Date"]).dt.normalize()
    )

    absent_market_days = sorted(
        set(weekdays)
        - observed_market_dates
        - holiday_dates
    )
else:
    absent_market_days = []

calendar_candidates = pd.DataFrame({
    "Date": absent_market_days,
    "Status": [
        "Unverified: possible market-wide data gap"
        for _ in absent_market_days
    ],
    "Action": [
        "Investigate data coverage and official trading calendar"
        for _ in absent_market_days
    ]
})

calendar_candidates.to_csv(
    OUT / "10_market_calendar_candidates.csv",
    index=False,
    date_format="%Y-%m-%d"
)

print(f"\nVerified exchange holidays: {len(verified_holidays)}")
print(f"Unverified market-wide missing weekdays: {len(calendar_candidates)}")
# 7. Detect outliers and create adjusted analysis columns.
# Original OHLC and Volume columns remain unchanged.
clean["DailyReturn"] = (
    clean.groupby("StockCode")["Close"]
    .pct_change(fill_method=None) * 100
)
clean["DailyReturn_Winsorized"] = clean["DailyReturn"]

clean["LogVolume"] = np.log1p(clean["Volume"].clip(lower=0))
clean["LogVolume_Winsorized"] = clean["LogVolume"]
clean["Volume_Winsorized"] = clean["Volume"].astype(float)

outlier_rows = []

for symbol, indexes in clean.groupby("StockCode").groups.items():
    idx = list(indexes)
    group = clean.loc[idx]

    # Daily return IQR and winsorization
    values = group["DailyReturn"].dropna()
    if len(values) >= 4:
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (group["DailyReturn"] < lower)
            | (group["DailyReturn"] > upper)
        )
        flagged = group.loc[mask]

        for _, row in flagged.iterrows():
            outlier_rows.append({
                "StockCode": symbol,
                "Date": row["Date"],
                "Metric": "DailyReturn",
                "OriginalValue": row["DailyReturn"],
                "LowerBound": lower,
                "UpperBound": upper,
                "Treatment": "Winsorized in adjusted analysis column only"
            })

        clean.loc[idx, "DailyReturn_Winsorized"] = (
            clean.loc[idx, "DailyReturn"].clip(lower, upper)
        )

    # Log volume IQR and winsorization
    values = group["LogVolume"].dropna()
    if len(values) >= 4:
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_log = max(0, q1 - 1.5 * iqr)
        upper_log = q3 + 1.5 * iqr

        mask = (
            (group["LogVolume"] < lower_log)
            | (group["LogVolume"] > upper_log)
        )
        flagged = group.loc[mask]

        lower_volume = np.expm1(lower_log)
        upper_volume = np.expm1(upper_log)

        for _, row in flagged.iterrows():
            outlier_rows.append({
                "StockCode": symbol,
                "Date": row["Date"],
                "Metric": "Volume",
                "OriginalValue": row["Volume"],
                "LowerBound": lower_volume,
                "UpperBound": upper_volume,
                "Treatment": "Winsorized in adjusted analysis column only"
            })

        clipped_log = clean.loc[idx, "LogVolume"].clip(
            lower_log, upper_log
        )
        clean.loc[idx, "LogVolume_Winsorized"] = clipped_log
        clean.loc[idx, "Volume_Winsorized"] = np.expm1(clipped_log)

outliers = pd.DataFrame(
    outlier_rows,
    columns=[
        "StockCode", "Date", "Metric", "OriginalValue",
        "LowerBound", "UpperBound", "Treatment"
    ]
)
outliers.to_csv(OUT / "10_outlier_treatment.csv", index=False)

# 8. Export cleaned analytical dataset
clean.to_csv(OUT / "10_cleaned_data.csv", index=False)

# 9. Summary

summary = pd.DataFrame([
    {"Metric": "Original records", "Value": original_count},
    {"Metric": "Rows quarantined for missing required values", "Value": len(missing_rows)},
    {"Metric": "Duplicate rows removed", "Value": duplicate_removed},
    {"Metric": "Invalid OHLC / negative volume rows quarantined", "Value": len(invalid_rows)},
    {"Metric": "Cleaned records retained", "Value": len(clean)},
    {"Metric": "Ticker-specific observed session gaps", "Value": len(gaps)},
    {"Metric": "Verified exchange holidays", "Value": len(verified_holidays)},
    {"Metric": "Unverified market-wide missing weekdays", "Value": len(calendar_candidates)},
    {"Metric": "DailyReturn outliers treated in adjusted column", "Value": int((outliers["Metric"] == "DailyReturn").sum()) if not outliers.empty else 0},
    {"Metric": "Volume outliers treated in adjusted column", "Value": int((outliers["Metric"] == "Volume").sum()) if not outliers.empty else 0},
    {"Metric": "Total outlier observations", "Value": len(outliers)}
])

summary.to_csv(OUT / "10_treatment_summary.csv", index=False)

print("\n========== DATA TREATMENT REPORT ==========")
print(summary.to_string(index=False))
print("\nOriginal OHLC and Volume values were not modified.")
print("Missing and invalid rows were quarantined, not fabricated.")
print("Outliers were winsorized only in additional analysis columns.")
print("Verified holidays were excluded from missing-day detection.")
print("\nOutput directory:", OUT)
print("\nTREATMENT GUIDE:")
print("- Required source fields: StockCode, Date, OHLC prices, and Volume.")
print("- Rows missing required source values are quarantined; values are not fabricated.")
print("- Duplicate key: StockCode + Date; the first occurrence is retained.")
print("- Invalid OHLC relationships and negative Volume are quarantined.")
print("- Holiday dates come from the supplied verified exchange-holiday list.")
print("- Calendar candidates are unverified alerts, not confirmed missing trading sessions.")
print("- Outlier thresholds use the per-stock IQR rule: Q1 - 1.5*IQR and Q3 + 1.5*IQR.")
print("- Winsorized values are stored in additional analysis columns; original OHLC and Volume remain unchanged.")
print("- First DailyReturn per stock is undefined because no previous Close is available.")
