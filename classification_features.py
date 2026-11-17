"""Nguoi 2 - Feature Engineering va ban giao du lieu Classification.

- Doc du lieu sach tu visualizations/10_cleaned_data.csv
- Tinh dac trung rieng tung ma co phieu, chi dung du lieu hien tai/qua khu.
- label=1 neu Close phien giao dich tiep theo > Close hien tai; nguoc lai label=0.
- Chia theo ngay giao dich chung; purge mau Train co target_date cham sang Test.
- StandardScaler chi fit tren X_train; X_test chi transform.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "return", "ret_lag1", "ret_lag2", "ret5",
    "ma_ratio", "close_vs_ma20", "range_pct", "close_pos",
    "volume_change", "vol_ratio", "vol10", "rsi14",
]
LABEL = "label"
META = ["StockCode", "Date", "target_date"]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    # Edge cases: all gains -> 100; no move -> 50.
    result = result.mask((loss == 0) & (gain > 0), 100)
    result = result.mask((loss == 0) & (gain == 0), 50)
    return result


def load_clean_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay du lieu sach cua Nguoi 1: {path}")
    return clean_dataframe(pd.read_csv(path, encoding="utf-8-sig"))


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Chuan hoa/kiem tra DataFrame OHLCV (dung chung cho huan luyen va du doan du lieu moi)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = {"StockCode", "Date", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Du lieu sach thieu cot bat buoc: {sorted(missing)}")
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    for col in ["High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["StockCode", "Date", "High", "Low", "Close", "Volume"])
    df = df[(df["Close"] > 0) & (df["High"] >= df["Low"]) & (df["Volume"] >= 0)]
    df = df.sort_values(["StockCode", "Date"]).drop_duplicates(
        ["StockCode", "Date"], keep="last"
    ).reset_index(drop=True)
    if df.empty:
        raise ValueError("Du lieu sach khong con ban ghi hop le.")
    return df


def build_classification_data(clean: pd.DataFrame, for_inference: bool = False) -> pd.DataFrame:
    """Feature engineering theo tung StockCode; label la huong phien tiep theo.

    for_inference=True: giu cac dong moi nhat chua co phien ke tiep (label/target_date = NaN)
    de predict.py dung cung 12 feature voi luc huan luyen.
    """
    df = clean.copy()
    output = []
    for symbol, group in df.groupby("StockCode", sort=False):
        g = group.sort_values("Date").copy()
        close = g["Close"].astype(float)
        volume = g["Volume"].astype(float)
        # Nguoi 1 cung ban giao cot winsorized rieng cho phan tich outlier.
        # Dung cot dieu chinh neu co; cot DailyReturn_Winsorized dang theo don vi %.
        if "DailyReturn_Winsorized" in g.columns:
            daily_return = pd.to_numeric(g["DailyReturn_Winsorized"], errors="coerce") / 100.0
            daily_return = daily_return.fillna(close.pct_change())
        else:
            daily_return = close.pct_change()

        volume_feature = (
            pd.to_numeric(g["Volume_Winsorized"], errors="coerce")
            if "Volume_Winsorized" in g.columns else volume
        )
        volume_feature = volume_feature.fillna(volume)

        g["return"] = daily_return
        g["ret_lag1"] = daily_return.shift(1)
        g["ret_lag2"] = daily_return.shift(2)
        g["ret5"] = (1 + daily_return).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1

        ma5 = close.rolling(5, min_periods=5).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        g["ma_ratio"] = ma5 / ma20
        g["close_vs_ma20"] = close / ma20 - 1
        g["range_pct"] = (g["High"] - g["Low"]) / close
        intraday_range = (g["High"] - g["Low"]).replace(0, np.nan)
        g["close_pos"] = ((close - g["Low"]) / intraday_range).fillna(0.5)
        g["volume_change"] = volume_feature.pct_change()
        g["vol_ratio"] = volume_feature / volume_feature.rolling(20, min_periods=20).mean()
        g["vol10"] = daily_return.rolling(10, min_periods=10).std()
        g["rsi14"] = _rsi(close, 14)

        next_close = close.shift(-1)
        g["target_date"] = g["Date"].shift(-1)
        # Label 1 = next session rises; label 0 = falls or unchanged.
        g[LABEL] = np.where(next_close.notna(), (next_close > close).astype(int), np.nan)
        output.append(g)

    result = pd.concat(output, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan)
    if for_inference:
        return result.dropna(subset=FEATURES).sort_values(["Date", "StockCode"]).reset_index(drop=True)
    result = result.dropna(subset=FEATURES + ["target_date", LABEL]).copy()
    result[LABEL] = result[LABEL].astype("int8")
    return result.sort_values(["Date", "StockCode"]).reset_index(drop=True)


def chronological_split(data: pd.DataFrame, test_size: float = 0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size phai nam trong khoang (0, 1).")
    dates = np.array(sorted(pd.to_datetime(data["Date"]).dt.normalize().unique()))
    if len(dates) < 3:
        raise ValueError("Khong du ngay giao dich de chia Train/Test.")
    split_idx = int(len(dates) * (1 - test_size))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    test_start = pd.Timestamp(dates[split_idx])

    # Purge: Train sample is retained only if its next-session label is still before Test.
    train = data[
        (data["Date"] < test_start) & (data["target_date"] < test_start)
    ].copy()
    test = data[data["Date"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Train/Test rong sau khi chia theo ngay.")
    overlap = set(train["Date"]).intersection(set(test["Date"]))
    if overlap:
        raise AssertionError(f"Ngay overlap phat hien: {sorted(overlap)[:5]}")
    if not (train["target_date"] < test_start).all():
        raise AssertionError("Label cua Train co the nhin sang giai doan Test.")
    return train.reset_index(drop=True), test.reset_index(drop=True), test_start


def run_pipeline(input_csv, output_dir, test_size=0.2):
    clean = load_clean_data(input_csv)
    features = build_classification_data(clean)
    train_raw, test_raw, test_start = chronological_split(features, test_size)

    scaler = StandardScaler()
    train_scaled = train_raw.copy()
    test_scaled = test_raw.copy()
    train_scaled.loc[:, FEATURES] = scaler.fit_transform(train_raw[FEATURES])
    test_scaled.loc[:, FEATURES] = scaler.transform(test_raw[FEATURES])

    # Main handoff files remain raw features so Person 3 can use fold-local scaling
    # inside a time-aware CV Pipeline. Scaled versions are supplied separately.
    columns = META + FEATURES + [LABEL]
    train_handoff = train_raw[columns].copy()
    test_handoff = test_raw[columns].copy()
    train_scaled_handoff = train_scaled[columns].copy()
    test_scaled_handoff = test_scaled[columns].copy()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / "classification_features.csv", index=False, encoding="utf-8-sig")
    train_handoff.to_csv(out / "clf_train.csv", index=False, encoding="utf-8-sig")
    test_handoff.to_csv(out / "clf_test.csv", index=False, encoding="utf-8-sig")
    train_scaled_handoff.to_csv(out / "clf_train_scaled.csv", index=False, encoding="utf-8-sig")
    test_scaled_handoff.to_csv(out / "clf_test_scaled.csv", index=False, encoding="utf-8-sig")
    joblib.dump(scaler, out / "standard_scaler.joblib")

    schema = {
        "owner": "Nguoi 2 - Feature Engineering",
        "source_file": "visualizations/10_cleaned_data.csv" if Path(input_csv).name == "10_cleaned_data.csv" else Path(input_csv).name,
        "task": "binary_classification_next_session_direction",
        "feature_columns": FEATURES,
        "label_column": LABEL,
        "label_definition": "1 neu Close phien giao dich tiep theo cung ma > Close hien tai; nguoc lai 0",
        "metadata_columns": META,
        "main_handoff_files": {
            "train": "clf_train.csv (feature chua scale, label 0/1)",
            "test": "clf_test.csv (feature chua scale, label 0/1)",
        },
        "scaled_files": {
            "train": "clf_train_scaled.csv",
            "test": "clf_test_scaled.csv",
            "scaler": "standard_scaler.joblib",
        },
        "scaler_rule": "StandardScaler.fit tren X_train; X_test chi duoc transform",
        "split_rule": "Chia theo ngay giao dich duy nhat tren tat ca ma; khong shuffle; purge Train neu target_date cham sang Test",
        "test_start_date": str(test_start.date()),
        "train_last_feature_date": str(train_handoff["Date"].max().date()),
        "test_first_feature_date": str(test_handoff["Date"].min().date()),
        "train_rows": len(train_handoff),
        "test_rows": len(test_handoff),
        "symbols": sorted(features["StockCode"].unique().tolist()),
        "train_label_distribution": {str(k): int(v) for k, v in train_handoff[LABEL].value_counts().sort_index().items()},
        "test_label_distribution": {str(k): int(v) for k, v in test_handoff[LABEL].value_counts().sort_index().items()},
    }
    (out / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    qa = {
        "clean_input_rows": int(len(clean)),
        "feature_rows": int(len(features)),
        "symbols": int(features["StockCode"].nunique()),
        "train_rows": int(len(train_handoff)),
        "test_rows": int(len(test_handoff)),
        "train_test_date_overlap": int(len(set(train_handoff["Date"]) & set(test_handoff["Date"]))),
        "train_target_dates_before_test": bool((pd.to_datetime(train_handoff["target_date"]) < test_start).all()),
        "train_feature_means_after_scaling": {
            k: float(v) for k, v in train_scaled_handoff[FEATURES].mean().round(10).items()
        },
        "label_values": sorted(features[LABEL].unique().astype(int).tolist()),
    }
    (out / "quality_checks.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "clean": clean, "features": features,
        "train": train_handoff, "test": test_handoff,
        "train_scaled": train_scaled_handoff, "test_scaled": test_scaled_handoff,
        "scaler": scaler, "schema": schema, "qa": qa,
        "test_start": test_start, "output_dir": out,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    source = root / "visualizations" / "10_cleaned_data.csv"
    output = root / "data" / "processed"
    result = run_pipeline(source, output, test_size=0.2)
    print("\n========== NGUOI 2 - CLASSIFICATION FEATURE ENGINEERING ==========")
    print("Du lieu dau vao:", source)
    print("So dong cleaned:", len(result["clean"]))
    print("So dong sau Feature Engineering:", len(result["features"]))
    print("So ma co phieu:", result["features"]["StockCode"].nunique())
    print("So dac trung:", len(FEATURES), "-", ", ".join(FEATURES))
    print("Label: 1 = Tang phien tiep theo; 0 = Giam/di ngang")
    print("\n========== TRAIN / TEST ==========")
    print("Train:", len(result["train"]))
    print("Test:", len(result["test"]))
    print("Ngay cuoi Train:", result["train"]["Date"].max())
    print("Ngay dau Test:", result["test"]["Date"].min())
    print("Ngay giao nhau:", result["qa"]["train_test_date_overlap"])
    print("Phan phoi label Train:", result["schema"]["train_label_distribution"])
    print("Phan phoi label Test:", result["schema"]["test_label_distribution"])
    print("\nScaler fit tren Train; Test chi transform.")
    print("Da luu bo ban giao tai:", result["output_dir"])
