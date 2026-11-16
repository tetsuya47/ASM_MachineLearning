"""Tao dac trung TUONG DOI + nhan nhi phan (Tang / Giam-Dung gia) cho bai toan phan loai.

Khac voi feature_engineering.py (gia tuyet doi), cac dac trung o day khong phu thuoc muc gia
nen so sanh duoc giua cac ma (VIC 100k+ vs VPB ~20k).
Nhan: 1 = Tang (return phien ke tiep > 0), 0 = Giam hoac dung gia.
"""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "return", "ret_lag2", "ret_lag3", "ret5",
    "ma_ratio", "close_vs_ma20", "range_pct", "close_pos",
    "volume_change", "vol_ratio", "vol10", "rsi14",
]
LABEL = "label"


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_classification_data(raw: pd.DataFrame) -> pd.DataFrame:
    need = {"StockCode", "Date", "High", "Low", "Close", "Volume"}
    if need - set(raw.columns):
        raise ValueError(f"Thieu cot: {sorted(need - set(raw.columns))}")
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["StockCode", "Date", "High", "Low", "Close", "Volume"])
            .sort_values(["StockCode", "Date"])
            .drop_duplicates(["StockCode", "Date"], keep="last")
            .reset_index(drop=True))

    out = []
    for code, g in df.groupby("StockCode", sort=False):
        g = g.copy()
        c, v = g["Close"], g["Volume"]
        g["return"] = c.pct_change()
        g["ret_lag2"] = g["return"].shift(1)
        g["ret_lag3"] = g["return"].shift(2)
        g["ret5"] = c.pct_change(5)
        ma5, ma20 = c.rolling(5).mean(), c.rolling(20).mean()
        g["ma_ratio"] = ma5 / ma20
        g["close_vs_ma20"] = c / ma20 - 1
        g["range_pct"] = (g["High"] - g["Low"]) / c
        g["close_pos"] = (c - g["Low"]) / (g["High"] - g["Low"]).replace(0, np.nan)
        g["volume_change"] = v.pct_change()
        g["vol_ratio"] = v / v.rolling(20).mean()
        g["vol10"] = g["return"].rolling(10).std()
        g["rsi14"] = _rsi(c)
        g["next_return"] = g["return"].shift(-1)       # chi dung de tao nhan / backtest
        g["target_date"] = g["Date"].shift(-1)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["next_return"])
    df[LABEL] = (df["next_return"] > 0).astype(int)
    return df.sort_values(["Date", "StockCode"]).reset_index(drop=True)


def split_by_date(df: pd.DataFrame, test_size=0.2):
    """Chia theo thoi gian; bo ngay cuoi cua train (nhan cua no roi vao test)."""
    dates = np.array(sorted(df["Date"].unique()))
    pos = int(len(dates) * (1 - test_size))
    test_start = pd.Timestamp(dates[pos])
    train_last = pd.Timestamp(dates[pos - 2])
    return df[df["Date"] <= train_last].copy(), df[df["Date"] >= test_start].copy()


def run(input_csv="data/raw/all_stocks_20241113_20261113.csv", out_dir="data/processed", test_size=0.2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_classification_data(pd.read_csv(input_csv))
    train, test = split_by_date(df, test_size)
    cols = ["StockCode", "Date"] + FEATURES + ["next_return", LABEL]
    train[cols].to_csv(out_dir / "clf_train.csv", index=False, encoding="utf-8-sig")
    test[cols].to_csv(out_dir / "clf_test.csv", index=False, encoding="utf-8-sig")
    print(f"Train: {len(train)} dong ({train.Date.min().date()} -> {train.Date.max().date()})")
    print(f"Test : {len(test)} dong ({test.Date.min().date()} -> {test.Date.max().date()})")
    print("Ty le Tang - train:", round(train[LABEL].mean(), 3), "| test:", round(test[LABEL].mean(), 3))
    return train, test


if __name__ == "__main__":
    run()
