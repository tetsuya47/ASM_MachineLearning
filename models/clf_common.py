"""Phan dung chung cho Nguoi 3 (mo hinh hoa). Tai su dung code cua Nguoi 2, khong tao lai feature/label.

- Doc dung clf_train.csv / clf_test.csv theo hop dong du lieu (feature_schema.json).
- CV chia theo NGAY giao dich (10 ma cung ngay nam cung fold), purge mau co target_date cham sang validation.
- Scaler duoc dat trong Pipeline cua tung script de fit rieng tren fold-train.
"""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

if sys.version_info[:2] != (3, 12):
    warnings.warn(f"Du an chuan hoa Python 3.12, dang chay {sys.version.split()[0]}.", RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classification_features import FEATURES, LABEL, load_clean_data  # noqa: E402  (code cua Nguoi 2)

PROCESSED = ROOT / "data" / "processed"
CLEAN_CSV = ROOT / "visualizations" / "10_cleaned_data.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CLASSES = ["Giam/Dung", "Tang"]
SEED = 42


def load_handoff():
    """Doc clf_train.csv, clf_test.csv; sap xep theo thoi gian; kiem tra hop dong du lieu."""
    out = []
    for name in ("clf_train.csv", "clf_test.csv"):
        df = pd.read_csv(PROCESSED / name, parse_dates=["Date", "target_date"], encoding="utf-8-sig")
        missing = set(FEATURES + [LABEL]) - set(df.columns)
        if missing:
            raise ValueError(f"{name} thieu cot: {sorted(missing)}")
        out.append(df.sort_values(["Date", "StockCode"], kind="stable").reset_index(drop=True))
    train, test = out
    if set(train["Date"]) & set(test["Date"]):
        raise ValueError("Train va Test co ngay trung nhau.")
    if train["target_date"].max() >= test["Date"].min():
        raise ValueError("target_date cua Train cham sang giai doan Test.")
    return train, test


def date_cv_splits(df, n_splits=5):
    """TimeSeriesSplit theo ngay giao dich + purge theo target_date. df phai da reset_index (vi tri 0..n-1)."""
    dates = pd.DatetimeIndex(sorted(df["Date"].unique()))
    splits = []
    for tr_d, va_d in TimeSeriesSplit(n_splits=n_splits).split(dates):
        val_start = dates[va_d[0]]
        tr = np.flatnonzero(df["Date"].isin(dates[tr_d]) & (df["target_date"] < val_start))
        va = np.flatnonzero(df["Date"].isin(dates[va_d]))
        splits.append((tr, va))
    return splits


def majority_class(y_train):
    """Lop da so xac dinh tu TRAIN (khong nhin phan phoi Test)."""
    return int(pd.Series(y_train).mean() >= 0.5)


def baseline_metrics(y_train, y_test):
    """Metric THAT cua baseline 'luon doan lop da so cua Train' tren Test (tinh bang evaluate, khong gan tay)."""
    pred = np.full(len(y_test), majority_class(y_train))
    return evaluate(y_test, pred)


def evaluate(y, pred, proba=None):
    row = {
        "Accuracy": accuracy_score(y, pred),
        "Balanced Acc": balanced_accuracy_score(y, pred),
        "Precision(Tang)": precision_score(y, pred, zero_division=0),
        "Recall(Tang)": recall_score(y, pred, zero_division=0),
        "F1-macro": f1_score(y, pred, average="macro"),
    }
    if proba is not None and len(np.unique(y)) > 1:
        row["ROC-AUC"] = roc_auc_score(y, proba)
    return row


def attach_next_return(df):
    """Them cot next_return = Close(target_date)/Close(Date)-1 tu du lieu sach, CHI de backtest
    (khong dung lam feature/label; label van la Close phien ke tiep > Close hien tai)."""
    clean = load_clean_data(CLEAN_CSV)[["StockCode", "Date", "Close"]]
    now = clean.rename(columns={"Close": "Close_now"})
    nxt = clean.rename(columns={"Date": "target_date", "Close": "Close_next"})
    out = df.merge(now, on=["StockCode", "Date"], how="left").merge(nxt, on=["StockCode", "target_date"], how="left")
    out["next_return"] = out["Close_next"] / out["Close_now"] - 1
    return out


MODEL_PATH = ROOT / "models" / "logreg_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"


def save_model_meta(threshold, model_name, path=META_PATH):
    """Luu nguong du doan (chon tren Train/CV) canh file .joblib de predict.py/dashboard dung lai."""
    path.write_text(json.dumps({"model": model_name, "threshold": float(threshold), "features": FEATURES,
                                "classes": CLASSES}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_meta(path=META_PATH):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"model": "LogReg", "threshold": 0.5, "features": FEATURES, "classes": CLASSES}
