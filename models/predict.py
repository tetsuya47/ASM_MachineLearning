"""Du doan xu huong phien ke tiep (Tang / Giam-Dung) cho du lieu co phieu MOI.

Quy trinh: doc OHLCV -> lam sach -> tinh 12 feature (dung chung code Nguoi 2) -> nap mo hinh .joblib
-> P(Tang) -> so voi nguong -> xuat CSV.

Dau vao: CSV co cot StockCode, Date, High, Low, Close, Volume (Open tuy chon); moi ma can >= 20 phien lich su.
Chay:
    python models/predict.py --input data/raw/all_stocks_20241113_20261113.csv --output results/predictions.csv
    python models/predict.py --input new.csv --all-dates      # du doan moi phien, khong chi phien moi nhat
"""
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd
from clf_common import FEATURES, CLASSES, MODEL_PATH, RESULTS, load_model_meta
from classification_features import clean_dataframe, build_classification_data

MIN_HISTORY = 20   # ma20 / vol_ratio can 20 phien


def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chua co mo hinh {path}. Chay: python models/logistic_regression.py")
    return joblib.load(path)


def predict_frame(raw: pd.DataFrame, model=None, threshold=None, latest_only=True) -> pd.DataFrame:
    """raw: DataFrame OHLCV. Tra ve DataFrame: StockCode, Date, <12 feature>, proba_up, label, prediction."""
    model = model if model is not None else load_model()
    threshold = load_model_meta()["threshold"] if threshold is None else threshold
    clean = clean_dataframe(raw)
    feats = build_classification_data(clean, for_inference=True)
    if feats.empty:
        n = clean.groupby("StockCode").size().max()
        raise ValueError(f"Khong du lich su de tinh feature (can >= {MIN_HISTORY} phien/ma, hien toi da {n}).")
    if latest_only:
        feats = feats.sort_values("Date").groupby("StockCode", as_index=False).tail(1)
    out = feats[["StockCode", "Date"] + FEATURES].reset_index(drop=True)
    out["proba_up"] = model.predict_proba(out[FEATURES])[:, 1]
    out["label"] = (out["proba_up"] > threshold).astype(int)
    out["prediction"] = out["label"].map(dict(enumerate(CLASSES)))
    out["threshold"] = threshold
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV OHLCV cua co phieu moi")
    ap.add_argument("--output", default=str(RESULTS / "predictions.csv"))
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--threshold", type=float, default=None, help="Mac dinh: nguong trong models/model_meta.json")
    ap.add_argument("--all-dates", action="store_true", help="Du doan cho moi ngay thay vi chi phien moi nhat")
    a = ap.parse_args(argv)

    out = predict_frame(pd.read_csv(a.input, encoding="utf-8-sig"), load_model(a.model), a.threshold,
                        latest_only=not a.all_dates)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False, encoding="utf-8-sig")
    show = out.groupby("StockCode").tail(1) if a.all_dates else out
    show = show[["StockCode", "Date", "proba_up", "prediction"]].assign(proba_up=lambda d: d["proba_up"].round(3))
    print(show.to_string(index=False))
    print(f"\nDa xuat {len(out)} dong -> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
