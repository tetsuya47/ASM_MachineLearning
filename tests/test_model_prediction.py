import json
import joblib
import numpy as np
import pandas as pd
import pytest
from clf_common import MODEL_PATH, META_PATH, FEATURES, CLASSES
from predict import predict_frame, load_model, main

RAW = None


@pytest.fixture(scope="module")
def raw():
    from pathlib import Path
    f = next(Path(__file__).resolve().parents[1].joinpath("data", "raw").glob("all_stocks_*.csv"))
    return pd.read_csv(f, encoding="utf-8-sig")


def test_model_reloads_from_joblib():
    m = joblib.load(MODEL_PATH)
    p = m.predict_proba(pd.DataFrame(np.zeros((2, len(FEATURES))), columns=FEATURES))
    assert p.shape == (2, 2) and np.allclose(p.sum(1), 1)


def test_meta_threshold_valid():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert 0 < meta["threshold"] < 1 and meta["features"] == FEATURES


def test_predict_latest_format(raw):
    out = predict_frame(raw, load_model())
    assert len(out) == raw["StockCode"].nunique()
    assert out["proba_up"].between(0, 1).all()
    assert set(out["label"]) <= {0, 1} and set(out["prediction"]) <= set(CLASSES)
    assert (out["label"] == (out["proba_up"] > out["threshold"]).astype(int)).all()
    assert out[FEATURES].notna().all().all()


def test_predict_all_dates_and_threshold_override(raw):
    all_ = predict_frame(raw, threshold=0.0, latest_only=False)
    assert len(all_) > raw["StockCode"].nunique() and (all_["label"] == 1).all()


def test_predict_too_little_history_raises(raw):
    with pytest.raises(ValueError):
        predict_frame(raw[raw["StockCode"] == "VCB"].head(10))


def test_cli_writes_csv(raw, tmp_path):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    raw.to_csv(src, index=False)
    assert main(["--input", str(src), "--output", str(dst)]) == 0
    assert {"StockCode", "Date", "proba_up", "prediction"} <= set(pd.read_csv(dst).columns)
