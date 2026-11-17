import json
import numpy as np
import pandas as pd
from classification_features import FEATURES, LABEL, load_clean_data, build_classification_data
from clf_common import CLEAN_CSV, PROCESSED, load_handoff


def test_twelve_features_no_missing_in_handoff():
    train, test = load_handoff()
    assert len(FEATURES) == 12
    for df in (train, test):
        assert df[FEATURES].notna().all().all()
        assert np.isfinite(df[FEATURES].to_numpy()).all()


def test_schema_matches_code():
    schema = json.loads((PROCESSED / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["feature_columns"] == FEATURES


def test_label_only_zero_one_and_matches_next_close():
    train, test = load_handoff()
    for df in (train, test):
        assert set(df[LABEL].unique()) <= {0, 1}
    clean = load_clean_data(CLEAN_CSV)[["StockCode", "Date", "Close"]]
    d = train.merge(clean, on=["StockCode", "Date"]).merge(
        clean.rename(columns={"Date": "target_date", "Close": "Close_next"}), on=["StockCode", "target_date"])
    assert ((d["Close_next"] > d["Close"]).astype(int) == d[LABEL]).all()


def test_handoff_reproducible_from_clean_data():
    """Tao lai feature tu du lieu sach phai khop file ban giao (khong sinh them logic moi)."""
    feats = build_classification_data(load_clean_data(CLEAN_CSV))
    saved = pd.read_csv(PROCESSED / "classification_features.csv", parse_dates=["Date"])
    assert len(feats) == len(saved)
    np.testing.assert_allclose(feats[FEATURES].to_numpy(), saved[FEATURES].to_numpy(), atol=1e-9)
