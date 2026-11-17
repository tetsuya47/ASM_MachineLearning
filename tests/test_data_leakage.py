import numpy as np
import pandas as pd
from classification_features import FEATURES, META, LABEL, load_clean_data, build_classification_data
from clf_common import CLEAN_CSV, load_handoff, date_cv_splits


def test_no_overlapping_dates_and_train_before_test():
    train, test = load_handoff()
    assert not set(train["Date"]) & set(test["Date"])
    assert train["Date"].max() < test["Date"].min()
    assert train["target_date"].max() < test["Date"].min()


def test_meta_and_label_not_in_features():
    assert not set(FEATURES) & set(META + [LABEL])


def test_cv_folds_purged():
    train, _ = load_handoff()
    for tr, va in date_cv_splits(train, 5):
        assert train.loc[tr, "target_date"].max() < train.loc[va, "Date"].min()
        assert not set(train.loc[tr, "Date"]) & set(train.loc[va, "Date"])


def test_features_ignore_the_future():
    """Doi gia cac phien SAU ngay t khong duoc lam doi feature cua ngay t va truoc do."""
    clean = load_clean_data(CLEAN_CSV)
    cut = clean["Date"].sort_values().unique()[300]
    base = build_classification_data(clean, for_inference=True)
    mod = clean.copy()
    mod["Volume"] = mod["Volume"].astype(float)
    after = mod["Date"] > cut
    mod.loc[after, ["High", "Low", "Close", "Volume"]] *= 1.37
    for c in ("DailyReturn_Winsorized", "Volume_Winsorized"):
        if c in mod:
            mod.loc[after, c] = mod.loc[after, c] * 1.37
    alt = build_classification_data(mod, for_inference=True)
    a = base[base["Date"] < cut].set_index(["StockCode", "Date"])[FEATURES]
    b = alt[alt["Date"] < cut].set_index(["StockCode", "Date"])[FEATURES]
    pd.testing.assert_frame_equal(a, b.loc[a.index])
