import pandas as pd
import pytest
from classification_features import clean_dataframe, load_clean_data
from clf_common import CLEAN_CSV


def _raw(**over):
    d = {"StockCode": [" vcb", "VCB", "VCB"], "Date": ["2025-01-02", "2025-01-03", "2025-01-03"],
         "High": [11, 12, 13], "Low": [9, 10, 11], "Close": [10, 11, 12], "Volume": [100, 200, 300]}
    d.update(over)
    return pd.DataFrame(d)


def test_clean_file_has_required_columns_and_no_missing():
    df = load_clean_data(CLEAN_CSV)
    assert df[["StockCode", "Date", "High", "Low", "Close", "Volume"]].notna().all().all()
    assert not df.duplicated(["StockCode", "Date"]).any()
    assert (df["Close"] > 0).all() and (df["High"] >= df["Low"]).all()


def test_clean_dataframe_normalises_and_dedups():
    out = clean_dataframe(_raw())
    assert set(out["StockCode"]) == {"VCB"}
    assert len(out) == 2                      # trung (VCB, 2025-01-03) bi loai
    assert out.loc[out["Date"] == "2025-01-03", "Close"].iloc[0] == 12   # giu ban ghi cuoi


def test_invalid_rows_removed():
    out = clean_dataframe(_raw(Close=[10, -1, 12], Volume=[100, 200, None]))
    assert len(out) == 1


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        clean_dataframe(_raw().drop(columns=["Volume"]))
