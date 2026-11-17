"""Dashboard Streamlit: gia, chi bao, du doan phien ke tiep va danh gia cac mo hinh.
Chay:  streamlit run app/dashboard.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "models"):
    sys.path.insert(0, str(p))
from clf_common import FEATURES, LABEL, CLASSES, RESULTS, load_handoff, load_model_meta, evaluate, baseline_metrics  # noqa: E402
from predict import predict_frame, load_model  # noqa: E402

st.set_page_config(page_title="Du doan xu huong co phieu", layout="wide")


@st.cache_data
def load_prices(path):
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["Date"])
    df.columns = df.columns.str.strip()
    return df.sort_values(["StockCode", "Date"])


@st.cache_resource
def get_model():
    return load_model()


def indicators(g):
    g = g.copy()
    c = g["Close"]
    g["MA20"] = c.rolling(20).mean()
    d = c.diff()
    rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean()
    g["RSI14"] = 100 - 100 / (1 + rs)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    g["MACD"] = ema12 - ema26
    g["Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
    g["Hist"] = g["MACD"] - g["Signal"]
    return g


# ---------------- Du lieu ----------------
st.sidebar.header("Du lieu")
up = st.sidebar.file_uploader("CSV OHLCV moi (StockCode, Date, High, Low, Close, Volume)", type="csv")
default = next((ROOT / "data" / "raw").glob("all_stocks_*.csv"), None)
if up is not None:
    prices = load_prices(up)
elif default is not None:
    prices = load_prices(default)
else:
    st.error("Khong tim thay du lieu. Hay tai len file CSV.")
    st.stop()

meta = load_model_meta()
threshold = st.sidebar.slider("Nguong P(Tang)", 0.30, 0.70, float(meta["threshold"]), 0.01,
                              help="Mac dinh la nguong chon tren Train/CV (models/model_meta.json)")
symbol = st.sidebar.selectbox("Ma co phieu", sorted(prices["StockCode"].unique()))

st.title("Du doan xu huong phien ke tiep")
st.caption("Mo hinh hoc may chi mang tinh hoc tap, khong phai khuyen nghi dau tu. "
           "Gia mot phien rat nhieu: xem muc 'Danh gia' de biet mo hinh co vuot baseline hay khong.")

tab_pred, tab_eval, tab_models = st.tabs(["Du doan", "Danh gia mo hinh", "So sanh mo hinh & Feature"])

# ---------------- Tab 1 ----------------
with tab_pred:
    g = indicators(prices[prices["StockCode"] == symbol])
    try:
        pred = predict_frame(prices[prices["StockCode"] == symbol], get_model(), threshold).iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Du doan phien ke tiep", pred["prediction"])
        c2.metric("Xac suat Tang", f"{pred['proba_up']:.1%}")
        c3.metric("Xac suat Giam/Dung", f"{1 - pred['proba_up']:.1%}")
        st.caption(f"Du lieu den {pred['Date']:%d/%m/%Y}; nguong = {threshold:.2f}.")
    except ValueError as e:
        st.warning(str(e))

    show = g.tail(st.sidebar.slider("So phien hien thi", 60, 500, 180, 20))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.55, .2, .25], vertical_spacing=.03,
                        subplot_titles=("Gia & MA20", "RSI(14)", "MACD"))
    fig.add_trace(go.Candlestick(x=show["Date"], open=show.get("Open", show["Close"]), high=show["High"],
                                 low=show["Low"], close=show["Close"], name="Gia"), 1, 1)
    fig.add_trace(go.Scatter(x=show["Date"], y=show["MA20"], name="MA20"), 1, 1)
    fig.add_trace(go.Scatter(x=show["Date"], y=show["RSI14"], name="RSI14"), 2, 1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Bar(x=show["Date"], y=show["Hist"], name="Histogram"), 3, 1)
    fig.add_trace(go.Scatter(x=show["Date"], y=show["MACD"], name="MACD"), 3, 1)
    fig.add_trace(go.Scatter(x=show["Date"], y=show["Signal"], name="Signal"), 3, 1)
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Du doan cho tat ca ma (phien moi nhat) & xuat CSV"):
        allp = predict_frame(prices, get_model(), threshold)
        st.dataframe(allp[["StockCode", "Date", "proba_up", "prediction"]], hide_index=True)
        st.download_button("Tai predictions.csv", allp.to_csv(index=False).encode("utf-8-sig"), "predictions.csv")

# ---------------- Tab 2 ----------------
with tab_eval:
    train, test = load_handoff()
    proba = get_model().predict_proba(test[FEATURES])[:, 1]
    pred_t = (proba > threshold).astype(int)
    m, b = evaluate(test[LABEL], pred_t, proba), baseline_metrics(train[LABEL], test[LABEL])
    st.subheader(f"Mo hinh {meta['model']} tren tap Test ({len(test)} mau)")
    table = pd.DataFrame({"Mo hinh": m, "Baseline (luon doan lop da so Train)": b}).round(3)
    st.dataframe(table)
    if m["Accuracy"] <= b["Accuracy"]:
        st.warning(f"Accuracy {m['Accuracy']:.1%} chua vuot baseline {b['Accuracy']:.1%}. "
                   "Hay doc results/ANALYSIS.md de biet nguyen nhan.")
    cm = confusion_matrix(test[LABEL], pred_t)
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Doan: {c}" for c in CLASSES], y=[f"Thuc: {c}" for c in CLASSES],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed"); fig.update_layout(title="Confusion Matrix (Test)", height=360)
    st.plotly_chart(fig, width="stretch")

# ---------------- Tab 3 ----------------
with tab_models:
    f = RESULTS / "model_comparison.csv"
    if f.exists():
        st.subheader("So sanh cac mo hinh (Test)")
        st.dataframe(pd.read_csv(f, encoding="utf-8-sig", index_col=0).round(3))
    f = RESULTS / "analysis_feature_contribution.csv"
    if f.exists():
        st.subheader("Dong gop cua tung feature (CV tren Train)")
        fc = pd.read_csv(f, encoding="utf-8-sig", index_col=0)
        st.bar_chart(fc["Permutation (giam AUC)"])
        st.dataframe(fc)
    else:
        coef = pd.Series(get_model().named_steps["model"].coef_[0], index=FEATURES)
        st.subheader("He so Logistic Regression"); st.bar_chart(coef)
    f = RESULTS / "analysis_final_test.csv"
    if f.exists():
        st.subheader("Mo hinh sau chon feature + toi uu sieu tham so + nguong (chon tren Train/CV)")
        st.dataframe(pd.read_csv(f, encoding="utf-8-sig", index_col=0))
