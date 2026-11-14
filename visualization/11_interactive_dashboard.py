from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "all_stocks_20241113_20261113.csv"
OUT_DIR = ROOT / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading stock data...")
df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Date", "StockCode", "Close"])
df = df.sort_values(["StockCode", "Date"])
df = df.drop_duplicates(["StockCode", "Date"], keep="last")

df["DailyReturn"] = df.groupby("StockCode")["Close"].pct_change() * 100
df["Month"] = df["Date"].dt.to_period("M").astype(str)

stocks = sorted(df["StockCode"].unique())
print(f"Loaded {len(df):,} rows | {len(stocks)} stocks")
print(f"Period: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ---------------------------------------------------------
# 1. PERFORMANCE COMPARISON - BASE 100
# ---------------------------------------------------------
performance = df.copy()
performance["IndexedPrice"] = (
    performance["Close"]
    / performance.groupby("StockCode")["Close"].transform("first")
    * 100
)

fig = px.line(
    performance,
    x="Date",
    y="IndexedPrice",
    color="StockCode",
    title="Stock Performance Comparison - Base 100",
    hover_data={
        "Date": "|%Y-%m-%d",
        "StockCode": True,
        "Close": ":,.2f",
        "IndexedPrice": ":.2f"
    },
    labels={
        "Date": "Trading date",
        "IndexedPrice": "Indexed price",
        "StockCode": "Stock"
    }
)
fig.add_hline(y=100, line_dash="dash", annotation_text="Starting value = 100")
fig.update_layout(
    hovermode="x unified",
    yaxis_title="Indexed price (base 100)",
    legend_title="Stock code"
)
fig.add_annotation(
    text="Interpretation: values above 100 indicate growth from the first observation.",
    xref="paper", yref="paper", x=0, y=-0.18, showarrow=False
)
fig.write_html(OUT_DIR / "11_01_performance_comparison.html", include_plotlyjs=True)

# ---------------------------------------------------------
# 2. MONTHLY RETURN HEATMAP
# ---------------------------------------------------------
monthly = (
    df.sort_values("Date")
      .groupby(["StockCode", "Month"])["Close"]
      .last()
      .reset_index()
)
monthly["MonthlyReturn"] = (
    monthly.groupby("StockCode")["Close"].pct_change() * 100
)
heat = monthly.pivot(
    index="StockCode",
    columns="Month",
    values="MonthlyReturn"
)

fig = px.imshow(
    heat,
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    aspect="auto",
    text_auto=".1f",
    title="Monthly Stock Returns Heatmap",
    labels={
        "x": "Month",
        "y": "Stock",
        "color": "Return (%)"
    }
)
fig.update_layout(
    hoverlabel=dict(namelength=-1),
    xaxis_title="Month",
    yaxis_title="Stock code"
)
fig.add_annotation(
    text="Green: positive return | Red: negative return | Values are monthly percentage changes.",
    xref="paper", yref="paper", x=0, y=-0.18, showarrow=False
)
fig.write_html(OUT_DIR / "11_02_monthly_return_heatmap.html", include_plotlyjs=True)

# ---------------------------------------------------------
# 3. CORRELATION HEATMAP
# ---------------------------------------------------------
returns_matrix = df.pivot(
    index="Date", columns="StockCode", values="DailyReturn"
)
corr = returns_matrix.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu_r",
    title="Daily Return Correlation Between Stocks",
    labels={"color": "Pearson correlation"}
)
fig.update_layout(
    xaxis_title="Stock code",
    yaxis_title="Stock code"
)
fig.add_annotation(
    text="Near +1: move in the same direction | Near -1: move in opposite directions | Near 0: weak linear relationship.",
    xref="paper", yref="paper", x=0, y=-0.18, showarrow=False
)
fig.write_html(OUT_DIR / "11_03_correlation_heatmap.html", include_plotlyjs=True)

# ---------------------------------------------------------
# 4. RETURN DISTRIBUTION - BOX PLOT
# ---------------------------------------------------------
fig = px.box(
    df.dropna(subset=["DailyReturn"]),
    x="StockCode",
    y="DailyReturn",
    color="StockCode",
    points="outliers",
    title="Daily Return Distribution by Stock",
    hover_data=["Date", "Close", "Volume"],
    labels={
        "StockCode": "Stock",
        "DailyReturn": "Daily return (%)"
    }
)
fig.add_hline(y=0, line_dash="dash")
fig.update_layout(
    showlegend=False,
    hovermode="closest",
    yaxis_title="Daily return (%)"
)
fig.add_annotation(
    text="Center line: median | Box: 25th-75th percentile | Points: potential extreme observations.",
    xref="paper", yref="paper", x=0, y=-0.18, showarrow=False
)
fig.write_html(OUT_DIR / "11_04_return_distribution.html", include_plotlyjs=True)

# ---------------------------------------------------------
# 5. RISK DASHBOARD - ROLLING VOLATILITY AND DRAWDOWN
# ---------------------------------------------------------
risk = df.copy()
risk["RollingVolatility"] = (
    risk.groupby("StockCode")["DailyReturn"]
        .transform(lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252))
)
risk["RollingVolatility"] *= 100

risk["RunningPeak"] = risk.groupby("StockCode")["Close"].cummax()
risk["Drawdown"] = (
    risk["Close"] / risk["RunningPeak"] - 1
) * 100

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=(
        "20-session Annualized Rolling Volatility",
        "Drawdown from Historical Running Peak"
    )
)

for stock in stocks:
    part = risk[risk["StockCode"] == stock]
    fig.add_trace(
        go.Scatter(
            x=part["Date"],
            y=part["RollingVolatility"],
            name=stock,
            mode="lines",
            legendgroup=stock,
            hovertemplate=(
                "Stock: " + stock +
                "<br>Date: %{x|%Y-%m-%d}" +
                "<br>Annualized volatility: %{y:.2f}%<extra></extra>"
            )
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=part["Date"],
            y=part["Drawdown"],
            name=stock,
            mode="lines",
            legendgroup=stock,
            showlegend=False,
            hovertemplate=(
                "Stock: " + stock +
                "<br>Date: %{x|%Y-%m-%d}" +
                "<br>Drawdown: %{y:.2f}%<extra></extra>"
            )
        ),
        row=2, col=1
    )

fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
fig.update_layout(
    title="Stock Risk Analysis",
    hovermode="x unified",
    height=850,
    legend_title="Stock code"
)
fig.write_html(OUT_DIR / "11_05_risk_dashboard.html", include_plotlyjs=True)

# ---------------------------------------------------------
# 6. TECHNICAL INDICATORS FOR EACH STOCK
# ---------------------------------------------------------
for stock in stocks:
    part = df[df["StockCode"] == stock].copy()
    part = part.sort_values("Date")

    part["MA20"] = part["Close"].rolling(20).mean()
    part["MA50"] = part["Close"].rolling(50).mean()
    part["MA200"] = part["Close"].rolling(200).mean()

    middle = part["Close"].rolling(20).mean()
    std = part["Close"].rolling(20).std()
    part["BB_Upper"] = middle + 2 * std
    part["BB_Lower"] = middle - 2 * std

    delta = part["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    part["RSI"] = 100 - (100 / (1 + rs))

    ema12 = part["Close"].ewm(span=12, adjust=False).mean()
    ema26 = part["Close"].ewm(span=26, adjust=False).mean()
    part["MACD"] = ema12 - ema26
    part["Signal"] = part["MACD"].ewm(span=9, adjust=False).mean()
    part["MACD_Hist"] = part["MACD"] - part["Signal"]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.48, 0.18, 0.18, 0.16],
        subplot_titles=(
            f"{stock} - Close Price, Moving Averages and Bollinger Bands",
            "Relative Strength Index (RSI)",
            "Moving Average Convergence Divergence (MACD)",
            "Trading Volume"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=part["Date"], y=part["Close"],
            name="Close",
            line=dict(width=1.8),
            customdata=np.column_stack([part["Open"], part["High"], part["Low"], part["Volume"]]),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Close: %{y:,.2f} VND"
                "<br>Open: %{customdata[0]:,.2f}"
                "<br>High: %{customdata[1]:,.2f}"
                "<br>Low: %{customdata[2]:,.2f}"
                "<br>Volume: %{customdata[3]:,.0f} shares<extra></extra>"
            )
        ), row=1, col=1
    )

    for col, name in [
        ("MA20", "MA20"),
        ("MA50", "MA50"),
        ("MA200", "MA200"),
        ("BB_Upper", "Bollinger Upper"),
        ("BB_Lower", "Bollinger Lower")
    ]:
        fig.add_trace(
            go.Scatter(
                x=part["Date"], y=part[col],
                name=name, mode="lines",
                hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{name}: %{{y:,.2f}}<extra></extra>"
            ), row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=part["Date"], y=part["RSI"],
            name="RSI(14)", line=dict(width=1.5),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>RSI: %{y:.2f}<extra></extra>"
        ), row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(
        go.Bar(
            x=part["Date"], y=part["MACD_Hist"],
            name="MACD Histogram",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Histogram: %{y:.3f}<extra></extra>"
        ), row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=part["Date"], y=part["MACD"],
            name="MACD"
        ), row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=part["Date"], y=part["Signal"],
            name="Signal"
        ), row=3, col=1
    )

    fig.add_trace(
        go.Bar(
            x=part["Date"], y=part["Volume"],
            name="Volume",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Volume: %{y:,.0f} shares<extra></extra>"
        ), row=4, col=1
    )

    fig.update_yaxes(title_text="Price (VND)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Shares", row=4, col=1)

    fig.update_layout(
        title=f"Interactive Technical Analysis - {stock}",
        height=1100,
        hovermode="x unified",
        legend_title="Indicator"
    )
    fig.add_annotation(
        text="MA: moving average | Bollinger Bands: price dispersion | RSI: momentum (70/30 reference levels) | MACD: trend momentum.",
        xref="paper", yref="paper", x=0, y=-0.06, showarrow=False
    )

    fig.write_html(
        OUT_DIR / f"11_06_technical_indicators_{stock}.html",
        include_plotlyjs=True
    )

# ---------------------------------------------------------
# 7. PRICE VS VOLUME SCATTER
# ---------------------------------------------------------
fig = px.scatter(
    df,
    x="Volume",
    y="Close",
    color="StockCode",
    size="Volume",
    size_max=16,
    hover_data={
        "Date": "|%Y-%m-%d",
        "StockCode": True,
        "Open": ":,.2f",
        "High": ":,.2f",
        "Low": ":,.2f",
        "Close": ":,.2f",
        "Volume": ":,.0f"
    },
    title="Stock Price vs Trading Volume",
    labels={
        "Volume": "Trading volume (shares)",
        "Close": "Closing price (VND)",
        "StockCode": "Stock"
    }
)
fig.update_layout(hovermode="closest")
fig.add_annotation(
    text="Each point represents one stock trading session. Hover over a point to inspect its date, price and volume.",
    xref="paper", yref="paper", x=0, y=-0.18, showarrow=False
)
fig.write_html(OUT_DIR / "11_07_price_volume_scatter.html", include_plotlyjs=True)

print("\nSUCCESS: Interactive visualization files created.")
print(f"Output directory: {OUT_DIR}")
print("Open the HTML files using Microsoft Edge or Google Chrome.")
