import plotly.graph_objects as go
from common import load_data, OUTPUT_DIR

df = load_data()
for stock in sorted(df["StockCode"].unique()):
    temp = df[df["StockCode"] == stock].sort_values("Date").tail(100)
    fig = go.Figure(data=[go.Candlestick(
        x=temp["Date"], open=temp["Open"], high=temp["High"], low=temp["Low"],
        close=temp["Close"], name=stock, customdata=temp[["Volume"]].to_numpy(),
        hovertemplate=("<b>%{x|%d/%m/%Y}</b><br>Open: %{open:,.2f} VND<br>"
                       "High: %{high:,.2f} VND<br>Low: %{low:,.2f} VND<br>"
                       "Close: %{close:,.2f} VND<br>Volume: %{customdata[0]:,.0f} shares"
                       "<extra>%{fullData.name}</extra>"))])
    fig.update_layout(
        title=f"CANDLESTICK CHART - {stock}<br><sup>Last 100 available observations | Hover for OHLCV details</sup>",
        xaxis_title="Trading date", yaxis_title="Price (VND)",
        xaxis_rangeslider_visible=False, template="plotly_white", hovermode="x unified",
        annotations=[dict(text="OHLC = Open, High, Low, Close | Volume = traded shares",
                          xref="paper", yref="paper", x=0, y=-0.18,
                          showarrow=False, align="left", font=dict(size=10, color="gray"))],
        margin=dict(b=65))
    output = OUTPUT_DIR / f"06_candlestick_{stock}.html"
    fig.write_html(str(output))
    print("Saved:", output)
print("CANDLESTICK CHARTS COMPLETED!")
