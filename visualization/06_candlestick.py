import plotly.graph_objects as go
from common import load_data, OUTPUT_DIR

df = load_data()
stocks = sorted(df["StockCode"].unique())

for stock in stocks:
    temp = df[df["StockCode"] == stock].tail(100)

    fig = go.Figure(data=[
        go.Candlestick(
            x=temp["Date"],
            open=temp["Open"],
            high=temp["High"],
            low=temp["Low"],
            close=temp["Close"],
            name=stock
        )
    ])

    fig.update_layout(
        title=f"CANDLESTICK CHART - {stock}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    output = OUTPUT_DIR / f"06_candlestick_{stock}.html"
    fig.write_html(str(output))
    print("Saved:", output)

print("CANDLESTICK CHARTS COMPLETED!")
