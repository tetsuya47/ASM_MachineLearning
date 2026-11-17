import matplotlib.pyplot as plt
from common import load_data, OUTPUT_DIR, add_chart_note

df = load_data()
fig, ax = plt.subplots(figsize=(15, 7))
for stock in sorted(df["StockCode"].unique()):
    temp = df[df["StockCode"] == stock].sort_values("Date")
    line, = ax.plot(temp["Date"], temp["Close"], label=stock, linewidth=1.3)
    latest = temp.iloc[-1]
    ax.annotate(f"{stock}: {latest['Close']:,.0f}", (latest["Date"], latest["Close"]),
                xytext=(4, 3), textcoords="offset points", fontsize=7, color=line.get_color())
ax.set_title("STOCK CLOSING PRICE\nDaily closing prices by stock")
ax.set_xlabel("Trading date")
ax.set_ylabel("Closing price (VND)")
ax.legend(title="Stock code", ncol=5)
ax.grid(True, alpha=0.25)
add_chart_note(fig, df, "End labels show latest available closing price")
fig.tight_layout(rect=(0, 0.04, 1, 1))
output = OUTPUT_DIR / "01_close_price.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", output)
