import matplotlib.pyplot as plt
from common import load_data, OUTPUT_DIR, add_chart_note

df = load_data()
fig, ax = plt.subplots(figsize=(15, 7))
for stock in sorted(df["StockCode"].unique()):
    temp = df[df["StockCode"] == stock].sort_values("Date").copy()
    temp["Growth"] = temp["Close"] / temp["Close"].iloc[0] * 100
    line, = ax.plot(temp["Date"], temp["Growth"], label=stock, linewidth=1.3)
    latest = temp.iloc[-1]
    ax.annotate(f"{stock}: {latest['Growth']:.1f}", (latest["Date"], latest["Growth"]),
                xytext=(4, 2), textcoords="offset points", fontsize=7, color=line.get_color())
ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Base = 100")
ax.set_title("STOCK PERFORMANCE COMPARISON\nIndexed closing price (first observation = 100)")
ax.set_xlabel("Trading date")
ax.set_ylabel("Indexed price (first observation = 100)")
ax.legend(title="Stock code / reference", ncol=5)
ax.grid(True, alpha=0.25)
add_chart_note(fig, df, "Above 100 = above first close; below 100 = below first close")
fig.tight_layout(rect=(0, 0.04, 1, 1))
output = OUTPUT_DIR / "02_growth.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", output)
