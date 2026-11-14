import matplotlib.pyplot as plt
import seaborn as sns
from common import load_data, OUTPUT_DIR, add_chart_note

df = load_data()
df["Return"] = df.groupby("StockCode")["Close"].pct_change() * 100
returns = df.pivot(index="Date", columns="StockCode", values="Return")
correlation = returns.corr()
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1,
            center=0, linewidths=0.5,
            cbar_kws={"label": "Pearson correlation coefficient"}, ax=ax)
ax.set_title("STOCK RETURN CORRELATION\nPearson correlation of daily percentage returns")
ax.set_xlabel("Stock code")
ax.set_ylabel("Stock code")
add_chart_note(fig, df, "+1 positive linear association; 0 no linear association; -1 negative linear association")
fig.tight_layout(rect=(0, 0.04, 1, 1))
output = OUTPUT_DIR / "05_correlation.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", output)
