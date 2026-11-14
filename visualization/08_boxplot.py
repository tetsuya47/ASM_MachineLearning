from common import load_data, OUTPUT_DIR, add_chart_note
import matplotlib.pyplot as plt
import seaborn as sns

df = load_data().sort_values(["StockCode", "Date"])
df["DailyReturn"] = df.groupby("StockCode")["Close"].pct_change(fill_method=None) * 100
returns = df.dropna(subset=["DailyReturn"])
fig, axes = plt.subplots(2, 1, figsize=(15, 12))
sns.boxplot(data=returns, x="StockCode", y="DailyReturn", ax=axes[0])
axes[0].set_title("DAILY STOCK RETURNS DISTRIBUTION - BOXPLOT")
axes[0].set_xlabel("Stock code")
axes[0].set_ylabel("Daily return (%)")
axes[0].axhline(0, linestyle="--", linewidth=1, color="gray")
axes[0].text(0.01, 0.98, "Box: Q1-Q3 (IQR) | Center line: median | Points beyond whiskers: potential outliers",
             transform=axes[0].transAxes, va="top", fontsize=8,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
sns.boxplot(data=df, x="StockCode", y="Volume", ax=axes[1])
axes[1].set_title("TRADING VOLUME DISTRIBUTION - BOXPLOT")
axes[1].set_xlabel("Stock code")
axes[1].set_ylabel("Trading volume (shares)")
axes[1].text(0.01, 0.98, "Points beyond whiskers are potential statistical outliers, not automatically errors",
             transform=axes[1].transAxes, va="top", fontsize=8,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
add_chart_note(fig, df, "First return per stock is undefined because no previous observation exists")
fig.tight_layout(rect=(0, 0.04, 1, 1))
output_path = OUTPUT_DIR / "08_boxplot.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("BOXPLOT CREATED:", output_path)
