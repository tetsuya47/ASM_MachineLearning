import matplotlib.pyplot as plt
import seaborn as sns
from common import load_data, OUTPUT_DIR, add_chart_note

df = load_data()
df["Return"] = df.groupby("StockCode")["Close"].pct_change() * 100
valid = df["Return"].dropna()
fig, ax = plt.subplots(figsize=(14, 7))
sns.histplot(data=df.dropna(subset=["Return"]), x="Return", hue="StockCode", bins=50,
             element="step", stat="density", common_norm=False, ax=ax)
mean_value, median_value = valid.mean(), valid.median()
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero return")
ax.axvline(mean_value, color="red", linestyle="-.", linewidth=1.2, label=f"Mean = {mean_value:.3f}%")
ax.axvline(median_value, color="green", linestyle=":", linewidth=1.2, label=f"Median = {median_value:.3f}%")
ax.set_title("DAILY RETURN DISTRIBUTION\nDistribution of daily percentage changes in closing price")
ax.set_xlabel("Daily return (%)")
ax.set_ylabel("Density (normalized within each stock)")
ax.legend(title="Reference lines")
ax.grid(axis="y", alpha=0.2)
add_chart_note(fig, df, "First observation per stock has no previous close and is excluded")
fig.tight_layout(rect=(0, 0.04, 1, 1))
output = OUTPUT_DIR / "04_returns.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)
print("Saved:", output)
