import matplotlib.pyplot as plt
import seaborn as sns
from common import load_data, OUTPUT_DIR

df = load_data()
df["Return"] = df.groupby("StockCode")["Close"].pct_change() * 100

plt.figure(figsize=(14, 7))

sns.histplot(
    data=df.dropna(subset=["Return"]),
    x="Return",
    hue="StockCode",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("DAILY RETURN DISTRIBUTION")
plt.xlabel("Daily Return (%)")
plt.ylabel("Density")
plt.tight_layout()

output = OUTPUT_DIR / "04_returns.png"
plt.savefig(output, dpi=200)
plt.show()
plt.close()

print("Saved:", output)
