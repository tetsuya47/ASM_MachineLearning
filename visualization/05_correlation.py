import matplotlib.pyplot as plt
import seaborn as sns
from common import load_data, OUTPUT_DIR

df = load_data()
df["Return"] = df.groupby("StockCode")["Close"].pct_change() * 100

returns = df.pivot(
    index="Date",
    columns="StockCode",
    values="Return"
)

correlation = returns.corr()

plt.figure(figsize=(11, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("STOCK RETURN CORRELATION")
plt.tight_layout()

output = OUTPUT_DIR / "05_correlation.png"
plt.savefig(output, dpi=200)
plt.show()
plt.close()

print("Saved:", output)
