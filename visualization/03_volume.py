import matplotlib.pyplot as plt
import seaborn as sns
from common import load_data, OUTPUT_DIR

df = load_data()
volume = df.groupby("StockCode")["Volume"].mean().sort_values()

plt.figure(figsize=(12, 6))
sns.barplot(x=volume.values, y=volume.index)

plt.title("AVERAGE TRADING VOLUME")
plt.xlabel("Average Volume")
plt.ylabel("Stock")
plt.tight_layout()

output = OUTPUT_DIR / "03_volume.png"
plt.savefig(output, dpi=200)
plt.show()
plt.close()

print("Saved:", output)
