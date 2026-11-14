import matplotlib.pyplot as plt
from common import load_data, OUTPUT_DIR

df = load_data()
stocks = sorted(df["StockCode"].unique())

plt.figure(figsize=(15, 7))

for stock in stocks:
    temp = df[df["StockCode"] == stock].copy()
    temp["Growth"] = temp["Close"] / temp["Close"].iloc[0] * 100
    plt.plot(temp["Date"], temp["Growth"], label=stock)

plt.title("STOCK PERFORMANCE COMPARISON")
plt.xlabel("Date")
plt.ylabel("Indexed Price (First day = 100)")
plt.legend(ncol=5)
plt.tight_layout()

output = OUTPUT_DIR / "02_growth.png"
plt.savefig(output, dpi=200)
plt.show()
plt.close()

print("Saved:", output)
