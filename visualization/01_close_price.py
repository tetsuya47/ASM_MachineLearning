import matplotlib.pyplot as plt
from common import load_data, OUTPUT_DIR

df = load_data()
stocks = sorted(df["StockCode"].unique())

plt.figure(figsize=(15, 7))

for stock in stocks:
    temp = df[df["StockCode"] == stock]
    plt.plot(temp["Date"], temp["Close"], label=stock)

plt.title("STOCK CLOSING PRICE")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend(ncol=5)
plt.tight_layout()

output = OUTPUT_DIR / "01_close_price.png"
plt.savefig(output, dpi=200)
plt.show()
plt.close()

print("Saved:", output)
