from common import load_data, OUTPUT_DIR

df = load_data()

summary = df.groupby("StockCode").agg(
    Average_Close=("Close", "mean"),
    Highest_Price=("High", "max"),
    Lowest_Price=("Low", "min"),
    Average_Volume=("Volume", "mean")
)

first = df.groupby("StockCode")["Close"].first()
last = df.groupby("StockCode")["Close"].last()

summary["Growth_Percent"] = (last / first - 1) * 100
summary = summary.round(2)

output = OUTPUT_DIR / "07_summary.csv"
summary.to_csv(output, encoding="utf-8-sig")

print("\n===== STOCK SUMMARY =====")
print(summary)
print("\nSaved:", output)

print("\nCOLUMN GUIDE:")
print("- Average_Close: arithmetic mean of daily Close (VND).")
print("- Highest_Price: maximum observed High (VND).")
print("- Lowest_Price: minimum observed Low (VND).")
print("- Average_Volume: arithmetic mean of daily Volume (shares).")
print("- Growth_Percent: (last observed Close / first observed Close - 1) * 100.")
print("- Growth_Percent is an endpoint comparison; corporate-action adjustments are not assessed here.")
