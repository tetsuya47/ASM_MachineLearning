from common import load_data, OUTPUT_DIR
import matplotlib.pyplot as plt
import seaborn as sns

df = load_data()

# Tính lợi suất hằng ngày theo từng cổ phiếu
df = df.sort_values(["StockCode", "Date"])
df["DailyReturn"] = (
    df.groupby("StockCode")["Close"]
    .pct_change(fill_method=None) * 100
)

# Chỉ sử dụng các giá trị hợp lệ để vẽ
returns = df.dropna(subset=["DailyReturn"])

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Boxplot lợi suất
sns.boxplot(
    data=returns,
    x="StockCode",
    y="DailyReturn",
    ax=axes[0]
)
axes[0].set_title("Daily Stock Returns Distribution - Boxplot")
axes[0].set_xlabel("Stock Code")
axes[0].set_ylabel("Daily Return (%)")
axes[0].axhline(0, linestyle="--", linewidth=1)

# Boxplot khối lượng giao dịch
sns.boxplot(
    data=df,
    x="StockCode",
    y="Volume",
    ax=axes[1]
)
axes[1].set_title("Trading Volume Distribution - Boxplot")
axes[1].set_xlabel("Stock Code")
axes[1].set_ylabel("Volume (Shares)")

plt.tight_layout()

output_path = OUTPUT_DIR / "08_boxplot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("BOXPLOT CREATED:", output_path)
