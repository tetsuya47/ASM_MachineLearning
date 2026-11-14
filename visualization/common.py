from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "all_stocks_20241113_20261113.csv"
OUTPUT_DIR = ROOT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "StockCode", "Close"])
    df = df.sort_values(["StockCode", "Date"])
    df = df.drop_duplicates(["StockCode", "Date"])

    print("DATA LOADED SUCCESSFULLY")
    print("Total records:", len(df))
    print("Stock symbols:", sorted(df["StockCode"].unique()))

    return df
