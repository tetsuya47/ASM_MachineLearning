from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_engineering import run_pipeline

INPUT = ROOT / "data" / "raw" / "all_stocks_20241113_20261113.csv"
OUTPUT = ROOT / "data" / "processed"

def main():
    result = run_pipeline(INPUT, OUTPUT, test_size=0.2)
    train = result["train"]
    test = result["test"]

    print("\n========== FEATURE ENGINEERING ==========")
    print("Số dòng sau tạo đặc trưng:", len(result["engineered_data"]))
    print("Số mã cổ phiếu:", result["engineered_data"]["StockCode"].nunique())
    print("Các đặc trưng:", ", ".join(result["feature_columns"]))
    print("\n========== TRAIN / TEST ==========")
    print("Train:", len(train))
    print("Test:", len(test))
    print("Ngày cuối Train:", result["train_last_date"])
    print("Ngày đầu Test:", result["test_first_date"])
    print("X_train:", train[result["feature_columns"]].shape)
    print("X_test:", test[result["feature_columns"]].shape)
    print("\n========== KIỂM TRA CHUẨN HÓA ==========")
    print("Trung bình từng đặc trưng trên Train (xấp xỉ 0):")
    print(train[result["feature_columns"]].mean().round(4).to_string())
    print("\nĐã lưu:")
    print(OUTPUT / "all_stocks_features.csv")
    print(OUTPUT / "train_scaled.csv")
    print(OUTPUT / "test_scaled.csv")

if __name__ == "__main__":
    main()
