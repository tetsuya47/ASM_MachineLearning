"""Entry point for Person 2. Main pipeline is Classification."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classification_features import run_pipeline, FEATURES

def main():
    source = ROOT / "visualizations" / "10_cleaned_data.csv"
    output = ROOT / "data" / "processed"
    result = run_pipeline(source, output, test_size=0.2)
    print("\\n========== NGUOI 2 - CLASSIFICATION ==========")
    print("Input:", source)
    print("Feature:", ", ".join(FEATURES))
    print("Train:", len(result["train"]), "| Test:", len(result["test"]))
    print("Ngay cuoi Train:", result["train"]["Date"].max())
    print("Ngay dau Test:", result["test"]["Date"].min())
    print("Overlap ngay:", result["qa"]["train_test_date_overlap"])
    print("Da tao clf_train.csv, clf_test.csv va cac file scaler/schema.")

if __name__ == "__main__":
    main()
