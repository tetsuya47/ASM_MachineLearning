"""Compatibility module: the official data pipeline is Classification, not Regression."""
from classification_features import (
    FEATURES,
    LABEL,
    load_clean_data,
    build_classification_data,
    chronological_split,
    run_pipeline,
)

if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parent
    result = run_pipeline(
        root / "visualizations" / "10_cleaned_data.csv",
        root / "data" / "processed",
        test_size=0.2,
    )
    print("Classification handoff generated:", result["output_dir"])
