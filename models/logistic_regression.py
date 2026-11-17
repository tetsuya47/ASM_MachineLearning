"""Logistic Regression (mo hinh chinh) du doan xu huong gia phien ke tiep: Tang / Giam-Dung.
Do luong theo de bai: Accuracy va Confusion Matrix (kem Precision, Recall, F1).
Chay:  python models/logistic_regression.py
"""
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from clf_common import (FEATURES, LABEL, CLASSES, RESULTS, ROOT, SEED,
                        load_handoff, date_cv_splits, baseline_metrics, evaluate, save_model_meta, META_PATH)

warnings.filterwarnings("ignore")


def main():
    train, test = load_handoff()
    Xtr, ytr, Xte, yte = train[FEATURES], train[LABEL], test[FEATURES], test[LABEL]
    base = baseline_metrics(ytr, yte)       # lop da so lay tu TRAIN
    baseline = base["Accuracy"]
    print(f"Train: {len(train)} mau | Test: {len(test)} mau")
    print(f"Ty le lop Tang - train {ytr.mean():.3f}, test {yte.mean():.3f}")
    print(f"Baseline (luon doan lop da so cua Train) tren Test: accuracy = {baseline:.3f}, F1-macro = {base['F1-macro']:.3f}\n")

    cv = date_cv_splits(train, n_splits=5)
    configs = {"LogReg (mac dinh)": None, "LogReg (class_weight=balanced)": "balanced"}
    rows, pipes, cms = [], {}, {}
    for name, cw in configs.items():
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("model", LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED))])
        cvr = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["accuracy", "f1_macro"])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)           # danh gia dung 1 lan tren Test
        pipes[name], cms[name] = pipe, confusion_matrix(yte, pred)
        r = evaluate(yte, pred, pipe.predict_proba(Xte)[:, 1])
        rows.append({"Model": name,
                     "CV Accuracy": cvr["test_accuracy"].mean(),
                     "Accuracy (Train)": (pipe.predict(Xtr) == ytr).mean(),
                     "Accuracy (Test)": r.pop("Accuracy"), **r,
                     "CV F1-macro": cvr["test_f1_macro"].mean()})
        print(f"=== {name} ===")
        print("Confusion Matrix (hang = thuc te, cot = du doan):")
        print(pd.DataFrame(cms[name], index=[f"Thuc: {c}" for c in CLASSES],
                           columns=[f"Doan: {c}" for c in CLASSES]), "\n")
        print(classification_report(yte, pred, target_names=CLASSES, zero_division=0))

    res = pd.DataFrame(rows).set_index("Model")
    res.loc["Baseline (luon doan Giam/Dung)"] = {"Accuracy (Test)": base.pop("Accuracy"), **base}
    res = res.round(3)
    print(res.to_string())
    res.to_csv(RESULTS / "logistic_regression_results.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, cm) in zip(axes, cms.items()):
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\nAccuracy Test = {res.loc[name, 'Accuracy (Test)']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "logreg_confusion_matrix.png", dpi=150)
    plt.close(fig)

    final = pipes["LogReg (class_weight=balanced)"]
    coef = pd.Series(final.named_steps["model"].coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    coef.plot(kind="barh", ax=ax, color=np.where(coef > 0, "tab:green", "tab:red"),
              title="He so Logistic Regression (dac trung da chuan hoa)")
    ax.axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS / "logreg_coefficients.png", dpi=150)
    plt.close(fig)
    print("\nHe so (>0: tang kha nang 'Tang', <0: giam):")
    print(coef.sort_values(ascending=False).round(3).to_string())

    joblib.dump(final, ROOT / "models" / "logreg_model.joblib")
    if not META_PATH.exists():   # giu nguong da chon boi analysis.py (Train/CV) neu co
        save_model_meta(threshold=0.5, model_name="LogReg (class_weight=balanced)")
    print("\nDa luu ket qua vao results/ va models/logreg_model.joblib")


if __name__ == "__main__":
    main()
