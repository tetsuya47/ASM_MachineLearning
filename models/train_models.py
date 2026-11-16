"""Huan luyen, danh gia va backtest cac mo hinh phan loai xu huong gia phien ke tiep.
Chay:  python models/train_models.py   (sau khi chay: python classification_features.py)
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classification_features import FEATURES, LABEL  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM (RBF)": SVC(probability=True, class_weight="balanced", random_state=SEED),
        "Naive Bayes": GaussianNB(),
    }


def metrics(name, y, pred, proba):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, pred),
        "Balanced Acc": balanced_accuracy_score(y, pred),
        "Precision(Tang)": precision_score(y, pred, zero_division=0),
        "Recall(Tang)": recall_score(y, pred, zero_division=0),
        "F1-macro": f1_score(y, pred, average="macro"),
        "ROC-AUC": roc_auc_score(y, proba),
    }


def backtest(test, proba, name, threshold=0.5):
    """Mua khi P(Tang) > nguong, nguoc lai dung ngoai thi truong. Chia deu von giua cac ma,
    chua tinh phi giao dich / thue / T+2 -> chi mang tinh minh hoa."""
    d = test[["Date", "StockCode", "next_return"]].copy()
    d["signal"] = (proba > threshold).astype(int)
    d["strat"] = d["signal"] * d["next_return"]
    daily = d.groupby("Date")[["next_return", "strat"]].mean()
    return (1 + daily).cumprod().rename(columns={"next_return": "Mua & giu", "strat": name})


def main():
    train = pd.read_csv(ROOT / "data/processed/clf_train.csv", parse_dates=["Date"]).sort_values("Date")
    test = pd.read_csv(ROOT / "data/processed/clf_test.csv", parse_dates=["Date"]).sort_values("Date")
    Xtr, ytr, Xte, yte = train[FEATURES], train[LABEL], test[FEATURES], test[LABEL]

    baseline = max(yte.mean(), 1 - yte.mean())
    print(f"Train {len(train)} | Test {len(test)} | Baseline (luon doan lop da so) = {baseline:.3f}\n")

    tscv = TimeSeriesSplit(n_splits=5)
    rows, probas, preds = [], {}, {}
    for name, model in get_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        cv = cross_val_score(pipe, Xtr, ytr, cv=tscv, scoring="f1_macro", n_jobs=-1).mean()
        pipe.fit(Xtr, ytr)
        probas[name] = pipe.predict_proba(Xte)[:, 1]
        preds[name] = pipe.predict(Xte)
        r = metrics(name, yte, preds[name], probas[name])
        r["CV F1-macro"] = cv
        rows.append(r)

    # Tinh chinh Random Forest
    grid = GridSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("model", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, 8],
         "model__min_samples_leaf": [5, 10, 20]},
        cv=tscv, scoring="f1_macro", n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    name = "Random Forest (tuned)"
    probas[name], preds[name] = best.predict_proba(Xte)[:, 1], best.predict(Xte)
    r = metrics(name, yte, preds[name], probas[name])
    r["CV F1-macro"] = grid.best_score_
    rows.append(r)
    print("Tham so tot nhat RF:", grid.best_params_, "\n")

    res = pd.DataFrame(rows).set_index("Model").round(3).sort_values("F1-macro", ascending=False)
    res.loc["[Baseline: luon doan Giam]"] = [baseline, 0.5, 0, 0, np.nan, 0.5, np.nan]
    print(res.to_string())
    res.to_csv(RES / "model_comparison.csv", encoding="utf-8-sig")

    # Chon mo hinh theo CV tren tap TRAIN (khong chon theo Test de tranh thien vi)
    top = res.drop(index="[Baseline: luon doan Giam]")["CV F1-macro"].idxmax()
    # Confusion matrix cua mo hinh tot nhat (theo F1-macro)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[top]),
                           display_labels=["Giam/Dung", "Tang"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {top}")
    fig.savefig(RES / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    # Feature importance (RF tuned)
    imp = pd.Series(best.named_steps["model"].feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot(kind="barh", ax=ax, title="Feature Importance - Random Forest (tuned)")
    fig.tight_layout()
    fig.savefig(RES / "feature_importance.png", dpi=150)

    # Backtest minh hoa
    curves = backtest(test, probas[top], f"Chien luoc {top}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    curves.plot(ax=ax, title="Backtest tren tap Test (von ban dau = 1, chua tinh phi)")
    ax.set_ylabel("Gia tri danh muc")
    fig.tight_layout()
    fig.savefig(RES / "backtest.png", dpi=150)
    final = curves.iloc[-1]
    print(f"\nBacktest: Mua & giu = {final.iloc[0]-1:+.1%} | {top} = {final.iloc[1]-1:+.1%}")
    print("\nDa luu vao thu muc results/")


if __name__ == "__main__":
    main()
