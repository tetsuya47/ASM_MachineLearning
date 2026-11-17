"""Huan luyen, danh gia va backtest cac mo hinh phan loai xu huong gia phien ke tiep.
Chay:  python models/train_models.py   (sau khi chay: python classification_features.py)
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from clf_common import (FEATURES, LABEL, CLASSES, RESULTS, SEED, load_handoff, date_cv_splits,
                        baseline_metrics, evaluate, attach_next_return)

warnings.filterwarnings("ignore")


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM (RBF)": SVC(probability=True, class_weight="balanced", random_state=SEED),
        "Naive Bayes": GaussianNB(),
    }


def backtest(test, proba, name, threshold=0.5):
    """Mua khi P(Tang) > nguong, nguoc lai dung ngoai thi truong. Chia deu von giua cac ma,
    chua tinh phi giao dich / thue / T+2 -> chi mang tinh minh hoa."""
    d = attach_next_return(test)[["Date", "StockCode", "next_return"]]
    d["signal"] = (proba > threshold).astype(int)
    d["strat"] = d["signal"] * d["next_return"]
    daily = d.groupby("Date")[["next_return", "strat"]].mean()
    return (1 + daily).cumprod().rename(columns={"next_return": "Mua & giu", "strat": name})


def main():
    train, test = load_handoff()
    Xtr, ytr, Xte, yte = train[FEATURES], train[LABEL], test[FEATURES], test[LABEL]
    base = baseline_metrics(ytr, yte)       # lop da so lay tu TRAIN
    baseline = base["Accuracy"]
    print(f"Train {len(train)} | Test {len(test)} | Baseline (lop da so cua Train) = {baseline:.3f} | F1-macro = {base['F1-macro']:.3f}\n")

    cv = date_cv_splits(train, n_splits=5)     # chia theo ngay giao dich + purge
    rows, probas, preds = [], {}, {}
    for name, model in get_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        cv_f1 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
        pipe.fit(Xtr, ytr)
        probas[name], preds[name] = pipe.predict_proba(Xte)[:, 1], pipe.predict(Xte)
        rows.append({"Model": name, **evaluate(yte, preds[name], probas[name]), "CV F1-macro": cv_f1})

    # Tinh chinh Random Forest
    grid = GridSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("model", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, 8],
         "model__min_samples_leaf": [5, 10, 20]},
        cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    name = "Random Forest (tuned)"
    probas[name], preds[name] = best.predict_proba(Xte)[:, 1], best.predict(Xte)
    rows.append({"Model": name, **evaluate(yte, preds[name], probas[name]), "CV F1-macro": grid.best_score_})
    print("Tham so tot nhat RF:", grid.best_params_, "\n")

    res = pd.DataFrame(rows).set_index("Model").round(3).sort_values("F1-macro", ascending=False)
    res.loc["[Baseline: luon doan Giam]"] = base
    print(res.to_string())
    res.to_csv(RESULTS / "model_comparison.csv", encoding="utf-8-sig")

    # Chon mo hinh theo CV tren tap TRAIN (khong chon theo Test de tranh thien vi)
    top = res.drop(index="[Baseline: luon doan Giam]")["CV F1-macro"].idxmax()
    # Confusion matrix cua mo hinh tot nhat (theo F1-macro)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[top]),
                           display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {top}")
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Feature importance (RF tuned)
    imp = pd.Series(best.named_steps["model"].feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot(kind="barh", ax=ax, title="Feature Importance - Random Forest (tuned)")
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Backtest minh hoa
    curves = backtest(test, probas[top], f"Chien luoc {top}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    curves.plot(ax=ax, title="Backtest tren tap Test (von ban dau = 1, chua tinh phi)")
    ax.set_ylabel("Gia tri danh muc")
    fig.tight_layout()
    fig.savefig(RESULTS / "backtest.png", dpi=150)
    plt.close(fig)
    final = curves.iloc[-1]
    print(f"\nBacktest: Mua & giu = {final.iloc[0]-1:+.1%} | {top} = {final.iloc[1]-1:+.1%}")
    print("\nDa luu vao thu muc results/")


if __name__ == "__main__":
    main()
