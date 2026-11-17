"""Phan tich nguyen nhan mo hinh chua vuot Baseline (Van de 2).
MOI lua chon (feature, sieu tham so, nguong) chi dua tren Train/CV; Test chi dung de bao cao 1 lan.
Chay:  python models/analysis.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from clf_common import (FEATURES, LABEL, RESULTS, SEED, load_handoff, date_cv_splits,
                        baseline_metrics, evaluate, save_model_meta)

warnings.filterwarnings("ignore")


def pipe(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def logreg(**kw):
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED, **kw)


def cv_scores(p, X, y, cv):
    """AUC & F1-macro tren cac fold validation (scaler fit rieng tung fold)."""
    auc, f1 = [], []
    for tr, va in cv:
        m = clone(p).fit(X.iloc[tr], y.iloc[tr])
        auc.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        f1.append(f1_score(y.iloc[va], m.predict(X.iloc[va]), average="macro"))
    return np.mean(auc), np.std(auc), np.mean(f1), np.std(f1)


def oof_proba(p, X, y, cv):
    out = pd.Series(np.nan, index=X.index)
    for tr, va in cv:
        out.iloc[va] = clone(p).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
    return out.dropna()


def main():
    train, test = load_handoff()
    Xtr, ytr, Xte, yte = train[FEATURES], train[LABEL], test[FEATURES], test[LABEL]
    cv = date_cv_splits(train, 5)
    base = baseline_metrics(ytr, yte)
    report = {"baseline_test": {k: round(v, 3) for k, v in base.items()}}

    # 1) Dong gop tung feature: AUC don bien + permutation importance tren fold validation (CHI Train)
    rng = np.random.default_rng(SEED)
    rows = []
    for f in FEATURES:
        aucs = [roc_auc_score(ytr.iloc[va], Xtr[f].iloc[va]) for _, va in cv]
        rows.append({"Feature": f, "AUC don bien (CV)": np.mean(aucs)})
    uni = pd.DataFrame(rows).set_index("Feature")
    perm = {f: [] for f in FEATURES}
    for tr, va in cv:
        m = pipe(logreg()).fit(Xtr.iloc[tr], ytr.iloc[tr])
        Xv, yv = Xtr.iloc[va], ytr.iloc[va]
        a0 = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        for f in FEATURES:
            drops = []
            for _ in range(5):
                Xp = Xv.copy(); Xp[f] = rng.permutation(Xp[f].values)
                drops.append(a0 - roc_auc_score(yv, m.predict_proba(Xp)[:, 1]))
            perm[f].append(np.mean(drops))
    uni["Permutation (giam AUC)"] = pd.Series({f: np.mean(v) for f, v in perm.items()})
    uni["|AUC-0.5|"] = (uni["AUC don bien (CV)"] - 0.5).abs()
    uni = uni.sort_values("Permutation (giam AUC)", ascending=False).round(4)
    uni.to_csv(RESULTS / "analysis_feature_contribution.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    uni["Permutation (giam AUC)"].sort_values().plot(kind="barh", ax=ax, title="Permutation importance (CV, Train) - LogReg")
    ax.axvline(0, color="k", lw=.8); fig.tight_layout(); fig.savefig(RESULTS / "analysis_feature_contribution.png", dpi=150)
    print("== Dong gop feature (Train/CV) ==\n", uni.to_string(), "\n")

    # 2) Lua chon dac trung: top-k theo permutation (tinh tren Train), so sanh CV AUC
    ranked = list(uni.index)
    sel_rows = []
    for k in (3, 5, 8, 12):
        feats = ranked[:k]
        a, sa, f, sf = cv_scores(pipe(logreg()), Xtr[feats], ytr, cv)
        sel_rows.append({"k": k, "Features": ", ".join(feats), "CV AUC": a, "CV AUC std": sa, "CV F1-macro": f})
    sel = pd.DataFrame(sel_rows).round(4)
    sel.to_csv(RESULTS / "analysis_feature_selection.csv", index=False, encoding="utf-8-sig")
    best_k = int(sel.loc[sel["CV AUC"].idxmax(), "k"])
    feats = ranked[:best_k]
    print("== Chon dac trung ==\n", sel.drop(columns="Features").to_string(index=False), f"\n-> k={best_k}: {feats}\n")

    # 3) Toi uu sieu tham so tren Train/CV (tieu chi ROC-AUC)
    grids = {
        "LogReg": (pipe(logreg()), {"model__C": [0.001, 0.01, 0.1, 1, 10]}),
        "SVM (RBF)": (pipe(SVC(probability=True, class_weight="balanced", random_state=SEED)),
                      {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1]}),
        "Random Forest": (pipe(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
                          {"model__n_estimators": [200], "model__max_depth": [2, 3, 5],
                           "model__min_samples_leaf": [20, 50]}),
    }
    hp_rows, tuned = [], {}
    for name, (p, g) in grids.items():
        gs = GridSearchCV(p, g, cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr[feats], ytr)
        tuned[name] = gs.best_estimator_
        hp_rows.append({"Model": name, "Best params": json.dumps(gs.best_params_), "CV AUC": gs.best_score_})
    hp = pd.DataFrame(hp_rows).round(4)
    hp.to_csv(RESULTS / "analysis_hyperparameters.csv", index=False, encoding="utf-8-sig")
    print("== Sieu tham so (Train/CV) ==\n", hp.to_string(index=False), "\n")

    # 4) Nguong du doan: chon tren xac suat out-of-fold cua Train (toi da F1-macro)
    name = hp.loc[hp["CV AUC"].idxmax(), "Model"]
    final = tuned[name]
    oof = oof_proba(final, Xtr[feats], ytr, cv)
    y_oof = ytr.loc[oof.index]
    grid_t = np.round(np.arange(0.30, 0.71, 0.01), 2)
    f1s = [f1_score(y_oof, (oof > t).astype(int), average="macro") for t in grid_t]
    thr = float(grid_t[int(np.argmax(f1s))])
    th = pd.DataFrame({"threshold": grid_t, "OOF F1-macro": np.round(f1s, 4)})
    th.to_csv(RESULTS / "analysis_threshold.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(grid_t, f1s); ax.axvline(thr, color="r", ls="--", label=f"chon = {thr:.2f}")
    ax.set(xlabel="Nguong P(Tang)", ylabel="F1-macro (OOF, Train)", title=f"Chon nguong tren Train/CV - {name}")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "analysis_threshold.png", dpi=150)
    print(f"== Nguong: {thr:.2f} (OOF F1-macro = {max(f1s):.3f}; nguong 0.5 = {f1s[list(grid_t).index(0.5)]:.3f})\n")

    # 5) Danh gia tong quat hoa: CV tung fold + Test DUNG 1 LAN + theo tung ma + bootstrap theo ngay
    a, sa, f, sf = cv_scores(final, Xtr[feats], ytr, cv)
    final.fit(Xtr[feats], ytr)
    proba = final.predict_proba(Xte[feats])[:, 1]
    pred_default, pred_thr = final.predict(Xte[feats]), (proba > thr).astype(int)
    res = pd.DataFrame({"Baseline (lop da so Train)": base,
                        f"{name} (nguong 0.5)": evaluate(yte, pred_default, proba),
                        f"{name} (nguong {thr:.2f})": evaluate(yte, pred_thr, proba)}).T.round(3)
    res.to_csv(RESULTS / "analysis_final_test.csv", encoding="utf-8-sig")
    print("== Test (danh gia 1 lan) ==\n", res.to_string(), "\n")

    dates = test["Date"].values
    ud = np.unique(dates)
    boots = []
    for _ in range(500):
        d = rng.choice(ud, len(ud), replace=True)
        idx = np.concatenate([np.flatnonzero(dates == x) for x in d])
        boots.append(roc_auc_score(yte.iloc[idx], proba[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per = (pd.DataFrame({"StockCode": test["StockCode"], "ok": (pred_thr == yte.values)})
           .groupby("StockCode")["ok"].mean().round(3).rename("Accuracy Test"))
    per.to_csv(RESULTS / "analysis_per_stock.csv", encoding="utf-8-sig")

    report.update({"selected_model": name, "selected_features": feats, "threshold": thr,
                   "cv_auc": [round(a, 4), round(sa, 4)], "cv_f1_macro": [round(f, 4), round(sf, 4)],
                   "test_auc": round(roc_auc_score(yte, proba), 4), "test_auc_ci95": [round(lo, 4), round(hi, 4)]})
    (RESULTS / "analysis_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nTheo ma (Test):\n", per.to_string())
    print("\nCanh bao: neu CI95 cua AUC chua 0.5 thi chua chung minh duoc kha nang du doan.")

    # Nguong cho mo hinh DANG TRIEN KHAI (logreg_model.joblib: LogReg balanced, 12 feature) - chon rieng
    # tren OOF cua chinh no (Train/CV), khong dung lai nguong cua mo hinh khac.
    dep_oof = oof_proba(pipe(logreg()), Xtr, ytr, cv)
    dep_y = ytr.loc[dep_oof.index]
    dep_f1 = [f1_score(dep_y, (dep_oof > t).astype(int), average="macro") for t in grid_t]
    dep_thr = float(grid_t[int(np.argmax(dep_f1))])
    print(f"Nguong cho LogReg trien khai: {dep_thr:.2f} (OOF F1-macro {max(dep_f1):.3f} vs 0.5: {dep_f1[list(grid_t).index(0.5)]:.3f})")
    save_model_meta(threshold=dep_thr, model_name="LogReg (class_weight=balanced)")
    report["deployed_logreg_threshold"] = dep_thr
    (RESULTS / "analysis_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
