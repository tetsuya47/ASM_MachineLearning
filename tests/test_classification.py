import numpy as np
import pandas as pd
from clf_common import baseline_metrics, majority_class, evaluate, load_handoff
from classification_features import LABEL


def test_baseline_f1_is_not_hardcoded_half():
    ytr = pd.Series([0] * 60 + [1] * 40)
    yte = pd.Series([0] * 573 + [1] * 427)
    m = baseline_metrics(ytr, yte)
    assert m["Accuracy"] == 0.573
    assert m["Balanced Acc"] == 0.5
    assert m["Precision(Tang)"] == 0 and m["Recall(Tang)"] == 0
    assert abs(m["F1-macro"] - 0.3643) < 1e-3          # ~36.4%, khong phai 0.5


def test_majority_class_comes_from_train_not_test():
    assert majority_class(pd.Series([1, 1, 0])) == 1
    m = baseline_metrics(pd.Series([1, 1, 0]), pd.Series([0, 0, 0, 1]))   # Test da so la 0 nhung Train la 1
    assert m["Accuracy"] == 0.25


def test_baseline_on_real_test_split():
    train, test = load_handoff()
    m = baseline_metrics(train[LABEL], test[LABEL])
    assert round(m["Accuracy"], 3) == 0.573 and round(m["F1-macro"], 3) == 0.364


def test_evaluate_keys():
    r = evaluate(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), np.array([.2, .8, .6, .9]))
    assert {"Accuracy", "Balanced Acc", "Precision(Tang)", "Recall(Tang)", "F1-macro", "ROC-AUC"} <= set(r)
