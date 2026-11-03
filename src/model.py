"""Ensemble models, training and evaluation."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def base_models(seed: int = 42, n_estimators: int = 150) -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                                random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1,
                                            random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=n_estimators,
                                                            random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=n_estimators, random_state=seed),
    }


def ensemble_models(seed: int = 42, n_estimators: int = 150) -> dict:
    base = list(base_models(seed, n_estimators).items())
    return {
        "Voting (soft)": VotingClassifier(base, voting="soft", n_jobs=1),
        "Stacking": StackingClassifier(base, final_estimator=LogisticRegression(max_iter=1000),
                                       cv=3, n_jobs=1),
    }


def all_models(seed=42, n_estimators=150) -> dict:
    models = {"Decision Tree (baseline)": DecisionTreeClassifier(random_state=seed)}
    models.update(base_models(seed, n_estimators))
    models.update(ensemble_models(seed, n_estimators))
    return models


def split(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def evaluate(model, X_test, y_test, multiclass: bool, class_names=None) -> dict:
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    latency_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    avg = "weighted" if multiclass else "binary"
    out = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, pred, average=avg, zero_division=0),
        "Latency (ms/sample)": latency_ms,
        "_cm": confusion_matrix(y_test, pred),
        "_report": classification_report(y_test, pred, target_names=class_names,
                                         zero_division=0, output_dict=True),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        try:
            if multiclass:
                out["ROC-AUC"] = roc_auc_score(y_test, proba, multi_class="ovr",
                                               average="weighted")
            else:
                out["ROC-AUC"] = roc_auc_score(y_test, proba[:, 1])
                fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                out["_roc"] = (fpr, tpr)
        except ValueError:
            pass
    if not multiclass:
        tn, fp, fn, tp = out["_cm"].ravel()
        out["False Alarm Rate"] = fp / (fp + tn) if (fp + tn) else 0.0
    return out


def train_and_evaluate(X_train, X_test, y_train, y_test, multiclass, class_names=None,
                       selected=None, seed=42, n_estimators=150, progress=None):
    models = all_models(seed, n_estimators)
    if selected:
        models = {k: v for k, v in models.items() if k in selected}
    fitted, results = {}, {}
    for i, (name, m) in enumerate(models.items()):
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        res = evaluate(m, X_test, y_test, multiclass, class_names)
        res["Train time (s)"] = time.perf_counter() - t0
        fitted[name], results[name] = m, res
        if progress:
            progress((i + 1) / len(models), name)
    return fitted, results


def results_table(results: dict) -> pd.DataFrame:
    rows = {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in results.items()}
    return pd.DataFrame(rows).T.sort_values("F1", ascending=False)


def feature_importance(model, feature_names) -> pd.Series | None:
    """Average importance across tree-based members where available."""
    imps = []
    members = []
    if hasattr(model, "estimators_"):
        members = list(model.estimators_)
    else:
        members = [model]
    for m in members:
        if hasattr(m, "feature_importances_"):
            imps.append(m.feature_importances_)
    if not imps:
        return None
    s = pd.Series(np.mean(imps, axis=0), index=feature_names)
    return s.sort_values(ascending=False)
