"""Порог выбирается на validation; test не участвует в выборе."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def choose_threshold(y, probabilities, thresholds):
    scores = [f1_score(y, probabilities >= t, zero_division=0) for t in thresholds]
    # При равном F1 выбираем первый (наименьший) порог, как в ДЗ №7.
    return float(thresholds[int(np.argmax(scores))])


def score(y, probabilities, threshold):
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "positive_rate": float(np.mean(y)), "rows": int(len(y)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def evaluate(model, splits, config, threshold=None):
    features, target = config["features"], config["target"]
    val = splits["validation"]
    p_val = model.predict_proba(val[features])[:, 1]
    if threshold is None:
        threshold = choose_threshold(val[target], p_val, config["evaluation"]["thresholds"])
    test = splits["test"]
    p_test = model.predict_proba(test[features])[:, 1]
    metrics = {
        "mode": config["mode"], "target": target,
        "target_meaning": "No CDR in next 30 days; not confirmed contract churn",
        "threshold": threshold, "threshold_selected_on": "validation",
        "validation": score(val[target], p_val, threshold),
        "test": score(test[target], p_test, threshold),
        "train_rows": len(splits["train"]),
    }
    predictions = pd.DataFrame({
        config["id_column"]: test[config["id_column"]].to_numpy(),
        target: test[target].to_numpy(), "probability": p_test,
        "prediction": (p_test >= threshold).astype(int),
    })
    return metrics, predictions
