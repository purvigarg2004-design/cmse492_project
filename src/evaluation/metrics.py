# src/evaluation/metrics.py

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def eval_model(y_true, y_pred, name: str = "model") -> Dict[str, float]:
    """
    Compute standard classification metrics and print a short summary.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels (0/1).
    name : str
        Name of the model (used in the printout and result dict).

    Returns
    -------
    dict
        Dictionary with accuracy, precision, recall, F1, and the confusion
        matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy :", f"{acc:.3f}")
    print("Precision:", f"{prec:.3f}")
    print("Recall   :", f"{rec:.3f}")
    print("F1       :", f"{f1:.3f}")
    print("Confusion matrix:\n", cm)

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def threshold_metrics(
    y_true,
    y_proba,
    thresholds: List[float],
    positive_label: int = 1,
) -> pd.DataFrame:
    """
    Compute precision, recall, and F1 for a list of decision thresholds.

    This is the helper behind the "Precision/Recall/F1 vs Threshold" plot.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    thresholds : list of float
        Threshold values in [0, 1]. For each threshold t, we predict
        positive if proba >= t.
    positive_label : int
        Label of the positive class (default = 1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'threshold', 'precision', 'recall', 'f1'.
    """
    rows = []
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        prec = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=positive_label)
        f1 = f1_score(y_true, y_pred, pos_label=positive_label)
        rows.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1})

    return pd.DataFrame(rows)
