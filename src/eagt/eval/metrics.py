import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

def compute_metrics(y_true, y_pred, labels=None):
    """
    Compute standard classification metrics.
    
    Parameters
    ----------
    y_true : list[int]
    y_pred : list[int]
    labels : list[str], optional
    
    Returns
    -------
    dict
        accuracy, precision, recall, f1, confusion_matrix, report
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, digits=4) if labels else ""
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
