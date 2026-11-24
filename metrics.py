"""Metrics: accuracy, macro-F1, ordinal MAE, calibration (ECE), confusion matrix."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def ece_score(probs, y, bins=10):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def summarize(probs, y, n_classes):
    pred = probs.argmax(1)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "mae": float(np.abs(pred - y).mean()),  # average class distance: ordinal "how far off"
        "ece": ece_score(probs, y),
        "confusion": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist(),
    }
