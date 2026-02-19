import numpy as np
from sklearn.metrics import roc_auc_score

def accuracy(y_true, y_pred):
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))

def expected_calibration_error(confidences, correct, n_bins=10):
    """
    confidences: list[float]
    correct: list[int] (1 if correct else 0)
    """
    confidences = np.array(confidences)
    correct = np.array(correct)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)

    return float(ece)

def auroc(confidences, correct):
    # If all labels same, AUROC is undefined
    if len(set(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, confidences))
