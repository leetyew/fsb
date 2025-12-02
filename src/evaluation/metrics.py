"""
Standard evaluation metrics for credit scoring models.

Implements:
- ROC AUC: Area under the ROC curve
- PAUC: Partial AUC in high-recall region (FNR in [0, 0.2])
- Brier: Mean squared error of probability predictions
- ABR: Average Bad Rate among accepted applicants
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC score.

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).

    Returns:
        AUC score in [0, 1].
    """
    return float(roc_auc_score(y_true, y_score))


def compute_pauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fnr: float = 0.2,
) -> float:
    """Compute Partial AUC in high-recall region.

    Measures AUC over the region where False Negative Rate (FNR) is in [0, max_fnr].
    This focuses on the high-recall region where missing bad loans is costly.

    The result is normalized to [0, 1] by dividing by max_fnr.

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).
        max_fnr: Maximum FNR to include (default 0.2 = top 80% recall).

    Returns:
        Normalized partial AUC in [0, 1].
    """
    # Get ROC curve: fpr, tpr at various thresholds
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # FNR = 1 - TPR, so we want TPR >= 1 - max_fnr
    min_tpr = 1.0 - max_fnr

    # Filter to region where TPR >= min_tpr (i.e., FNR <= max_fnr)
    mask = tpr >= min_tpr

    if not mask.any():
        return 0.0

    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]

    # Compute area using trapezoidal rule
    # We integrate FPR as function of TPR in the restricted region
    area = np.trapz(fpr_partial, tpr_partial)

    # The maximum possible area in this region is max_fnr * 1.0
    # Normalize so perfect classifier gets 1.0
    # Note: lower FPR is better, so we compute 1 - (area / max_fnr)
    normalized_pauc = 1.0 - (area / max_fnr)

    return float(np.clip(normalized_pauc, 0.0, 1.0))


def compute_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Lower is better. Perfect calibration = 0.

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).

    Returns:
        Brier score in [0, 1].
    """
    return float(brier_score_loss(y_true, y_score))


def compute_abr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    accept_rate: float = 0.15,
) -> float:
    """Compute Average Bad Rate among accepted applicants.

    Simulates accepting the top α fraction (lowest scores) and computes
    the actual bad rate among those accepted.

    Lower ABR is better (fewer defaults among accepted applicants).

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).
        accept_rate: Fraction to accept (α). Default 0.15.

    Returns:
        Bad rate among accepted applicants.
    """
    n_accept = max(1, int(len(y_true) * accept_rate))

    # Accept applicants with lowest predicted PD
    accept_indices = np.argsort(y_score)[:n_accept]
    y_accepted = y_true[accept_indices]

    return float(y_accepted.mean())


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: List[str],
    accept_rate: float = 0.15,
) -> Dict[str, float]:
    """Compute multiple evaluation metrics.

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).
        metrics: List of metric names to compute.
            Supported: "auc", "pauc", "brier", "abr".
        accept_rate: Acceptance rate for ABR calculation.

    Returns:
        Dictionary mapping metric name to value.
    """
    results = {}

    metric_funcs = {
        "auc": lambda: compute_auc(y_true, y_score),
        "pauc": lambda: compute_pauc(y_true, y_score),
        "brier": lambda: compute_brier(y_true, y_score),
        "abr": lambda: compute_abr(y_true, y_score, accept_rate),
    }

    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower in metric_funcs:
            results[metric_lower] = metric_funcs[metric_lower]()
        else:
            raise ValueError(f"Unknown metric: {metric}. Supported: {list(metric_funcs.keys())}")

    return results
