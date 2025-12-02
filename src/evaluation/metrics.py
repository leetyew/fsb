"""
Standard evaluation metrics for credit scoring models.

Implements:
- ROC AUC: Area under the ROC curve
- PAUC: Partial AUC in high-recall region (FNR in [0, 0.2])
- Brier: Mean squared error of probability predictions
- ABR: Average Bad Rate among accepted applicants
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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
    accept_rate_min: float = 0.2,
    accept_rate_max: float = 0.4,
    n_points: int = 21,
) -> float:
    """Compute Average Bad Rate integrated over acceptance range.

    Per paper Section 6.3: "Specifically, we integrate the ABR over acceptance
    between 20% and 40%, which reflects historical policies at Monedo."

    Computes bad rate at multiple acceptance thresholds and integrates
    using trapezoidal rule.

    Lower ABR is better (fewer defaults among accepted applicants).

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).
        accept_rate_min: Minimum acceptance rate for integration (default 0.2).
        accept_rate_max: Maximum acceptance rate for integration (default 0.4).
        n_points: Number of points for numerical integration (default 21).

    Returns:
        Integrated bad rate over acceptance range, normalized by range width.
    """
    # Sort by score to efficiently compute bad rates at different thresholds
    sorted_indices = np.argsort(y_score)
    y_sorted = y_true[sorted_indices]
    n = len(y_true)

    # Compute bad rate at each acceptance threshold
    accept_rates = np.linspace(accept_rate_min, accept_rate_max, n_points)
    bad_rates = []

    for rate in accept_rates:
        n_accept = max(1, int(n * rate))
        bad_rate = y_sorted[:n_accept].mean()
        bad_rates.append(bad_rate)

    # Integrate using trapezoidal rule and normalize by range width
    integrated = np.trapz(bad_rates, accept_rates)
    normalized = integrated / (accept_rate_max - accept_rate_min)

    return float(normalized)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: List[str],
    abr_range: Tuple[float, float] = (0.2, 0.4),
) -> Dict[str, float]:
    """Compute multiple evaluation metrics.

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).
        metrics: List of metric names to compute.
            Supported: "auc", "pauc", "brier", "abr".
        abr_range: (min, max) acceptance rate range for ABR integration.
            Per paper Section 6.3, default is (0.2, 0.4).

    Returns:
        Dictionary mapping metric name to value.
    """
    results = {}

    metric_funcs = {
        "auc": lambda: compute_auc(y_true, y_score),
        "pauc": lambda: compute_pauc(y_true, y_score),
        "brier": lambda: compute_brier(y_true, y_score),
        "abr": lambda: compute_abr(y_true, y_score, abr_range[0], abr_range[1]),
    }

    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower in metric_funcs:
            results[metric_lower] = metric_funcs[metric_lower]()
        else:
            raise ValueError(f"Unknown metric: {metric}. Supported: {list(metric_funcs.keys())}")

    return results
