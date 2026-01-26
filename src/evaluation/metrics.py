"""
Standard evaluation metrics for credit scoring models.

Implements:
- ROC AUC: Area under the ROC curve
- PAUC: Partial AUC in high-recall region (FNR in [0, 0.2])
- Brier: Mean squared error of probability predictions
- ABR: Average Bad Rate among accepted applicants
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src.evaluation.thresholds import ThresholdSpec


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC score.

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).

    Returns:
        AUC score in [0, 1], or NaN if only one class is present.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
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
        Normalized partial AUC in [0, 1], or NaN if only one class is present.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
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


def compute_abr_breakdown(
    y_true: np.ndarray,
    y_score: np.ndarray,
    accept_rate_min: float = 0.2,
    accept_rate_max: float = 0.4,
    n_points: int = 21,
) -> Dict[str, Any]:
    """Compute ABR breakdown with detailed diagnostics.

    Returns detailed information about the ABR computation for verification,
    including both mean and trapezoidal integration methods (TWEAK 2).

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).
        accept_rate_min: Minimum acceptance rate for integration (default 0.2).
        accept_rate_max: Maximum acceptance rate for integration (default 0.4).
        n_points: Number of points for numerical integration (default 21).

    Returns:
        Dictionary containing:
        - accept_rates: List of acceptance rate grid points
        - bad_rates: List of bad rates at each level
        - integrated_abr_mean: Mean of bad_rates (primary metric)
        - integrated_abr_trapz: trapz normalized by range (secondary, verification)
        - k_values: Number of accepts at each level
        - n_total: Total number of samples
        - note: Description of primary metric
    """
    # Validate inputs
    if len(y_true) == 0 or len(y_score) == 0:
        return {
            "accept_rates": [],
            "bad_rates": [],
            "integrated_abr_mean": float("nan"),
            "integrated_abr_trapz": float("nan"),
            "k_values": [],
            "n_total": 0,
            "note": "Empty input",
        }

    if len(y_true) != len(y_score):
        raise ValueError(f"Shape mismatch: y_true={len(y_true)}, y_score={len(y_score)}")

    # Check for finite scores
    if not np.all(np.isfinite(y_score)):
        n_nonfinite = (~np.isfinite(y_score)).sum()
        raise ValueError(f"{n_nonfinite} non-finite scores detected")

    # Sort by score ascending (lowest PD first = best applicants)
    sorted_indices = np.argsort(y_score)
    y_sorted = y_true[sorted_indices]
    n_total = len(y_true)

    # Compute bad rate at each acceptance threshold
    accept_rates = np.linspace(accept_rate_min, accept_rate_max, n_points)
    bad_rates = []
    k_values = []

    for rate in accept_rates:
        # Use ceil to ensure at least 1 accept at min rate
        k = max(1, int(np.ceil(rate * n_total)))
        k_values.append(k)
        bad_rate = float(y_sorted[:k].mean())
        bad_rates.append(bad_rate)

    # TWEAK 2: Compute BOTH integration methods
    # Primary: arithmetic mean (simple average over grid)
    integrated_abr_mean = float(np.mean(bad_rates))

    # Secondary: trapezoidal integration normalized by range width
    integrated_abr_trapz = float(
        np.trapz(bad_rates, accept_rates) / (accept_rate_max - accept_rate_min)
    )

    return {
        "accept_rates": [float(r) for r in accept_rates],
        "bad_rates": [float(br) for br in bad_rates],
        "integrated_abr_mean": integrated_abr_mean,
        "integrated_abr_trapz": integrated_abr_trapz,
        "k_values": k_values,
        "n_total": n_total,
        "note": "mean over grid (primary)",
    }


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
    threshold_spec: Optional[ThresholdSpec] = None,
    abr_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Compute multiple evaluation metrics.

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted probability of bad (y=1).
        metrics: List of metric names to compute.
            Supported: "auc", "pauc", "brier", "abr".
        threshold_spec: ThresholdSpec with ABR range and pAUC parameters.
            If provided, takes precedence over abr_range.
        abr_range: (min, max) acceptance rate range for ABR integration.
            Deprecated: use threshold_spec instead.

    Returns:
        Dictionary mapping metric name to value.
    """
    # Resolve thresholds: prefer ThresholdSpec, fall back to abr_range for compatibility
    if threshold_spec is not None:
        _abr_range = threshold_spec.abr_range
        _pauc_max_fnr = threshold_spec.pauc_max_fnr
    elif abr_range is not None:
        _abr_range = abr_range
        _pauc_max_fnr = 0.2  # Legacy default
    else:
        # Use paper defaults
        _abr_range = (0.2, 0.4)
        _pauc_max_fnr = 0.2

    results = {}

    metric_funcs = {
        "auc": lambda: compute_auc(y_true, y_score),
        "pauc": lambda: compute_pauc(y_true, y_score, max_fnr=_pauc_max_fnr),
        "brier": lambda: compute_brier(y_true, y_score),
        "abr": lambda: compute_abr(y_true, y_score, _abr_range[0], _abr_range[1]),
    }

    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower in metric_funcs:
            results[metric_lower] = metric_funcs[metric_lower]()
        else:
            raise ValueError(f"Unknown metric: {metric}. Supported: {list(metric_funcs.keys())}")

    return results
