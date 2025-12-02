"""
BASL Stage 1: Reject filtering via novelty detection.

Uses Isolation Forest to identify and remove outlier rejects that are
too dissimilar to accepts (extreme outliers) or too similar (already
well-represented in accepts).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import BASLFilteringConfig


def filter_rejects(
    X_accepts: np.ndarray,
    X_rejects: np.ndarray,
    cfg: BASLFilteringConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter rejects using Isolation Forest novelty detection.

    Fits an Isolation Forest on accepts to learn the "normal" distribution,
    then scores rejects by similarity. Keeps only rejects in the middle
    percentile range [beta_lower, beta_upper].

    Args:
        X_accepts: Feature matrix of accepted applicants (n_accepts, n_features).
        X_rejects: Feature matrix of rejected applicants (n_rejects, n_features).
        cfg: Filtering configuration with beta_lower, beta_upper, random_seed.

    Returns:
        Tuple of (X_rejects_filtered, indices_kept):
        - X_rejects_filtered: Filtered reject features.
        - indices_kept: Original indices of kept rejects (for tracking).
    """
    # Fit Isolation Forest on accepts to learn normal distribution
    iforest = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=cfg.random_seed,
    )
    iforest.fit(X_accepts)

    # Score rejects: higher score = more similar to accepts (less outlier-like)
    outlier_scores = iforest.decision_function(X_rejects)

    # Compute thresholds based on reject score distribution
    lower_thresh = np.percentile(outlier_scores, cfg.beta_lower * 100)
    upper_thresh = np.percentile(outlier_scores, cfg.beta_upper * 100)

    # Keep rejects in [beta_lower, beta_upper] percentile range
    keep_mask = (outlier_scores >= lower_thresh) & (outlier_scores <= upper_thresh)
    indices_kept = np.where(keep_mask)[0]

    return X_rejects[keep_mask], indices_kept
