"""
BASL Stage 2: Pseudo-labeling via weak learner.

Uses L1-regularized logistic regression to iteratively pseudo-label
confident rejects based on percentile thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.config import BASLLabelingConfig
from src.models.logistic_regression import LogisticRegressionConfig, LogisticRegressionModel


@dataclass
class BASLIterationStats:
    """Statistics from a single BASL labeling iteration.

    Tracks threshold values and counts for diagnostic verification.
    Paper-faithful invariant: thresholds must be fixed after iteration 1.
    """

    tau_good: float               # Bottom γ percentile threshold (score <= tau_good -> good)
    tau_bad: float                # Top γθ percentile threshold (score >= tau_bad -> bad)
    n_labeled_good: int           # Number of rejects labeled as good
    n_labeled_bad: int            # Number of rejects labeled as bad
    thresholds_recomputed: bool   # True only if thresholds were computed this iteration
    remaining_pool_size: int      # Size of reject pool after labeling
    n_subsampled: int             # Number of rejects subsampled this iteration
    score_min: float              # Min score among subsampled rejects
    score_max: float              # Max score among subsampled rejects
    score_mean: float             # Mean score among subsampled rejects


def label_rejects_iteration(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_rejects_pool: np.ndarray,
    cfg: BASLLabelingConfig,
    rng: np.random.Generator,
    fixed_thresholds: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], BASLIterationStats]:
    """Perform one iteration of pseudo-labeling on rejects.

    Steps:
    1. Subsample ρ fraction from reject pool.
    2. Fit weak learner (L1 LogReg) on current labeled data.
    3. Score subsampled rejects.
    4. Compute or use fixed thresholds for confident labeling.
    5. Label confident rejects as good (y=0) or bad (y=1).

    Args:
        X_labeled: Current labeled feature matrix (accepts + pseudo-labeled).
        y_labeled: Current labels.
        X_rejects_pool: Remaining unlabeled reject features.
        cfg: Labeling configuration.
        rng: Random number generator for subsampling.
        fixed_thresholds: If provided, use these thresholds instead of computing
            new ones. Should be (tau_good, tau_bad) from first iteration.

    Returns:
        Tuple of (X_new, y_new, remaining_indices, thresholds, stats):
        - X_new: Features of newly labeled rejects.
        - y_new: Pseudo-labels for newly labeled rejects.
        - remaining_indices: Indices of rejects still in pool (not labeled).
        - thresholds: (tau_good, tau_bad) thresholds used this iteration.
        - stats: BASLIterationStats with diagnostic information.
    """
    n_pool = len(X_rejects_pool)
    if n_pool == 0:
        # No rejects left to label
        thresholds = fixed_thresholds or (0.0, 1.0)
        stats = BASLIterationStats(
            tau_good=thresholds[0],
            tau_bad=thresholds[1],
            n_labeled_good=0,
            n_labeled_bad=0,
            thresholds_recomputed=False,
            remaining_pool_size=0,
            n_subsampled=0,
            score_min=float("nan"),
            score_max=float("nan"),
            score_mean=float("nan"),
        )
        return (
            np.array([]).reshape(0, X_labeled.shape[1]),
            np.array([]),
            np.array([]),
            thresholds,
            stats,
        )

    # Step 1: Subsample rejects
    n_subsample = max(1, int(n_pool * cfg.subsample_ratio))
    subsample_indices = rng.choice(n_pool, size=n_subsample, replace=False)
    X_subsample = X_rejects_pool[subsample_indices]

    # Step 2: Fit weak learner on current labeled data
    weak_learner_cfg = LogisticRegressionConfig(random_seed=cfg.random_seed)
    weak_learner = LogisticRegressionModel(weak_learner_cfg)
    weak_learner.fit(X_labeled, y_labeled)

    # Step 3: Score subsampled rejects (probability of being bad, y=1)
    scores = weak_learner.predict_proba(X_subsample)

    # Step 4: Compute or use fixed thresholds
    thresholds_recomputed = fixed_thresholds is None
    if fixed_thresholds is None:
        # First iteration: compute thresholds from score distribution
        tau_good = float(np.percentile(scores, cfg.gamma * 100))
        tau_bad = float(np.percentile(scores, (1 - cfg.theta * cfg.gamma) * 100))
        thresholds = (tau_good, tau_bad)
    else:
        tau_good, tau_bad = fixed_thresholds
        thresholds = fixed_thresholds

    # Step 5: Label confident rejects
    good_mask = scores <= tau_good
    bad_mask = scores >= tau_bad
    labeled_mask = good_mask | bad_mask

    # Count labels before extracting data
    n_labeled_good = int(good_mask.sum())
    n_labeled_bad = int(bad_mask.sum())

    # Extract newly labeled data
    X_new_list = []
    y_new_list = []

    if good_mask.any():
        X_new_list.append(X_subsample[good_mask])
        y_new_list.append(np.zeros(good_mask.sum(), dtype=int))

    if bad_mask.any():
        X_new_list.append(X_subsample[bad_mask])
        y_new_list.append(np.ones(bad_mask.sum(), dtype=int))

    if X_new_list:
        X_new = np.vstack(X_new_list)
        y_new = np.concatenate(y_new_list)
    else:
        X_new = np.array([]).reshape(0, X_labeled.shape[1])
        y_new = np.array([], dtype=int)

    # Determine which rejects remain in pool (not in subsample, or in subsample but not labeled)
    # Rejects that were subsampled and labeled are removed from pool
    labeled_subsample_indices = subsample_indices[labeled_mask]

    # Create mask for remaining pool
    remaining_mask = np.ones(n_pool, dtype=bool)
    remaining_mask[labeled_subsample_indices] = False
    remaining_indices = np.where(remaining_mask)[0]

    # Build stats object for diagnostics
    stats = BASLIterationStats(
        tau_good=tau_good,
        tau_bad=tau_bad,
        n_labeled_good=n_labeled_good,
        n_labeled_bad=n_labeled_bad,
        thresholds_recomputed=thresholds_recomputed,
        remaining_pool_size=len(remaining_indices),
        n_subsampled=n_subsample,
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        score_mean=float(scores.mean()),
    )

    return X_new, y_new, remaining_indices, thresholds, stats
