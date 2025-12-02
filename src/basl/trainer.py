"""
BASL Trainer: Orchestrates filtering and labeling stages.

Provides methods for iterative pseudo-labeling integrated with AcceptanceLoop:
1. filter_rejects_once(): Filter outlier rejects (called once at start)
2. label_one_iteration(): One iteration of weak learner labeling
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.basl.filtering import filter_rejects
from src.basl.labeling import label_rejects_iteration
from src.config import BASLConfig


class BASLTrainer:
    """Orchestrates BASL filtering and pseudo-labeling stages.

    BASL (Bias-Aware Self-Learning) augments the training data by:
    1. Filtering rejects via novelty detection to remove outliers (once).
    2. Iteratively pseudo-labeling confident rejects using a weak learner.

    Used by AcceptanceLoop which controls iteration count via early stopping.
    """

    def __init__(self, cfg: BASLConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.labeling.random_seed)
        self._fixed_thresholds: Optional[Tuple[float, float]] = None

    def filter_rejects_once(
        self,
        X_accepts: np.ndarray,
        X_rejects: np.ndarray,
    ) -> np.ndarray:
        """Filter rejects using novelty detection (Isolation Forest).

        Should be called ONCE at the start before iterative labeling.

        Args:
            X_accepts: Accept features for training novelty detector.
            X_rejects: Reject features to filter.

        Returns:
            Filtered reject features (outliers removed).
        """
        X_r_filtered, _ = filter_rejects(X_accepts, X_rejects, self.cfg.filtering)
        return X_r_filtered

    def label_one_iteration(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_rejects_pool: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
        """Perform one iteration of pseudo-labeling.

        Called iteratively by AcceptanceLoop. Thresholds are fixed after
        the first call.

        Args:
            X_labeled: Current labeled features (accepts + pseudo-labeled rejects).
            y_labeled: Current labels.
            X_rejects_pool: Remaining unlabeled reject features.

        Returns:
            Tuple of (X_new, y_new, remaining_pool_indices, thresholds):
            - X_new: Features of newly labeled rejects this iteration.
            - y_new: Pseudo-labels for newly labeled rejects.
            - remaining_pool_indices: Indices of rejects still in pool.
            - thresholds: (tau_good, tau_bad) used this iteration.
        """
        if len(X_rejects_pool) == 0:
            return (
                np.array([]).reshape(0, X_labeled.shape[1]),
                np.array([], dtype=int),
                np.array([], dtype=int),
                self._fixed_thresholds or (0.0, 1.0),
            )

        X_new, y_new, remaining_indices, thresholds = label_rejects_iteration(
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            X_rejects_pool=X_rejects_pool,
            cfg=self.cfg.labeling,
            rng=self.rng,
            fixed_thresholds=self._fixed_thresholds,
        )

        # Fix thresholds after first iteration
        if self._fixed_thresholds is None:
            self._fixed_thresholds = thresholds

        return X_new, y_new, remaining_indices, thresholds

    def reset_thresholds(self) -> None:
        """Reset fixed thresholds for a new training session."""
        self._fixed_thresholds = None
