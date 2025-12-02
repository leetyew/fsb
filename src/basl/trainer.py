"""
BASL Trainer: Orchestrates filtering and labeling stages.

Returns augmented data (accepts + pseudo-labeled rejects) for
external model training.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.basl.filtering import filter_rejects
from src.basl.labeling import label_rejects_iteration
from src.config import BASLConfig


class BASLTrainer:
    """Orchestrates BASL filtering and pseudo-labeling stages.

    BASL (Bias-Aware Self-Learning) augments the training data by:
    1. Filtering rejects via novelty detection to remove outliers.
    2. Iteratively pseudo-labeling confident rejects using a weak learner.

    The caller handles final model training with the augmented data.
    """

    def __init__(self, cfg: BASLConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.labeling.random_seed)

    def run(
        self,
        X_a: np.ndarray,
        y_a: np.ndarray,
        X_r: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run BASL to create augmented training data.

        Args:
            X_a: Accept features (n_accepts, n_features).
            y_a: Accept labels (n_accepts,).
            X_r: Reject features (n_rejects, n_features). No labels available.

        Returns:
            Tuple of (X_augmented, y_augmented):
            - X_augmented: Accepts + pseudo-labeled rejects.
            - y_augmented: Labels for augmented data.
        """
        # Stage 1: Filter rejects via novelty detection
        X_r_filtered, _ = filter_rejects(X_a, X_r, self.cfg.filtering)

        if len(X_r_filtered) == 0:
            # No rejects passed filtering, return accepts only
            return X_a.copy(), y_a.copy()

        # Stage 2: Initialize labeled and unlabeled sets
        L_X = X_a.copy()
        L_y = y_a.copy()
        U_X = X_r_filtered.copy()

        # Stage 3: Iterative pseudo-labeling
        fixed_thresholds = None

        for j in range(self.cfg.max_iterations):
            if len(U_X) == 0:
                # No more rejects to label
                break

            # Perform one labeling iteration
            X_new, y_new, remaining_indices, thresholds = label_rejects_iteration(
                X_labeled=L_X,
                y_labeled=L_y,
                X_rejects_pool=U_X,
                cfg=self.cfg.labeling,
                rng=self.rng,
                fixed_thresholds=fixed_thresholds,
            )

            # Fix thresholds after first iteration
            if j == 0:
                fixed_thresholds = thresholds

            # Append newly labeled rejects to labeled set
            if len(X_new) > 0:
                L_X = np.vstack([L_X, X_new])
                L_y = np.concatenate([L_y, y_new])

            # Update pool to remaining unlabeled rejects
            U_X = U_X[remaining_indices]

        return L_X, L_y
