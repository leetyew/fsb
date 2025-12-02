"""
L1-regularized Logistic Regression model (weak learner for BASL).

Used for pseudo-labeling rejects in BASL Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class LogisticRegressionConfig:
    """Configuration for L1 Logistic Regression."""

    penalty: str = "l1"
    C: float = 1.0  # Inverse regularization strength
    solver: str = "saga"  # Required for L1 penalty
    max_iter: int = 1000
    random_seed: int = 42


class LogisticRegressionModel:
    """L1-regularized Logistic Regression for BASL weak learner."""

    def __init__(self, cfg: LogisticRegressionConfig | None = None):
        """Initialize model with config.

        Args:
            cfg: Configuration. Uses defaults if None.
        """
        self.cfg = cfg or LogisticRegressionConfig()
        self._model = LogisticRegression(
            penalty=self.cfg.penalty,
            C=self.cfg.C,
            solver=self.cfg.solver,
            max_iter=self.cfg.max_iter,
            random_state=self.cfg.random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model on training data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (0=good, 1=bad).
        """
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of bad (y=1).

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Probability of class 1 (bad) for each sample.
        """
        return self._model.predict_proba(X)[:, 1]
