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
    """L1-regularized Logistic Regression for BASL weak learner.

    Note: This wrapper returns P(y=1) as a 1D array.
    """

    def __init__(self, cfg: LogisticRegressionConfig | None = None):
        self.cfg = cfg or LogisticRegressionConfig()
        self._model: LogisticRegression | None = None
        self._single_class: int | None = None  # 0 or 1 if degenerate
        self._eps = getattr(self.cfg, "eps", 1e-6)

    def _create_model(self) -> LogisticRegression:
        """Create a fresh LogisticRegression instance."""
        return LogisticRegression(
            penalty=self.cfg.penalty,
            C=self.cfg.C,
            solver=self.cfg.solver,
            max_iter=self.cfg.max_iter,
            random_state=self.cfg.random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            self._single_class = int(unique_classes[0])
            self._model = None  # Clear any previous model to prevent accidental use
            return
        self._single_class = None
        self._model = self._create_model()
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1) for each sample."""
        if self._single_class is not None:
            if self._single_class == 0:
                return np.full(len(X), self._eps, dtype=float)
            else:
                return np.full(len(X), 1.0 - self._eps, dtype=float)
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)[:, 1]
