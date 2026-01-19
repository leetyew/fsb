"""
XGBoost model wrapper for credit scoring.

Provides a simple interface for XGBoost binary classification with
default hyperparameters from the paper (Table E.9).
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.config import XGBoostConfig


class XGBoostModel:
    """XGBoost binary classifier wrapper.

    Wraps xgboost.XGBClassifier with a simplified interface for
    credit scoring. Outputs probability of default (PD) which is
    then used with an α-percentile cutoff for acceptance decisions.

    Paper-faithful (Table E.8): No early stopping for synthetic experiments.
    Uses fixed 100 trees with subsample=0.8 and colsample=0.8.
    """

    def __init__(self, cfg: XGBoostConfig) -> None:
        self.cfg = cfg
        self._model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on labeled data.

        Paper-faithful (Table E.8): No early stopping, no validation split.
        Uses full training data with fixed number of trees.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,), where 1=bad, 0=good.
        """
        self._model = xgb.XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            random_state=self.cfg.random_seed,
            objective="binary:logistic",
            eval_metric="auc",
            # No early_stopping_rounds per Table E.8 (synthetic)
        )
        # Train on full data without validation split
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of default (PD) for each applicant.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) with PD estimates (P(y=1|X)).
            Lower values indicate lower risk applicants.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        # XGBClassifier.predict_proba returns (n_samples, 2) for binary
        return self._model.predict_proba(X)[:, 1]
