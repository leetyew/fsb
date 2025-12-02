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

    Uses early stopping with a validation split to prevent overfitting,
    which is important when training on BASL-augmented data with
    pseudo-labeled rejects.
    """

    def __init__(self, cfg: XGBoostConfig) -> None:
        self.cfg = cfg
        self._model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on labeled data with early stopping.

        Splits data into train/validation for early stopping to prevent
        overfitting, especially important for BASL-augmented training.

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
            early_stopping_rounds=self.cfg.early_stopping_rounds,
        )

        # Split data for early stopping validation
        if len(X) > 50 and self.cfg.validation_fraction > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.cfg.validation_fraction,
                random_state=self.cfg.random_seed,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            # Not enough data for split, train without early stopping
            self._model.set_params(early_stopping_rounds=None)
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
