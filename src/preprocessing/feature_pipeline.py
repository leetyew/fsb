"""
Minimal feature pipeline (pass-through for synthetic data).

Extend with actual transformations when real data format is confirmed.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class FeaturePipeline:
    """Minimal feature pipeline that converts DataFrame to numpy array.

    Currently a pass-through for synthetic data. Extend with transformers
    (imputation, scaling, encoding) when real data format is confirmed.
    """

    def __init__(self, feature_cols: List[str] | None = None):
        """Initialize pipeline.

        Args:
            feature_cols: List of feature column names to use.
                If None, uses all columns except 'y'.
        """
        self.feature_cols = feature_cols
        self._fitted_cols: List[str] | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePipeline":
        """Fit pipeline (store feature columns).

        Args:
            df: Input dataframe.
            y: Target series (unused, for sklearn compatibility).

        Returns:
            Self for chaining.
        """
        if self.feature_cols is not None:
            self._fitted_cols = self.feature_cols
        else:
            # Use all columns except 'y'
            self._fitted_cols = [c for c in df.columns if c != "y"]

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe to numpy array.

        Args:
            df: Input dataframe.

        Returns:
            Numpy array of features.
        """
        if self._fitted_cols is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        return df[self._fitted_cols].values.astype(np.float64)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            df: Input dataframe.
            y: Target series (unused).

        Returns:
            Numpy array of features.
        """
        return self.fit(df, y).transform(df)

    @property
    def feature_names(self) -> List[str]:
        """Get fitted feature column names."""
        if self._fitted_cols is None:
            raise RuntimeError("Pipeline not fitted.")
        return self._fitted_cols
