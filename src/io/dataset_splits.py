"""
Dataset splitting utilities for train/val/test partitioning.

Supports:
- Random splits by row
- Time-based splits (requires datetime column)
- Stratified splits (maintains label distribution)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplits(NamedTuple):
    """Container for train/val/test splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_random_splits(
    df: pd.DataFrame,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    stratify_col: str | None = None,
    random_seed: int = 42,
) -> DatasetSplits:
    """Split dataframe randomly into train/val/test.

    Args:
        df: Input dataframe.
        train_fraction: Fraction for training.
        val_fraction: Fraction for validation.
        test_fraction: Fraction for test.
        stratify_col: Column name to stratify by (e.g., "y" for label).
        random_seed: Random seed for reproducibility.

    Returns:
        DatasetSplits with train, val, test dataframes.
    """
    # Validate fractions sum to 1
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    stratify = df[stratify_col] if stratify_col else None

    # First split: train vs (val + test)
    val_test_fraction = val_fraction + test_fraction
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_fraction,
        stratify=stratify,
        random_state=random_seed,
    )

    # Second split: val vs test
    # Adjust fraction for second split: test_fraction / (val_fraction + test_fraction)
    test_of_remainder = test_fraction / val_test_fraction
    stratify_vt = val_test_df[stratify_col] if stratify_col else None

    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_of_remainder,
        stratify=stratify_vt,
        random_state=random_seed,
    )

    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def make_time_splits(
    df: pd.DataFrame,
    time_col: str,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> DatasetSplits:
    """Split dataframe by time (chronological order).

    Sorts by time column and takes sequential chunks.
    This prevents data leakage from future to past.

    Args:
        df: Input dataframe.
        time_col: Column name containing datetime/timestamp.
        train_fraction: Fraction for training (earliest data).
        val_fraction: Fraction for validation (middle data).
        test_fraction: Fraction for test (latest data).

    Returns:
        DatasetSplits with train, val, test dataframes.
    """
    # Validate fractions sum to 1
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    # Sort by time
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)

    # Compute split indices
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    train_df = df_sorted.iloc[:train_end].reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].reset_index(drop=True)

    return DatasetSplits(train=train_df, val=val_df, test=test_df)
