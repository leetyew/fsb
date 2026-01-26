"""
Splitting utilities for creating replicates.

Implements:
- KFold splits on accepts (Da)
- Bootstrap sampling on holdout (H)
- Replicate index builder for 4x25 = 100 replicates

Per plan Part B: Dr is fixed (not bootstrapped) across all replicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import KFold


@dataclass
class ReplicateIndex:
    """Index for a single replicate.

    Stores indices rather than data to allow lazy loading.
    """

    replicate_id: int
    fold_id: int
    bootstrap_id: int
    replicate_key: str
    train_indices: np.ndarray    # Indices into Da for training
    val_indices: np.ndarray      # Indices into Da for validation
    holdout_indices: np.ndarray  # Indices into H (with replacement)


def create_kfold_splits(
    n_samples: int,
    n_folds: int = 4,
    random_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create KFold train/val splits on accepts.

    Args:
        n_samples: Number of accept samples.
        n_folds: Number of folds (default 4).
        random_seed: Random seed for reproducibility.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    indices = np.arange(n_samples)
    return [(train_idx, val_idx) for train_idx, val_idx in kf.split(indices)]


def create_bootstrap_samples(
    n_samples: int,
    n_bootstraps: int = 25,
    random_seed: int = 42,
) -> List[np.ndarray]:
    """Create bootstrap samples of holdout.

    Args:
        n_samples: Number of holdout samples.
        n_bootstraps: Number of bootstrap samples (default 25).
        random_seed: Random seed for reproducibility.

    Returns:
        List of index arrays (sampling with replacement).
    """
    rng = np.random.default_rng(random_seed)
    return [
        rng.choice(n_samples, size=n_samples, replace=True)
        for _ in range(n_bootstraps)
    ]


def build_replicate_index(
    n_accepts: int,
    n_holdout: int,
    n_folds: int = 4,
    n_bootstraps: int = 25,
    random_seed: int = 42,
) -> List[ReplicateIndex]:
    """Build complete replicate index for 4x25 replicates.

    Per plan Part B.3:
    - KFold split is on rows of Da only
    - Bootstrap sampling is on rows of H only
    - Dr is fixed (not bootstrapped)

    Args:
        n_accepts: Number of accept samples (Da).
        n_holdout: Number of holdout samples (H).
        n_folds: Number of folds for CV (default 4).
        n_bootstraps: Number of bootstrap samples (default 25).
        random_seed: Random seed for reproducibility.

    Returns:
        List of ReplicateIndex objects, one per replicate.
    """
    # Create KFold splits on accepts
    # Use separate seed offset to ensure different from bootstrap
    kfold_splits = create_kfold_splits(n_accepts, n_folds, random_seed)

    # Create bootstrap samples of holdout
    # Use offset seed to ensure different from KFold
    bootstrap_samples = create_bootstrap_samples(
        n_holdout, n_bootstraps, random_seed + 1000
    )

    # Build Cartesian product: n_folds x n_bootstraps replicates
    replicates = []
    replicate_id = 0

    for fold_id, (train_idx, val_idx) in enumerate(kfold_splits):
        for boot_id, holdout_idx in enumerate(bootstrap_samples):
            replicate_key = f"fold={fold_id}|boot={boot_id}"
            rep = ReplicateIndex(
                replicate_id=replicate_id,
                fold_id=fold_id,
                bootstrap_id=boot_id,
                replicate_key=replicate_key,
                train_indices=train_idx,
                val_indices=val_idx,
                holdout_indices=holdout_idx,
            )
            replicates.append(rep)
            replicate_id += 1

    return replicates
