"""
Abstract data backend interface and replicate data contract.

Defines the contract that both real and synthetic backends must fulfill,
ensuring the experiment runners are truly backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ReplicateData:
    """Data for a single replicate in the experiment.

    Per plan Part B.1: This is the contract that all backends must expose.
    Contains train/val accepts, unlabeled rejects, and labeled holdout.

    Attributes:
        Da_train_X: Features for training accepts.
        Da_train_y: Labels for training accepts.
        Da_val_X: Features for validation accepts (held out from training).
        Da_val_y: Labels for validation accepts.
        Dr_X: Reject features (unlabeled - no y column).
        H_X: Holdout features.
        H_y: Holdout labels.
        feature_names: List of feature column names.
        replicate_id: Global replicate ID (0 to n_replicates-1).
        fold_id: KFold ID (0 to n_folds-1).
        bootstrap_id: Bootstrap ID (0 to n_bootstraps-1).
        replicate_key: Human-readable key like "fold=2|boot=13".
    """

    Da_train_X: np.ndarray
    Da_train_y: np.ndarray
    Da_val_X: np.ndarray
    Da_val_y: np.ndarray
    Dr_X: np.ndarray
    H_X: np.ndarray
    H_y: np.ndarray
    feature_names: List[str]
    replicate_id: int
    fold_id: int
    bootstrap_id: int
    replicate_key: str

    def __post_init__(self) -> None:
        """Validate data integrity after initialization."""
        # Labels must be binary
        for arr, name in [(self.Da_train_y, "Da_train_y"),
                          (self.Da_val_y, "Da_val_y"),
                          (self.H_y, "H_y")]:
            unique = set(np.unique(arr))
            if not unique.issubset({0, 1}):
                raise ValueError(f"{name} must be binary (0,1), got {unique}")

        # Feature dimensions must match label counts
        assert len(self.Da_train_X) == len(self.Da_train_y), (
            f"Da_train shape mismatch: X={len(self.Da_train_X)}, y={len(self.Da_train_y)}"
        )
        assert len(self.Da_val_X) == len(self.Da_val_y), (
            f"Da_val shape mismatch: X={len(self.Da_val_X)}, y={len(self.Da_val_y)}"
        )
        assert len(self.H_X) == len(self.H_y), (
            f"H shape mismatch: X={len(self.H_X)}, y={len(self.H_y)}"
        )

        # Feature column count must be consistent
        n_features = len(self.feature_names)
        for arr, name in [(self.Da_train_X, "Da_train_X"),
                          (self.Da_val_X, "Da_val_X"),
                          (self.Dr_X, "Dr_X"),
                          (self.H_X, "H_X")]:
            if arr.ndim != 2 or arr.shape[1] != n_features:
                raise ValueError(
                    f"{name} must have {n_features} columns, got shape {arr.shape}"
                )

    @property
    def n_accepts_train(self) -> int:
        """Number of training accepts."""
        return len(self.Da_train_y)

    @property
    def n_accepts_val(self) -> int:
        """Number of validation accepts."""
        return len(self.Da_val_y)

    @property
    def n_accepts(self) -> int:
        """Total number of accepts (train + val)."""
        return self.n_accepts_train + self.n_accepts_val

    @property
    def n_rejects(self) -> int:
        """Number of rejects."""
        return len(self.Dr_X)

    @property
    def n_holdout(self) -> int:
        """Number of holdout samples."""
        return len(self.H_y)

    @property
    def accepts_bad_rate(self) -> float:
        """Bad rate among all accepts (train + val)."""
        all_y = np.concatenate([self.Da_train_y, self.Da_val_y])
        return float(all_y.mean())

    @property
    def holdout_bad_rate(self) -> float:
        """Bad rate in holdout."""
        return float(self.H_y.mean())


class DataBackend(ABC):
    """Abstract base class for data backends.

    Per plan Part B.2: Both real and synthetic backends must implement
    this interface to provide replicates to the experiment runners.
    """

    @abstractmethod
    def n_replicates(self) -> int:
        """Return total number of replicates.

        Default: 4 folds x 25 bootstraps = 100 replicates.
        """
        pass

    @abstractmethod
    def get_replicate(self, i: int) -> ReplicateData:
        """Return data for replicate index i.

        Args:
            i: Replicate index (0 to n_replicates-1).

        Returns:
            ReplicateData containing all data for this replicate.
        """
        pass

    def __len__(self) -> int:
        """Number of replicates."""
        return self.n_replicates()

    def __iter__(self):
        """Iterate over all replicates."""
        for i in range(self.n_replicates()):
            yield self.get_replicate(i)

    def get_replicate_by_key(self, fold_id: int, bootstrap_id: int) -> ReplicateData:
        """Get replicate by fold and bootstrap ID.

        Args:
            fold_id: KFold ID (0 to n_folds-1).
            bootstrap_id: Bootstrap ID (0 to n_bootstraps-1).

        Returns:
            ReplicateData for the specified fold and bootstrap.
        """
        # Default implementation: linear search
        for i in range(self.n_replicates()):
            rep = self.get_replicate(i)
            if rep.fold_id == fold_id and rep.bootstrap_id == bootstrap_id:
                return rep
        raise ValueError(f"No replicate found for fold={fold_id}, bootstrap={bootstrap_id}")
