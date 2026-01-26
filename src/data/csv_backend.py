"""
CSV-based data backend for loading static data bundles.

Loads Da.csv, Dr.csv, H.csv and builds 4x25 replicates per plan Part B.3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.backend import DataBackend, ReplicateData
from src.data.splitters import ReplicateIndex, build_replicate_index


class CSVBackend(DataBackend):
    """Backend that loads from static CSV bundle.

    Per plan Part B.3:
    - Loads Da.csv, Dr.csv, H.csv from data_dir
    - Da has labels (y column)
    - Dr is unlabeled (no y column)
    - H has labels (y column)
    - Builds 4-fold x 25-bootstrap = 100 replicates
    """

    def __init__(
        self,
        data_dir: str,
        n_folds: int = 4,
        n_bootstraps: int = 25,
        random_seed: int = 42,
    ):
        """Initialize CSV backend.

        Args:
            data_dir: Path to directory containing Da.csv, Dr.csv, H.csv.
            n_folds: Number of folds for CV (default 4).
            n_bootstraps: Number of bootstrap samples (default 25).
            random_seed: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.n_folds = n_folds
        self.n_bootstraps = n_bootstraps
        self.random_seed = random_seed

        # Load data files
        self._load_data()

        # Build replicate index
        self._replicate_index = build_replicate_index(
            n_accepts=len(self._Da),
            n_holdout=len(self._H),
            n_folds=n_folds,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
        )

    def _load_data(self) -> None:
        """Load Da.csv, Dr.csv, H.csv and validate."""
        da_path = self.data_dir / "Da.csv"
        dr_path = self.data_dir / "Dr.csv"
        h_path = self.data_dir / "H.csv"

        # Check files exist
        for path in [da_path, dr_path, h_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        # Load CSVs
        self._Da = pd.read_csv(da_path)
        self._Dr = pd.read_csv(dr_path)
        self._H = pd.read_csv(h_path)

        # Validate label integrity per plan Part A.5
        self._validate_labels()

        # Load metadata if available
        meta_path = self.data_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {}

    def _validate_labels(self) -> None:
        """Validate label integrity per plan sanity gates."""
        # Da must have y column with binary labels
        if "y" not in self._Da.columns:
            raise ValueError("Da.csv must have 'y' column")
        da_y_unique = set(self._Da["y"].unique())
        if not da_y_unique.issubset({0, 1}):
            raise ValueError(f"Da.y must be binary (0,1), got {da_y_unique}")

        # H must have y column with binary labels
        if "y" not in self._H.columns:
            raise ValueError("H.csv must have 'y' column")
        h_y_unique = set(self._H["y"].unique())
        if not h_y_unique.issubset({0, 1}):
            raise ValueError(f"H.y must be binary (0,1), got {h_y_unique}")

        # Dr must NOT have y column (unlabeled)
        if "y" in self._Dr.columns:
            raise ValueError(
                "Dr.csv must NOT have 'y' column. "
                "Rejects must be unlabeled per plan Part A.5."
            )

        # Check bad_rate(Da) < bad_rate(H) per plan sanity gate
        da_bad_rate = self._Da["y"].mean()
        h_bad_rate = self._H["y"].mean()
        if da_bad_rate >= h_bad_rate:
            raise ValueError(
                f"Sanity gate failed: bad_rate(Da)={da_bad_rate:.4f} >= "
                f"bad_rate(H)={h_bad_rate:.4f}. "
                "Selection bias should result in lower bad rate among accepts."
            )

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature column names."""
        # All columns except 'y' are features
        return [c for c in self._Da.columns if c != "y"]

    @property
    def metadata(self) -> dict:
        """Return metadata from meta.json if available."""
        return self._meta

    def n_replicates(self) -> int:
        """Return total number of replicates."""
        return len(self._replicate_index)

    def get_replicate(self, i: int) -> ReplicateData:
        """Return data for replicate index i.

        Per plan:
        - KFold split is on Da rows
        - Bootstrap sampling is on H rows
        - Dr is fixed (same for all replicates)
        """
        if i < 0 or i >= len(self._replicate_index):
            raise IndexError(f"Replicate index {i} out of range [0, {len(self._replicate_index)})")

        idx = self._replicate_index[i]
        features = self.feature_names

        # Get Da features and labels
        Da_X = self._Da[features].values
        Da_y = self._Da["y"].values

        # Apply KFold split to Da
        Da_train_X = Da_X[idx.train_indices]
        Da_train_y = Da_y[idx.train_indices]
        Da_val_X = Da_X[idx.val_indices]
        Da_val_y = Da_y[idx.val_indices]

        # Dr is fixed (unlabeled)
        Dr_X = self._Dr[features].values

        # Apply bootstrap to H
        H_X = self._H[features].values[idx.holdout_indices]
        H_y = self._H["y"].values[idx.holdout_indices]

        return ReplicateData(
            Da_train_X=Da_train_X,
            Da_train_y=Da_train_y,
            Da_val_X=Da_val_X,
            Da_val_y=Da_val_y,
            Dr_X=Dr_X,
            H_X=H_X,
            H_y=H_y,
            feature_names=features,
            replicate_id=idx.replicate_id,
            fold_id=idx.fold_id,
            bootstrap_id=idx.bootstrap_id,
            replicate_key=idx.replicate_key,
        )

    def get_summary(self) -> dict:
        """Return summary statistics about the loaded data."""
        return {
            "n_accepts": len(self._Da),
            "n_rejects": len(self._Dr),
            "n_holdout": len(self._H),
            "n_features": len(self.feature_names),
            "n_folds": self.n_folds,
            "n_bootstraps": self.n_bootstraps,
            "n_replicates": self.n_replicates(),
            "accepts_bad_rate": float(self._Da["y"].mean()),
            "holdout_bad_rate": float(self._H["y"].mean()),
            "feature_names": self.feature_names,
        }
