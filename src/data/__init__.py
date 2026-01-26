"""
Data loading and replication protocol for backend-agnostic experiments.

This module provides:
- DataBackend: Abstract base class for data sources
- ReplicateData: Dataclass for single replicate with train/val/rejects/holdout
- CSVBackend: Loads from static CSV bundle (Da.csv, Dr.csv, H.csv)
- Splitters: KFold and bootstrap helpers for 4x25 replicates
"""

from src.data.backend import DataBackend, ReplicateData
from src.data.csv_backend import CSVBackend
from src.data.splitters import (
    build_replicate_index,
    create_kfold_splits,
    create_bootstrap_samples,
)

__all__ = [
    "DataBackend",
    "ReplicateData",
    "CSVBackend",
    "build_replicate_index",
    "create_kfold_splits",
    "create_bootstrap_samples",
]
