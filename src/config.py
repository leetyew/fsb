"""
Configuration management using pydantic.

All config classes use pydantic for validation and YAML loading.
Config files are stored in configs/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import yaml
from pydantic import BaseModel, Field


# Base path for config files
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


class GaussianMixtureConfig(BaseModel):
    """Configuration for Gaussian mixture components."""

    mu_good_base: List[float] = Field(default=[0.0, 0.0])
    mu_bad_base: List[float] = Field(default=[2.0, 1.0])
    component_offset: float = 1.0
    sigma_max: float = 1.0


class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic data generation."""

    random_seed: int = 42
    n_features: int = 2
    n_components: int = 2
    bad_rate: float = 0.70
    n_holdout: int = 3000
    gaussian_mixture: GaussianMixtureConfig = Field(
        default_factory=GaussianMixtureConfig
    )

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> SyntheticDataConfig:
        """Load config from YAML file.

        Args:
            path: Path to YAML file. If None, uses configs/synthetic_data.yaml.
        """
        if path is None:
            path = CONFIGS_DIR / "synthetic_data.yaml"
        return cls(**load_yaml(path))


class XGBoostConfig(BaseModel):
    """Configuration for XGBoost model.

    Default values from paper Table E.9 (synthetic experiments).
    """

    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10  # Paper default for synthetic
    validation_fraction: float = 0.2  # Fraction of training data for early stopping
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> XGBoostConfig:
        """Load config from YAML file.

        Args:
            path: Path to YAML file. If None, uses configs/model_xgboost.yaml.
        """
        if path is None:
            path = CONFIGS_DIR / "model_xgboost.yaml"
        return cls(**load_yaml(path))


class AcceptanceLoopConfig(BaseModel):
    """Configuration for acceptance loop simulation.

    Default values from paper (Appendix C.1).
    """

    n_periods: int = 500
    batch_size: int = 100
    target_accept_rate: float = 0.15  # α
    initial_batch_size: int = 100
    x_v_feature: str = "x0"  # Feature for initial acceptance (before model exists)
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> AcceptanceLoopConfig:
        """Load config from YAML file.

        Args:
            path: Path to YAML file. If None, uses configs/acceptance_loop.yaml.
        """
        if path is None:
            path = CONFIGS_DIR / "acceptance_loop.yaml"
        return cls(**load_yaml(path))


class BASLFilteringConfig(BaseModel):
    """Configuration for BASL reject filtering via novelty detection."""

    beta_lower: float = 0.05  # Remove bottom 5% outliers (most dissimilar to accepts)
    beta_upper: float = 1.0  # No upper filtering (paper default)
    random_seed: int = 42


class BASLLabelingConfig(BaseModel):
    """Configuration for BASL pseudo-labeling.

    Default values from paper for synthetic experiments.
    """

    subsample_ratio: float = 0.8  # ρ: 0.8 for synthetic, 0.3 for real data
    gamma: float = 0.01  # Percentile threshold for confident labels
    theta: float = 2.0  # Imbalance multiplier: label γ as good, θ*γ as bad
    random_seed: int = 42


class BASLConfig(BaseModel):
    """Configuration for Bias-Aware Self-Learning.

    Default values from paper.
    """

    max_iterations: int = 5  # j_max from paper
    filtering: BASLFilteringConfig = Field(default_factory=BASLFilteringConfig)
    labeling: BASLLabelingConfig = Field(default_factory=BASLLabelingConfig)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> BASLConfig:
        """Load config from YAML file.

        Args:
            path: Path to YAML file. If None, uses configs/basl.yaml.
        """
        if path is None:
            path = CONFIGS_DIR / "basl.yaml"
        return cls(**load_yaml(path))


class ExperimentConfig(BaseModel):
    """Configuration for experiments.

    Default values from paper Table E.9 and Section 7.1.1.
    """

    n_seeds: int = 100  # Paper uses 100 simulation trials
    start_seed: int = 0
    n_bayes_samples: int = 1000  # Monte Carlo samples for Bayesian eval
    track_every: int = 10  # Iteration tracking interval for Figure 2

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "ExperimentConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML file. If None, uses configs/experiment.yaml.
        """
        if path is None:
            path = CONFIGS_DIR / "experiment.yaml"
        return cls(**load_yaml(path))
