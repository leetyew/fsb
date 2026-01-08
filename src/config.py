"""
Configuration management using pydantic.

All config classes use pydantic for validation and YAML loading.
Config files are stored in configs/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import yaml
from pydantic import BaseModel, Field


# Base path for config files
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


class GaussianMixtureConfig(BaseModel):
    """Configuration for 2-component GMM per class.

    Per paper Algorithm C.1:
    - Each class has 2 mixture components for realistic overlap
    - Informative features (X1, X2): class-conditional means
    - Noise features (N1, N2): N(0, I) identical for both classes

    Default means create overlap between good/bad classes:
    - Good (y=0): μ_g1=(0,0), μ_g2=(1,1)
    - Bad (y=1): μ_b1=(2,1), μ_b2=(3,2)
    """

    # Covariance scale for informative features
    # Per Appendix E.1: σ_max = 1 for MAR baseline
    sigma_max: float = 1.0

    # Good class (y=0) mixture component means for X1, X2
    mu_good_1: List[float] = Field(default=[0.0, 0.0])
    mu_good_2: List[float] = Field(default=[1.0, 1.0])

    # Bad class (y=1) mixture component means for X1, X2
    mu_bad_1: List[float] = Field(default=[2.0, 1.0])
    mu_bad_2: List[float] = Field(default=[3.0, 2.0])

    # Mixture weights (equal by default)
    weight_good: List[float] = Field(default=[0.5, 0.5])
    weight_bad: List[float] = Field(default=[0.5, 0.5])


class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic data generation.

    Paper-faithful implementation per Algorithm C.1:
    1. Generate y ~ Bernoulli(bad_rate)
    2. Generate X|y from class-conditional Gaussians:
       - X1, X2: informative features (different means for good/bad)
       - N1, N2: noise features (same distribution for good/bad)
    3. x_v = the most separating feature (largest |mu_good - mu_bad|)

    The model f_a(X) trains on ALL features [X1, X2, N1, N2].
    x_v is part of X (typically X1), used for initial acceptance ranking.
    """

    random_seed: int = 42
    n_components: int = 2  # Number of GMM components
    bad_rate: float = 0.70  # Target bad rate
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

    Per paper Section 6.1 and Algorithm C.2:
    - Runs for all n_periods without early stopping
    - All accepts (D^a) used for training
    - Separate external holdout for oracle evaluation

    Acceptance modes:
    - "feature": Use x_v (most separating feature) threshold (for Exp I)
    - "model": Use f_a(X) model scores after first batch (for Exp II - BASL dynamics)

    Note: x_v is determined by the generator as the feature with largest
    |mu_good - mu_bad|, typically X1.
    """

    n_periods: int = 500
    batch_size: int = 100
    target_accept_rate: float = 0.15  # α
    initial_batch_size: int = 100
    acceptance_mode: str = "feature"  # "feature" for Exp I, "model" for Exp II
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

    Default values from paper (Appendix E.1).
    """

    max_iterations: int = 3  # j_max from paper (Appendix E.1: "jmax = 3")
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


class BayesianEvalConfig(BaseModel):
    """Configuration for Bayesian evaluation (Algorithm 1).

    Default values from paper Table E.9 and Section 6.3.

    Two pseudo-labeling modes:
    - Direct (use_banding=False, default): Paper-faithful approach where each
      reject's label is sampled as Binomial(1, P(y^r|X^r)) using model predictions.
    - Banded (use_banding=True): Variance reduction via score stratification,
      using Beta posteriors estimated from accepts in each band.
    """

    # Pseudo-labeling mode
    use_banding: bool = False  # False = paper-faithful direct mode

    # Banding parameters (only used when use_banding=True)
    n_bands: int = 10  # K: number of score bands
    prior_alpha: float = 1.0  # Beta prior alpha (uninformative)
    prior_beta: float = 1.0  # Beta prior beta (uninformative)

    # MC convergence parameters (from paper Table E.9)
    j_min: int = 100  # Minimum MC samples before convergence check
    j_max: int = 100000  # Maximum MC samples (paper: 10^6, using 10^5 for speed)
    epsilon: float = 1e-6  # Convergence threshold

    random_seed: int = 42

    # ABR integration range (paper Section 6.3: "integrate ABR over acceptance
    # between 20% and 40%, which reflects historical policies at Monedo")
    abr_range: Tuple[float, float] = (0.2, 0.4)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "BayesianEvalConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML file. If None, uses configs/bayesian_eval.yaml.
        """
        if path is None:
            path = CONFIGS_DIR / "bayesian_eval.yaml"
        return cls(**load_yaml(path))
