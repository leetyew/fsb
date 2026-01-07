"""
Synthetic data generator for credit scoring experiments.

Paper-faithful implementation per Algorithm C.1:
1. Generate y ~ Bernoulli(bad_rate)
2. For each sample, select mixture component (2 per class)
3. Generate informative features X1, X2 from class/component-specific Gaussian
4. Generate noise features N1, N2 from N(0, I) - identical for all classes
5. x_v = the most separating feature (largest |E[X|y=0] - E[X|y=1]|)

Key design decisions per paper directives:
- Noise features are INDEPENDENT of class and informative features
- Covariance for informative features is non-degenerate and PSD
- Classes must overlap (Oracle AUC < 1.0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SyntheticDataConfig


class SyntheticGenerator:
    """
    Generates synthetic credit data per paper Algorithm C.1.

    Uses 2-component GMM per class for informative features:
    - Good (y=0): μ_g1=(0,0), μ_g2=(1,1)
    - Bad (y=1): μ_b1=(2,1), μ_b2=(3,2)

    Noise features N1, N2 are drawn from N(0, I) identically for all samples.
    x_v is the feature with largest class-mean separation (typically X1).
    """

    # Feature names matching paper Figure 2(b)
    INFORMATIVE_FEATURES = ["X1", "X2"]
    NOISE_FEATURES = ["N1", "N2"]
    ALL_FEATURES = INFORMATIVE_FEATURES + NOISE_FEATURES

    def __init__(self, cfg: SyntheticDataConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

        # Store GMM parameters for informative features
        gm = cfg.gaussian_mixture
        self._mu_good = [np.array(gm.mu_good_1), np.array(gm.mu_good_2)]
        self._mu_bad = [np.array(gm.mu_bad_1), np.array(gm.mu_bad_2)]
        self._weight_good = np.array(gm.weight_good)
        self._weight_bad = np.array(gm.weight_bad)

        # Normalize weights
        self._weight_good = self._weight_good / self._weight_good.sum()
        self._weight_bad = self._weight_bad / self._weight_bad.sum()

        # Generate covariance for informative features (shared across components)
        self._sigma_info = self._generate_informative_covariance(gm.sigma_max)

        # Determine x_v: feature with largest mean separation
        self._x_v_feature = self._find_most_separating_feature()

    @property
    def x_v_feature(self) -> str:
        """Return the name of the visible/acceptance feature (most separating)."""
        return self._x_v_feature

    @property
    def feature_cols(self) -> list[str]:
        """Return all feature column names."""
        return self.ALL_FEATURES.copy()

    def _generate_informative_covariance(self, sigma_max: float) -> np.ndarray:
        """Generate non-degenerate PSD covariance for informative features.

        Uses random matrix construction: Σ = A @ A.T + diag_jitter
        This guarantees PSD and creates realistic correlations.
        """
        n_info = len(self.INFORMATIVE_FEATURES)

        # Generate random matrix with entries in (0, sqrt(sigma_max))
        # This ensures variance is proportional to sigma_max
        A = self.rng.uniform(0.3, np.sqrt(sigma_max), size=(n_info, n_info))

        # Create PSD matrix
        cov = A @ A.T

        # Add diagonal jitter for numerical stability and ensure non-degeneracy
        min_var = 0.5  # Minimum variance to prevent tight clusters
        cov += min_var * np.eye(n_info)

        return cov

    def _find_most_separating_feature(self) -> str:
        """Find the feature with largest |E[X|y=0] - E[X|y=1]|.

        Computes expected mean for each class using mixture weights.
        This is x_v - the 'bureau score' feature used for acceptance ranking.
        """
        # Expected mean for good class (weighted avg of component means)
        mu_good_expected = (
            self._weight_good[0] * self._mu_good[0] +
            self._weight_good[1] * self._mu_good[1]
        )

        # Expected mean for bad class
        mu_bad_expected = (
            self._weight_bad[0] * self._mu_bad[0] +
            self._weight_bad[1] * self._mu_bad[1]
        )

        # Find most separating informative feature
        separation = np.abs(mu_good_expected - mu_bad_expected)
        most_sep_idx = np.argmax(separation)
        return self.INFORMATIVE_FEATURES[most_sep_idx]

    def _sample_applicants(self, n_samples: int) -> pd.DataFrame:
        """Generate applicants per Algorithm C.1.

        Order:
        1. Sample labels y ~ Bernoulli(bad_rate)
        2. For each sample, select mixture component
        3. Sample informative features from component-specific Gaussian
        4. Sample noise features from N(0, I) - class-independent
        """
        n_info = len(self.INFORMATIVE_FEATURES)
        n_noise = len(self.NOISE_FEATURES)

        # Step 1: Sample labels
        labels = self.rng.binomial(1, self.cfg.bad_rate, size=n_samples)

        # Step 2 & 3: Sample informative features from GMM
        informative = np.zeros((n_samples, n_info))

        # Good applicants (y=0)
        mask_good = labels == 0
        n_good = mask_good.sum()
        if n_good > 0:
            # Select component for each good sample
            components_good = self.rng.choice(
                2, size=n_good, p=self._weight_good
            )
            for comp in [0, 1]:
                comp_mask = components_good == comp
                n_comp = comp_mask.sum()
                if n_comp > 0:
                    informative[np.where(mask_good)[0][comp_mask]] = (
                        self.rng.multivariate_normal(
                            self._mu_good[comp], self._sigma_info, size=n_comp
                        )
                    )

        # Bad applicants (y=1)
        mask_bad = labels == 1
        n_bad = mask_bad.sum()
        if n_bad > 0:
            # Select component for each bad sample
            components_bad = self.rng.choice(
                2, size=n_bad, p=self._weight_bad
            )
            for comp in [0, 1]:
                comp_mask = components_bad == comp
                n_comp = comp_mask.sum()
                if n_comp > 0:
                    informative[np.where(mask_bad)[0][comp_mask]] = (
                        self.rng.multivariate_normal(
                            self._mu_bad[comp], self._sigma_info, size=n_comp
                        )
                    )

        # Step 4: Sample noise features from N(0, I) - INDEPENDENT of class
        # This is critical: noise must be identical distribution for y=0 and y=1
        noise = self.rng.standard_normal((n_samples, n_noise))

        # Combine into DataFrame
        features = np.hstack([informative, noise])
        df = pd.DataFrame(features, columns=self.ALL_FEATURES)
        df["y"] = labels

        return df

    def generate_population(self, n_samples: int) -> pd.DataFrame:
        """Generate a population of applicants.

        Args:
            n_samples: Number of applicants to generate.

        Returns:
            DataFrame with features [X1, X2, N1, N2] and label y.
        """
        return self._sample_applicants(n_samples)

    def generate_holdout(self) -> pd.DataFrame:
        """Generate a representative holdout set.

        Returns:
            DataFrame with n_holdout samples.
        """
        return self._sample_applicants(self.cfg.n_holdout)
