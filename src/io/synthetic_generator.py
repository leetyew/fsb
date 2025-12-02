"""
Synthetic data generator for credit scoring experiments.

Implements Gaussian Mixture Model generation per paper Appendix E.1.
Key parameters from Table E.9:
- C=2 components
- μ_g1=(0,0), μ_b1=(2,1) for goods/bads in component 1
- Covariance entries from U(0, σ_max), σ_max=1.0
- Default bad_rate=0.70, n_holdout=3000
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SyntheticDataConfig


class SyntheticGenerator:
    """
    Generates synthetic credit data using Gaussian Mixture Models.

    Each applicant belongs to one of C mixture components and is either
    'good' (y=0) or 'bad' (y=1). Label is assigned first via Bernoulli(bad_rate),
    then features are sampled from class-specific Gaussian distributions.
    """

    def __init__(self, cfg: SyntheticDataConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

        # Pre-generate GMM parameters at init for consistency across samples
        self._mu_good, self._mu_bad = self._generate_means()
        self._sigma_good, self._sigma_bad = self._generate_covariances()

    def _generate_means(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate component-specific means for goods and bads.

        From paper Table E.9:
        - μ_g1 = (0, 0), μ_b1 = (2, 1)
        - μ_gc = μ_g1 + (c-1), μ_bc = μ_b1 + (c-1) for c > 1

        Returns:
            (mu_good, mu_bad): Arrays of shape (n_components, n_features)
        """
        n_comp = self.cfg.n_components
        n_feat = self.cfg.n_features
        offset = self.cfg.gaussian_mixture.component_offset

        mu_g_base = np.array(self.cfg.gaussian_mixture.mu_good_base)
        mu_b_base = np.array(self.cfg.gaussian_mixture.mu_bad_base)

        # Pad or truncate to match n_features
        mu_g_base = self._adjust_array_length(mu_g_base, n_feat)
        mu_b_base = self._adjust_array_length(mu_b_base, n_feat)

        mu_good = np.zeros((n_comp, n_feat))
        mu_bad = np.zeros((n_comp, n_feat))

        for c in range(n_comp):
            mu_good[c] = mu_g_base + c * offset
            mu_bad[c] = mu_b_base + c * offset

        return mu_good, mu_bad

    def _adjust_array_length(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Pad with zeros or truncate array to target length."""
        if len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)))
        return arr[:target_len]

    def _generate_covariances(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate component-specific covariance matrices.

        Per paper: entries sampled from U(0, σ_max), then made
        symmetric positive-definite via Σ = A @ A.T + εI

        Returns:
            (sigma_good, sigma_bad): Arrays of shape (n_components, n_features, n_features)
        """
        n_comp = self.cfg.n_components
        n_feat = self.cfg.n_features
        sigma_max = self.cfg.gaussian_mixture.sigma_max
        eps = 1e-6

        sigma_good = np.zeros((n_comp, n_feat, n_feat))
        sigma_bad = np.zeros((n_comp, n_feat, n_feat))

        for c in range(n_comp):
            A_good = self.rng.uniform(0, sigma_max, size=(n_feat, n_feat))
            A_bad = self.rng.uniform(0, sigma_max, size=(n_feat, n_feat))

            sigma_good[c] = A_good @ A_good.T + eps * np.eye(n_feat)
            sigma_bad[c] = A_bad @ A_bad.T + eps * np.eye(n_feat)

        return sigma_good, sigma_bad

    def _sample_applicants(self, n_samples: int) -> pd.DataFrame:
        """
        Sample applicants from the GMM population.

        Steps per paper:
        1. Sample component assignment uniformly
        2. Sample label y from Bernoulli(bad_rate)
        3. Sample features from N(μ, Σ) for assigned component and class

        Args:
            n_samples: Number of applicants to generate.

        Returns:
            DataFrame with feature columns (x0, x1, ...) and 'y' label.
        """
        n_feat = self.cfg.n_features
        n_comp = self.cfg.n_components

        components = self.rng.integers(0, n_comp, size=n_samples)
        labels = self.rng.binomial(1, self.cfg.bad_rate, size=n_samples)

        features = np.zeros((n_samples, n_feat))

        for c in range(n_comp):
            # Good applicants (y=0) in component c
            mask_good = (components == c) & (labels == 0)
            n_good = mask_good.sum()
            if n_good > 0:
                features[mask_good] = self.rng.multivariate_normal(
                    self._mu_good[c], self._sigma_good[c], size=n_good
                )

            # Bad applicants (y=1) in component c
            mask_bad = (components == c) & (labels == 1)
            n_bad = mask_bad.sum()
            if n_bad > 0:
                features[mask_bad] = self.rng.multivariate_normal(
                    self._mu_bad[c], self._sigma_bad[c], size=n_bad
                )

        feature_cols = [f"x{i}" for i in range(n_feat)]
        df = pd.DataFrame(features, columns=feature_cols)
        df["y"] = labels

        return df

    def generate_population(self, n_samples: int) -> pd.DataFrame:
        """
        Generate a population of applicants.

        Args:
            n_samples: Number of applicants to generate.

        Returns:
            DataFrame with features (x0, x1, ...) and label (y).
        """
        return self._sample_applicants(n_samples)

    def generate_holdout(self) -> pd.DataFrame:
        """
        Generate a representative holdout set.

        The holdout represents the true population distribution with
        observed labels for all applicants (no selection bias).

        Returns:
            DataFrame with n_holdout samples.
        """
        return self._sample_applicants(self.cfg.n_holdout)
