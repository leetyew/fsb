#!/usr/bin/env python
"""
Offline realistic synthetic data generator.

Per plan Part A: Generates a static CSV bundle with Da.csv, Dr.csv, H.csv
and snapshot diagnostics. The acceptance loop exists ONLY here - experiments
consume the static output.

Output structure:
    data/synthetic/<run_id>/
        Da.csv          # accepts, labeled (y in {0,1})
        Dr.csv          # rejects, unlabeled (NO y column)
        H.csv           # holdout, labeled
        meta.json       # parameters + achieved rates + seed + schema
        snapshots/
            iter_001.json
            iter_005.json
            ...

Usage:
    python scripts/generate_synthetic_like_real.py
    python scripts/generate_synthetic_like_real.py --config configs/synthetic_generator.yaml
    python scripts/generate_synthetic_like_real.py --run-id my_run --seed 123
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import XGBoostConfig
from src.models.xgboost_model import XGBoostModel


class RealisticSyntheticGenerator:
    """Generates realistic synthetic credit data with 50 features.

    Per plan Part A.3-A.4:
    - 30 continuous (skewed + bounded ratios)
    - 10 count/integer
    - 5 binary
    - 5 categorical (4-8 levels)
    """

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Feature counts
        self.n_continuous = cfg["features"]["n_continuous"]
        self.n_count = cfg["features"]["n_count"]
        self.n_binary = cfg["features"]["n_binary"]
        self.n_categorical = cfg["features"]["n_categorical"]
        self.n_features = cfg["features"]["total"]

        # Build feature names
        self.feature_names = self._build_feature_names()

        # Generate latent correlation structure
        self._setup_latent_structure()

    def _build_feature_names(self) -> List[str]:
        """Build list of feature names by type."""
        names = []
        names.extend([f"cont_{i}" for i in range(self.n_continuous)])
        names.extend([f"count_{i}" for i in range(self.n_count)])
        names.extend([f"bin_{i}" for i in range(self.n_binary)])
        names.extend([f"cat_{i}" for i in range(self.n_categorical)])
        return names

    def _setup_latent_structure(self) -> None:
        """Setup Gaussian copula correlation structure."""
        # Generate random correlation matrix for latent variables
        n = self.n_features
        # Random matrix -> symmetric -> ensure PSD
        A = self.rng.standard_normal((n, n))
        cov = A @ A.T / n
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(cov))
        self.latent_corr = cov / np.outer(d, d)

        # Coefficients for label generation (informative features have non-zero)
        self.label_coefs = np.zeros(n)
        # First 15 continuous features are informative
        self.label_coefs[:15] = self.rng.uniform(0.3, 0.8, 15) * self.rng.choice([-1, 1], 15)
        # Some count features are informative (randomly signed for realism)
        start_count = self.n_continuous
        self.label_coefs[start_count:start_count + 5] = (
            self.rng.uniform(0.2, 0.5, 5) * self.rng.choice([-1, 1], 5)
        )

    def _sample_latent(self, n_samples: int) -> np.ndarray:
        """Sample from Gaussian copula."""
        return self.rng.multivariate_normal(
            np.zeros(self.n_features),
            self.latent_corr,
            size=n_samples,
        )

    def _transform_features(self, z: np.ndarray) -> np.ndarray:
        """Transform latent z to observed features via nonlinear marginals."""
        n = z.shape[0]
        X = np.zeros((n, self.n_features))
        idx = 0

        # Continuous features: mix of normal, lognormal, beta-like
        for i in range(self.n_continuous):
            u = stats.norm.cdf(z[:, idx])  # Uniform via copula
            if i % 3 == 0:
                # Lognormal-like (skewed right)
                X[:, idx] = stats.lognorm.ppf(u, s=0.5, scale=1.0)
            elif i % 3 == 1:
                # Beta-like (bounded)
                X[:, idx] = stats.beta.ppf(u, a=2, b=5)
            else:
                # Normal with shift
                X[:, idx] = stats.norm.ppf(u, loc=0, scale=1)
            idx += 1

        # Count features: Poisson-like
        for i in range(self.n_count):
            u = stats.norm.cdf(z[:, idx])
            lam = 3 + i  # Different rates
            X[:, idx] = stats.poisson.ppf(np.clip(u, 0.001, 0.999), mu=lam)
            idx += 1

        # Binary features
        for i in range(self.n_binary):
            prob = 0.3 + 0.1 * i  # Different base rates
            X[:, idx] = (stats.norm.cdf(z[:, idx]) > (1 - prob)).astype(float)
            idx += 1

        # Categorical features (stored as integers)
        for i in range(self.n_categorical):
            n_levels = 4 + (i % 5)  # 4-8 levels
            u = stats.norm.cdf(z[:, idx])
            X[:, idx] = np.floor(u * n_levels).astype(int)
            idx += 1

        return X

    def _generate_labels(self, X: np.ndarray) -> np.ndarray:
        """Generate labels per plan A.4 formula.

        logit(PD) = intercept + w^T phi(x) + sparse interactions + u + eps
        """
        n = X.shape[0]
        bad_rate = self.cfg["labels"]["population_bad_rate"]

        # Base logit from features
        logit = np.dot(X, self.label_coefs)

        # Add sparse interactions (first 5 pairs of continuous)
        for i in range(5):
            logit += 0.1 * X[:, 2 * i] * X[:, 2 * i + 1]

        # Unobserved risk factor u ~ N(0,1)
        u = self.rng.standard_normal(n)
        logit += 0.7 * u

        # Noise
        eps = self.rng.normal(0, 0.25, n)
        logit += eps

        # Calibrate intercept to achieve target bad rate
        # Binary search for intercept
        def get_bad_rate(intercept):
            prob = 1 / (1 + np.exp(-(logit + intercept)))
            return prob.mean()

        lo, hi = -100, 100
        for _ in range(50):
            mid = (lo + hi) / 2
            if get_bad_rate(mid) < bad_rate:
                lo = mid
            else:
                hi = mid

        prob = 1 / (1 + np.exp(-(logit + mid)))
        y = self.rng.binomial(1, prob)

        return y

    def generate_population(self, n_samples: int) -> pd.DataFrame:
        """Generate population with features and labels."""
        z = self._sample_latent(n_samples)
        X = self._transform_features(z)
        y = self._generate_labels(X)

        df = pd.DataFrame(X, columns=self.feature_names)
        df["y"] = y
        return df

    def generate_holdout(self, n_samples: int) -> pd.DataFrame:
        """Generate representative holdout set."""
        return self.generate_population(n_samples)


def compute_ks_drift(Da: pd.DataFrame, H: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    """Compute KS statistics between Da and H for each feature."""
    ks_stats = {}
    for f in features:
        stat, _ = stats.ks_2samp(Da[f].values, H[f].values)
        ks_stats[f] = float(stat)
    return ks_stats


def run_acceptance_loop(
    generator: RealisticSyntheticGenerator,
    cfg: dict,
    model_cfg: XGBoostConfig,
    snapshot_iters: List[int],
) -> Dict[str, Any]:
    """Run the acceptance loop per plan Part A.4.

    Returns dict with Da, Dr, and snapshots.
    """
    n_periods = cfg["acceptance_loop"]["n_periods"]
    batch_size = cfg["acceptance_loop"]["batch_size"]
    sigma = cfg["acceptance_loop"]["sigma_policy"]
    mode = cfg["acceptance_loop"]["acceptance_mode"]
    initial_size = cfg["acceptance_loop"]["initial_seed_size"]

    features = generator.feature_names
    rng = generator.rng

    # Generate initial seed accepts via noisy policy
    pool = generator.generate_population(initial_size * 5)
    X_pool = pool[features].values
    y_pool = pool["y"].values

    # Initial accept via random noisy threshold on first feature (proxy for bureau)
    scores_init = -X_pool[:, 0] + rng.normal(0, sigma, len(pool))
    k_init = initial_size
    idx_sorted = np.argsort(scores_init)[::-1]
    accept_idx = idx_sorted[:k_init]
    reject_idx = idx_sorted[k_init:]

    Da = pool.iloc[accept_idx].copy()
    remaining_pool = pool.iloc[reject_idx].copy()

    snapshots = []

    # Train initial model
    model = XGBoostModel(model_cfg)
    X_Da = Da[features].values
    y_Da = Da["y"].values
    model.fit(X_Da, y_Da)

    for t in tqdm(range(1, n_periods + 1), desc="Acceptance loop"):
        # Generate new batch
        batch = generator.generate_population(batch_size * 5)
        X_batch = batch[features].values

        # Score with current model
        scores = model.predict_proba(X_batch)

        # Accept based on mode
        if mode == "topk":
            # Accept lowest PD
            k = batch_size
            idx = np.argsort(scores)[:k]
        else:
            # Stochastic top-k: add noise to scores
            noisy_scores = scores + rng.normal(0, sigma, len(scores))
            k = batch_size
            idx = np.argsort(noisy_scores)[:k]

        batch_accepts = batch.iloc[idx]
        batch_rejects = batch.iloc[~np.isin(np.arange(len(batch)), idx)]

        # Add to Da
        Da = pd.concat([Da, batch_accepts], ignore_index=True)
        remaining_pool = pd.concat([remaining_pool, batch_rejects], ignore_index=True)

        # Retrain model
        X_Da = Da[features].values
        y_Da = Da["y"].values
        model.fit(X_Da, y_Da)

        # Record snapshot if needed
        if t in snapshot_iters:
            snapshot = {
                "iter": t,
                "n_accepts": len(Da),
                "bad_rate_accepts": float(Da["y"].mean()),
            }
            snapshots.append(snapshot)

    # Final Dr = remaining pool (drop y column)
    Dr = remaining_pool.drop(columns=["y"])

    return {
        "Da": Da,
        "Dr": Dr,
        "snapshots": snapshots,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate realistic synthetic data")
    parser.add_argument("--config", type=str, default="configs/synthetic_generator.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: timestamp)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg.get("random_seed", 42)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Realistic Synthetic Data Generator")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Run ID: {run_id}")
    print(f"Seed: {seed}")

    # Create output directory
    output_dir = PROJECT_ROOT / "data" / "synthetic" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots").mkdir(exist_ok=True)

    # Initialize generator
    generator = RealisticSyntheticGenerator(cfg, seed)

    # Generate holdout first (before acceptance loop)
    n_holdout = cfg["sizes"]["n_holdout"]
    H = generator.generate_holdout(n_holdout)
    print(f"Generated holdout: {len(H)} samples, bad_rate={H['y'].mean():.4f}")

    # Run acceptance loop
    xgb_cfg = cfg["xgboost"]
    model_cfg = XGBoostConfig(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_seed=seed,
    )

    snapshot_iters = cfg["acceptance_loop"]["snapshot_iterations"]
    print(f"Running acceptance loop: {cfg['acceptance_loop']['n_periods']} periods...")

    result = run_acceptance_loop(generator, cfg, model_cfg, snapshot_iters)
    Da = result["Da"]
    Dr = result["Dr"]
    snapshots = result["snapshots"]

    print(f"Accepts (Da): {len(Da)} samples, bad_rate={Da['y'].mean():.4f}")
    print(f"Rejects (Dr): {len(Dr)} samples (unlabeled)")

    # Compute KS drift for final snapshot
    features = generator.feature_names
    ks_stats = compute_ks_drift(Da, H, features)
    avg_ks = np.mean(list(ks_stats.values()))
    max_ks = np.max(list(ks_stats.values()))
    top_ks_features = sorted(ks_stats.items(), key=lambda x: -x[1])[:5]

    # Add drift stats to snapshots
    for snap in snapshots:
        snap["avg_ks_Da_vs_H"] = avg_ks
        snap["max_ks_Da_vs_H"] = max_ks
        snap["top_ks_features"] = top_ks_features

    # Sanity gates per plan A.5
    da_bad = Da["y"].mean()
    h_bad = H["y"].mean()
    assert set(Da["y"].unique()).issubset({0, 1}), "Da.y must be binary"
    assert set(H["y"].unique()).issubset({0, 1}), "H.y must be binary"
    assert "y" not in Dr.columns, "Dr must not have y column"
    assert da_bad < h_bad, f"bad_rate(Da)={da_bad:.4f} >= bad_rate(H)={h_bad:.4f}"
    print(f"Sanity gates passed: bad_rate(Da)={da_bad:.4f} < bad_rate(H)={h_bad:.4f}")

    # Save outputs
    Da.to_csv(output_dir / "Da.csv", index=False)
    Dr.to_csv(output_dir / "Dr.csv", index=False)
    H.to_csv(output_dir / "H.csv", index=False)

    # Save snapshots
    for snap in snapshots:
        snap_path = output_dir / "snapshots" / f"iter_{snap['iter']:03d}.json"
        with open(snap_path, "w") as f:
            json.dump(snap, f, indent=2)

    # Save metadata
    meta = {
        "run_id": run_id,
        "seed": seed,
        "config": cfg,
        "n_accepts": len(Da),
        "n_rejects": len(Dr),
        "n_holdout": len(H),
        "bad_rate_accepts": float(da_bad),
        "bad_rate_holdout": float(h_bad),
        "avg_ks_drift": float(avg_ks),
        "max_ks_drift": float(max_ks),
        "feature_names": features,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput saved to: {output_dir}")
    print(f"  - Da.csv: {len(Da)} rows")
    print(f"  - Dr.csv: {len(Dr)} rows (no y column)")
    print(f"  - H.csv: {len(H)} rows")
    print(f"  - meta.json")
    print(f"  - snapshots/: {len(snapshots)} files")


if __name__ == "__main__":
    main()
