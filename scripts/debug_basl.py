#!/usr/bin/env python
"""Debug script to understand BASL behavior."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.basl.filtering import filter_rejects
from src.basl.labeling import label_rejects_iteration
from src.config import (
    AcceptanceLoopConfig,
    BASLConfig,
    SyntheticDataConfig,
    XGBoostConfig,
)
from src.io.synthetic_acceptance_loop import AcceptanceLoop
from src.io.synthetic_generator import SyntheticGenerator


def main():
    # Load configs
    data_cfg = SyntheticDataConfig.from_yaml()
    model_cfg = XGBoostConfig.from_yaml()
    loop_cfg = AcceptanceLoopConfig.from_yaml()
    loop_cfg = AcceptanceLoopConfig(**{**loop_cfg.model_dump(), "n_periods": 50})  # Smaller run
    basl_cfg = BASLConfig.from_yaml()

    print("=" * 60)
    print("BASL Configuration")
    print("=" * 60)
    print(f"  max_iterations: {basl_cfg.max_iterations}")
    print(f"  filtering.beta_lower: {basl_cfg.filtering.beta_lower}")
    print(f"  filtering.beta_upper: {basl_cfg.filtering.beta_upper}")
    print(f"  labeling.subsample_ratio: {basl_cfg.labeling.subsample_ratio}")
    print(f"  labeling.gamma: {basl_cfg.labeling.gamma}")
    print(f"  labeling.theta: {basl_cfg.labeling.theta}")

    # Generate data
    print("\n" + "=" * 60)
    print("Generating data via acceptance loop")
    print("=" * 60)

    generator = SyntheticGenerator(data_cfg)
    loop = AcceptanceLoop(generator, model_cfg, loop_cfg, basl_cfg=None)
    D_a, D_r, H = loop.run()

    feature_cols = [f"x{i}" for i in range(data_cfg.n_features)]
    X_a = D_a[feature_cols].values
    y_a = D_a["y"].values
    X_r = D_r[feature_cols].values

    print(f"  Accepts: {len(X_a)} (bad rate: {y_a.mean():.3f})")
    print(f"  Rejects: {len(X_r)}")

    # Stage 1: Filtering
    print("\n" + "=" * 60)
    print("Stage 1: Reject Filtering")
    print("=" * 60)

    X_r_filtered, kept_indices = filter_rejects(X_a, X_r, basl_cfg.filtering)
    print(f"  Rejects after filtering: {len(X_r_filtered)} ({len(X_r_filtered)/len(X_r)*100:.1f}% kept)")

    # Stage 2: Pseudo-labeling
    print("\n" + "=" * 60)
    print("Stage 2: Iterative Pseudo-labeling")
    print("=" * 60)

    L_X = X_a.copy()
    L_y = y_a.copy()
    U_X = X_r_filtered.copy()

    rng = np.random.default_rng(basl_cfg.labeling.random_seed)
    fixed_thresholds = None
    total_good_added = 0
    total_bad_added = 0

    for j in range(basl_cfg.max_iterations):
        if len(U_X) == 0:
            print(f"  Iteration {j+1}: No rejects left in pool")
            break

        X_new, y_new, remaining_indices, thresholds = label_rejects_iteration(
            X_labeled=L_X,
            y_labeled=L_y,
            X_rejects_pool=U_X,
            cfg=basl_cfg.labeling,
            rng=rng,
            fixed_thresholds=fixed_thresholds,
        )

        if j == 0:
            fixed_thresholds = thresholds
            print(f"  Thresholds computed: tau_good={thresholds[0]:.4f}, tau_bad={thresholds[1]:.4f}")

        n_good = (y_new == 0).sum()
        n_bad = (y_new == 1).sum()
        total_good_added += n_good
        total_bad_added += n_bad

        print(f"  Iteration {j+1}: Added {n_good} good, {n_bad} bad (pool: {len(U_X)} -> {len(remaining_indices)})")

        if len(X_new) > 0:
            L_X = np.vstack([L_X, X_new])
            L_y = np.concatenate([L_y, y_new])

        U_X = U_X[remaining_indices]

    print(f"\n  Total pseudo-labeled: {total_good_added + total_bad_added}")
    print(f"    Good (y=0): {total_good_added}")
    print(f"    Bad (y=1): {total_bad_added}")
    print(f"  Final augmented data: {len(L_X)} samples")
    print(f"  Augmented bad rate: {L_y.mean():.3f}")

    # Compare with true labels in holdout
    print("\n" + "=" * 60)
    print("Holdout Statistics")
    print("=" * 60)
    y_H = H["y"].values
    print(f"  Holdout size: {len(H)}")
    print(f"  Holdout bad rate: {y_H.mean():.3f}")


if __name__ == "__main__":
    main()
