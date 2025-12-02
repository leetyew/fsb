#!/usr/bin/env python
"""Compare post-hoc BASL vs during-loop BASL as described in paper A9."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.basl.trainer import BASLTrainer
from src.config import (
    AcceptanceLoopConfig,
    BASLConfig,
    SyntheticDataConfig,
    XGBoostConfig,
)
from src.evaluation.metrics import compute_metrics
from src.io.synthetic_acceptance_loop import AcceptanceLoop
from src.io.synthetic_generator import SyntheticGenerator
from src.models.xgboost_model import XGBoostModel


def run_comparison(n_periods: int = 100):
    """Run comparison between baseline, post-hoc BASL, and during-loop BASL."""

    # Load configs
    data_cfg = SyntheticDataConfig.from_yaml()
    model_cfg = XGBoostConfig.from_yaml()
    loop_cfg = AcceptanceLoopConfig.from_yaml()
    loop_cfg = AcceptanceLoopConfig(**{**loop_cfg.model_dump(), "n_periods": n_periods})
    basl_cfg = BASLConfig.from_yaml()

    generator = SyntheticGenerator(data_cfg)
    feature_cols = [f"x{i}" for i in range(data_cfg.n_features)]

    print("=" * 70)
    print(f"Running comparison with n_periods={n_periods}")
    print("=" * 70)

    # Mode 1: Baseline (no BASL)
    print("\n[Mode 1] Baseline - training on accepts only")
    loop_baseline = AcceptanceLoop(generator, model_cfg, loop_cfg, basl_cfg=None)
    D_a_base, D_r_base, H = loop_baseline.run()

    X_a = D_a_base[feature_cols].values
    y_a = D_a_base["y"].values
    X_H = H[feature_cols].values
    y_H = H["y"].values

    baseline_model = XGBoostModel(model_cfg)
    baseline_model.fit(X_a, y_a)

    scores_H_base = baseline_model.predict_proba(X_H)
    metrics_base = compute_metrics(y_H, scores_H_base, ["auc", "pauc", "brier", "abr"],
                                   accept_rate=loop_cfg.target_accept_rate)

    print(f"  Accepts: {len(D_a_base)}, Rejects: {len(D_r_base)}")
    print(f"  Accept bad rate: {y_a.mean():.3f}")

    # Mode 2: Post-hoc BASL (apply after loop)
    print("\n[Mode 2] Post-hoc BASL - same data, apply BASL after loop")
    X_r = D_r_base[feature_cols].values

    basl_trainer = BASLTrainer(basl_cfg)
    X_aug, y_aug = basl_trainer.run(X_a, y_a, X_r)

    posthoc_model = XGBoostModel(model_cfg)
    posthoc_model.fit(X_aug, y_aug)

    scores_H_posthoc = posthoc_model.predict_proba(X_H)
    metrics_posthoc = compute_metrics(y_H, scores_H_posthoc, ["auc", "pauc", "brier", "abr"],
                                      accept_rate=loop_cfg.target_accept_rate)

    print(f"  Augmented: {len(X_aug)} (+{len(X_aug) - len(X_a)} pseudo-labeled)")
    print(f"  Augmented bad rate: {y_aug.mean():.3f}")

    # Mode 3: During-loop BASL (apply at each iteration)
    print("\n[Mode 3] During-loop BASL - apply BASL at each iteration (paper A9)")
    # Need to create a new generator with same seed for fair comparison
    generator_loop = SyntheticGenerator(data_cfg)
    loop_basl = AcceptanceLoop(generator_loop, model_cfg, loop_cfg, basl_cfg=basl_cfg)
    D_a_loop, D_r_loop, H_loop, loop_model = loop_basl.run(return_model=True)

    X_a_loop = D_a_loop[feature_cols].values
    y_a_loop = D_a_loop["y"].values
    X_H_loop = H_loop[feature_cols].values
    y_H_loop = H_loop["y"].values

    scores_H_loop = loop_model.predict_proba(X_H_loop)
    metrics_loop = compute_metrics(y_H_loop, scores_H_loop, ["auc", "pauc", "brier", "abr"],
                                   accept_rate=loop_cfg.target_accept_rate)

    print(f"  Accepts: {len(D_a_loop)}, Rejects: {len(D_r_loop)}")
    print(f"  Accept bad rate: {y_a_loop.mean():.3f}")

    # Compare results
    print("\n" + "=" * 70)
    print("Results Comparison (Oracle - Holdout Evaluation)")
    print("=" * 70)
    print(f"  Holdout bad rate: {y_H.mean():.3f}")
    print()
    print(f"  {'Metric':<10} {'Baseline':>12} {'Post-hoc':>12} {'During-loop':>12}")
    print("  " + "-" * 50)

    for m in ["auc", "pauc", "brier", "abr"]:
        base_val = metrics_base[m]
        post_val = metrics_posthoc[m]
        loop_val = metrics_loop[m]
        print(f"  {m:<10} {base_val:>12.4f} {post_val:>12.4f} {loop_val:>12.4f}")

    print("\n  Improvement over baseline:")
    print(f"  {'Metric':<10} {'Post-hoc':>12} {'During-loop':>12}")
    print("  " + "-" * 36)
    for m in ["auc", "pauc"]:  # Higher is better
        post_diff = metrics_posthoc[m] - metrics_base[m]
        loop_diff = metrics_loop[m] - metrics_base[m]
        print(f"  {m:<10} {post_diff:>+12.4f} {loop_diff:>+12.4f}")
    for m in ["brier", "abr"]:  # Lower is better
        post_diff = metrics_base[m] - metrics_posthoc[m]
        loop_diff = metrics_base[m] - metrics_loop[m]
        print(f"  {m:<10} {post_diff:>+12.4f} {loop_diff:>+12.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-periods", type=int, default=100)
    args = parser.parse_args()

    run_comparison(args.n_periods)
