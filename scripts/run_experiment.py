#!/usr/bin/env python
"""
Main entry point for synthetic experiments.

Runs the full pipeline:
1. Generate biased data via acceptance loop (baseline mode)
2. Train baseline model on accepts only
3. Apply BASL to get augmented data, train BASL model
4. Evaluate both models (oracle on holdout + Bayesian evaluation)
5. Print comparison results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import (
    AcceptanceLoopConfig,
    BASLConfig,
    SyntheticDataConfig,
    XGBoostConfig,
)
from src.evaluation.bayesian_eval import BayesianEvalConfig, bayesian_evaluate
from src.evaluation.metrics import compute_metrics
from src.io.synthetic_acceptance_loop import AcceptanceLoop
from src.io.synthetic_generator import SyntheticGenerator
from src.models.xgboost_model import XGBoostModel


def run_experiment(
    data_cfg: SyntheticDataConfig,
    model_cfg: XGBoostConfig,
    loop_cfg: AcceptanceLoopConfig,
    basl_cfg: BASLConfig,
    eval_cfg: BayesianEvalConfig,
) -> dict:
    """Run full synthetic experiment comparing baseline vs BASL.

    Uses during-loop BASL (paper page A9) where BASL is applied at each
    iteration of the acceptance loop, not post-hoc.

    Args:
        data_cfg: Synthetic data generation config.
        model_cfg: XGBoost model config.
        loop_cfg: Acceptance loop config.
        basl_cfg: BASL config.
        eval_cfg: Bayesian evaluation config.

    Returns:
        Dictionary with all results.
    """
    results = {}
    feature_cols = [f"x{i}" for i in range(data_cfg.n_features)]

    # Step 1: Run baseline acceptance loop (no BASL)
    print("=" * 60)
    print("Step 1: Running baseline acceptance loop")
    print("=" * 60)

    generator_base = SyntheticGenerator(data_cfg)
    loop_base = AcceptanceLoop(generator_base, model_cfg, loop_cfg, basl_cfg=None)
    D_a_base, D_r_base, H, baseline_model = loop_base.run(return_model=True)

    X_a_base = D_a_base[feature_cols].values
    y_a_base = D_a_base["y"].values
    X_H = H[feature_cols].values
    y_H = H["y"].values

    print(f"  Accepts (D_a): {len(D_a_base)} samples (bad rate: {y_a_base.mean():.3f})")
    print(f"  Rejects (D_r): {len(D_r_base)} samples")
    print(f"  Holdout (H):   {len(H)} samples (bad rate: {y_H.mean():.3f})")

    # Step 2: Run BASL acceptance loop (BASL at each iteration per paper A9)
    print("\n" + "=" * 60)
    print("Step 2: Running BASL acceptance loop (during-loop mode)")
    print("=" * 60)

    generator_basl = SyntheticGenerator(data_cfg)
    loop_basl = AcceptanceLoop(generator_basl, model_cfg, loop_cfg, basl_cfg=basl_cfg)
    D_a_basl, D_r_basl, _, basl_model = loop_basl.run(return_model=True)

    X_a_basl = D_a_basl[feature_cols].values
    y_a_basl = D_a_basl["y"].values

    print(f"  Accepts (D_a): {len(D_a_basl)} samples (bad rate: {y_a_basl.mean():.3f})")
    print(f"  Rejects (D_r): {len(D_r_basl)} samples")

    # Step 3: Oracle evaluation on holdout
    print("\n" + "=" * 60)
    print("Step 3: Oracle evaluation on holdout")
    print("=" * 60)

    scores_H_baseline = baseline_model.predict_proba(X_H)
    scores_H_basl = basl_model.predict_proba(X_H)

    oracle_metrics = eval_cfg.metrics
    oracle_baseline = compute_metrics(y_H, scores_H_baseline, oracle_metrics, eval_cfg.accept_rate)
    oracle_basl = compute_metrics(y_H, scores_H_basl, oracle_metrics, eval_cfg.accept_rate)

    results["oracle_baseline"] = oracle_baseline
    results["oracle_basl"] = oracle_basl

    print("\n  Oracle Metrics (Holdout):")
    print(f"  {'Metric':<10} {'Baseline':>12} {'BASL':>12} {'Diff':>12}")
    print("  " + "-" * 48)
    for m in oracle_metrics:
        base_val = oracle_baseline[m]
        basl_val = oracle_basl[m]
        diff = basl_val - base_val
        print(f"  {m:<10} {base_val:>12.4f} {basl_val:>12.4f} {diff:>+12.4f}")

    # Step 4: Bayesian evaluation (using baseline data for fair comparison)
    print("\n" + "=" * 60)
    print("Step 4: Bayesian evaluation (accepts + rejects)")
    print("=" * 60)

    X_r_base = D_r_base[feature_cols].values

    scores_a_baseline = baseline_model.predict_proba(X_a_base)
    scores_r_baseline = baseline_model.predict_proba(X_r_base)

    scores_a_basl = basl_model.predict_proba(X_a_base)
    scores_r_basl = basl_model.predict_proba(X_r_base)

    print(f"  Running {eval_cfg.n_samples} Monte Carlo samples...")

    bayes_baseline = bayesian_evaluate(y_a_base, scores_a_baseline, scores_r_baseline, eval_cfg)
    bayes_basl = bayesian_evaluate(y_a_base, scores_a_basl, scores_r_basl, eval_cfg)

    results["bayesian_baseline"] = bayes_baseline
    results["bayesian_basl"] = bayes_basl

    print("\n  Bayesian Metrics (Posterior Mean [95% CI]):")
    print(f"  {'Metric':<10} {'Baseline':>24} {'BASL':>24}")
    print("  " + "-" * 60)
    for m in eval_cfg.metrics:
        base_m = bayes_baseline["metrics"][m]
        basl_m = bayes_basl["metrics"][m]
        base_str = f"{base_m['mean']:.4f} [{base_m['q2.5']:.4f}, {base_m['q97.5']:.4f}]"
        basl_str = f"{basl_m['mean']:.4f} [{basl_m['q2.5']:.4f}, {basl_m['q97.5']:.4f}]"
        print(f"  {m:<10} {base_str:>24} {basl_str:>24}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Baseline: {len(D_a_base)} accepts, {len(D_r_base)} rejects")
    print(f"  BASL: {len(D_a_basl)} accepts, {len(D_r_basl)} rejects")
    print(f"  Holdout: {len(H)} samples")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run synthetic credit scoring experiment")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file for results")
    parser.add_argument("--n-periods", type=int, help="Override n_periods in acceptance loop")
    parser.add_argument("--n-samples", type=int, help="Override n_samples in Bayesian eval")
    args = parser.parse_args()

    # Load configs from YAML files
    data_cfg = SyntheticDataConfig.from_yaml()
    model_cfg = XGBoostConfig.from_yaml()
    loop_cfg = AcceptanceLoopConfig.from_yaml()
    basl_cfg = BASLConfig.from_yaml()

    # Override from command line
    if args.n_periods:
        loop_cfg = AcceptanceLoopConfig(**{**loop_cfg.model_dump(), "n_periods": args.n_periods})

    # Bayesian eval config (not in YAML yet, use defaults)
    n_samples = args.n_samples or 5000
    eval_cfg = BayesianEvalConfig(
        n_samples=n_samples,
        n_score_bands=10,
        metrics=["auc", "pauc", "brier", "abr"],
        accept_rate=loop_cfg.target_accept_rate,
    )

    # Run experiment
    results = run_experiment(data_cfg, model_cfg, loop_cfg, basl_cfg, eval_cfg)

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
