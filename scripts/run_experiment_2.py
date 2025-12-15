#!/usr/bin/env python
"""
Experiment II: BASL Training Effectiveness (Figures 3-4, Table 2).

Per paper Appendix E.1, B, and Algorithm C.2:
- Uses AcceptanceLoop: 500 periods × 100 applicants per batch
- All scorecards (f_a, f_o, f_c) are XGBoost
- LR used ONLY for BASL weak learner (pseudo-labeling)
- BASL applied EACH iteration (pseudo-labels discarded after use)
- Compares: Oracle, Accepts-only, BASL models

CRITICAL: BASL pseudo-labels are ephemeral and discarded every iteration.
They are NEVER added to D_a or D_r, and NEVER persist across iterations.

Purpose: Show that BASL reduces sampling bias in training data and improves
predictive performance (AUC, ABR) compared to accepts-only baselines.

Usage:
    python scripts/run_experiment_2.py
    python scripts/run_experiment_2.py --seed 42
    python scripts/run_experiment_2.py --n-seeds 10
    python scripts/run_experiment_2.py --config configs/experiment_2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from src.config import (
    AcceptanceLoopConfig,
    BASLConfig,
    BayesianEvalConfig,
    SyntheticDataConfig,
    XGBoostConfig,
)
from src.evaluation.metrics import compute_metrics
from src.io.synthetic_acceptance_loop import AcceptanceLoop
from src.io.synthetic_generator import SyntheticGenerator


# Global config loaded from YAML
CONFIG: dict[str, Any] = {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load experiment configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "experiment_2.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_feature_bias_analysis(
    model,
    oracle_model,
    holdout,
    feature_cols: list[str],
    n_bins: int = 10,
) -> dict[str, list[dict]]:
    """Compute predicted vs true bad rate across x0 bins for Figure 4.

    Args:
        model: BASL model (f_c)
        oracle_model: Oracle model (f_o)
        holdout: Holdout data with true labels
        feature_cols: List of feature column names
        n_bins: Number of bins for x0

    Returns:
        Dict of model_name -> list of bin results
    """
    x0 = holdout["x0"].values
    y_true = holdout["y"].values
    X_h = holdout[feature_cols].values

    # Create bins based on x0 percentiles
    bin_edges = np.percentile(x0, np.linspace(0, 100, n_bins + 1))

    results = {}
    for model_name, m in [("model", model), ("oracle", oracle_model)]:
        scores = m.predict_proba(X_h)
        model_results = []

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (x0 >= bin_edges[i]) & (x0 <= bin_edges[i + 1])
            else:
                mask = (x0 >= bin_edges[i]) & (x0 < bin_edges[i + 1])

            if mask.sum() > 0:
                model_results.append({
                    "bin": i,
                    "x0_min": float(bin_edges[i]),
                    "x0_max": float(bin_edges[i + 1]),
                    "n_samples": int(mask.sum()),
                    "true_bad_rate": float(y_true[mask].mean()),
                    "predicted_bad_rate": float(scores[mask].mean()),
                    "bias": float(scores[mask].mean() - y_true[mask].mean()),
                })

        results[model_name] = model_results

    return results


def run_trial(seed: int) -> dict[str, Any]:
    """Run single trial of Experiment II using AcceptanceLoop with BASL.

    Per Algorithm C.2 and Appendix B:
    - 500 periods × 100 applicants per batch
    - f_a (accepts-based XGBoost) drives acceptance
    - f_o (oracle XGBoost) trained on D_a ∪ D_r with true labels
    - f_c (BASL-corrected XGBoost) trained on D_a + temp pseudo-labels
    - Pseudo-labels discarded each iteration
    """
    # Configure data generator
    data_cfg = SyntheticDataConfig(
        random_seed=seed,
        n_features=CONFIG["synthetic_data"]["n_features"],
        n_components=CONFIG["synthetic_data"]["n_components"],
        bad_rate=CONFIG["synthetic_data"]["bad_rate"],
        n_holdout=CONFIG["synthetic_data"]["n_holdout"],
    )
    generator = SyntheticGenerator(data_cfg)

    # Configure AcceptanceLoop
    loop_cfg = AcceptanceLoopConfig(
        n_periods=CONFIG["acceptance_loop"]["n_periods"],
        batch_size=CONFIG["acceptance_loop"]["batch_size"],
        initial_batch_size=CONFIG["acceptance_loop"]["initial_batch_size"],
        target_accept_rate=CONFIG["acceptance_loop"]["target_accept_rate"],
        x_v_feature=CONFIG["acceptance_loop"]["x_v_feature"],
        random_seed=seed,
    )

    # Configure XGBoost model (f_a, f_o, f_c scorecards)
    model_cfg = XGBoostConfig(
        n_estimators=CONFIG["xgboost"]["n_estimators"],
        max_depth=CONFIG["xgboost"]["max_depth"],
        learning_rate=CONFIG["xgboost"]["learning_rate"],
        random_seed=seed,
    )

    # Configure BASL (with LR weak learner for pseudo-labeling)
    basl_cfg = BASLConfig(
        filter_proportion=CONFIG["basl"]["filter_proportion"],
        max_iterations=CONFIG["basl"]["max_iterations"],
        pseudo_threshold=CONFIG["basl"]["pseudo_threshold"],
        lr_C=CONFIG["logistic_regression"]["C"],
        lr_penalty=CONFIG["logistic_regression"]["penalty"],
        lr_solver=CONFIG["logistic_regression"]["solver"],
        random_seed=seed,
    )

    # Configure Bayesian evaluation (optional - for consistency with Exp I)
    abr_range = tuple(CONFIG["evaluation"]["abr_range"])
    bayesian_cfg = BayesianEvalConfig(
        n_bands=10,
        j_min=100,
        j_max=10000,
        epsilon=1e-6,
        prior_alpha=1.0,
        prior_beta=1.0,
        random_seed=seed,
        abr_range=abr_range,
    )

    # Generate holdout (separate from loop data per Algorithm C.2)
    holdout = generator.generate_holdout()

    # Run AcceptanceLoop with BASL
    loop = AcceptanceLoop(
        generator=generator,
        model_cfg=model_cfg,
        cfg=loop_cfg,
        basl_cfg=basl_cfg,  # Enable BASL for Experiment II
        bayesian_cfg=bayesian_cfg,
    )

    track_every = CONFIG["experiment"].get("track_every", 50)
    D_a, D_r, holdout, model, metrics_history = loop.run(
        holdout=holdout,
        track_every=track_every,
        show_progress=False,
    )

    # Extract final metrics
    final_metrics = metrics_history[-1]
    feature_cols = [f"x{i}" for i in range(data_cfg.n_features)]

    # Also run accepts-only baseline for comparison
    # Train accepts-only XGBoost on same final D_a
    from src.models.xgboost_model import XGBoostModel
    accepts_only_model = XGBoostModel(model_cfg)
    X_accepts = D_a[feature_cols].values
    y_accepts = D_a["y"].values
    accepts_only_model.fit(X_accepts, y_accepts)

    # Evaluate accepts-only on holdout
    X_holdout = holdout[feature_cols].values
    y_holdout = holdout["y"].values
    accepts_scores = accepts_only_model.predict_proba(X_holdout)
    accepts_metrics = compute_metrics(
        y_holdout, accepts_scores,
        metrics=["auc", "abr"],
        abr_range=abr_range,
    )

    # Compute feature bias analysis for Figure 4
    # Note: We need the oracle model from the loop - it's tracked in metrics_history
    # For now, train it here for the feature bias analysis
    oracle_model = XGBoostModel(model_cfg)
    X_rejects = D_r[feature_cols].values
    y_rejects = D_r["y"].values
    X_all = np.vstack([X_accepts, X_rejects])
    y_all = np.hstack([y_accepts, y_rejects])
    oracle_model.fit(X_all, y_all)

    n_bins = CONFIG["evaluation"].get("n_bins", 10)
    feature_bias = compute_feature_bias_analysis(
        model, oracle_model, holdout, feature_cols, n_bins
    )

    return {
        "seed": seed,
        "n_periods": CONFIG["acceptance_loop"]["n_periods"],
        "n_accepts": len(D_a),
        "n_rejects": len(D_r),
        "accepts_bad_rate": float(D_a["y"].mean()),
        "rejects_bad_rate": float(D_r["y"].mean()),
        "holdout_bad_rate": float(holdout["y"].mean()),
        # Oracle (f_o on holdout - benchmark)
        "oracle_auc": final_metrics["oracle"]["auc"],
        "oracle_abr": final_metrics["oracle"]["abr"],
        # BASL model on holdout (f_c on holdout)
        "basl_auc": final_metrics["model_holdout"]["auc"],
        "basl_abr": final_metrics["model_holdout"]["abr"],
        # Accepts-only XGBoost (for Table 2 comparison)
        "xgb_accepts_auc": accepts_metrics["auc"],
        "xgb_accepts_abr": accepts_metrics["abr"],
        # Full metrics history for Figures 3-4
        "metrics_history": metrics_history,
        # Feature bias analysis for Figure 4
        "feature_bias_analysis": feature_bias,
    }


def main():
    global CONFIG

    parser = argparse.ArgumentParser(
        description="Experiment II: BASL Training Effectiveness (Figures 3-4, Table 2)"
    )
    parser.add_argument("--seed", type=int, help="Single seed to run")
    parser.add_argument("--n-seeds", type=int, help="Number of seeds (overrides config)")
    parser.add_argument("--start-seed", type=int, help="Starting seed (overrides config)")
    parser.add_argument("--name", type=str, default="", help="Experiment name suffix")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    CONFIG = load_config(config_path)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp2_basl_training_{timestamp}"
    if args.name:
        exp_name += f"_{args.name}"
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Determine seeds (CLI overrides config)
    n_seeds = args.n_seeds if args.n_seeds is not None else CONFIG["experiment"]["n_seeds"]
    start_seed = args.start_seed if args.start_seed is not None else CONFIG["experiment"]["start_seed"]

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(start_seed, start_seed + n_seeds))

    n_periods = CONFIG["acceptance_loop"]["n_periods"]
    batch_size = CONFIG["acceptance_loop"]["batch_size"]
    track_every = CONFIG["experiment"].get("track_every", 50)

    print("=" * 70)
    print("Experiment II: BASL Training Effectiveness (Figures 3-4, Table 2)")
    print("=" * 70)
    print(f"  Config: {config_path or 'configs/experiment_2.yaml'}")
    print(f"  Output: {exp_dir}")
    print(f"  Seeds: {seeds}")
    print(f"  AcceptanceLoop: {n_periods} periods × {batch_size} applicants")
    print(f"  Holdout size: {CONFIG['synthetic_data']['n_holdout']}")
    print(f"  Acceptance rate: {CONFIG['acceptance_loop']['target_accept_rate']}")
    print(f"  BASL filter proportion: {CONFIG['basl']['filter_proportion']}")
    print(f"  ABR range: {CONFIG['evaluation']['abr_range']}")
    print(f"  Track every: {track_every} iterations")
    print("=" * 70)

    # Save config to output directory
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

    # Run trials
    trials = []
    all_feature_bias = []
    for seed in tqdm(seeds, desc="Running trials"):
        trial = run_trial(seed)

        # Extract feature bias and metrics history separately
        feature_bias = trial.pop("feature_bias_analysis")
        metrics_history = trial.pop("metrics_history")

        all_feature_bias.append({"seed": seed, **feature_bias})
        trials.append(trial)

        # Save individual trial summary
        trial_path = exp_dir / f"trial_seed{seed}.json"
        with open(trial_path, "w") as f:
            json.dump(trial, f, indent=2)

        # Save full metrics history separately for Figures 3-4
        history_path = exp_dir / f"metrics_history_seed{seed}.json"
        with open(history_path, "w") as f:
            json.dump(metrics_history, f, indent=2)

    # Save feature bias analysis (for Figure 4)
    with open(exp_dir / "feature_bias_analysis.json", "w") as f:
        json.dump(all_feature_bias, f, indent=2)

    # Aggregate results (Table 2)
    def mean_std(key):
        values = [t[key] for t in trials]
        return float(np.mean(values)), float(np.std(values))

    aggregated = {
        "n_trials": len(trials),
        "n_periods": n_periods,
    }

    for method in ["oracle", "xgb_accepts", "basl"]:
        for metric in ["auc", "abr"]:
            key = f"{method}_{metric}"
            mean, std = mean_std(key)
            aggregated[f"{key}_mean"] = mean
            aggregated[f"{key}_std"] = std

    with open(exp_dir / "aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print summary (Table 2 format)
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Table 2 format)")
    print("=" * 70)
    print(f"{'Method':<20} {'AUC':>18} {'ABR':>18}")
    print("-" * 58)

    oracle_auc_m, oracle_auc_s = mean_std("oracle_auc")
    oracle_abr_m, oracle_abr_s = mean_std("oracle_abr")
    print(f"{'Oracle (f_o)':<20} {oracle_auc_m:.4f}+/-{oracle_auc_s:.4f}  {oracle_abr_m:.4f}+/-{oracle_abr_s:.4f}")

    xgb_auc_m, xgb_auc_s = mean_std("xgb_accepts_auc")
    xgb_abr_m, xgb_abr_s = mean_std("xgb_accepts_abr")
    print(f"{'Accepts-only (f_a)':<20} {xgb_auc_m:.4f}+/-{xgb_auc_s:.4f}  {xgb_abr_m:.4f}+/-{xgb_abr_s:.4f}")

    basl_auc_m, basl_auc_s = mean_std("basl_auc")
    basl_abr_m, basl_abr_s = mean_std("basl_abr")
    print(f"{'BASL (f_c)':<20} {basl_auc_m:.4f}+/-{basl_auc_s:.4f}  {basl_abr_m:.4f}+/-{basl_abr_s:.4f}")

    print("=" * 70)
    print(f"\nResults saved to: {exp_dir}")
    print(f"  - aggregated.json: Table 2 data")
    print(f"  - metrics_history_seed*.json: Figures 3-4 data")
    print(f"  - feature_bias_analysis.json: Figure 4 feature bias data")


if __name__ == "__main__":
    main()
