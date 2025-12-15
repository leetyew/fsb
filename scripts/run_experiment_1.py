#!/usr/bin/env python
"""
Experiment I: Reliability of Bayesian Evaluation (Figure 2).

Per paper Appendix E.1 and Algorithm C.2:
- Uses AcceptanceLoop: 500 periods × 100 applicants per batch
- All scorecards (f_a, f_o) are XGBoost
- LR used ONLY for Bayesian prior calibration
- Compares: Oracle, Accepts-only, Bayesian evaluation methods

Purpose: Show that Bayesian evaluation produces reliable unbiased
performance estimates compared to accepts-only evaluation.

Usage:
    python scripts/run_experiment_1.py
    python scripts/run_experiment_1.py --seed 42
    python scripts/run_experiment_1.py --n-seeds 10
    python scripts/run_experiment_1.py --config configs/experiment_1.yaml
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
    BayesianEvalConfig,
    SyntheticDataConfig,
    XGBoostConfig,
)
from src.io.synthetic_acceptance_loop import AcceptanceLoop
from src.io.synthetic_generator import SyntheticGenerator


# Global config loaded from YAML
CONFIG: dict[str, Any] = {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load experiment configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "experiment_1.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_trial(seed: int) -> dict[str, Any]:
    """Run single trial of Experiment I using AcceptanceLoop.

    Per Algorithm C.2:
    - 500 periods × 100 applicants per batch
    - f_a (accepts-based XGBoost) drives acceptance
    - f_o (oracle XGBoost) trained on D_a ∪ D_r with true labels
    - Bayesian evaluation uses LR-calibrated prior
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

    # Configure XGBoost model (f_a scorecard)
    model_cfg = XGBoostConfig(
        n_estimators=CONFIG["xgboost"]["n_estimators"],
        max_depth=CONFIG["xgboost"]["max_depth"],
        learning_rate=CONFIG["xgboost"]["learning_rate"],
        random_seed=seed,
    )

    # Configure Bayesian evaluation (with LR-calibrated prior)
    abr_range = tuple(CONFIG["bayesian"]["abr_range"])
    bayesian_cfg = BayesianEvalConfig(
        n_bands=CONFIG["bayesian"]["n_bands"],
        j_min=CONFIG["bayesian"]["j_min"],
        j_max=CONFIG["bayesian"]["j_max"],
        epsilon=CONFIG["bayesian"]["epsilon"],
        prior_alpha=CONFIG["bayesian"]["prior_alpha"],
        prior_beta=CONFIG["bayesian"]["prior_beta"],
        random_seed=seed,
        abr_range=abr_range,
        use_banding=CONFIG["bayesian"].get("use_banding", False),
    )

    # Generate holdout (separate from loop data per Algorithm C.2)
    holdout = generator.generate_holdout()

    # Run AcceptanceLoop (baseline mode: no BASL)
    loop = AcceptanceLoop(
        generator=generator,
        model_cfg=model_cfg,
        cfg=loop_cfg,
        basl_cfg=None,  # Experiment I does NOT use BASL
        bayesian_cfg=bayesian_cfg,
    )

    track_every = CONFIG["experiment"].get("track_every", 50)
    D_a, D_r, holdout, model, metrics_history = loop.run(
        holdout=holdout,
        track_every=track_every,
        show_progress=False,
    )

    # Extract final metrics for summary
    final_metrics = metrics_history[-1]

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
        # Model on holdout (f_a on holdout - true performance)
        "model_holdout_auc": final_metrics["model_holdout"]["auc"],
        "model_holdout_abr": final_metrics["model_holdout"]["abr"],
        # Accepts-only (biased estimate)
        "accepts_auc": final_metrics["accepts"]["auc"],
        "accepts_abr": final_metrics["accepts"]["abr"],
        "accepts_abr_bias": final_metrics["accepts"]["abr"] - final_metrics["oracle"]["abr"],
        # Bayesian (proposed estimate)
        "bayesian_auc": final_metrics["bayesian"]["auc"],
        "bayesian_abr": final_metrics["bayesian"]["abr"],
        "bayesian_abr_bias": final_metrics["bayesian"]["abr"] - final_metrics["oracle"]["abr"],
        # Full metrics history for Figure 2
        "metrics_history": metrics_history,
    }


def main():
    global CONFIG

    parser = argparse.ArgumentParser(
        description="Experiment I: Reliability of Bayesian Evaluation (Figure 2)"
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
    exp_name = f"exp1_bayesian_eval_{timestamp}"
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
    print("Experiment I: Reliability of Bayesian Evaluation (Figure 2)")
    print("=" * 70)
    print(f"  Config: {config_path or 'configs/experiment_1.yaml'}")
    print(f"  Output: {exp_dir}")
    print(f"  Seeds: {seeds}")
    print(f"  AcceptanceLoop: {n_periods} periods × {batch_size} applicants")
    print(f"  Holdout size: {CONFIG['synthetic_data']['n_holdout']}")
    print(f"  Acceptance rate: {CONFIG['acceptance_loop']['target_accept_rate']}")
    print(f"  ABR range: {CONFIG['bayesian']['abr_range']}")
    print(f"  Track every: {track_every} iterations")
    print("=" * 70)

    # Save config to output directory
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)

    # Run trials
    trials = []
    for seed in tqdm(seeds, desc="Running trials"):
        trial = run_trial(seed)
        trials.append(trial)

        # Save individual trial (without metrics_history for summary)
        trial_summary = {k: v for k, v in trial.items() if k != "metrics_history"}
        trial_path = exp_dir / f"trial_seed{seed}.json"
        with open(trial_path, "w") as f:
            json.dump(trial_summary, f, indent=2)

        # Save full metrics history separately for Figure 2
        history_path = exp_dir / f"metrics_history_seed{seed}.json"
        with open(history_path, "w") as f:
            json.dump(trial["metrics_history"], f, indent=2)

    # Aggregate results (final iteration)
    aggregated = {
        "n_trials": len(trials),
        "n_periods": n_periods,
        # Oracle (f_o - benchmark)
        "oracle_abr_mean": float(np.mean([t["oracle_abr"] for t in trials])),
        "oracle_abr_std": float(np.std([t["oracle_abr"] for t in trials])),
        # Model on holdout (f_a true performance)
        "model_holdout_abr_mean": float(np.mean([t["model_holdout_abr"] for t in trials])),
        "model_holdout_abr_std": float(np.std([t["model_holdout_abr"] for t in trials])),
        # Accepts-only (biased)
        "accepts_abr_mean": float(np.mean([t["accepts_abr"] for t in trials])),
        "accepts_abr_std": float(np.std([t["accepts_abr"] for t in trials])),
        "accepts_abr_bias_mean": float(np.mean([t["accepts_abr_bias"] for t in trials])),
        "accepts_abr_bias_std": float(np.std([t["accepts_abr_bias"] for t in trials])),
        # Bayesian (proposed)
        "bayesian_abr_mean": float(np.mean([t["bayesian_abr"] for t in trials])),
        "bayesian_abr_std": float(np.std([t["bayesian_abr"] for t in trials])),
        "bayesian_abr_bias_mean": float(np.mean([t["bayesian_abr_bias"] for t in trials])),
        "bayesian_abr_bias_std": float(np.std([t["bayesian_abr_bias"] for t in trials])),
    }

    with open(exp_dir / "aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Figure 2 data)")
    print("=" * 70)
    print(f"  Oracle ABR (f_o):      {aggregated['oracle_abr_mean']:.4f} +/- {aggregated['oracle_abr_std']:.4f}")
    print(f"  Model Holdout (f_a):   {aggregated['model_holdout_abr_mean']:.4f} +/- {aggregated['model_holdout_abr_std']:.4f}")
    print(f"  Accepts-only ABR:      {aggregated['accepts_abr_mean']:.4f} +/- {aggregated['accepts_abr_std']:.4f}")
    print(f"    Bias:                {aggregated['accepts_abr_bias_mean']:+.4f} +/- {aggregated['accepts_abr_bias_std']:.4f}")
    print(f"  Bayesian ABR:          {aggregated['bayesian_abr_mean']:.4f} +/- {aggregated['bayesian_abr_std']:.4f}")
    print(f"    Bias:                {aggregated['bayesian_abr_bias_mean']:+.4f} +/- {aggregated['bayesian_abr_bias_std']:.4f}")
    print("=" * 70)
    print(f"\nResults saved to: {exp_dir}")
    print(f"  - Per-iteration metrics: metrics_history_seed*.json")
    print(f"  - Trial summaries: trial_seed*.json")
    print(f"  - Aggregated: aggregated.json")


if __name__ == "__main__":
    main()
