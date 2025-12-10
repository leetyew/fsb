#!/usr/bin/env python
"""
Run BASL experiment for paper replication.

Runs acceptance loops with per-iteration metric tracking to generate:
- Panels (a-d): Oracle vs Accepts-based vs Bayesian evaluation over iterations
- Panel (e): Baseline vs BASL training comparison over iterations

Also stores data for Figure 2 visualizations:
- Panel (a): Feature distributions (accepts/rejects/population)
- Panel (b): Surrogate model coefficients
- Panel (c): Score distributions

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --seed 42 --track-every 10
    python scripts/run_experiment.py --n-seeds 3

Results saved to experiments/experiment_{timestamp}/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.config import (
    AcceptanceLoopConfig,
    BASLConfig,
    BASLFilteringConfig,
    BASLLabelingConfig,
    BayesianEvalConfig,
    ExperimentConfig,
    SyntheticDataConfig,
    XGBoostConfig,
)
from src.evaluation.metrics import compute_metrics
from src.io.synthetic_acceptance_loop import AcceptanceLoop
from src.io.synthetic_generator import SyntheticGenerator
from src.models.xgboost_model import XGBoostModel


def compute_surrogate_coefficients(
    model: XGBoostModel,
    X: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Fit logistic regression on XGBoost predictions as a surrogate model.

    Per paper Figure 2(b), this gives interpretable coefficients to visualize
    bias in model learned from accepts vs BASL vs oracle training.

    Args:
        model: Trained XGBoost model.
        X: Feature matrix to get predictions on.
        feature_names: Names of features for coefficient dict.

    Returns:
        Dict mapping feature name to coefficient (including intercept).
    """
    scores = model.predict_proba(X)
    # Create pseudo-labels: threshold at 0.5 for binary classification
    y_pseudo = (scores > 0.5).astype(int)

    # Fit logistic regression on predictions
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(X, y_pseudo)

    coefficients = {"intercept": float(lr.intercept_[0])}
    for name, coef in zip(feature_names, lr.coef_[0]):
        coefficients[name] = float(coef)

    return coefficients


def set_seed_in_configs(
    seed: int,
    data_cfg: SyntheticDataConfig,
    model_cfg: XGBoostConfig,
    loop_cfg: AcceptanceLoopConfig,
    basl_cfg: BASLConfig,
    bayesian_cfg: BayesianEvalConfig,
) -> tuple[SyntheticDataConfig, XGBoostConfig, AcceptanceLoopConfig, BASLConfig, BayesianEvalConfig]:
    """Create new config instances with updated random seed."""
    data_cfg_new = SyntheticDataConfig(**{**data_cfg.model_dump(), "random_seed": seed})
    model_cfg_new = XGBoostConfig(**{**model_cfg.model_dump(), "random_seed": seed})
    loop_cfg_new = AcceptanceLoopConfig(**{**loop_cfg.model_dump(), "random_seed": seed})

    basl_filtering = BASLFilteringConfig(
        **{**basl_cfg.filtering.model_dump(), "random_seed": seed}
    )
    basl_labeling = BASLLabelingConfig(
        **{**basl_cfg.labeling.model_dump(), "random_seed": seed}
    )
    basl_cfg_new = BASLConfig(
        max_iterations=basl_cfg.max_iterations,
        filtering=basl_filtering,
        labeling=basl_labeling,
    )

    bayesian_cfg_new = BayesianEvalConfig(**{**bayesian_cfg.model_dump(), "random_seed": seed})

    return data_cfg_new, model_cfg_new, loop_cfg_new, basl_cfg_new, bayesian_cfg_new


def run_trial(
    seed: int,
    data_cfg: SyntheticDataConfig,
    model_cfg: XGBoostConfig,
    loop_cfg: AcceptanceLoopConfig,
    basl_cfg: BASLConfig,
    bayesian_cfg: BayesianEvalConfig,
    track_every: int = 10,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run a single trial with iteration tracking.

    Runs both baseline and BASL loops with the same holdout for fair comparison.
    Returns iteration history for both, plus data for Figure 2 visualizations.
    """
    data_cfg, model_cfg, loop_cfg, basl_cfg, bayesian_cfg = set_seed_in_configs(
        seed, data_cfg, model_cfg, loop_cfg, basl_cfg, bayesian_cfg
    )

    feature_cols = [f"x{i}" for i in range(data_cfg.n_features)]

    # Generate holdout once for both loops (ensures fair comparison)
    generator_holdout = SyntheticGenerator(data_cfg)
    holdout = generator_holdout.generate_holdout()
    X_holdout = holdout[feature_cols].values
    y_holdout = holdout["y"].values

    if show_progress:
        tqdm.write(f"  Seed {seed}: Running baseline loop...")
    generator_base = SyntheticGenerator(data_cfg)
    loop_base = AcceptanceLoop(
        generator_base, model_cfg, loop_cfg, basl_cfg=None, bayesian_cfg=bayesian_cfg
    )
    D_a_base, D_r_base, _, baseline_model, baseline_history = loop_base.run(
        holdout=holdout, track_every=track_every, show_progress=show_progress
    )

    if show_progress:
        tqdm.write(f"  Seed {seed}: Running BASL loop...")
    generator_basl = SyntheticGenerator(data_cfg)
    loop_basl = AcceptanceLoop(
        generator_basl, model_cfg, loop_cfg, basl_cfg=basl_cfg, bayesian_cfg=bayesian_cfg
    )
    D_a_basl, D_r_basl, _, basl_model, basl_history = loop_basl.run(
        holdout=holdout, track_every=track_every, show_progress=show_progress
    )

    # Train an oracle model on D_a ∪ D_r with true labels (per paper Algorithm C.2)
    if show_progress:
        tqdm.write(f"  Seed {seed}: Training oracle model...")

    # Combine accepts and rejects from baseline loop
    X_oracle = np.vstack([
        D_a_base[feature_cols].values,
        D_r_base[feature_cols].values
    ])
    y_oracle = np.hstack([
        D_a_base["y"].values,
        D_r_base["y"].values
    ])

    oracle_model = XGBoostModel(model_cfg)
    oracle_model.fit(X_oracle, y_oracle)

    # Compute surrogate model coefficients for Figure 2(b)
    baseline_coeffs = compute_surrogate_coefficients(baseline_model, X_holdout, feature_cols)
    basl_coeffs = compute_surrogate_coefficients(basl_model, X_holdout, feature_cols)
    oracle_coeffs = compute_surrogate_coefficients(oracle_model, X_holdout, feature_cols)

    # Compute score distributions for Figure 2(c)
    scores_baseline = baseline_model.predict_proba(X_holdout).tolist()
    scores_basl = basl_model.predict_proba(X_holdout).tolist()
    scores_oracle = oracle_model.predict_proba(X_holdout).tolist()

    # Compute Oracle model metrics for Figure 2(e) reference line
    oracle_metrics = compute_metrics(
        y_holdout, np.array(scores_oracle),
        ["auc", "pauc", "brier", "abr"],
        abr_range=bayesian_cfg.abr_range
    )

    # Store feature distributions for Figure 2(a)
    # Use a subset of features to keep file size manageable
    X_accepts = D_a_base[feature_cols].values
    X_rejects = D_r_base[feature_cols].values

    # Sample features for KDE (store x0 and x1 for 2D case)
    n_sample = min(1000, len(X_accepts), len(X_rejects), len(X_holdout))
    rng = np.random.default_rng(seed)
    idx_accepts = rng.choice(len(X_accepts), size=min(n_sample, len(X_accepts)), replace=False)
    idx_rejects = rng.choice(len(X_rejects), size=min(n_sample, len(X_rejects)), replace=False)
    idx_holdout = rng.choice(len(X_holdout), size=min(n_sample, len(X_holdout)), replace=False)

    return {
        "seed": seed,
        "baseline_history": baseline_history,
        "basl_history": basl_history,
        "n_accepts_base": len(D_a_base),
        "n_rejects_base": len(D_r_base),
        "n_accepts_basl": len(D_a_basl),
        "n_rejects_basl": len(D_r_basl),
        "holdout_bad_rate": float(holdout["y"].mean()),
        # Figure 2(a) data: feature distributions
        "feature_distributions": {
            "accepts_x0": X_accepts[idx_accepts, 0].tolist(),
            "accepts_x1": X_accepts[idx_accepts, 1].tolist() if data_cfg.n_features > 1 else [],
            "rejects_x0": X_rejects[idx_rejects, 0].tolist(),
            "rejects_x1": X_rejects[idx_rejects, 1].tolist() if data_cfg.n_features > 1 else [],
            "population_x0": X_holdout[idx_holdout, 0].tolist(),
            "population_x1": X_holdout[idx_holdout, 1].tolist() if data_cfg.n_features > 1 else [],
        },
        # Figure 2(b) data: surrogate model coefficients
        "surrogate_coefficients": {
            "baseline": baseline_coeffs,
            "basl": basl_coeffs,
            "oracle": oracle_coeffs,
        },
        # Figure 2(c) data: score distributions (sampled)
        "score_distributions": {
            "baseline": [scores_baseline[i] for i in idx_holdout],
            "basl": [scores_basl[i] for i in idx_holdout],
            "oracle": [scores_oracle[i] for i in idx_holdout],
        },
        # Figure 2(e) data: Oracle model metrics (trained on full population)
        "oracle_metrics": oracle_metrics,
    }


def aggregate_histories(trials: list[dict]) -> dict[str, Any]:
    """Aggregate iteration histories across trials.

    Returns mean and std for each metric at each iteration.
    """
    if not trials:
        return {}

    # Get iteration points from first trial
    iterations = [h["iteration"] for h in trials[0]["baseline_history"]]
    metrics = list(trials[0]["baseline_history"][0]["oracle"].keys())

    def aggregate_metric_series(
        histories: list[list[dict]], eval_type: str, metric: str
    ) -> dict:
        """Aggregate a single metric across trials."""
        values_by_iter = {}
        for history in histories:
            for h in history:
                it = h["iteration"]
                if it not in values_by_iter:
                    values_by_iter[it] = []
                values_by_iter[it].append(h[eval_type][metric])

        return {
            "iterations": sorted(values_by_iter.keys()),
            "mean": [float(np.mean(values_by_iter[i])) for i in sorted(values_by_iter.keys())],
            "std": [float(np.std(values_by_iter[i])) for i in sorted(values_by_iter.keys())],
        }

    result = {
        "n_trials": len(trials),
        "iterations": iterations,
        "metrics": metrics,
        "baseline": {},
        "basl": {},
    }

    # Aggregate baseline histories
    baseline_histories = [t["baseline_history"] for t in trials]
    for eval_type in ["oracle", "accepts", "bayesian"]:
        result["baseline"][eval_type] = {}
        for metric in metrics:
            result["baseline"][eval_type][metric] = aggregate_metric_series(
                baseline_histories, eval_type, metric
            )

    # Aggregate BASL histories
    basl_histories = [t["basl_history"] for t in trials]
    for eval_type in ["oracle", "accepts", "bayesian"]:
        result["basl"][eval_type] = {}
        for metric in metrics:
            result["basl"][eval_type][metric] = aggregate_metric_series(
                basl_histories, eval_type, metric
            )

    return result


def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Run BASL experiment for paper replication"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for single trial (overrides config)"
    )
    parser.add_argument(
        "--n-seeds", type=int, help="Number of seeds to run (overrides config)"
    )
    parser.add_argument(
        "--start-seed", type=int, help="Starting seed (overrides config)"
    )
    parser.add_argument(
        "--track-every", type=int, help="Track metrics every N iterations (overrides config)"
    )
    parser.add_argument(
        "--n-periods", type=int, help="Override n_periods from acceptance_loop.yaml"
    )
    parser.add_argument(
        "--name", type=str, default="", help="Optional experiment name suffix"
    )
    args = parser.parse_args()

    # Load configs from YAML
    data_cfg = SyntheticDataConfig.from_yaml()
    model_cfg = XGBoostConfig.from_yaml()
    loop_cfg = AcceptanceLoopConfig.from_yaml()
    basl_cfg = BASLConfig.from_yaml()
    bayesian_cfg = BayesianEvalConfig.from_yaml()
    exp_cfg = ExperimentConfig.from_yaml()

    # CLI overrides
    n_seeds = args.n_seeds if args.n_seeds is not None else exp_cfg.n_seeds
    start_seed = args.start_seed if args.start_seed is not None else exp_cfg.start_seed
    track_every = args.track_every if args.track_every is not None else exp_cfg.track_every

    if args.n_periods:
        loop_cfg = AcceptanceLoopConfig(
            **{**loop_cfg.model_dump(), "n_periods": args.n_periods}
        )

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"experiment_{timestamp}"
    if args.name:
        exp_name += f"_{args.name}"
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Determine seeds
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(start_seed, start_seed + n_seeds))

    print("=" * 70)
    print("BASL Experiment")
    print("=" * 70)
    print(f"  Output: {exp_dir}")
    print(f"  Seeds: {seeds}")
    print(f"  n_periods: {loop_cfg.n_periods}")
    print(f"  track_every: {track_every}")
    print(f"  basl_max_iterations: {basl_cfg.max_iterations} (jmax)")
    print(f"  bayesian_eval: j_max={bayesian_cfg.j_max}")
    print("=" * 70)

    # Run trials
    # Show detailed progress for single seed, only outer tqdm for multi-seed
    single_seed = len(seeds) == 1
    trials = []

    if single_seed:
        # Single seed: show detailed progress
        for seed in seeds:
            trial = run_trial(
                seed, data_cfg, model_cfg, loop_cfg, basl_cfg, bayesian_cfg,
                track_every, show_progress=True
            )
            trials.append(trial)
            trial_path = exp_dir / f"trial_seed{seed}.json"
            with open(trial_path, "w") as f:
                json.dump(convert_numpy(trial), f, indent=2)
    else:
        # Multi-seed: show only seeds progress bar
        for seed in tqdm(seeds, desc="Seeds"):
            trial = run_trial(
                seed, data_cfg, model_cfg, loop_cfg, basl_cfg, bayesian_cfg,
                track_every, show_progress=False
            )
            trials.append(trial)
            trial_path = exp_dir / f"trial_seed{seed}.json"
            with open(trial_path, "w") as f:
                json.dump(convert_numpy(trial), f, indent=2)

    # Aggregate if multiple trials
    if len(trials) > 1:
        aggregated = aggregate_histories(trials)
        agg_path = exp_dir / "aggregated.json"
        with open(agg_path, "w") as f:
            json.dump(convert_numpy(aggregated), f, indent=2)
        print(f"\nAggregated results saved to: {agg_path}")

    # Save config
    config = {
        "seeds": seeds,
        "track_every": track_every,
        "n_periods": loop_cfg.n_periods,
        "data_cfg": data_cfg.model_dump(),
        "model_cfg": model_cfg.model_dump(),
        "loop_cfg": loop_cfg.model_dump(),
        "basl_cfg": basl_cfg.model_dump(),
        "bayesian_cfg": bayesian_cfg.model_dump(),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(convert_numpy(config), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
