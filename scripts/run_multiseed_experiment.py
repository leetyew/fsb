#!/usr/bin/env python
"""
Multi-seed experiment runner for robust evaluation.

Runs the full experiment pipeline multiple times with different random seeds
and aggregates results, following the paper's methodology of averaging over
100 simulation trials (Section 7.1.1).

Usage:
    python scripts/run_multiseed_experiment.py --n-seeds 10
    python scripts/run_multiseed_experiment.py --n-seeds 100

Resume interrupted experiment:
    python scripts/run_multiseed_experiment.py --resume experiments/experiment_20231201_120000

Results are automatically saved to experiments/experiment_{timestamp}/.
Trials are saved incrementally, allowing resume after interruption.
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
from tqdm import tqdm

from src.config import (
    AcceptanceLoopConfig,
    BASLConfig,
    BASLFilteringConfig,
    BASLLabelingConfig,
    ExperimentConfig,
    SyntheticDataConfig,
    XGBoostConfig,
)
from src.evaluation.bayesian_eval import BayesianEvalConfig, bayesian_evaluate
from src.evaluation.metrics import compute_metrics
from src.io.synthetic_acceptance_loop import AcceptanceLoop
from src.io.synthetic_generator import SyntheticGenerator


def set_seed_in_configs(
    seed: int,
    data_cfg: SyntheticDataConfig,
    model_cfg: XGBoostConfig,
    loop_cfg: AcceptanceLoopConfig,
    basl_cfg: BASLConfig,
) -> tuple[SyntheticDataConfig, XGBoostConfig, AcceptanceLoopConfig, BASLConfig]:
    """Create new config instances with updated random seed.

    Propagates the seed to all config objects to ensure reproducibility
    while allowing different seeds across trials.
    """
    data_cfg_new = SyntheticDataConfig(**{**data_cfg.model_dump(), "random_seed": seed})
    model_cfg_new = XGBoostConfig(**{**model_cfg.model_dump(), "random_seed": seed})
    loop_cfg_new = AcceptanceLoopConfig(**{**loop_cfg.model_dump(), "random_seed": seed})

    # BASL has nested configs that need seed updates
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

    return data_cfg_new, model_cfg_new, loop_cfg_new, basl_cfg_new


def run_single_trial(
    seed: int,
    data_cfg: SyntheticDataConfig,
    model_cfg: XGBoostConfig,
    loop_cfg: AcceptanceLoopConfig,
    basl_cfg: BASLConfig,
    eval_cfg: BayesianEvalConfig,
) -> dict[str, Any]:
    """Run a single trial with given seed.

    Returns oracle, accepts-based, and Bayesian metrics for both baseline and BASL.
    This enables Table C.3 comparison of evaluation methods.
    """
    # Update configs with this trial's seed
    data_cfg, model_cfg, loop_cfg, basl_cfg = set_seed_in_configs(
        seed, data_cfg, model_cfg, loop_cfg, basl_cfg
    )

    feature_cols = [f"x{i}" for i in range(data_cfg.n_features)]

    # Run baseline acceptance loop (no BASL)
    generator_base = SyntheticGenerator(data_cfg)
    loop_base = AcceptanceLoop(generator_base, model_cfg, loop_cfg, basl_cfg=None)
    D_a_base, D_r_base, H, baseline_model = loop_base.run(return_model=True)

    X_H = H[feature_cols].values
    y_H = H["y"].values
    X_a_base = D_a_base[feature_cols].values
    y_a_base = D_a_base["y"].values
    X_r_base = D_r_base[feature_cols].values

    # Run BASL acceptance loop
    generator_basl = SyntheticGenerator(data_cfg)
    loop_basl = AcceptanceLoop(generator_basl, model_cfg, loop_cfg, basl_cfg=basl_cfg)
    D_a_basl, D_r_basl, _, basl_model = loop_basl.run(return_model=True)

    X_a_basl = D_a_basl[feature_cols].values
    y_a_basl = D_a_basl["y"].values
    X_r_basl = D_r_basl[feature_cols].values

    # Compute scores for all evaluation types
    scores_H_baseline = baseline_model.predict_proba(X_H)
    scores_H_basl = basl_model.predict_proba(X_H)
    scores_a_baseline = baseline_model.predict_proba(X_a_base)
    scores_r_baseline = baseline_model.predict_proba(X_r_base)
    scores_a_basl = basl_model.predict_proba(X_a_basl)
    scores_r_basl = basl_model.predict_proba(X_r_basl)

    # 1. Oracle evaluation (on holdout - ground truth)
    oracle_baseline = compute_metrics(
        y_H, scores_H_baseline, eval_cfg.metrics, eval_cfg.accept_rate
    )
    oracle_basl = compute_metrics(
        y_H, scores_H_basl, eval_cfg.metrics, eval_cfg.accept_rate
    )

    # 2. Accepts-based evaluation (on D_a only - biased)
    accepts_baseline = compute_metrics(
        y_a_base, scores_a_baseline, eval_cfg.metrics, eval_cfg.accept_rate
    )
    accepts_basl = compute_metrics(
        y_a_basl, scores_a_basl, eval_cfg.metrics, eval_cfg.accept_rate
    )

    # 3. Bayesian evaluation (posterior sampling on D_a + D_r)
    bayes_result_baseline = bayesian_evaluate(
        y_a_base, scores_a_baseline, scores_r_baseline, eval_cfg
    )
    bayes_result_basl = bayesian_evaluate(
        y_a_basl, scores_a_basl, scores_r_basl, eval_cfg
    )
    bayesian_baseline = {m: bayes_result_baseline["metrics"][m]["mean"] for m in eval_cfg.metrics}
    bayesian_basl = {m: bayes_result_basl["metrics"][m]["mean"] for m in eval_cfg.metrics}

    return {
        "seed": seed,
        # Oracle (ground truth on holdout)
        "oracle_baseline": oracle_baseline,
        "oracle_basl": oracle_basl,
        # Accepts-based (biased, on D_a only)
        "accepts_baseline": accepts_baseline,
        "accepts_basl": accepts_basl,
        # Bayesian (posterior sampling on D_a + D_r)
        "bayesian_baseline": bayesian_baseline,
        "bayesian_basl": bayesian_basl,
        # Metadata
        "n_accepts_base": len(D_a_base),
        "n_rejects_base": len(D_r_base),
        "n_accepts_basl": len(D_a_basl),
        "n_rejects_basl": len(D_r_basl),
        "holdout_bad_rate": float(y_H.mean()),
    }


def aggregate_results(trials: list[dict]) -> dict[str, Any]:
    """Aggregate results across multiple trials.

    Computes mean, std, and 95% confidence intervals for each metric.
    Also computes Table C.3 stats: Bias, Variance, RMSE for each evaluation method.
    """
    metrics = ["auc", "pauc", "brier", "abr"]

    def summarize_array(arr: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q2.5": float(np.percentile(arr, 2.5)),
            "q97.5": float(np.percentile(arr, 97.5)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    results = {"n_trials": len(trials), "baseline": {}, "basl": {}, "improvement": {}}

    for metric in metrics:
        baseline_vals = np.array([t["oracle_baseline"][metric] for t in trials])
        basl_vals = np.array([t["oracle_basl"][metric] for t in trials])

        results["baseline"][metric] = summarize_array(baseline_vals)
        results["basl"][metric] = summarize_array(basl_vals)

        # Improvement: positive means BASL is better
        # For AUC/PAUC: higher is better, so diff = basl - baseline
        # For Brier/ABR: lower is better, so diff = baseline - basl
        if metric in ["auc", "pauc"]:
            diff = basl_vals - baseline_vals
        else:
            diff = baseline_vals - basl_vals

        results["improvement"][metric] = summarize_array(diff)

    # Additional stats
    results["holdout_bad_rate"] = summarize_array(
        np.array([t["holdout_bad_rate"] for t in trials])
    )

    # Table C.3: Evaluation method comparison (Bias, Variance, RMSE)
    # Compare accepts-based vs Bayesian evaluation against Oracle (ground truth)
    results["table_c3"] = {}

    for metric in metrics:
        oracle_vals = np.array([t["oracle_baseline"][metric] for t in trials])
        accepts_vals = np.array([t["accepts_baseline"][metric] for t in trials])
        bayesian_vals = np.array([t["bayesian_baseline"][metric] for t in trials])

        # Bias = |mean(Estimated) - mean(Oracle)|
        # Variance = var(Estimated)
        # RMSE = sqrt(Bias^2 + Variance)
        oracle_mean = np.mean(oracle_vals)

        accepts_bias = abs(np.mean(accepts_vals) - oracle_mean)
        accepts_var = np.var(accepts_vals)
        accepts_rmse = np.sqrt(accepts_bias**2 + accepts_var)

        bayesian_bias = abs(np.mean(bayesian_vals) - oracle_mean)
        bayesian_var = np.var(bayesian_vals)
        bayesian_rmse = np.sqrt(bayesian_bias**2 + bayesian_var)

        results["table_c3"][metric] = {
            "accepts": {
                "bias": float(accepts_bias),
                "variance": float(accepts_var),
                "rmse": float(accepts_rmse),
            },
            "bayesian": {
                "bias": float(bayesian_bias),
                "variance": float(bayesian_var),
                "rmse": float(bayesian_rmse),
            },
            "oracle_mean": float(oracle_mean),
        }

    # Also store raw accepts and bayesian metrics for detailed analysis
    results["accepts_baseline"] = {}
    results["bayesian_baseline"] = {}
    for metric in metrics:
        accepts_vals = np.array([t["accepts_baseline"][metric] for t in trials])
        bayesian_vals = np.array([t["bayesian_baseline"][metric] for t in trials])
        results["accepts_baseline"][metric] = summarize_array(accepts_vals)
        results["bayesian_baseline"][metric] = summarize_array(bayesian_vals)

    return results


def print_results(results: dict) -> None:
    """Print formatted results summary."""
    print("\n" + "=" * 70)
    print(f"Multi-Seed Results Summary ({results['n_trials']} trials)")
    print("=" * 70)

    print("\nOracle Metrics (Mean ± Std [95% CI]):")
    print(f"  {'Metric':<10} {'Baseline':>28} {'BASL':>28}")
    print("  " + "-" * 68)

    for metric in ["auc", "pauc", "brier", "abr"]:
        base = results["baseline"][metric]
        basl = results["basl"][metric]
        base_str = f"{base['mean']:.4f} ± {base['std']:.4f} [{base['q2.5']:.4f}, {base['q97.5']:.4f}]"
        basl_str = f"{basl['mean']:.4f} ± {basl['std']:.4f} [{basl['q2.5']:.4f}, {basl['q97.5']:.4f}]"
        print(f"  {metric:<10} {base_str:>28} {basl_str:>28}")

    print("\nImprovement (BASL over Baseline):")
    print(f"  {'Metric':<10} {'Mean':>10} {'Std':>10} {'95% CI':>24}")
    print("  " + "-" * 56)

    for metric in ["auc", "pauc", "brier", "abr"]:
        imp = results["improvement"][metric]
        sign = "+" if imp["mean"] > 0 else ""
        print(
            f"  {metric:<10} {sign}{imp['mean']:>9.4f} {imp['std']:>10.4f} "
            f"[{imp['q2.5']:>+.4f}, {imp['q97.5']:>+.4f}]"
        )

    # Table C.3: Evaluation Method Comparison
    if "table_c3" in results:
        print("\n" + "-" * 70)
        print("Table C.3: Evaluation Method Comparison (Baseline Model)")
        print("-" * 70)
        print(f"  {'Metric':<8} {'Eval Method':<12} {'Bias':>10} {'Variance':>10} {'RMSE':>10}")
        print("  " + "-" * 52)

        for metric in ["auc", "pauc", "brier", "abr"]:
            c3 = results["table_c3"][metric]
            print(f"  {metric:<8} {'Accepts':<12} {c3['accepts']['bias']:>10.4f} {c3['accepts']['variance']:>10.4f} {c3['accepts']['rmse']:>10.4f}")
            print(f"  {'':<8} {'Bayesian':<12} {c3['bayesian']['bias']:>10.4f} {c3['bayesian']['variance']:>10.4f} {c3['bayesian']['rmse']:>10.4f}")

    print(f"\nHoldout bad rate: {results['holdout_bad_rate']['mean']:.3f} ± {results['holdout_bad_rate']['std']:.3f}")


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


def load_completed_trials(exp_dir: Path) -> list[dict]:
    """Load completed trials from trials.jsonl file."""
    trials_path = exp_dir / "trials.jsonl"
    trials = []
    if trials_path.exists():
        with open(trials_path) as f:
            for line in f:
                if line.strip():
                    trials.append(json.loads(line))
    return trials


def save_trial(exp_dir: Path, trial: dict) -> None:
    """Append a single trial to trials.jsonl file."""
    trials_path = exp_dir / "trials.jsonl"
    with open(trials_path, "a") as f:
        f.write(json.dumps(convert_numpy(trial)) + "\n")


def save_final_results(
    exp_dir: Path,
    trials: list[dict],
    data_cfg: SyntheticDataConfig,
    model_cfg: XGBoostConfig,
    loop_cfg: AcceptanceLoopConfig,
    basl_cfg: BASLConfig,
    eval_cfg: BayesianEvalConfig,
    exp_args: dict[str, Any],
    seeds: list[int],
) -> None:
    """Save final aggregated results and config."""
    # Aggregate results
    results = aggregate_results(trials)
    results["config"] = {
        "n_periods": loop_cfg.n_periods,
        "batch_size": loop_cfg.batch_size,
        "accept_rate": loop_cfg.target_accept_rate,
        "bad_rate": data_cfg.bad_rate,
        "holdout_size": data_cfg.n_holdout,
        "seeds": seeds,
    }
    results["trials"] = trials

    # Print summary
    print_results(results)

    # 1. Full results with all trials
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    # 2. Summary only (for quick review)
    summary = {
        "n_trials": results["n_trials"],
        "config": results["config"],
        "baseline": results["baseline"],
        "basl": results["basl"],
        "improvement": results["improvement"],
        "holdout_bad_rate": results["holdout_bad_rate"],
        # Table C.3: Evaluation method comparison
        "table_c3": results.get("table_c3", {}),
        "accepts_baseline": results.get("accepts_baseline", {}),
        "bayesian_baseline": results.get("bayesian_baseline", {}),
    }
    summary_path = exp_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(convert_numpy(summary), f, indent=2)

    # 3. Config snapshot for reproducibility
    config_snapshot = {
        "args": exp_args,
        "data_cfg": data_cfg.model_dump(),
        "model_cfg": model_cfg.model_dump(),
        "loop_cfg": loop_cfg.model_dump(),
        "basl_cfg": basl_cfg.model_dump(),
        "eval_cfg": {
            "n_samples": eval_cfg.n_samples,
            "n_score_bands": eval_cfg.n_score_bands,
            "metrics": eval_cfg.metrics,
            "accept_rate": eval_cfg.accept_rate,
        },
    }
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(convert_numpy(config_snapshot), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {exp_dir}")
    print(f"  - results.json  (full results with all trials)")
    print(f"  - summary.json  (aggregated metrics only)")
    print(f"  - config.json   (experiment configuration)")
    print(f"  - trials.jsonl  (incremental trial results)")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed synthetic experiment"
    )
    parser.add_argument(
        "--n-seeds", type=int, help="Number of random seeds (overrides config)"
    )
    parser.add_argument(
        "--start-seed", type=int, help="Starting seed value (overrides config)"
    )
    parser.add_argument(
        "--n-periods", type=int, help="Override n_periods from acceptance_loop.yaml"
    )
    parser.add_argument(
        "--n-bayes-samples", type=int, help="Bayesian eval samples (overrides config)"
    )
    parser.add_argument(
        "--name", type=str, default="", help="Optional experiment name suffix"
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to experiment directory to resume"
    )
    args = parser.parse_args()

    # Load configs from YAML
    data_cfg = SyntheticDataConfig.from_yaml()
    model_cfg = XGBoostConfig.from_yaml()
    loop_cfg = AcceptanceLoopConfig.from_yaml()
    basl_cfg = BASLConfig.from_yaml()
    exp_cfg = ExperimentConfig.from_yaml()

    # CLI overrides (or use config defaults)
    n_seeds = args.n_seeds if args.n_seeds is not None else exp_cfg.n_seeds
    start_seed = args.start_seed if args.start_seed is not None else exp_cfg.start_seed
    n_bayes_samples = args.n_bayes_samples if args.n_bayes_samples is not None else exp_cfg.n_bayes_samples

    # Determine experiment directory
    if args.resume:
        exp_dir = Path(args.resume)
        if not exp_dir.exists():
            print(f"Error: Resume directory does not exist: {exp_dir}")
            sys.exit(1)
        # Load config from existing experiment
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)
            # Restore settings from saved config
            n_seeds = saved_config["args"]["n_seeds"]
            start_seed = saved_config["args"]["start_seed"]
            if saved_config["args"]["n_periods"]:
                loop_cfg = AcceptanceLoopConfig(
                    **{**loop_cfg.model_dump(), "n_periods": saved_config["args"]["n_periods"]}
                )
            n_bayes_samples = saved_config["args"]["n_bayes_samples"]
            args.name = saved_config["args"]["name"]
        print(f"Resuming experiment from: {exp_dir}")
    else:
        # Create new experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"experiment_{timestamp}"
        if args.name:
            exp_name += f"_{args.name}"
        exp_dir = PROJECT_ROOT / "experiments" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Override n_periods if specified via CLI
    if args.n_periods:
        loop_cfg = AcceptanceLoopConfig(
            **{**loop_cfg.model_dump(), "n_periods": args.n_periods}
        )

    # Evaluation config
    eval_cfg = BayesianEvalConfig(
        n_samples=n_bayes_samples,
        n_score_bands=10,
        metrics=["auc", "pauc", "brier", "abr"],
        accept_rate=loop_cfg.target_accept_rate,
    )

    # Build experiment args dict for reproducibility
    exp_args = {
        "n_seeds": n_seeds,
        "start_seed": start_seed,
        "n_periods": loop_cfg.n_periods,
        "n_bayes_samples": n_bayes_samples,
        "name": args.name,
    }

    # Load completed trials if resuming
    completed_trials = load_completed_trials(exp_dir)
    completed_seeds = {t["seed"] for t in completed_trials}

    # Determine which seeds to run
    all_seeds = list(range(start_seed, start_seed + n_seeds))
    remaining_seeds = [s for s in all_seeds if s not in completed_seeds]

    print("=" * 70)
    print("Multi-Seed Experiment Configuration")
    print("=" * 70)
    print(f"  Output: {exp_dir}")
    print(f"  Seeds: {start_seed} to {start_seed + n_seeds - 1}")
    print(f"  n_periods: {loop_cfg.n_periods}")
    print(f"  batch_size: {loop_cfg.batch_size}")
    print(f"  accept_rate: {loop_cfg.target_accept_rate}")
    print(f"  bad_rate: {data_cfg.bad_rate}")
    print(f"  holdout_size: {data_cfg.n_holdout}")
    print(f"  early_stopping_rounds: {model_cfg.early_stopping_rounds}")

    if completed_trials:
        print(f"\n  Resuming: {len(completed_trials)} trials already completed")
        print(f"  Remaining: {len(remaining_seeds)} trials to run")

    if not remaining_seeds:
        print("\nAll trials already completed!")
        # Just regenerate final results
        save_final_results(
            exp_dir, completed_trials, data_cfg, model_cfg,
            loop_cfg, basl_cfg, eval_cfg, exp_args, all_seeds
        )
        return

    # Run remaining trials
    trials = list(completed_trials)  # Start with completed trials

    print(f"\nRunning {len(remaining_seeds)} trials...")
    for seed in tqdm(remaining_seeds, desc="Trials"):
        trial_result = run_single_trial(
            seed, data_cfg, model_cfg, loop_cfg, basl_cfg, eval_cfg
        )
        trials.append(trial_result)
        # Save incrementally after each trial
        save_trial(exp_dir, trial_result)

    # Save final aggregated results
    save_final_results(
        exp_dir, trials, data_cfg, model_cfg,
        loop_cfg, basl_cfg, eval_cfg, exp_args, all_seeds
    )


if __name__ == "__main__":
    main()
