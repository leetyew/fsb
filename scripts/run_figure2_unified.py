#!/usr/bin/env python
"""
Unified Figure 2: All five panels from single acceptance loop.

Data consistency guarantee: All panels use the SAME holdout H,
training snapshot (D_a, D_r), and models f_a, f_o, f_c.

Panels:
  (a) Bias in Data: X1 distributions for H (ref), D_a, D_r
  (b) Bias in Model: LR surrogate coefficients + R² on XGB predictions
  (c) Bias in Predictions: P(BAD) score arrays on H for all 3 models
  (d) Impact on Evaluation: ABR over iterations (Bayesian tracking)
  (e) Impact on Training: ABR over iterations (BASL convergence)

Key insight: AcceptanceLoop already tracks three models:
  - model (f_c): BASL-corrected XGBoost
  - accepts_only_model (f_a): Accepts-only XGBoost
  - oracle_model (f_o): Oracle XGBoost (D_a + D_r with true labels)

Usage:
    python scripts/run_figure2_unified.py
    python scripts/run_figure2_unified.py --seed 42
    python scripts/run_figure2_unified.py --config configs/figure2_unified.yaml
    python scripts/run_figure2_unified.py --n-seeds 10  # Multi-seed run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AcceptanceLoopConfig,
    BASLConfig,
    BASLFilteringConfig,
    BASLLabelingConfig,
    BayesianEvalConfig,
    SyntheticDataConfig,
    XGBoostConfig,
)
from src.io.synthetic_acceptance_loop import AcceptanceLoop
from src.io.synthetic_generator import SyntheticGenerator
from src.models.xgboost_model import XGBoostModel


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


@dataclass
class PanelAData:
    """Panel (a): Feature distribution data.

    Uses holdout H as the unbiased reference population.
    Raw X1 stored; plotting code negates to show bureau score x_v = -X1.
    """

    x_v_feature: str  # Feature name (typically "X1")
    ref_xv: List[float]  # Holdout H (unbiased reference)
    Da_xv: List[float]  # Accepts at snapshot
    Dr_xv: List[float]  # Rejects at snapshot


@dataclass
class SurrogateCoefs:
    """LinearRegression surrogate coefficients for a single model."""

    coefs: List[float]  # [intercept, X1, X2, N1, N2]
    r2: float  # R² to validate fit quality


@dataclass
class PanelBData:
    """Panel (b): LinearRegression surrogate coefficients.

    Locked-down specification:
    - Dataset: Holdout H
    - Target: model.predict_proba(H) (P(BAD))
    - Model: LinearRegression(fit_intercept=True)
    - Features: [X1, X2, N1, N2] (no standardization)
    """

    feature_names: List[str]  # ["Intercept", "X1", "X2", "N1", "N2"]
    fa: SurrogateCoefs  # Accepts-only surrogate
    fo: SurrogateCoefs  # Oracle surrogate
    fc: SurrogateCoefs  # BASL surrogate


@dataclass
class PanelCData:
    """Panel (c): P(BAD) score distributions on holdout H."""

    fa_scores: List[float]  # f_a P(BAD) scores on H
    fo_scores: List[float]  # f_o P(BAD) scores on H
    fc_scores: List[float]  # f_c P(BAD) scores on H


@dataclass
class IterationMetrics:
    """Metrics at a single tracked iteration."""

    iteration: int
    # ABR metrics (panel d, e)
    fo_H_abr: float
    fa_H_abr: float
    fc_H_abr: float
    fa_DaVal_abr: float
    bayesian_abr: float
    # AUC metrics
    fo_H_auc: float
    fa_H_auc: float
    fc_H_auc: float
    # Diagnostic fields for panel (d) debugging
    n_Da_train: Optional[int] = None
    n_Da_val: Optional[int] = None
    n_H: Optional[int] = None
    bad_rate_Da_val: Optional[float] = None
    bad_rate_H: Optional[float] = None
    count_bad_Da_val: Optional[int] = None
    pred_Da_val_min: Optional[float] = None
    pred_Da_val_p5: Optional[float] = None
    pred_Da_val_p50: Optional[float] = None
    pred_Da_val_p95: Optional[float] = None
    pred_Da_val_max: Optional[float] = None


@dataclass
class PanelCSummary:
    """Summary statistics for panel (c) scores."""

    mean: float
    p95: float
    p99: float


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "figure2_unified.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_panel_a_data(
    holdout: pd.DataFrame,
    D_a: pd.DataFrame,
    D_r: pd.DataFrame,
    x_v_feature: str,
) -> PanelAData:
    """Collect Panel (a) feature distribution data.

    Per plan: Uses holdout H as the unbiased reference population.
    Raw X1 stored; plotting code negates to show bureau score.

    Args:
        holdout: Holdout H (unbiased reference)
        D_a: Accepts at snapshot
        D_r: Rejects at snapshot
        x_v_feature: Feature name (typically "X1")

    Returns:
        PanelAData with X1 values from each dataset
    """
    return PanelAData(
        x_v_feature=x_v_feature,
        ref_xv=holdout[x_v_feature].tolist(),
        Da_xv=D_a[x_v_feature].tolist(),
        Dr_xv=D_r[x_v_feature].tolist(),
    )


def fit_lr_surrogate(
    X: np.ndarray,
    y_scores: np.ndarray,
) -> SurrogateCoefs:
    """Fit LinearRegression surrogate to approximate XGB model.

    Per plan locked-down specification:
    - Model: LinearRegression(fit_intercept=True)
    - Features: raw features (no standardization)
    - Target: XGB P(BAD) scores (continuous)

    Args:
        X: Feature matrix [n_samples, n_features]
        y_scores: XGB predict_proba output (P(BAD))

    Returns:
        SurrogateCoefs with [intercept, coef1, ...] and R²
    """
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y_scores)
    r2 = lr.score(X, y_scores)

    # Return coefficients as [intercept, X1, X2, N1, N2]
    coefs = [float(lr.intercept_)] + lr.coef_.tolist()

    return SurrogateCoefs(coefs=coefs, r2=float(r2))


def collect_panel_b_data(
    fa_model: XGBoostModel,
    fo_model: XGBoostModel,
    fc_model: XGBoostModel,
    holdout: pd.DataFrame,
    feature_cols: List[str],
) -> PanelBData:
    """Collect Panel (b) LinearRegression surrogate coefficients.

    Per plan locked-down specification:
    - Dataset: Holdout H
    - Target: model.predict_proba(H) (P(BAD))
    - Model: LinearRegression(fit_intercept=True)
    - Features: [X1, X2, N1, N2] (no standardization)

    Args:
        fa_model: Accepts-only XGBoost (f_a)
        fo_model: Oracle XGBoost (f_o)
        fc_model: BASL XGBoost (f_c)
        holdout: Holdout dataframe
        feature_cols: Feature column names [X1, X2, N1, N2]

    Returns:
        PanelBData with surrogate coefficients and R² for all three models
    """
    X_h = holdout[feature_cols].values

    # Get XGB scores (P(BAD)) on holdout
    fa_scores = fa_model.predict_proba(X_h)
    fo_scores = fo_model.predict_proba(X_h)
    fc_scores = fc_model.predict_proba(X_h)

    # Fit LR surrogates
    fa_surrogate = fit_lr_surrogate(X_h, fa_scores)
    fo_surrogate = fit_lr_surrogate(X_h, fo_scores)
    fc_surrogate = fit_lr_surrogate(X_h, fc_scores)

    return PanelBData(
        feature_names=["Intercept"] + list(feature_cols),
        fa=fa_surrogate,
        fo=fo_surrogate,
        fc=fc_surrogate,
    )


def collect_panel_c_data(
    fa_model: XGBoostModel,
    fo_model: XGBoostModel,
    fc_model: XGBoostModel,
    holdout: pd.DataFrame,
    feature_cols: List[str],
) -> PanelCData:
    """Collect Panel (c) P(BAD) score distributions.

    Args:
        fa_model: Accepts-only XGBoost (f_a)
        fo_model: Oracle XGBoost (f_o)
        fc_model: BASL XGBoost (f_c)
        holdout: Holdout dataframe
        feature_cols: Feature column names

    Returns:
        PanelCData with P(BAD) scores on H for all three models
    """
    X_h = holdout[feature_cols].values

    fa_scores = fa_model.predict_proba(X_h)
    fo_scores = fo_model.predict_proba(X_h)
    fc_scores = fc_model.predict_proba(X_h)

    return PanelCData(
        fa_scores=fa_scores.tolist(),
        fo_scores=fo_scores.tolist(),
        fc_scores=fc_scores.tolist(),
    )


def compute_panel_c_summary(panel_c: PanelCData) -> Dict[str, PanelCSummary]:
    """Compute summary statistics for panel (c) scores."""
    summaries = {}
    for name, scores in [
        ("fa", panel_c.fa_scores),
        ("fo", panel_c.fo_scores),
        ("fc", panel_c.fc_scores),
    ]:
        arr = np.array(scores)
        summaries[name] = PanelCSummary(
            mean=float(np.mean(arr)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )
    return summaries


def extract_iteration_data(
    metrics_history: List[Dict[str, Any]],
) -> List[IterationMetrics]:
    """Extract metrics from all tracked iterations.

    Args:
        metrics_history: Full metrics history from AcceptanceLoop

    Returns:
        List of IterationMetrics for each tracked iteration
    """
    iteration_data = []
    for m in metrics_history:
        iteration = m["iteration"]

        # Use paper-faithful keys (fo_H, fa_H, fc_H) with fallback to legacy
        fo_H = m.get("fo_H", m.get("oracle", {}))
        fa_H = m.get("fa_H", {})
        fc_H = m.get("fc_H", m.get("model_holdout", {}))
        fa_DaVal = m.get("fa_DaVal", m.get("accepts", {}))
        bayesian = m.get("bayesian", {})
        diagnostic = m.get("diagnostic", {})

        iteration_data.append(
            IterationMetrics(
                iteration=iteration,
                fo_H_abr=fo_H.get("abr", float("nan")),
                fa_H_abr=fa_H.get("abr", float("nan")),
                fc_H_abr=fc_H.get("abr", float("nan")),
                fa_DaVal_abr=fa_DaVal.get("abr", float("nan")),
                bayesian_abr=bayesian.get("abr", float("nan")),
                fo_H_auc=fo_H.get("auc", float("nan")),
                fa_H_auc=fa_H.get("auc", float("nan")),
                fc_H_auc=fc_H.get("auc", float("nan")),
                # Diagnostic fields for panel (d) debugging
                n_Da_train=diagnostic.get("n_Da_train"),
                n_Da_val=diagnostic.get("n_Da_val"),
                n_H=diagnostic.get("n_H"),
                bad_rate_Da_val=diagnostic.get("bad_rate_Da_val"),
                bad_rate_H=diagnostic.get("bad_rate_H"),
                count_bad_Da_val=diagnostic.get("count_bad_Da_val"),
                pred_Da_val_min=diagnostic.get("pred_Da_val_min"),
                pred_Da_val_p5=diagnostic.get("pred_Da_val_p5"),
                pred_Da_val_p50=diagnostic.get("pred_Da_val_p50"),
                pred_Da_val_p95=diagnostic.get("pred_Da_val_p95"),
                pred_Da_val_max=diagnostic.get("pred_Da_val_max"),
            )
        )

    return iteration_data


def run_unified_figure2(
    config: Dict[str, Any],
    seed: int,
    track_every: int,
) -> Dict[str, Any]:
    """Run unified Figure 2 data collection.

    Main orchestrator that:
    1. Runs AcceptanceLoop with BASL enabled
    2. At panel_snapshot_iter, trains all three models from same (D_a, D_r) snapshot
    3. Collects static panel data (a, b, c)
    4. Extracts iteration_data from metrics_history for dynamic panels (d, e)

    Args:
        config: Configuration dictionary
        seed: Random seed
        track_every: Track metrics every N iterations

    Returns:
        Complete Figure 2 data artifact
    """
    n_periods = config["acceptance_loop"]["n_periods"]
    panel_snapshot_iter = n_periods

    # MANDATORY constraint per plan (Case C): only final state available
    assert panel_snapshot_iter == n_periods, (
        f"panel_snapshot_iter={panel_snapshot_iter} but only final state "
        f"(n_periods={n_periods}) is available without modifying AcceptanceLoop"
    )

    # Configure data generator
    data_cfg = SyntheticDataConfig(
        random_seed=seed,
        n_components=config["synthetic_data"]["n_components"],
        bad_rate=config["synthetic_data"]["bad_rate"],
        n_holdout=config["synthetic_data"]["n_holdout"],
    )
    generator = SyntheticGenerator(data_cfg)

    # Configure AcceptanceLoop
    loop_cfg = AcceptanceLoopConfig(
        n_periods=n_periods,
        batch_size=config["acceptance_loop"]["batch_size"],
        initial_batch_size=config["acceptance_loop"]["initial_batch_size"],
        target_accept_rate=config["acceptance_loop"]["target_accept_rate"],
        random_seed=seed,
    )

    # Configure XGBoost model
    model_cfg = XGBoostConfig(
        n_estimators=config["xgboost"]["n_estimators"],
        max_depth=config["xgboost"]["max_depth"],
        learning_rate=config["xgboost"]["learning_rate"],
        random_seed=seed,
    )

    # Configure BASL
    basl_filtering = config["basl"].get("filtering", {})
    basl_labeling = config["basl"].get("labeling", {})

    basl_cfg = BASLConfig(
        max_iterations=config["basl"]["max_iterations"],
        filtering=BASLFilteringConfig(
            beta_lower=basl_filtering.get("beta_lower", 0.05),
            beta_upper=basl_filtering.get("beta_upper", 1.0),
            random_seed=seed,
        ),
        labeling=BASLLabelingConfig(
            subsample_ratio=basl_labeling.get("subsample_ratio", 0.8),
            gamma=basl_labeling.get("gamma", 0.01),
            theta=basl_labeling.get("theta", 2.0),
            random_seed=seed,
        ),
    )

    # Configure Bayesian evaluation
    abr_range = tuple(config["evaluation"]["abr_range"])
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
    # This tracks all three models: f_c (model), f_a (accepts_only_model), f_o (oracle_model)
    loop = AcceptanceLoop(
        generator=generator,
        model_cfg=model_cfg,
        cfg=loop_cfg,
        basl_cfg=basl_cfg,
        bayesian_cfg=bayesian_cfg,
    )

    print(f"Running AcceptanceLoop with BASL ({n_periods} periods)...")
    D_a, D_r, holdout, fc_model, fo_model, fa_model, metrics_history = loop.run(
        holdout=holdout,
        track_every=track_every,
        show_progress=True,
    )

    feature_cols = generator.feature_cols
    x_v_feature = generator.x_v_feature

    # All three models (fa, fo, fc) are now returned directly from AcceptanceLoop:
    # - fa_model: accepts-only model (trained on D_a_train)
    # - fo_model: oracle model (trained on D_a ∪ D_r)
    # - fc_model: BASL model (trained on D_a_train + pseudo-labeled rejects)

    # Verify model identity by checking scores differ
    X_h = holdout[feature_cols].values
    fa_scores = fa_model.predict_proba(X_h)
    fo_scores = fo_model.predict_proba(X_h)
    fc_scores = fc_model.predict_proba(X_h)

    print(f"  f_a mean score: {fa_scores.mean():.4f}")
    print(f"  f_o mean score: {fo_scores.mean():.4f}")
    print(f"  f_c mean score: {fc_scores.mean():.4f}")

    # Collect static panel data (a, b, c)
    print("Collecting panel data...")

    panel_a = collect_panel_a_data(holdout, D_a, D_r, x_v_feature)
    panel_b = collect_panel_b_data(fa_model, fo_model, fc_model, holdout, feature_cols)

    # Panel (c): Use final iteration predictions (paper-faithful)
    # The paper shows Figure 2 at iteration 500 for all panels
    panel_c = collect_panel_c_data(fa_model, fo_model, fc_model, holdout, feature_cols)
    panel_c_summary = compute_panel_c_summary(panel_c)

    # Extract iteration data for panels (d), (e)
    iteration_data = extract_iteration_data(metrics_history)

    # Compute accept rate for snapshot sizes
    accept_rate = len(D_a) / (len(D_a) + len(D_r))

    # Build output artifact
    result = {
        "seed": seed,
        "n_periods": n_periods,
        "track_every": track_every,
        "panel_snapshot_iter": panel_snapshot_iter,
        # Panel (a): Feature distributions
        "panel_a": asdict(panel_a),
        # Panel (b): LR surrogate coefficients with R²
        "panel_b": {
            "feature_names": panel_b.feature_names,
            "fa": asdict(panel_b.fa),
            "fo": asdict(panel_b.fo),
            "fc": asdict(panel_b.fc),
        },
        # Panel (c): P(BAD) scores on H (final iteration, paper-faithful)
        "panel_c": asdict(panel_c),
        # Panels (d), (e): Metrics at each tracked iteration
        "iteration_data": [asdict(c) for c in iteration_data],
        # Dataset sizes
        "n_holdout": len(holdout),
        "n_accepts_final": len(D_a),
        "n_rejects_final": len(D_r),
        # Snapshot info
        "snapshot_sizes": {
            "iteration": panel_snapshot_iter,
            "Da_size": len(D_a),
            "Dr_size": len(D_r),
            "accept_rate": float(accept_rate),
        },
        # Panel (c) summary stats
        "panel_c_summary": {k: asdict(v) for k, v in panel_c_summary.items()},
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Figure 2: All five panels from single acceptance loop"
    )
    parser.add_argument("--seed", type=int, help="Single seed (overrides n-seeds)")
    parser.add_argument("--n-seeds", type=int, help="Number of seeds (overrides config)")
    parser.add_argument("--start-seed", type=int, help="Starting seed (overrides config)")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--track-every", type=int, help="Track metrics every N iterations (overrides config)")
    parser.add_argument("--output", type=str, help="Output directory path")
    parser.add_argument("--name", type=str, default="", help="Experiment name suffix")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Determine seeds (CLI overrides config)
    n_seeds = args.n_seeds if args.n_seeds is not None else config["experiment"]["n_seeds"]
    start_seed = args.start_seed if args.start_seed is not None else config["experiment"]["start_seed"]

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(start_seed, start_seed + n_seeds))

    # Determine track_every (CLI overrides config)
    track_every = args.track_every if args.track_every is not None else config["experiment"].get("track_every", 10)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"figure2_unified_{timestamp}"
    if args.name:
        exp_name += f"_{args.name}"
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    if args.output:
        exp_dir = Path(args.output)
    exp_dir.mkdir(parents=True, exist_ok=True)

    n_periods = config["acceptance_loop"]["n_periods"]
    batch_size = config["acceptance_loop"]["batch_size"]

    print("=" * 70)
    print("Unified Figure 2: All five panels from single acceptance loop")
    print("=" * 70)
    print(f"  Config: {config_path or 'configs/figure2_unified.yaml'}")
    print(f"  Output: {exp_dir}")
    print(f"  Seeds: {seeds}")
    print(f"  AcceptanceLoop: {n_periods} periods x {batch_size} applicants")
    print(f"  Holdout size: {config['synthetic_data']['n_holdout']}")
    print(f"  Acceptance rate: {config['acceptance_loop']['target_accept_rate']}")
    print(f"  BASL max_iterations: {config['basl']['max_iterations']}")
    print(f"  Track every: {track_every} iterations")
    print("=" * 70)

    # Save config to output directory
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run trials for each seed
    all_results = []
    for seed in tqdm(seeds, desc="Running seeds"):
        result = run_unified_figure2(config, seed, track_every)

        # Save individual seed result
        seed_path = exp_dir / f"figure2_unified_seed{seed}.json"
        with open(seed_path, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)

        all_results.append(result)

    # Aggregate iteration data across seeds (mean ± std)
    aggregated = aggregate_iteration_data(all_results)

    # Save aggregated results
    agg_path = exp_dir / "figure2_aggregated.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2, cls=NumpyEncoder)

    # Print summary
    print_results_summary(all_results, aggregated, seeds, exp_dir)


def aggregate_iteration_data(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate iteration metrics across multiple seeds.

    Computes mean and std for ABR and AUC metrics at each tracked iteration.

    Args:
        all_results: List of results from each seed

    Returns:
        Aggregated data with mean/std for each metric at each iteration
    """
    n_seeds = len(all_results)

    # Get all iterations from first result (all seeds should have same iterations)
    iterations = [it["iteration"] for it in all_results[0]["iteration_data"]]

    # Build iteration -> list of metrics across seeds
    iteration_metrics: Dict[int, List[Dict[str, Any]]] = {it: [] for it in iterations}

    for result in all_results:
        for it_data in result["iteration_data"]:
            iteration = it_data["iteration"]
            if iteration in iteration_metrics:
                iteration_metrics[iteration].append(it_data)

    # Compute mean/std for each iteration
    aggregated_data = []
    metric_keys = [
        "fo_H_abr", "fa_H_abr", "fc_H_abr", "fa_DaVal_abr", "bayesian_abr",
        "fo_H_auc", "fa_H_auc", "fc_H_auc",
        "n_Da_train", "n_Da_val", "n_H", "bad_rate_Da_val", "bad_rate_H", "count_bad_Da_val",
    ]

    for iteration in iterations:
        it_list = iteration_metrics.get(iteration, [])
        if not it_list:
            continue

        agg_it = {"iteration": iteration}
        for key in metric_keys:
            values = [it.get(key) for it in it_list if it.get(key) is not None]
            if values:
                agg_it[f"{key}_mean"] = float(np.mean(values))
                agg_it[f"{key}_std"] = float(np.std(values))
            else:
                agg_it[f"{key}_mean"] = None
                agg_it[f"{key}_std"] = None

        aggregated_data.append(agg_it)

    return {
        "n_seeds": n_seeds,
        "seeds": [r["seed"] for r in all_results],
        "iterations": iterations,
        "aggregated_data": aggregated_data,
    }


def print_results_summary(
    all_results: List[Dict[str, Any]],
    aggregated: Dict[str, Any],
    seeds: List[int],
    exp_dir: Path,
) -> None:
    """Print summary of results across all seeds."""
    n_seeds = len(seeds)
    result = all_results[0]  # Use first result for structure info

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Seeds run: {n_seeds}")
    print(f"  Panel snapshot iteration: {result['panel_snapshot_iter']}")
    print(f"  Holdout size: {result['n_holdout']}")

    # Aggregate panel (b) R² values
    r2_fa = [r["panel_b"]["fa"]["r2"] for r in all_results]
    r2_fo = [r["panel_b"]["fo"]["r2"] for r in all_results]
    r2_fc = [r["panel_b"]["fc"]["r2"] for r in all_results]

    print("\n  Panel (b) R² values (mean ± std):")
    print(f"    f_a: {np.mean(r2_fa):.4f} ± {np.std(r2_fa):.4f}")
    print(f"    f_o: {np.mean(r2_fo):.4f} ± {np.std(r2_fo):.4f}")
    print(f"    f_c: {np.mean(r2_fc):.4f} ± {np.std(r2_fc):.4f}")

    # Show aggregated iteration data
    agg_data = aggregated["aggregated_data"]
    print(f"\n  Tracked iterations: {len(agg_data)}")
    if agg_data:
        first = agg_data[0]
        last = agg_data[-1]
        print(f"    First (iter={first['iteration']}):")
        print(f"      fo_H_abr: {first['fo_H_abr_mean']:.4f} ± {first['fo_H_abr_std']:.4f}")
        print(f"      fc_H_abr: {first['fc_H_abr_mean']:.4f} ± {first['fc_H_abr_std']:.4f}")
        print(f"    Last (iter={last['iteration']}):")
        print(f"      fo_H_abr: {last['fo_H_abr_mean']:.4f} ± {last['fo_H_abr_std']:.4f}")
        print(f"      fc_H_abr: {last['fc_H_abr_mean']:.4f} ± {last['fc_H_abr_std']:.4f}")

    # Panel (d) diagnostic dump (from first seed as example)
    print("\n" + "=" * 70)
    print("PANEL (D) DIAGNOSTIC DUMP (first seed example)")
    print("=" * 70)
    diagnostic_iters = [0, 10, 50, 100]
    iter_to_data = {c["iteration"]: c for c in result["iteration_data"]}
    for it in diagnostic_iters:
        if it not in iter_to_data:
            print(f"  Iteration {it}: NOT TRACKED")
            continue
        c = iter_to_data[it]
        print(f"\n  Iteration {it}:")
        print(f"    n_Da_train: {c.get('n_Da_train', 'N/A')}")
        print(f"    n_Da_val: {c.get('n_Da_val', 'N/A')}")
        print(f"    bad_rate_Da_val: {c.get('bad_rate_Da_val', 'N/A')}")
        print(f"    count_bad_Da_val: {c.get('count_bad_Da_val', 'N/A')}")
        print(f"    fa_DaVal_abr: {c.get('fa_DaVal_abr', 'N/A')}")
        print(f"    fo_H_abr: {c.get('fo_H_abr', 'N/A')}")
        print(f"    bayesian_abr: {c.get('bayesian_abr', 'N/A')}")

    print("\n" + "=" * 70)
    print(f"\nResults saved to: {exp_dir}")
    print(f"  - Individual seeds: figure2_unified_seed*.json")
    print(f"  - Aggregated: figure2_aggregated.json")


if __name__ == "__main__":
    main()
