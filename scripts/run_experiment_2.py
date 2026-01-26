#!/usr/bin/env python
"""
CLI entrypoint for Experiment II (Table 3).

Runs training comparison experiment over all replicates and saves
replicate-level results. Aggregation happens in notebooks only.

Usage:
    python scripts/run_exp2.py --data-dir synthetic_runs/my_run
    python scripts/run_exp2.py --data-dir synthetic_runs/my_run --config configs/exp2.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BASLConfig, BASLFilteringConfig, BASLLabelingConfig, XGBoostConfig
from src.data.csv_backend import CSVBackend
from src.evaluation.thresholds import ThresholdSpec
from src.experiments.exp2_runner import run_exp2


def main():
    parser = argparse.ArgumentParser(description="Experiment II: Training Comparison")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data bundle")
    parser.add_argument("--config", type=str, default="configs/experiment_2.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--n-folds", type=int, default=4, help="Number of CV folds")
    parser.add_argument("--n-bootstraps", type=int, default=25, help="Number of bootstraps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Setup output
    run_name = args.run_name or datetime.now().strftime("exp2_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "experiments" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment II: Training Comparison (Table 3)")
    print("=" * 70)
    print(f"Data: {args.data_dir}")
    print(f"Output: {output_dir}")
    print(f"Replicates: {args.n_folds} folds x {args.n_bootstraps} bootstraps = {args.n_folds * args.n_bootstraps}")

    # Load data backend
    backend = CSVBackend(
        data_dir=args.data_dir,
        n_folds=args.n_folds,
        n_bootstraps=args.n_bootstraps,
        random_seed=args.seed,
    )
    print(f"Loaded: {backend.get_summary()}")

    # Setup configs
    xgb_cfg = cfg.get("xgboost", {})
    model_cfg = XGBoostConfig(
        n_estimators=xgb_cfg.get("n_estimators", 100),
        max_depth=xgb_cfg.get("max_depth", 3),
        learning_rate=xgb_cfg.get("learning_rate", 0.1),
        random_seed=args.seed,
    )

    basl_raw = cfg.get("basl", {})
    basl_filtering = basl_raw.get("filtering", {})
    basl_labeling = basl_raw.get("labeling", {})
    basl_cfg = BASLConfig(
        max_iterations=basl_raw.get("max_iterations", 3),
        filtering=BASLFilteringConfig(
            beta_lower=basl_filtering.get("beta_lower", 0.05),
            beta_upper=basl_filtering.get("beta_upper", 1.0),
            random_seed=args.seed,
        ),
        labeling=BASLLabelingConfig(
            subsample_ratio=basl_labeling.get("subsample_ratio", 0.8),
            gamma=basl_labeling.get("gamma", 0.01),
            theta=basl_labeling.get("theta", 2.0),
            random_seed=args.seed,
        ),
    )

    # Parse ThresholdSpec from config (or use paper defaults)
    thresh_cfg = cfg.get("thresholds", {})
    if thresh_cfg:
        threshold_spec = ThresholdSpec(
            abr_range=tuple(thresh_cfg.get("abr_range", [0.2, 0.4])),
            pauc_max_fnr=thresh_cfg.get("pauc_max_fnr", 0.2),
            policy=thresh_cfg.get("policy", "fixed_value"),
        )
    else:
        threshold_spec = ThresholdSpec.paper_default()

    print(f"Thresholds: abr_range={threshold_spec.abr_range}, pauc_max_fnr={threshold_spec.pauc_max_fnr}")

    # Run experiment
    rows = run_exp2(
        backend=backend,
        model_cfg=model_cfg,
        basl_cfg=basl_cfg,
        threshold_spec=threshold_spec,
        backend_name="synthetic",
        run_name=run_name,
        show_progress=True,
    )

    # Save results as CSV
    output_file = output_dir / "exp2_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"\nResults saved to: {output_file}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
