"""Verification script for Experiments I and II results.

Validates structural correctness, metric bounds, paper-faithful patterns,
and statistical reasonableness.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def verify_experiment_results(exp1_path: str, exp2_path: str) -> bool:
    """Comprehensive verification of experiment results.

    Args:
        exp1_path: Path to Experiment I results CSV.
        exp2_path: Path to Experiment II results CSV.

    Returns:
        True if all critical checks pass, False otherwise.
    """
    exp1 = pd.read_csv(exp1_path)
    exp2 = pd.read_csv(exp2_path)

    issues = []
    warnings = []

    # =================================================================
    # 1. STRUCTURAL CHECKS
    # =================================================================
    print("=" * 60)
    print("1. STRUCTURAL CHECKS")
    print("=" * 60)

    # Row counts
    exp1_expected = 800  # 100 replicates × 4 metrics × 2 methods
    exp2_expected = 800  # 100 replicates × 4 metrics × 2 methods

    print(f"Exp1 rows: {len(exp1)} (expected {exp1_expected})")
    print(f"Exp2 rows: {len(exp2)} (expected {exp2_expected})")

    if len(exp1) != exp1_expected:
        issues.append(f"Exp1 has {len(exp1)} rows, expected {exp1_expected}")
    if len(exp2) != exp2_expected:
        issues.append(f"Exp2 has {len(exp2)} rows, expected {exp2_expected}")

    # Metrics present
    expected_metrics = {"auc", "brier", "pauc", "abr"}
    exp1_metrics = set(exp1["metric"].unique())
    exp2_metrics = set(exp2["metric"].unique())

    print(f"\nExp1 metrics: {sorted(exp1_metrics)}")
    print(f"Exp2 metrics: {sorted(exp2_metrics)}")

    if exp1_metrics != expected_metrics:
        missing = expected_metrics - exp1_metrics
        extra = exp1_metrics - expected_metrics
        if missing:
            issues.append(f"Exp1 missing metrics: {missing}")
        if extra:
            warnings.append(f"Exp1 extra metrics: {extra}")

    if exp2_metrics != expected_metrics:
        missing = expected_metrics - exp2_metrics
        extra = exp2_metrics - expected_metrics
        if missing:
            issues.append(f"Exp2 missing metrics: {missing}")
        if extra:
            warnings.append(f"Exp2 extra metrics: {extra}")

    # Methods present
    exp1_methods = set(exp1["method"].unique())
    exp2_methods = set(exp2["train_method"].unique())
    print(f"\nExp1 methods: {sorted(exp1_methods)}")
    print(f"Exp2 train_methods: {sorted(exp2_methods)}")

    if exp1_methods != {"biased", "bayesian"}:
        issues.append(f"Exp1 unexpected methods: {exp1_methods}")
    if exp2_methods != {"accepts_only", "basl"}:
        issues.append(f"Exp2 unexpected train_methods: {exp2_methods}")

    # Threshold consistency
    if "abr_range" in exp1.columns:
        n_unique = exp1["abr_range"].nunique()
        print(f"\nExp1 abr_range values: {n_unique} unique")
        if n_unique != 1:
            issues.append(f"Exp1 has {n_unique} different abr_range values (expected 1)")
        else:
            print(f"  Value: {exp1['abr_range'].iloc[0]}")
    else:
        warnings.append("Exp1 missing abr_range column (pre-hardening run)")

    if "abr_range" in exp2.columns:
        n_unique = exp2["abr_range"].nunique()
        print(f"Exp2 abr_range values: {n_unique} unique")
        if n_unique != 1:
            issues.append(f"Exp2 has {n_unique} different abr_range values (expected 1)")
        else:
            print(f"  Value: {exp2['abr_range'].iloc[0]}")
    else:
        warnings.append("Exp2 missing abr_range column (pre-hardening run)")

    # NaN check
    exp1_value_cols = ["estimate", "truth"]
    nan_exp1 = exp1[exp1_value_cols].isna().sum().sum()
    nan_exp2 = exp2["value"].isna().sum()

    print(f"\nNaN values: Exp1={nan_exp1}, Exp2={nan_exp2}")
    if nan_exp1 > 0:
        issues.append(f"Exp1 has {nan_exp1} NaN values in estimate/truth")
    if nan_exp2 > 0:
        issues.append(f"Exp2 has {nan_exp2} NaN values in value")

    # =================================================================
    # 2. METRIC BOUNDS [0, 1]
    # =================================================================
    print("\n" + "=" * 60)
    print("2. METRIC BOUNDS [0, 1]")
    print("=" * 60)

    for col in ["estimate", "truth"]:
        vals = exp1[col]
        min_val, max_val = vals.min(), vals.max()
        status = "OK" if 0 <= min_val and max_val <= 1 else "FAIL"
        print(f"Exp1 {col}: [{min_val:.6f}, {max_val:.6f}] - {status}")
        if min_val < 0 or max_val > 1:
            issues.append(f"Exp1 {col} out of bounds: [{min_val:.4f}, {max_val:.4f}]")

    vals = exp2["value"]
    min_val, max_val = vals.min(), vals.max()
    status = "OK" if 0 <= min_val and max_val <= 1 else "FAIL"
    print(f"Exp2 value: [{min_val:.6f}, {max_val:.6f}] - {status}")
    if min_val < 0 or max_val > 1:
        issues.append(f"Exp2 value out of bounds: [{min_val:.4f}, {max_val:.4f}]")

    # =================================================================
    # 3. PAPER-FAITHFUL PATTERNS (Exp I)
    # =================================================================
    print("\n" + "=" * 60)
    print("3. PAPER-FAITHFUL PATTERNS (Exp I)")
    print("=" * 60)

    # Compute RMSE for each method/metric combination
    exp1["error"] = exp1["estimate"] - exp1["truth"]
    exp1["sq_error"] = exp1["error"] ** 2

    rmse_table = (
        exp1.groupby(["method", "metric"])
        .agg(RMSE=("sq_error", lambda x: np.sqrt(x.mean())))
        .unstack()
    )

    print("\nRMSE by Method and Metric:")
    print("-" * 50)
    rmse_display = rmse_table["RMSE"].round(6)
    print(rmse_display.to_string())

    # Extract RMSE values
    biased_brier = rmse_table.loc["biased", ("RMSE", "brier")]
    bayesian_brier = rmse_table.loc["bayesian", ("RMSE", "brier")]
    biased_auc = rmse_table.loc["biased", ("RMSE", "auc")]
    bayesian_auc = rmse_table.loc["bayesian", ("RMSE", "auc")]

    print("\n--- Key Comparisons ---")

    # Paper's main claim: Bayesian Brier RMSE < Biased Brier RMSE
    if bayesian_brier < biased_brier:
        print(
            f"Bayesian Brier RMSE ({bayesian_brier:.6f}) < "
            f"Biased ({biased_brier:.6f}): EXPECTED ✓"
        )
    else:
        print(
            f"Bayesian Brier RMSE ({bayesian_brier:.6f}) >= "
            f"Biased ({biased_brier:.6f}): UNEXPECTED ✗"
        )
        issues.append(
            f"Bayesian should have lower Brier RMSE than Biased "
            f"(got {bayesian_brier:.4f} vs {biased_brier:.4f})"
        )

    # AUC comparison (expected to be similar)
    auc_diff = abs(bayesian_auc - biased_auc)
    print(
        f"AUC RMSE difference: {auc_diff:.6f} "
        f"(Bayesian={bayesian_auc:.6f}, Biased={biased_auc:.6f})"
    )
    if bayesian_auc < biased_auc:
        print("  Bayesian AUC RMSE < Biased: Paper-faithful ✓")
    else:
        print("  Bayesian AUC RMSE >= Biased: Acceptable (AUC less affected)")

    # pAUC and ABR (threshold-dependent, biased may win)
    for metric in ["pauc", "abr"]:
        biased_val = rmse_table.loc["biased", ("RMSE", metric)]
        bayesian_val = rmse_table.loc["bayesian", ("RMSE", metric)]
        winner = "Bayesian" if bayesian_val < biased_val else "Biased"
        print(
            f"{metric.upper()} RMSE: Bayesian={bayesian_val:.6f}, "
            f"Biased={biased_val:.6f} (Winner: {winner})"
        )

    # =================================================================
    # 4. STATISTICAL REASONABLENESS (Exp II)
    # =================================================================
    print("\n" + "=" * 60)
    print("4. STATISTICAL REASONABLENESS (Exp II)")
    print("=" * 60)

    stats = exp2.groupby(["train_method", "metric"]).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        SE=("value", lambda x: x.std() / np.sqrt(len(x))),
    )

    print("\nStatistics by Training Method and Metric:")
    print("-" * 60)
    print(stats.round(6).to_string())

    # Check AUC is reasonable (> 0.7)
    auc_values = exp2[exp2["metric"] == "auc"]["value"]
    auc_mean = auc_values.mean()
    auc_std = auc_values.std()

    print(f"\n--- AUC Reasonableness ---")
    print(f"Overall AUC mean: {auc_mean:.6f}")
    print(f"Overall AUC std:  {auc_std:.6f}")

    if auc_mean < 0.7:
        issues.append(f"AUC mean ({auc_mean:.4f}) < 0.7 - model may not be learning")
        print(f"  AUC mean < 0.7: CONCERN ✗")
    else:
        print(f"  AUC mean >= 0.7: OK ✓")

    if auc_std > 0.05:
        warnings.append(f"AUC std ({auc_std:.4f}) > 0.05 - high variance")
        print(f"  AUC std > 0.05: WARNING")
    else:
        print(f"  AUC std <= 0.05: OK ✓")

    # Check Brier is reasonable (< 0.15)
    brier_values = exp2[exp2["metric"] == "brier"]["value"]
    brier_mean = brier_values.mean()

    print(f"\n--- Brier Reasonableness ---")
    print(f"Overall Brier mean: {brier_mean:.6f}")

    if brier_mean > 0.15:
        warnings.append(f"Brier mean ({brier_mean:.4f}) > 0.15 - poor calibration")
        print(f"  Brier mean > 0.15: WARNING")
    else:
        print(f"  Brier mean <= 0.15: OK ✓")

    # Compare BASL vs Accepts-only
    print("\n--- BASL vs Accepts-only Comparison ---")
    for metric in ["auc", "brier", "pauc", "abr"]:
        accepts = stats.loc[("accepts_only", metric), "mean"]
        basl = stats.loc[("basl", metric), "mean"]
        diff = basl - accepts
        pct_diff = (diff / accepts) * 100 if accepts != 0 else 0
        print(f"{metric.upper()}: accepts_only={accepts:.6f}, basl={basl:.6f}, diff={diff:+.6f} ({pct_diff:+.2f}%)")

    # Known tradeoff: BASL Brier may be worse due to pseudo-labels
    basl_brier = stats.loc[("basl", "brier"), "mean"]
    accepts_brier = stats.loc[("accepts_only", "brier"), "mean"]
    if basl_brier > accepts_brier:
        print(f"\nBASL Brier ({basl_brier:.6f}) > Accepts-only ({accepts_brier:.6f}): ")
        print("  This is EXPECTED (pseudo-labels hurt calibration)")

    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    if issues:
        print(f"\nISSUES FOUND ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n>>> VERIFICATION FAILED <<<")
        return False
    else:
        print("\n>>> ALL CRITICAL CHECKS PASSED <<<")
        return True


def main():
    """Find and verify the most recent experiment results."""
    import re

    experiments_dir = Path("experiments")

    # Find experiment directories with timestamp format (exp1_YYYYMMDD_HHMMSS)
    timestamp_pattern = re.compile(r"exp[12]_\d{8}_\d{6}$")

    exp1_dirs = sorted(
        [d for d in experiments_dir.glob("exp1_*") if timestamp_pattern.match(d.name)]
    )
    exp2_dirs = sorted(
        [d for d in experiments_dir.glob("exp2_*") if timestamp_pattern.match(d.name)]
    )

    if not exp1_dirs or not exp2_dirs:
        print("ERROR: Could not find experiment result directories with timestamp format")
        print("Expected format: exp1_YYYYMMDD_HHMMSS, exp2_YYYYMMDD_HHMMSS")
        sys.exit(1)

    # Use the most recent (by name, which includes timestamp)
    exp1_path = exp1_dirs[-1] / "exp1_results.csv"
    exp2_path = exp2_dirs[-1] / "exp2_results.csv"

    print(f"Verifying experiments:")
    print(f"  Exp1: {exp1_path}")
    print(f"  Exp2: {exp2_path}")
    print()

    if not exp1_path.exists():
        print(f"ERROR: {exp1_path} does not exist")
        sys.exit(1)
    if not exp2_path.exists():
        print(f"ERROR: {exp2_path} does not exist")
        sys.exit(1)

    success = verify_experiment_results(str(exp1_path), str(exp2_path))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
