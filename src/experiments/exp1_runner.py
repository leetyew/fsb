"""
Experiment I: Evaluation Accuracy (Table 2).

Per plan Part C.3: For each replicate and evaluation method:
- Model under evaluation: f (trained on Da_train)
- Truth = metric(f, H_boot) - model f evaluated on replicate's holdout
- Estimate = evaluation method output (biased or Bayesian)

This runner:
- Loops over replicates (4-fold CV x 25 bootstraps = 100)
- Never aggregates (writes replicate-level rows only)
- Produces output suitable for Table 2 aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.config import BayesianEvalConfig, XGBoostConfig
from src.data.backend import DataBackend, ReplicateData
from src.evaluation.bayesian_eval import bayesian_evaluate
from src.evaluation.metrics import compute_metrics
from src.evaluation.thresholds import ThresholdSpec
from src.models.xgboost_model import XGBoostModel


@dataclass
class Exp1Row:
    """Single row of Experiment I output.

    Per plan Part C.3 output schema.
    """

    experiment: int
    backend: str
    method: str
    metric: str
    estimate: float
    truth: float
    fold_id: int
    bootstrap_id: int
    replicate_id: int
    replicate_key: str
    run_name: str
    # Threshold metadata for auditability
    abr_range: str
    pauc_max_fnr: float
    threshold_policy: str


def run_exp1_replicate(
    rep: ReplicateData,
    model_cfg: XGBoostConfig,
    bayesian_cfg: BayesianEvalConfig,
    threshold_spec: ThresholdSpec,
    backend_name: str,
    run_name: str,
) -> List[Exp1Row]:
    """Run Experiment I for a single replicate.

    For each evaluation method (biased, bayesian), computes:
    - estimate: what the method outputs
    - truth: actual metric on this replicate's holdout

    Args:
        rep: ReplicateData with train/val/rejects/holdout.
        model_cfg: XGBoost configuration.
        bayesian_cfg: Bayesian evaluation configuration.
        threshold_spec: ThresholdSpec for ABR/pAUC thresholds (ensures consistency).
        backend_name: "synthetic" or "real".
        run_name: Name for this run.

    Returns:
        List of Exp1Row, one per (method, metric) combination.
    """
    rows: List[Exp1Row] = []

    # Train model on accepts (Da_train)
    model = XGBoostModel(model_cfg)
    model.fit(rep.Da_train_X, rep.Da_train_y)

    # Score all sets
    scores_train = model.predict_proba(rep.Da_train_X)
    scores_val = model.predict_proba(rep.Da_val_X)
    scores_rejects = model.predict_proba(rep.Dr_X)
    scores_holdout = model.predict_proba(rep.H_X)

    # Truth: model f evaluated on this replicate's bootstrapped holdout
    # Uses same ThresholdSpec as estimates for consistent comparison
    metrics_list = ["auc", "brier", "pauc", "abr"]
    truth_metrics = compute_metrics(
        rep.H_y, scores_holdout, metrics_list, threshold_spec=threshold_spec
    )

    # Method 1: Biased estimate (accepts-only)
    # Evaluate on Da_val (held-out accepts from this fold)
    # Uses same ThresholdSpec as truth for consistent comparison
    biased_metrics = compute_metrics(
        rep.Da_val_y, scores_val, metrics_list, threshold_spec=threshold_spec
    )

    # Threshold metadata for auditability
    abr_range_str = str(threshold_spec.abr_range)

    for metric in ["auc", "brier", "pauc", "abr"]:
        rows.append(Exp1Row(
            experiment=1,
            backend=backend_name,
            method="biased",
            metric=metric,
            estimate=biased_metrics[metric],
            truth=truth_metrics[metric],
            fold_id=rep.fold_id,
            bootstrap_id=rep.bootstrap_id,
            replicate_id=rep.replicate_id,
            replicate_key=rep.replicate_key,
            run_name=run_name,
            abr_range=abr_range_str,
            pauc_max_fnr=threshold_spec.pauc_max_fnr,
            threshold_policy=threshold_spec.policy,
        ))

    # Method 2: Bayesian evaluation
    # Uses accepts (train) with true labels + pseudo-labeled rejects
    # to estimate what performance would be on full population

    # Combine train + val for Bayesian (all accepts we have labels for)
    accepts_X = np.vstack([rep.Da_train_X, rep.Da_val_X])
    accepts_y = np.concatenate([rep.Da_train_y, rep.Da_val_y])
    accepts_scores = model.predict_proba(accepts_X)

    bayesian_result = bayesian_evaluate(
        y_accepts=accepts_y,
        scores_accepts=accepts_scores,
        scores_rejects=scores_rejects,
        cfg=bayesian_cfg,
        metrics_list=["auc", "brier", "pauc", "abr"],
        threshold_spec=threshold_spec,
    )

    for metric in ["auc", "brier", "pauc", "abr"]:
        rows.append(Exp1Row(
            experiment=1,
            backend=backend_name,
            method="bayesian",
            metric=metric,
            estimate=bayesian_result[metric]["mean"],
            truth=truth_metrics[metric],
            fold_id=rep.fold_id,
            bootstrap_id=rep.bootstrap_id,
            replicate_id=rep.replicate_id,
            replicate_key=rep.replicate_key,
            run_name=run_name,
            abr_range=abr_range_str,
            pauc_max_fnr=threshold_spec.pauc_max_fnr,
            threshold_policy=threshold_spec.policy,
        ))

    return rows


def run_exp1(
    backend: DataBackend,
    model_cfg: XGBoostConfig,
    bayesian_cfg: BayesianEvalConfig,
    threshold_spec: Optional[ThresholdSpec] = None,
    backend_name: str = "synthetic",
    run_name: str = "exp1",
    show_progress: bool = True,
) -> List[Exp1Row]:
    """Run Experiment I over all replicates.

    Per plan Part C:
    - Loops over replicate_id
    - Never aggregates
    - Writes replicate-level rows only

    Args:
        backend: DataBackend providing replicates.
        model_cfg: XGBoost configuration.
        bayesian_cfg: Bayesian evaluation configuration.
        threshold_spec: ThresholdSpec for ABR/pAUC thresholds.
            If None, uses paper defaults.
        backend_name: "synthetic" or "real".
        run_name: Name for this run.
        show_progress: Whether to show progress bar.

    Returns:
        List of all Exp1Row results (one per replicate x method x metric).
    """
    # Use paper defaults if not provided
    if threshold_spec is None:
        threshold_spec = ThresholdSpec.paper_default()

    all_rows: List[Exp1Row] = []
    n_replicates = backend.n_replicates()

    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(range(n_replicates), desc="Exp I replicates")
    else:
        iterator = range(n_replicates)

    for i in iterator:
        rep = backend.get_replicate(i)
        rows = run_exp1_replicate(
            rep=rep,
            model_cfg=model_cfg,
            bayesian_cfg=bayesian_cfg,
            threshold_spec=threshold_spec,
            backend_name=backend_name,
            run_name=run_name,
        )
        all_rows.extend(rows)

    return all_rows
