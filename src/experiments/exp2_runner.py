"""
Experiment II: Training Comparison (Table 3).

Per plan Part C.4: Train corrected models (accepts-only, BASL) and
evaluate ONLY on H_boot.

This runner:
- Loops over replicates (4-fold CV x 25 bootstraps = 100)
- Never aggregates (writes replicate-level rows only)
- Produces output suitable for Table 3 aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.basl.trainer import BASLTrainer
from src.config import BASLConfig, XGBoostConfig
from src.data.backend import DataBackend, ReplicateData
from src.evaluation.metrics import compute_metrics
from src.evaluation.thresholds import ThresholdSpec
from src.models.xgboost_model import XGBoostModel


@dataclass
class Exp2Row:
    """Single row of Experiment II output.

    Per plan Part C.4 output schema.
    """

    experiment: int
    backend: str
    train_method: str
    metric: str
    value: float
    fold_id: int
    bootstrap_id: int
    replicate_id: int
    replicate_key: str
    run_name: str
    # Threshold metadata for auditability
    abr_range: str
    pauc_max_fnr: float
    threshold_policy: str


def run_exp2_replicate(
    rep: ReplicateData,
    model_cfg: XGBoostConfig,
    basl_cfg: BASLConfig,
    threshold_spec: ThresholdSpec,
    backend_name: str,
    run_name: str,
) -> List[Exp2Row]:
    """Run Experiment II for a single replicate.

    Trains two models:
    - accepts_only: trained on Da_train
    - basl: trained on Da_train + pseudo-labeled rejects

    Both evaluated on H_boot (this replicate's bootstrapped holdout).

    Args:
        rep: ReplicateData with train/val/rejects/holdout.
        model_cfg: XGBoost configuration.
        basl_cfg: BASL configuration.
        threshold_spec: ThresholdSpec for ABR/pAUC thresholds (ensures consistency).
        backend_name: "synthetic" or "real".
        run_name: Name for this run.

    Returns:
        List of Exp2Row, one per (train_method, metric) combination.
    """
    rows: List[Exp2Row] = []
    metrics_list = ["auc", "brier", "pauc", "abr"]

    # Threshold metadata for auditability
    abr_range_str = str(threshold_spec.abr_range)

    # Method 1: Accepts-only
    accepts_model = XGBoostModel(model_cfg)
    accepts_model.fit(rep.Da_train_X, rep.Da_train_y)

    scores_holdout_accepts = accepts_model.predict_proba(rep.H_X)
    accepts_metrics = compute_metrics(
        rep.H_y, scores_holdout_accepts, metrics_list, threshold_spec=threshold_spec
    )

    for metric in metrics_list:
        rows.append(Exp2Row(
            experiment=2,
            backend=backend_name,
            train_method="accepts_only",
            metric=metric,
            value=accepts_metrics[metric],
            fold_id=rep.fold_id,
            bootstrap_id=rep.bootstrap_id,
            replicate_id=rep.replicate_id,
            replicate_key=rep.replicate_key,
            run_name=run_name,
            abr_range=abr_range_str,
            pauc_max_fnr=threshold_spec.pauc_max_fnr,
            threshold_policy=threshold_spec.policy,
        ))

    # Method 2: BASL
    basl_trainer = BASLTrainer(basl_cfg)

    # Filter rejects
    X_rejects_filtered = basl_trainer.filter_rejects_once(
        rep.Da_train_X, rep.Dr_X
    )

    # Run BASL labeling iterations
    X_labeled = rep.Da_train_X.copy()
    y_labeled = rep.Da_train_y.copy()
    X_pool = X_rejects_filtered.copy()

    for _ in range(basl_cfg.max_iterations):
        if len(X_pool) == 0:
            break

        X_new, y_new, remaining_indices, _ = basl_trainer.label_one_iteration(
            X_labeled, y_labeled, X_pool
        )

        if len(X_new) == 0:
            break

        X_labeled = np.vstack([X_labeled, X_new])
        y_labeled = np.concatenate([y_labeled, y_new])
        X_pool = X_pool[remaining_indices]

    # Train BASL model
    basl_model = XGBoostModel(model_cfg)
    basl_model.fit(X_labeled, y_labeled)

    scores_holdout_basl = basl_model.predict_proba(rep.H_X)
    basl_metrics = compute_metrics(
        rep.H_y, scores_holdout_basl, metrics_list, threshold_spec=threshold_spec
    )

    for metric in metrics_list:
        rows.append(Exp2Row(
            experiment=2,
            backend=backend_name,
            train_method="basl",
            metric=metric,
            value=basl_metrics[metric],
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


def run_exp2(
    backend: DataBackend,
    model_cfg: XGBoostConfig,
    basl_cfg: BASLConfig,
    threshold_spec: Optional[ThresholdSpec] = None,
    backend_name: str = "synthetic",
    run_name: str = "exp2",
    show_progress: bool = True,
) -> List[Exp2Row]:
    """Run Experiment II over all replicates.

    Per plan Part C:
    - Loops over replicate_id
    - Never aggregates
    - Writes replicate-level rows only

    Args:
        backend: DataBackend providing replicates.
        model_cfg: XGBoost configuration.
        basl_cfg: BASL configuration.
        threshold_spec: ThresholdSpec for ABR/pAUC thresholds.
            If None, uses paper defaults.
        backend_name: "synthetic" or "real".
        run_name: Name for this run.
        show_progress: Whether to show progress bar.

    Returns:
        List of all Exp2Row results (one per replicate x method x metric).
    """
    # Use paper defaults if not provided
    if threshold_spec is None:
        threshold_spec = ThresholdSpec.paper_default()

    all_rows: List[Exp2Row] = []
    n_replicates = backend.n_replicates()

    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(range(n_replicates), desc="Exp II replicates")
    else:
        iterator = range(n_replicates)

    for i in iterator:
        rep = backend.get_replicate(i)
        rows = run_exp2_replicate(
            rep=rep,
            model_cfg=model_cfg,
            basl_cfg=basl_cfg,
            threshold_spec=threshold_spec,
            backend_name=backend_name,
            run_name=run_name,
        )
        all_rows.extend(rows)

    return all_rows
