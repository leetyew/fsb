"""
Acceptance loop simulation with integrated training and evaluation.

Implements the paper's approach (Algorithm C.2) where:
- Training side: BASL pseudo-labels rejects (up to jmax iterations per training call)
- Evaluation side: Monte Carlo pseudo-labeling with convergence (Algorithm 1)
- Loop runs for all n_periods without early stopping (per paper Section 6.1)
- All accepts used for training; separate holdout for oracle evaluation
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.basl.trainer import BASLTrainer
from src.config import AcceptanceLoopConfig, BASLConfig, BayesianEvalConfig, XGBoostConfig
from src.evaluation.bayesian_eval import bayesian_evaluate
from src.evaluation.metrics import compute_metrics
from src.io.synthetic_generator import SyntheticGenerator
from src.models.xgboost_model import XGBoostModel


class AcceptanceLoop:
    """Integrated training and evaluation loop for paper replication.

    Supports two training modes:
    1. Baseline (basl_cfg=None): Train only on accepts
    2. BASL (basl_cfg provided): Train on accepts + pseudo-labeled rejects

    Tracks three evaluation types per iteration for Figure 2:
    - Oracle: Metrics on external holdout with true labels
    - Accepts: Metrics on accepts only (biased)
    - Bayesian: Metrics using MC pseudo-labeling (Algorithm 1)
    """

    def __init__(
        self,
        generator: SyntheticGenerator,
        model_cfg: XGBoostConfig,
        cfg: AcceptanceLoopConfig,
        basl_cfg: Optional[BASLConfig] = None,
        bayesian_cfg: Optional[BayesianEvalConfig] = None,
    ) -> None:
        self.generator = generator
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.basl_cfg = basl_cfg
        self.bayesian_cfg = bayesian_cfg or BayesianEvalConfig()
        self.rng = np.random.default_rng(cfg.random_seed)

        # BASL trainer only if config provided
        self.basl_trainer = BASLTrainer(basl_cfg) if basl_cfg is not None else None

    def _accept_by_feature(
        self, df: pd.DataFrame, feature: str, accept_rate: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Accept top α percentile by feature value (lower = lower risk)."""
        threshold = np.percentile(df[feature], accept_rate * 100)
        accept_mask = df[feature] <= threshold
        return df[accept_mask].copy(), df[~accept_mask].copy()

    def run(
        self,
        holdout: pd.DataFrame,
        track_every: int = 1,
        show_progress: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBoostModel, List[Dict[str, Any]]]:
        """Run the integrated training and evaluation loop per Algorithm C.2.

        Per paper Section 6.1:
        - New batch of applicants arrives each period
        - Model makes accept/reject decisions based on target_accept_rate
        - Accepts get observed labels, training data grows over time
        - Loop runs for all n_periods without early stopping

        Args:
            holdout: External holdout for oracle evaluation.
            track_every: Track metrics every N iterations.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            (D_a, D_r, holdout, model, metrics_history)
            where metrics_history contains oracle/accepts/bayesian metrics per iteration
        """
        feature_cols = [f"x{i}" for i in range(self.generator.cfg.n_features)]
        alpha = self.cfg.target_accept_rate

        # External holdout for oracle evaluation (drawn separately per Algorithm C.2)
        X_holdout = holdout[feature_cols].values
        y_holdout = holdout["y"].values

        # Step 1: Generate initial batch and accept by feature (no model yet)
        initial_batch = self.generator.generate_population(self.cfg.initial_batch_size)
        initial_accepts, initial_rejects = self._accept_by_feature(
            initial_batch, self.cfg.x_v_feature, alpha
        )

        # Initialize cumulative accepts and rejects
        all_accepts = [initial_accepts]
        all_rejects = [initial_rejects]

        X_accepts = initial_accepts[feature_cols].values
        y_accepts = initial_accepts["y"].values
        X_rejects = initial_rejects[feature_cols].values

        # Initialize model on first accepts
        model = XGBoostModel(self.model_cfg)
        model.fit(X_accepts, y_accepts)

        # For BASL: filter rejects and prepare pseudo-labeling pool
        if self.basl_trainer is not None:
            X_rejects_filtered = self.basl_trainer.filter_rejects_once(
                X_accepts, X_rejects
            )
            X_labeled = X_accepts.copy()
            y_labeled = y_accepts.copy()
            X_rejects_pool = X_rejects_filtered.copy()
        else:
            X_labeled = X_accepts.copy()
            y_labeled = y_accepts.copy()
            X_rejects_pool = X_rejects.copy()

        # Tracking state
        metrics_history: List[Dict[str, Any]] = []

        # Initial evaluation (iteration 0)
        initial_metrics = self._evaluate(
            model,
            X_accepts, y_accepts,
            X_rejects_pool,
            X_holdout, y_holdout,
            0
        )
        metrics_history.append(initial_metrics)

        # Step 2: Main loop - process new batches each period
        iterator = range(1, self.cfg.n_periods + 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Training loop", leave=True, dynamic_ncols=True)

        for iteration in iterator:
            # Generate new batch of applicants this period
            batch = self.generator.generate_population(self.cfg.batch_size)

            # Model makes accept/reject decisions (accept top α by predicted score)
            batch_X = batch[feature_cols].values
            batch_scores = model.predict_proba(batch_X)

            # Accept those with lowest predicted bad rate (top α percentile)
            threshold = np.percentile(batch_scores, alpha * 100)
            accept_mask = batch_scores <= threshold

            batch_accepts = batch[accept_mask].copy()
            batch_rejects = batch[~accept_mask].copy()

            # Accumulate for final return
            all_accepts.append(batch_accepts)
            all_rejects.append(batch_rejects)

            # Add new accepts to training data (observed labels)
            if len(batch_accepts) > 0:
                new_X = batch_accepts[feature_cols].values
                new_y = batch_accepts["y"].values
                X_accepts = np.vstack([X_accepts, new_X])
                y_accepts = np.concatenate([y_accepts, new_y])
                X_labeled = np.vstack([X_labeled, new_X])
                y_labeled = np.concatenate([y_labeled, new_y])

            # Add new rejects to pool (for Bayesian eval and BASL)
            if len(batch_rejects) > 0:
                new_X_rejects = batch_rejects[feature_cols].values
                if self.basl_trainer is not None:
                    # Filter new rejects before adding to pool
                    new_filtered = self.basl_trainer.filter_rejects_once(
                        X_accepts, new_X_rejects
                    )
                    X_rejects_pool = np.vstack([X_rejects_pool, new_filtered])
                else:
                    X_rejects_pool = np.vstack([X_rejects_pool, new_X_rejects])

            # BASL mode: run pseudo-labeling iterations
            if self.basl_trainer is not None and len(X_rejects_pool) > 0:
                X_labeled, y_labeled, X_rejects_pool = self._run_basl_labeling(
                    X_labeled, y_labeled, X_rejects_pool
                )

            # Retrain model on updated labeled set
            model.fit(X_labeled, y_labeled)

            # Track metrics at specified intervals
            should_track = (iteration % track_every == 0) or (iteration == self.cfg.n_periods)

            if should_track:
                iteration_metrics = self._evaluate(
                    model,
                    X_accepts, y_accepts,
                    X_rejects_pool,
                    X_holdout, y_holdout,
                    iteration
                )
                metrics_history.append(iteration_metrics)

        # Combine all accepts/rejects for return
        D_a = pd.concat(all_accepts, ignore_index=True)
        D_r = pd.concat(all_rejects, ignore_index=True)

        return D_a, D_r, holdout, model, metrics_history

    def _run_basl_labeling(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_rejects_pool: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run BASL labeling for up to max_iterations (jmax from paper).

        Per paper Section 5.4: BASL iterates up to jmax times, stopping early
        if no new rejects are labeled in an iteration.

        Args:
            X_labeled: Current labeled features (accepts + pseudo-labeled rejects).
            y_labeled: Current labels.
            X_rejects_pool: Remaining unlabeled reject features.

        Returns:
            (X_labeled, y_labeled, X_rejects_pool) after labeling iterations.
        """
        max_iters = self.basl_cfg.max_iterations if self.basl_cfg else 1

        for _ in range(max_iters):
            if len(X_rejects_pool) == 0:
                break

            X_new, y_new, remaining_indices, _ = self.basl_trainer.label_one_iteration(
                X_labeled, y_labeled, X_rejects_pool
            )

            # Early stopping for BASL: no new labels this iteration
            if len(X_new) == 0:
                break

            # Update labeled set with newly labeled rejects
            X_labeled = np.vstack([X_labeled, X_new])
            y_labeled = np.concatenate([y_labeled, y_new])

            # Update reject pool (shrinks as rejects get labeled)
            X_rejects_pool = X_rejects_pool[remaining_indices]

        return X_labeled, y_labeled, X_rejects_pool

    def _evaluate(
        self,
        model: XGBoostModel,
        X_accepts: np.ndarray,
        y_accepts: np.ndarray,
        X_rejects: np.ndarray,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        iteration: int,
    ) -> Dict[str, Any]:
        """Three-way evaluation for Figure 2.

        Computes:
        - Oracle: Metrics on external holdout with true labels
        - Accepts: Metrics on accepts only (biased)
        - Bayesian: Metrics using MC pseudo-labeling (Algorithm 1)
        """
        metrics_list = ["auc", "pauc", "brier", "abr"]
        abr_range = self.bayesian_cfg.abr_range

        # Oracle: evaluate on external holdout with true labels
        scores_holdout = model.predict_proba(X_holdout)
        oracle_metrics = compute_metrics(
            y_holdout, scores_holdout, metrics_list, abr_range=abr_range
        )

        # Accepts-only: evaluate on internal holdout accepts only (biased)
        scores_accepts = model.predict_proba(X_accepts)
        accepts_metrics = compute_metrics(
            y_accepts, scores_accepts, metrics_list, abr_range=abr_range
        )

        # Bayesian: MC pseudo-labeling on internal holdout (Algorithm 1)
        scores_rejects = (
            model.predict_proba(X_rejects) if len(X_rejects) > 0 else np.array([])
        )

        # Use unique seed per iteration to ensure different MC samples
        iter_cfg = BayesianEvalConfig(
            n_bands=self.bayesian_cfg.n_bands,
            j_min=self.bayesian_cfg.j_min,
            j_max=self.bayesian_cfg.j_max,
            epsilon=self.bayesian_cfg.epsilon,
            prior_alpha=self.bayesian_cfg.prior_alpha,
            prior_beta=self.bayesian_cfg.prior_beta,
            random_seed=self.bayesian_cfg.random_seed + iteration,
            abr_range=self.bayesian_cfg.abr_range,
        )

        bayesian_result = bayesian_evaluate(
            y_accepts, scores_accepts, scores_rejects,
            cfg=iter_cfg,
            metrics_list=metrics_list,
        )

        # Extract mean values for consistency with other metrics
        bayesian_metrics = {
            metric: bayesian_result[metric]["mean"]
            for metric in metrics_list
        }

        return {
            "iteration": iteration,
            "oracle": oracle_metrics,
            "accepts": accepts_metrics,
            "bayesian": bayesian_metrics,
            "bayesian_full": bayesian_result,  # Include full posterior stats
        }
