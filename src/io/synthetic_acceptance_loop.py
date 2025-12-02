"""
Acceptance loop simulation with integrated training and evaluation.

Implements the paper's approach where:
- Training side: Weak learner pseudo-labels rejects (deterministic via BASL)
- Evaluation side: Monte Carlo pseudo-labeling with convergence (Algorithm 1)
- Configurable train/holdout split at accepts AND rejects level
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

    def _split_train_holdout(
        self, df: pd.DataFrame, train_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train and holdout sets."""
        n = len(df)
        n_train = int(n * train_ratio)
        indices = self.rng.permutation(n)
        train_indices = indices[:n_train]
        holdout_indices = indices[n_train:]
        return df.iloc[train_indices].copy(), df.iloc[holdout_indices].copy()

    def _check_early_stopping(
        self,
        current_metric: float,
        best_metric: Optional[float],
        patience_counter: int,
        metric_name: str,
    ) -> Tuple[bool, Optional[float], int]:
        """Check if early stopping should trigger.

        Returns:
            (should_stop, new_best_metric, new_patience_counter)
        """
        es_cfg = self.cfg.early_stopping

        # Higher is better for AUC/PAUC, lower is better for Brier/ABR
        higher_is_better = metric_name in ["auc", "pauc"]

        if best_metric is None:
            return False, current_metric, 0

        if higher_is_better:
            improved = current_metric > best_metric + es_cfg.min_delta
        else:
            improved = current_metric < best_metric - es_cfg.min_delta

        if improved:
            return False, current_metric, 0
        else:
            new_counter = patience_counter + 1
            should_stop = new_counter >= es_cfg.patience
            return should_stop, best_metric, new_counter

    def run(
        self,
        holdout: pd.DataFrame,
        track_every: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBoostModel, List[Dict[str, Any]]]:
        """Run the integrated training and evaluation loop.

        Args:
            holdout: External holdout for oracle evaluation.
            track_every: Track metrics every N iterations.

        Returns:
            (D_a, D_r, holdout, model, metrics_history)
            where metrics_history contains oracle/accepts/bayesian metrics per iteration
        """
        feature_cols = [f"x{i}" for i in range(self.generator.cfg.n_features)]
        alpha = self.cfg.target_accept_rate
        train_ratio = self.cfg.train_holdout_split
        es_cfg = self.cfg.early_stopping

        # Step 1: Generate initial population
        total_applicants = self.cfg.initial_batch_size + (
            self.cfg.batch_size * self.cfg.n_periods
        )
        population = self.generator.generate_population(total_applicants)

        # Step 2: Initial acceptance using x_v (simulates initial lending decision)
        accepts_df, rejects_df = self._accept_by_feature(
            population, self.cfg.x_v_feature, alpha
        )

        # Step 3: Split at accepts AND rejects level using configured ratio
        train_accepts, internal_holdout_accepts = self._split_train_holdout(
            accepts_df, train_ratio
        )
        train_rejects, internal_holdout_rejects = self._split_train_holdout(
            rejects_df, train_ratio
        )

        # Extract features and labels for training
        X_train_accepts = train_accepts[feature_cols].values
        y_train_accepts = train_accepts["y"].values
        X_train_rejects = train_rejects[feature_cols].values

        # Internal holdout for Bayesian evaluation (coin flip)
        X_internal_holdout_accepts = internal_holdout_accepts[feature_cols].values
        y_internal_holdout_accepts = internal_holdout_accepts["y"].values
        X_internal_holdout_rejects = internal_holdout_rejects[feature_cols].values

        # External holdout for oracle evaluation
        X_holdout = holdout[feature_cols].values
        y_holdout = holdout["y"].values

        # Step 4: Filter train_rejects with Isolation Forest (once) - only for BASL mode
        if self.basl_trainer is not None:
            X_train_rejects_filtered = self.basl_trainer.filter_rejects_once(
                X_train_accepts, X_train_rejects
            )
        else:
            X_train_rejects_filtered = X_train_rejects

        # Initialize training state
        X_labeled = X_train_accepts.copy()
        y_labeled = y_train_accepts.copy()
        X_rejects_pool = X_train_rejects_filtered.copy()

        # Track accumulated D_a and D_r as DataFrames for return
        D_a = train_accepts.copy()
        D_r = train_rejects.copy()

        # Initialize XGBoost model
        model = XGBoostModel(self.model_cfg)
        model.fit(X_labeled, y_labeled)

        # Tracking state
        metrics_history: List[Dict[str, Any]] = []
        best_metric: Optional[float] = None
        patience_counter = 0

        # Initial evaluation (iteration 0)
        initial_metrics = self._evaluate(
            model,
            X_internal_holdout_accepts, y_internal_holdout_accepts,
            X_internal_holdout_rejects,
            X_holdout, y_holdout,
            0
        )
        metrics_history.append(initial_metrics)
        best_metric = initial_metrics["bayesian"][es_cfg.metric]

        # Step 5: Main training loop
        for iteration in tqdm(range(1, self.cfg.n_periods + 1), desc="Training loop"):
            # BASL mode: weak learner labels confident rejects
            if self.basl_trainer is not None and len(X_rejects_pool) > 0:
                X_new, y_new, remaining_indices, _ = self.basl_trainer.label_one_iteration(
                    X_labeled, y_labeled, X_rejects_pool
                )

                # Update labeled set with newly labeled rejects
                if len(X_new) > 0:
                    X_labeled = np.vstack([X_labeled, X_new])
                    y_labeled = np.concatenate([y_labeled, y_new])

                # Update reject pool (shrinks as rejects get labeled)
                X_rejects_pool = X_rejects_pool[remaining_indices]

            # Retrain XGBoost on updated labeled set
            model.fit(X_labeled, y_labeled)

            # Track metrics at specified intervals
            should_track = (iteration % track_every == 0) or (iteration == self.cfg.n_periods)

            if should_track:
                iteration_metrics = self._evaluate(
                    model,
                    X_internal_holdout_accepts, y_internal_holdout_accepts,
                    X_internal_holdout_rejects,
                    X_holdout, y_holdout,
                    iteration
                )
                metrics_history.append(iteration_metrics)

            # Early stopping check
            if es_cfg.enabled:
                if should_track:
                    current_metric = iteration_metrics["bayesian"][es_cfg.metric]
                else:
                    # Compute metrics just for early stopping check
                    current_eval = self._evaluate(
                        model,
                        X_internal_holdout_accepts, y_internal_holdout_accepts,
                        X_internal_holdout_rejects,
                        X_holdout, y_holdout,
                        iteration
                    )
                    current_metric = current_eval["bayesian"][es_cfg.metric]

                should_stop, best_metric, patience_counter = self._check_early_stopping(
                    current_metric, best_metric, patience_counter, es_cfg.metric
                )
                if should_stop:
                    print(f"\nEarly stopping at iteration {iteration} "
                          f"(no improvement for {es_cfg.patience} iterations)")
                    break

            # Stop if no more rejects to label (BASL mode only)
            if self.basl_trainer is not None and len(X_rejects_pool) == 0:
                print(f"\nStopping at iteration {iteration} (no more rejects to label)")
                break

        return D_a, D_r, holdout, model, metrics_history

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
