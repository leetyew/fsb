"""
Acceptance loop simulation with integrated training and evaluation.

Implements the paper's approach (Algorithm C.2) where:
- Training side: BASL pseudo-labels rejects (up to jmax iterations per training call)
- Evaluation side: Monte Carlo pseudo-labeling with convergence (Algorithm 1)
- Loop runs for all n_periods without early stopping (per paper Section 6.1)
- All accepts used for training; separate holdout for oracle evaluation
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.basl.trainer import BASLTrainer
from src.config import AcceptanceLoopConfig, BASLConfig, BayesianEvalConfig, XGBoostConfig
from src.evaluation.bayesian_eval import bayesian_evaluate
from src.evaluation.metrics import compute_metrics
from src.io.synthetic_generator import SyntheticGenerator
from src.models.logistic_regression import LogisticRegressionConfig, LogisticRegressionModel
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

    def _add_bureau_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bureau-like score column x_v based on the raw synthetic x0.

        Per paper Appendix E.1, bad borrowers have higher x0 than good borrowers.
        We define x_v = -x0 so that:
          - good borrowers (lower x0) get higher x_v (better score)
          - bad borrowers (higher x0) get lower x_v (worse score)

        This keeps the generator faithful to Appendix E.1 while making
        'highest x_v' correspond to lowest risk, matching Algorithm C.2.
        """
        df = df.copy()
        df["x_v"] = -df["x0"]
        return df

    def _accept_by_feature(
        self, df: pd.DataFrame, feature: str, accept_rate: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Accept top α percentile by score (higher x_v = lower risk).

        Per Algorithm C.2:
          - τ is the (1-α)-th percentile of x_v (e.g. 85th percentile for α=0.15)
          - D_a = { (X_i, y_i): x_{i,v} >= τ }  (top α fraction by score)
          - D_r = { (X_i, y_i): x_{i,v} <  τ }
        """
        # For α=0.15, compute 85th percentile threshold
        threshold = np.percentile(df[feature], (1 - accept_rate) * 100)
        accept_mask = df[feature] >= threshold
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
        # Add bureau score x_v = -x0 for paper-faithful acceptance
        initial_batch = self._add_bureau_score(initial_batch)
        initial_accepts, initial_rejects = self._accept_by_feature(
            initial_batch, self.cfg.x_v_feature, alpha
        )

        # Initialize cumulative accepts and rejects (true labels only)
        all_accepts = [initial_accepts]
        all_rejects = [initial_rejects]

        # X_accepts/y_accepts: cumulative TRUE accepts with observed labels
        # These grow each iteration and NEVER include pseudo-labels
        X_accepts = initial_accepts[feature_cols].values
        y_accepts = initial_accepts["y"].values

        # X_rejects_for_eval: cumulative rejects for Bayesian evaluation
        # This is the full D_r^(j) used for evaluation, grows each iteration
        X_rejects_for_eval = initial_rejects[feature_cols].values
        # y_rejects_true: true labels for rejects (for oracle training in simulation)
        y_rejects_true = initial_rejects["y"].values

        # Initialize model on first accepts (f_a: accepts-based scorecard)
        model = XGBoostModel(self.model_cfg)
        model.fit(X_accepts, y_accepts)

        # Initialize oracle model (f_o: trained on D_a ∪ D_r with true labels)
        # Per Algorithm C.2: f_o is the benchmark that has access to all true labels
        oracle_model = XGBoostModel(self.model_cfg)
        X_all = np.vstack([X_accepts, X_rejects_for_eval])
        y_all = np.concatenate([y_accepts, y_rejects_true])
        oracle_model.fit(X_all, y_all)

        # ALWAYS create f_prior (baseline XGBoost + LR calibrator) for Bayesian evaluation
        # Per paper Section 4.3: f_prior is SHARED by ALL methods (baseline and BASL)
        # The difference between methods is ONLY in how f_eval is trained, not f_prior
        # f_prior represents the "legacy/historical acceptance model" for reject priors
        # CRITICAL: f_prior is trained ONLY on accepts with true labels, never pseudo-labels
        baseline_model = XGBoostModel(self.model_cfg)
        baseline_model.fit(X_accepts, y_accepts)
        baseline_scores = baseline_model.predict_proba(X_accepts).reshape(-1, 1)
        lr_cfg = LogisticRegressionConfig(
            C=1.0, penalty="l2", solver="lbfgs", random_seed=self.model_cfg.random_seed
        )
        prior_calibrator = LogisticRegressionModel(lr_cfg)
        prior_calibrator.fit(baseline_scores, y_accepts)

        # Tracking state
        metrics_history: List[Dict[str, Any]] = []

        # Best model tracking (Section 4.2-4.3 from paper)
        best_model = None
        best_bayesian_abr = float('inf')
        best_iteration = 0

        # Initial evaluation (iteration 0)
        # Use X_rejects_for_eval (full D_r) for Bayesian evaluation, NOT the BASL pool
        initial_metrics = self._evaluate(
            model,
            oracle_model,
            X_accepts, y_accepts,
            X_rejects_for_eval,
            X_holdout, y_holdout,
            0,
            baseline_model,
            prior_calibrator,
        )
        metrics_history.append(initial_metrics)

        # Initialize best model with iteration 0
        if self.basl_trainer is not None:
            bayesian_abr = initial_metrics['bayesian']['abr']
            if bayesian_abr < best_bayesian_abr:
                best_model = copy.deepcopy(model)
                best_bayesian_abr = bayesian_abr
                best_iteration = 0

        # Step 2: Main loop - process new batches each period
        iterator = range(1, self.cfg.n_periods + 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Training loop", leave=True, dynamic_ncols=True)

        for iteration in iterator:
            # Generate new batch of applicants this period
            batch = self.generator.generate_population(self.cfg.batch_size)
            # Add bureau score x_v = -x0 for paper-faithful acceptance
            batch = self._add_bureau_score(batch)

            # MAR acceptance: accept by feature x_v only (same rule as initial batch)
            # Per paper Section 1.2 / Appendix C.2: acceptance depends on x_v, NOT model
            # "Acceptance NEVER uses model outputs" - this rule is FIXED for all iterations
            batch_accepts, batch_rejects = self._accept_by_feature(
                batch, self.cfg.x_v_feature, alpha
            )

            # Accumulate for final return
            all_accepts.append(batch_accepts)
            all_rejects.append(batch_rejects)

            # Add new accepts to cumulative accepts (true labels only)
            if len(batch_accepts) > 0:
                new_X = batch_accepts[feature_cols].values
                new_y = batch_accepts["y"].values
                X_accepts = np.vstack([X_accepts, new_X])
                y_accepts = np.concatenate([y_accepts, new_y])

                # Update baseline model and LR calibrator on all accumulated accepts
                # Per Section 4.3: prior comes from historical acceptance model
                if baseline_model is not None:
                    baseline_model.fit(X_accepts, y_accepts)
                    baseline_scores = baseline_model.predict_proba(X_accepts).reshape(-1, 1)
                    prior_calibrator.fit(baseline_scores, y_accepts)

            # Add new rejects to cumulative reject set for evaluation
            if len(batch_rejects) > 0:
                new_X_rejects = batch_rejects[feature_cols].values
                new_y_rejects = batch_rejects["y"].values
                X_rejects_for_eval = np.vstack([X_rejects_for_eval, new_X_rejects])
                y_rejects_true = np.concatenate([y_rejects_true, new_y_rejects])

            # Train oracle model on D_a ∪ D_r with true labels (per Algorithm C.2)
            # f_o represents the benchmark with access to all true labels
            X_all = np.vstack([X_accepts, X_rejects_for_eval])
            y_all = np.concatenate([y_accepts, y_rejects_true])
            oracle_model.fit(X_all, y_all)

            # Train model for this iteration
            # Per paper Algorithm C.2 and two-loop separation:
            # - Baseline: train on accepts only (f_a)
            # - BASL: run fresh on current D_a^(j), D_r^(j), pseudo-labels are ephemeral
            if self.basl_trainer is not None:
                # BASL mode: run fresh each iteration on current snapshot
                # Filter the FULL current reject set (not an incrementally shrinking pool)
                X_rejects_filtered = self.basl_trainer.filter_rejects_once(
                    X_accepts, X_rejects_for_eval
                )

                if len(X_rejects_filtered) > 0:
                    # Initialize local BASL state from accepts only (no carryover)
                    X_basl = X_accepts.copy()
                    y_basl = y_accepts.copy()

                    # Run BASL pseudo-labeling on filtered rejects
                    X_basl, y_basl, _ = self._run_basl_labeling(
                        X_basl, y_basl, X_rejects_filtered.copy()
                    )

                    # Train BASL evaluation model on accepts + pseudo-labeled rejects
                    model.fit(X_basl, y_basl)
                    # Pseudo-labels are now discarded (not stored for next iteration)
                else:
                    # No filtered rejects available, train on accepts only
                    model.fit(X_accepts, y_accepts)
            else:
                # Baseline mode: train only on true accepts (no pseudo-labels)
                model.fit(X_accepts, y_accepts)

            # Track metrics at specified intervals
            should_track = (iteration % track_every == 0) or (iteration == self.cfg.n_periods)

            if should_track:
                # Use X_rejects_for_eval (full D_r) for Bayesian evaluation
                # This ensures baseline and BASL are evaluated on the same population
                iteration_metrics = self._evaluate(
                    model,
                    oracle_model,
                    X_accepts, y_accepts,
                    X_rejects_for_eval,
                    X_holdout, y_holdout,
                    iteration,
                    baseline_model,
                    prior_calibrator,
                )
                metrics_history.append(iteration_metrics)

                # Track best model based on Bayesian ABR (Section 4.2 from paper)
                if self.basl_trainer is not None:
                    bayesian_abr = iteration_metrics['bayesian']['abr']
                    if bayesian_abr < best_bayesian_abr:
                        best_model = copy.deepcopy(model)
                        best_bayesian_abr = bayesian_abr
                        best_iteration = iteration

        # Combine all accepts/rejects for return
        D_a = pd.concat(all_accepts, ignore_index=True)
        D_r = pd.concat(all_rejects, ignore_index=True)

        # Return best model for BASL, last model for baseline (Section 4.3 from paper)
        final_model = best_model if (self.basl_trainer is not None and best_model is not None) else model

        if self.basl_trainer is not None and best_model is not None:
            print(f"  BASL: Using best model from iteration {best_iteration} (Bayesian ABR={best_bayesian_abr:.4f})")

        return D_a, D_r, holdout, final_model, metrics_history

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
        oracle_model: XGBoostModel,
        X_accepts: np.ndarray,
        y_accepts: np.ndarray,
        X_rejects: np.ndarray,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        iteration: int,
        baseline_model: Optional[XGBoostModel] = None,
        prior_calibrator: Optional[LogisticRegressionModel] = None,
    ) -> Dict[str, Any]:
        """Four-way evaluation for Figures 2-4.

        CRITICAL: X_rejects must be the FULL reject population (D_r), NOT the
        shrinking BASL labeling pool. Both baseline and BASL must evaluate on
        the same reject set for fair comparison.

        Computes:
        - Oracle: f_o (trained on D_a ∪ D_r) metrics on external holdout
        - Model: f_a/f_c (accepts-based or BASL) metrics on holdout
        - Accepts: f_a/f_c metrics on accepts only (biased)
        - Bayesian: f_a/f_c metrics using MC pseudo-labeling (Algorithm 1)

        Args:
            model: The model being evaluated (f_a for baseline, f_c for BASL).
            oracle_model: f_o trained on D_a ∪ D_r with true labels.
            X_rejects: Full reject population for Bayesian evaluation (must NOT
                       be the BASL pool which shrinks during training).
            baseline_model: Accepts-only XGBoost for generating prior scores (f_prior).
                            ALWAYS provided - shared by baseline and BASL.
            prior_calibrator: LR calibrator that maps XGBoost scores to probabilities.
                              ALWAYS provided - part of f_prior pipeline.
        """
        metrics_list = ["auc", "pauc", "brier", "abr"]
        abr_range = self.bayesian_cfg.abr_range

        # Oracle model (f_o): evaluate on external holdout with true labels
        # Per Algorithm C.2: f_o is trained on D_a ∪ D_r and represents the benchmark
        oracle_scores_holdout = oracle_model.predict_proba(X_holdout)
        oracle_metrics = compute_metrics(
            y_holdout, oracle_scores_holdout, metrics_list, abr_range=abr_range
        )

        # Model (f_a/f_c): evaluate on external holdout with true labels
        # This shows how the accepts-based or BASL model performs on unbiased data
        model_scores_holdout = model.predict_proba(X_holdout)
        model_holdout_metrics = compute_metrics(
            y_holdout, model_scores_holdout, metrics_list, abr_range=abr_range
        )

        # Accepts-only evaluation: evaluate on biased D_a population
        # This shows the optimistically biased view from evaluating only on accepts.
        # Per paper: Accepts ABR is intentionally biased low because rejects are missing.
        scores_accepts = model.predict_proba(X_accepts)
        accepts_metrics = compute_metrics(
            y_accepts, scores_accepts, metrics_list, abr_range=abr_range
        )

        # Bayesian: MC pseudo-labeling on internal holdout (Algorithm 1)
        # Per paper: f_eval scores are used for metrics, f_prior scores for pseudo-labeling
        # - f_eval = model (the model being evaluated, differs between baseline/BASL)
        # - f_prior = baseline XGBoost + LR calibrator (shared, always trained on accepts only)
        if len(X_rejects) > 0:
            # f_eval scores for rejects (used in metric computation)
            eval_scores_rejects = model.predict_proba(X_rejects)

            # f_prior scores for rejects (used for pseudo-label sampling)
            raw_prior_scores = baseline_model.predict_proba(X_rejects)
            prior_scores_rejects = prior_calibrator.predict_proba(raw_prior_scores.reshape(-1, 1))
        else:
            eval_scores_rejects = np.array([])
            prior_scores_rejects = np.array([])

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
            y_accepts, scores_accepts, eval_scores_rejects,
            cfg=iter_cfg,
            metrics_list=metrics_list,
            prior_scores_rejects=prior_scores_rejects,
        )

        # Extract mean values for consistency with other metrics
        bayesian_metrics = {
            metric: bayesian_result[metric]["mean"]
            for metric in metrics_list
        }

        return {
            "iteration": iteration,
            "oracle": oracle_metrics,           # f_o on holdout (benchmark)
            "model_holdout": model_holdout_metrics,  # f_a/f_c on holdout
            "accepts": accepts_metrics,         # f_a/f_c on accepts only (biased)
            "bayesian": bayesian_metrics,       # Bayesian estimate
            "bayesian_full": bayesian_result,   # Include full posterior stats
        }
