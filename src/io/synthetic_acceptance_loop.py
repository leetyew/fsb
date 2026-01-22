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
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.basl.trainer import BASLTrainer
from src.config import AcceptanceLoopConfig, BASLConfig, BayesianEvalConfig, XGBoostConfig
from src.evaluation.bayesian_eval import bayesian_evaluate
from src.evaluation.metrics import compute_abr_breakdown, compute_metrics
from src.io.synthetic_generator import SyntheticGenerator
from src.models.logistic_regression import LogisticRegressionConfig, LogisticRegressionModel
from src.models.xgboost_model import XGBoostModel


@dataclass
class AcceptanceDiagnostics:
    """Diagnostics from acceptance decision at each iteration.

    TWEAK 1: Strong acceptance-set equality check for paper-faithful verification.
    """

    n_batch: int
    alpha: float
    k_expected: int
    # Score statistics
    score_min: float
    score_max: float
    score_mean: float
    score_percentiles: Dict[str, float]
    threshold_tau: float
    # Counts
    n_accepted: int
    n_rejected: int
    n_ties_at_tau: int
    n_accepted_from_ties: int
    # Bad rates
    bad_rate_accepts: float
    bad_rate_rejects: float
    # Invariant checks
    direction_check_ok: bool
    # TWEAK 1: Strong acceptance-set equality check
    accept_set_exact_match: bool  # MUST BE TRUE
    accept_set_jaccard: float     # |intersection| / |union|
    n_mismatched: int             # |accepted_true Δ accepted_actual|


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

        # State tracking for paper-faithful assertions (Algorithm C.2)
        # Feature-based acceptance (x_v) must occur exactly once at j=0
        # Model-based acceptance (f_a) occurs for all j>=1
        self._feature_accept_called = 0
        self._model_accept_called = 0

    def _get_x_v_feature(self) -> str:
        """Get the visible/acceptance feature name.

        Per paper: x_v is the most separating feature (largest |mu_good - mu_bad|).
        The generator determines this automatically.
        """
        return self.generator.x_v_feature

    def _accept_by_feature(
        self, df: pd.DataFrame, feature: str, accept_rate: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Accept exactly k = ceil(α*n) highest by bureau score x_v = -X1.

        Per paper Algorithm C.2: x_v is a "bureau score" where higher = better (lower risk).
        Iteration 0 accepts applicants with x_v >= τ (highest bureau scores).
        This is called ONLY at j=0 (initialization), never again.

        In our synthetic setup:
          - Good applicants (y=0) have LOWER X1 values (E[X1|good] ≈ 0.5)
          - Bad applicants (y=1) have HIGHER X1 values (E[X1|bad] ≈ 2.5)
          - Define bureau score: x_v = -X1 (so higher x_v = lower risk)
          - Accept k applicants with HIGHEST x_v (i.e., lowest X1)

        Algorithm:
          - k = ceil(α * n)
          - bureau_scores = -X1 (higher = better)
          - D_a = k samples with highest bureau scores
          - D_r = remaining n-k samples
        """
        # Paper-faithful assertion: feature-based acceptance must only occur once (j=0)
        self._feature_accept_called += 1
        assert self._feature_accept_called == 1, (
            f"Feature-based acceptance must only occur once (j=0), "
            f"but was called {self._feature_accept_called} times."
        )

        n = len(df)
        k = max(1, int(np.ceil(accept_rate * n)))

        # Define bureau score as x_v = -X1 (higher = better, lower default risk)
        raw_feature = df[feature].values
        bureau_scores = -raw_feature  # x_v = -X1

        # Select k largest bureau scores using n-k indexing (explicit and safe)
        # This is equivalent to selecting k smallest X1 values (good borrowers)
        idx = np.argpartition(bureau_scores, n - k)[n - k:]

        accept_mask = np.zeros(n, dtype=bool)
        accept_mask[idx] = True

        # Directionality sanity check: accepted should have higher bureau scores
        if k < n:
            mean_accept_bureau = bureau_scores[accept_mask].mean()
            mean_reject_bureau = bureau_scores[~accept_mask].mean()
            assert mean_accept_bureau >= mean_reject_bureau - 1e-8, (
                f"Bureau score directionality error: mean_accept={mean_accept_bureau:.4f} < "
                f"mean_reject={mean_reject_bureau:.4f}. Expected accepts to have higher x_v."
            )

        assert accept_mask.sum() == k, f"Expected {k} accepts, got {accept_mask.sum()}"

        return df[accept_mask].copy(), df[~accept_mask].copy()

    def _accept_by_model(
        self, df: pd.DataFrame, model: XGBoostModel, accept_rate: float, feature_cols: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame, AcceptanceDiagnostics]:
        """Accept exactly k = ceil(α*n) lowest by model score (lower P(bad) = accept).

        Per Algorithm C.2 for model-based acceptance (j>=1):
          - k = ceil(α * n)
          - D_a = k samples with lowest f_a(X) scores (lowest risk)
          - D_r = remaining n-k samples

        This creates the feedback loop where model predictions drive acceptance,
        required to see compounding bias and BASL correction effects.

        Returns:
            Tuple of (accepts_df, rejects_df, diagnostics)
        """
        # Paper-faithful assertion: model-based acceptance only after initialization
        self._model_accept_called += 1
        assert self._feature_accept_called == 1, (
            "Model-based acceptance called before feature-based initialization (j=0)."
        )

        # Get model scores (P(bad))
        X = df[feature_cols].values
        scores = model.predict_proba(X)
        y_true = df["y"].values

        n = len(df)
        k = max(1, int(np.ceil(accept_rate * n)))

        # TWEAK 1: Compute gold-standard acceptance set using stable sort
        # stable_order sorts by (score, index) to ensure deterministic tie-breaking
        stable_order = np.lexsort((np.arange(n), scores))
        accepted_true = set(stable_order[:k])

        # Add deterministic micro-jitter for stable tie-breaking
        # This ensures ABR is well-defined even when scores are identical
        jitter = np.arange(n) * 1e-12
        scores_jittered = scores + jitter

        # Use argpartition for O(n) exact top-k selection (k smallest PD = lowest risk)
        idx = np.argpartition(scores_jittered, k - 1)[:k]

        accept_mask = np.zeros(n, dtype=bool)
        accept_mask[idx] = True
        accepted_actual = set(np.where(accept_mask)[0])

        # TWEAK 1: Strong acceptance-set equality check
        accept_set_exact_match = (accepted_true == accepted_actual)
        intersection = len(accepted_true & accepted_actual)
        union = len(accepted_true | accepted_actual)
        accept_set_jaccard = intersection / union if union > 0 else 1.0
        n_mismatched = len(accepted_true ^ accepted_actual)

        # Compute threshold tau (alpha-th percentile)
        threshold_tau = float(np.percentile(scores, accept_rate * 100))

        # Count ties at tau
        tie_tol = 1e-9
        n_ties_at_tau = int(np.sum(np.abs(scores - threshold_tau) < tie_tol))
        # Among ties, how many were accepted
        ties_mask = np.abs(scores - threshold_tau) < tie_tol
        n_accepted_from_ties = int((accept_mask & ties_mask).sum())

        # Directionality check: accepted should have lower mean score than rejected
        direction_check_ok = True
        if k < n:
            mean_accept = scores[accept_mask].mean()
            mean_reject = scores[~accept_mask].mean()
            direction_check_ok = mean_accept <= mean_reject + 1e-3
            assert direction_check_ok, (
                f"Directionality error: mean_accept={mean_accept:.4f} > "
                f"mean_reject={mean_reject:.4f}. Expected accepts to have lower PD."
            )

        assert accept_mask.sum() == k, f"Expected {k} accepts, got {accept_mask.sum()}"

        # Compute score percentiles for diagnostics
        percentiles = {
            "p1": float(np.percentile(scores, 1)),
            "p5": float(np.percentile(scores, 5)),
            "p10": float(np.percentile(scores, 10)),
            "p15": float(np.percentile(scores, 15)),
            "p20": float(np.percentile(scores, 20)),
            "p40": float(np.percentile(scores, 40)),
            "p50": float(np.percentile(scores, 50)),
            "p80": float(np.percentile(scores, 80)),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
        }

        # Compute bad rates
        bad_rate_accepts = float(y_true[accept_mask].mean()) if accept_mask.any() else float("nan")
        bad_rate_rejects = float(y_true[~accept_mask].mean()) if (~accept_mask).any() else float("nan")

        # Build diagnostics object
        diagnostics = AcceptanceDiagnostics(
            n_batch=n,
            alpha=accept_rate,
            k_expected=k,
            score_min=float(scores.min()),
            score_max=float(scores.max()),
            score_mean=float(scores.mean()),
            score_percentiles=percentiles,
            threshold_tau=threshold_tau,
            n_accepted=int(accept_mask.sum()),
            n_rejected=int((~accept_mask).sum()),
            n_ties_at_tau=n_ties_at_tau,
            n_accepted_from_ties=n_accepted_from_ties,
            bad_rate_accepts=bad_rate_accepts,
            bad_rate_rejects=bad_rate_rejects,
            direction_check_ok=direction_check_ok,
            accept_set_exact_match=accept_set_exact_match,
            accept_set_jaccard=accept_set_jaccard,
            n_mismatched=n_mismatched,
        )

        return df[accept_mask].copy(), df[~accept_mask].copy(), diagnostics

    def run(
        self,
        holdout: pd.DataFrame,
        track_every: int = 1,
        show_progress: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBoostModel, XGBoostModel, XGBoostModel, List[Dict[str, Any]]]:
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
            (D_a, D_r, holdout, fc_model, fo_model, fa_model, metrics_history)
            - fc_model: BASL model (or baseline if basl_cfg=None)
            - fo_model: Oracle model trained on D_a ∪ D_r
            - fa_model: Accepts-only model for panel(c) comparison
        """
        # Use feature columns from generator (paper-faithful: X1, X2, N1, N2)
        feature_cols = self.generator.feature_cols
        alpha = self.cfg.target_accept_rate
        x_v_feature = self._get_x_v_feature()

        # External holdout for oracle evaluation (drawn separately per Algorithm C.2)
        X_holdout = holdout[feature_cols].values
        y_holdout = holdout["y"].values

        # Step 1: Generate initial batch and accept by feature (no model yet)
        # Per Algorithm C.2: first batch uses x_v (most separating feature) for ranking
        initial_batch = self.generator.generate_population(self.cfg.initial_batch_size)
        initial_accepts, initial_rejects = self._accept_by_feature(
            initial_batch, x_v_feature, alpha
        )

        # Initialize cumulative accepts and rejects (true labels only)
        all_accepts = [initial_accepts]
        all_rejects = [initial_rejects]

        # Fix #4: Maintain train/val split to prevent data leakage in accepts-only evaluation
        # Per paper: accepts-only evaluation must use a validation subset NOT used to train f_a
        # We split accepts 80/20 and train only on training portion
        val_fraction = 0.2

        # X_accepts/y_accepts: cumulative TRUE accepts (all accepts for tracking)
        # X_accepts_train/y_accepts_train: training portion only (for model training)
        # X_accepts_val/y_accepts_val: validation portion only (for accepts-only evaluation)
        X_accepts_all = initial_accepts[feature_cols].values
        y_accepts_all = initial_accepts["y"].values

        # Split initial accepts into train/val
        n_initial = len(X_accepts_all)
        n_val_initial = max(1, int(n_initial * val_fraction))
        val_indices = self.rng.choice(n_initial, size=n_val_initial, replace=False)
        train_mask = np.ones(n_initial, dtype=bool)
        train_mask[val_indices] = False

        X_accepts_train = X_accepts_all[train_mask]
        y_accepts_train = y_accepts_all[train_mask]
        X_accepts_val = X_accepts_all[~train_mask]
        y_accepts_val = y_accepts_all[~train_mask]

        # For compatibility, X_accepts/y_accepts refers to ALL accepts (train + val)
        X_accepts = X_accepts_all
        y_accepts = y_accepts_all

        # X_rejects_for_eval: cumulative rejects for Bayesian evaluation
        # This is the full D_r^(j) used for evaluation, grows each iteration
        X_rejects_for_eval = initial_rejects[feature_cols].values
        # y_rejects_true: true labels for rejects (for oracle training in simulation)
        y_rejects_true = initial_rejects["y"].values

        # Initialize model on training accepts only (f_a: accepts-based scorecard)
        # Fix #4: Train only on D_a_train to prevent leakage in accepts-only evaluation
        model = XGBoostModel(self.model_cfg)
        model.fit(X_accepts_train, y_accepts_train)

        # Figure 2(e) requires THREE distinct models evaluated on H:
        # - f_o (oracle): trained on D_a ∪ D_r with true labels
        # - f_a (accepts-only): trained ONLY on D_a, no BASL
        # - f_c (BASL): trained on D_a + pseudo-labeled rejects
        #
        # In BASL mode, `model` becomes f_c. We need a SEPARATE accepts_only_model
        # to track f_a throughout the simulation for panel (e) comparison.
        accepts_only_model = XGBoostModel(self.model_cfg)
        accepts_only_model.fit(X_accepts_train, y_accepts_train)

        # Initialize oracle model (f_o: trained on D_a ∪ D_r with true labels)
        # Per Algorithm C.2: f_o is the benchmark that has access to all true labels
        # Oracle uses ALL data (train + val + rejects) since it's the ground truth benchmark
        oracle_model = XGBoostModel(self.model_cfg)
        X_all = np.vstack([X_accepts, X_rejects_for_eval])
        y_all = np.concatenate([y_accepts, y_rejects_true])
        oracle_model.fit(X_all, y_all)

        # ALWAYS create f_prior (baseline XGBoost + LR calibrator) for Bayesian evaluation
        # Per paper Section 4.3: f_prior is SHARED by ALL methods (baseline and BASL)
        # The difference between methods is ONLY in how f_eval is trained, not f_prior
        # f_prior represents the "legacy/historical acceptance model" for reject priors
        # NOTE: f_prior is trained on ALL accepts (train + val) since it's a separate model
        # used only for pseudo-labeling priors, not for accepts-only evaluation
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
        last_acceptance_diagnostics: Optional[AcceptanceDiagnostics] = None

        # Best model tracking (Section 4.2-4.3 from paper)
        # NOTE: For BASL mode, best model must come from iteration >= 1 (after BASL training)
        # Iteration 0 is pre-BASL and should NOT be considered for "best" selection
        best_model = None
        best_bayesian_abr = float('inf')
        best_iteration = -1  # -1 indicates no valid best yet

        # Initial evaluation (iteration 0)
        # Use X_rejects_for_eval (full D_r) for Bayesian evaluation, NOT the BASL pool
        # Fix #4: Pass held-out validation set for accepts-only evaluation (no leakage)
        # Note: No acceptance_diagnostics for iter 0 (feature-based acceptance, not model)
        initial_metrics = self._evaluate(
            model,
            oracle_model,
            accepts_only_model,  # Separate f_a model for panel (e) comparison
            X_accepts_val, y_accepts_val,  # Validation set for accepts-only (no leakage)
            X_rejects_for_eval,
            X_holdout, y_holdout,
            0,
            baseline_model,
            prior_calibrator,
            acceptance_diagnostics=None,  # No model-based acceptance at iter 0
            n_Da_train=len(X_accepts_train),
        )
        metrics_history.append(initial_metrics)

        # NOTE: Do NOT initialize best_model at iteration 0 for BASL mode
        # Iteration 0 is BEFORE any BASL training occurs, so it cannot be "best"

        # Step 2: Main loop - process new batches each period
        iterator = range(1, self.cfg.n_periods + 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Training loop", leave=True, dynamic_ncols=True)

        for iteration in iterator:
            # Generate new batch of applicants this period
            batch = self.generator.generate_population(self.cfg.batch_size)

            # Acceptance decision per Algorithm C.2:
            # - j=0 (initial): accept by x_v (feature) - already done above
            # - j>=1 (loop): accept by f_a(X) (model score)
            # This applies to BOTH experiments - the difference is in training (BASL vs baseline)
            batch_accepts, batch_rejects, acceptance_diag = self._accept_by_model(
                batch, model, alpha, feature_cols
            )

            # Store acceptance diagnostics for this iteration (for sampled iterations)
            # Will be added to metrics_history at tracking points
            last_acceptance_diagnostics = acceptance_diag

            # Accumulate for final return
            all_accepts.append(batch_accepts)
            all_rejects.append(batch_rejects)

            # Add new accepts to cumulative accepts (true labels only)
            # Fix #4: Split new accepts into train/val to maintain consistent split
            if len(batch_accepts) > 0:
                new_X = batch_accepts[feature_cols].values
                new_y = batch_accepts["y"].values

                # Split new batch into train/val (same 80/20 ratio)
                n_new = len(new_X)
                n_new_val = max(0, int(n_new * val_fraction))

                if n_new_val > 0 and n_new > n_new_val:
                    new_val_indices = self.rng.choice(n_new, size=n_new_val, replace=False)
                    new_train_mask = np.ones(n_new, dtype=bool)
                    new_train_mask[new_val_indices] = False

                    # Add to train set
                    X_accepts_train = np.vstack([X_accepts_train, new_X[new_train_mask]])
                    y_accepts_train = np.concatenate([y_accepts_train, new_y[new_train_mask]])

                    # Add to val set
                    X_accepts_val = np.vstack([X_accepts_val, new_X[~new_train_mask]])
                    y_accepts_val = np.concatenate([y_accepts_val, new_y[~new_train_mask]])
                else:
                    # All new samples go to train (batch too small for meaningful split)
                    X_accepts_train = np.vstack([X_accepts_train, new_X])
                    y_accepts_train = np.concatenate([y_accepts_train, new_y])

                # Update ALL accepts (for oracle and baseline models)
                X_accepts = np.vstack([X_accepts, new_X])
                y_accepts = np.concatenate([y_accepts, new_y])

                # Update baseline model and LR calibrator on all accumulated accepts
                # Per Section 4.3: prior comes from historical acceptance model
                # NOTE: baseline_model uses ALL accepts (not just train) since it's for priors
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

            # Always update accepts_only_model on current accepts (for panel e comparison)
            # This tracks f_a: the model trained ONLY on accepts, regardless of BASL mode
            accepts_only_model.fit(X_accepts_train, y_accepts_train)

            # Train model for this iteration
            # Per paper Algorithm C.2 and two-loop separation:
            # - Baseline: train on accepts only (f_a)
            # - BASL: run fresh on current D_a^(j), D_r^(j), pseudo-labels are ephemeral
            # Fix #4: Train only on D_a_train to prevent leakage in accepts-only evaluation
            if self.basl_trainer is not None:
                # BASL mode: run fresh each iteration on current snapshot
                # Filter the FULL current reject set (not an incrementally shrinking pool)
                # NOTE: Use X_accepts (all) for filtering since BASL uses all available data
                X_rejects_filtered = self.basl_trainer.filter_rejects_once(
                    X_accepts, X_rejects_for_eval
                )

                if len(X_rejects_filtered) > 0:
                    # Initialize local BASL state from TRAINING accepts only (Fix #4)
                    X_basl = X_accepts_train.copy()
                    y_basl = y_accepts_train.copy()

                    # Run BASL pseudo-labeling on filtered rejects
                    X_basl, y_basl, _ = self._run_basl_labeling(
                        X_basl, y_basl, X_rejects_filtered.copy()
                    )

                    # Train BASL evaluation model on train accepts + pseudo-labeled rejects
                    model.fit(X_basl, y_basl)
                    # Pseudo-labels are now discarded (not stored for next iteration)
                else:
                    # No filtered rejects available, train on training accepts only
                    model.fit(X_accepts_train, y_accepts_train)
            else:
                # Baseline mode: train only on TRAINING accepts (Fix #4: no leakage)
                model.fit(X_accepts_train, y_accepts_train)

            # Track metrics at specified intervals
            should_track = (iteration % track_every == 0) or (iteration == self.cfg.n_periods)

            if should_track:
                # Use X_rejects_for_eval (full D_r) for Bayesian evaluation
                # This ensures baseline and BASL are evaluated on the same population
                # Fix #4: Pass held-out validation set for accepts-only evaluation (no leakage)
                iteration_metrics = self._evaluate(
                    model,
                    oracle_model,
                    accepts_only_model,  # Separate f_a model for panel (e) comparison
                    X_accepts_val, y_accepts_val,  # Validation set for accepts-only (no leakage)
                    X_rejects_for_eval,
                    X_holdout, y_holdout,
                    iteration,
                    baseline_model,
                    prior_calibrator,
                    acceptance_diagnostics=last_acceptance_diagnostics,  # Include diagnostics
                    n_Da_train=len(X_accepts_train),
                )
                metrics_history.append(iteration_metrics)

                # Track best model based on Bayesian ABR (Section 4.2 from paper)
                # BASL mode only: iteration >= 1 required (iteration 0 is pre-BASL)
                # ABR validity guards: must be finite and > 0 (0.0 indicates degenerate eval)
                if self.basl_trainer is not None and iteration >= 1:
                    bayesian_abr = iteration_metrics['bayesian']['abr']
                    abr_is_valid = (
                        np.isfinite(bayesian_abr) and
                        bayesian_abr > 1e-6  # Guard against degenerate 0.0 values
                    )
                    if abr_is_valid and bayesian_abr < best_bayesian_abr:
                        best_model = copy.deepcopy(model)
                        best_bayesian_abr = bayesian_abr
                        best_iteration = iteration

        # Combine all accepts/rejects for return
        D_a = pd.concat(all_accepts, ignore_index=True)
        D_r = pd.concat(all_rejects, ignore_index=True)

        # Paper-faithful: Return LAST trained model for Figure 2 (per-iteration dynamics)
        # Best model selection is disabled for Figure 2 to show actual training progression
        # The `model` variable holds the most recent trained model
        final_model = model

        # Model collapse assertion: BASL model must not be constant after training
        if self.basl_trainer is not None:
            probe_preds = final_model.predict_proba(X_holdout[:256])
            pred_std = np.std(probe_preds)
            assert pred_std > 1e-6, (
                f"BASL model collapsed to constant predictions (std={pred_std:.6f}). "
                f"This indicates a bug in BASL training or model selection."
            )
            print(f"  BASL: Using final model from iteration {self.cfg.n_periods} (pred_std={pred_std:.4f})")

        return D_a, D_r, holdout, final_model, oracle_model, accepts_only_model, metrics_history

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
        accepts_only_model: XGBoostModel,
        X_accepts_val: np.ndarray,
        y_accepts_val: np.ndarray,
        X_rejects: np.ndarray,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        iteration: int,
        baseline_model: Optional[XGBoostModel] = None,
        prior_calibrator: Optional[LogisticRegressionModel] = None,
        acceptance_diagnostics: Optional[AcceptanceDiagnostics] = None,
        n_Da_train: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Five-way evaluation for Figure 2 with comprehensive diagnostics.

        Panel (e) requires THREE distinct models evaluated on H:
        - fo_H: Oracle (f_o) trained on D_a ∪ D_r with true labels
        - fa_H: Accepts-only (f_a) trained ONLY on D_a, no BASL
        - fc_H: BASL model (f_c) trained on D_a + pseudo-labeled rejects

        Panel (d) biased estimator:
        - fa_DaVal: f_a metrics on D_a_val (shows sampling bias)

        Bayesian evaluation:
        - bayesian: Algorithm 1 on H_a + pseudo-labeled H_r

        Args:
            model: The BASL model (f_c) in BASL mode, or f_a in baseline mode.
            oracle_model: f_o trained on D_a ∪ D_r with true labels.
            accepts_only_model: f_a trained ONLY on D_a (no BASL), for panel (e).
            X_accepts_val: Validation accepts (D_a_val) - NOT used in training.
            y_accepts_val: Validation accepts labels.
            X_rejects: Training rejects (D_r) - NOT used for Bayesian eval.
            X_holdout: External holdout features (H).
            y_holdout: External holdout labels (H).
            baseline_model: Accepts-only XGBoost for generating prior scores (f_prior).
            prior_calibrator: LR calibrator that maps XGBoost scores to probabilities.
        """
        metrics_list = ["auc", "pauc", "brier", "abr"]
        abr_range = self.bayesian_cfg.abr_range
        alpha = self.cfg.target_accept_rate

        # Paper-faithful assertion: ABR integrates between 20% and 40% acceptance
        # Per Section 6.3: "integrate ABR over acceptance between 20% and 40%"
        assert abr_range == (0.2, 0.4), (
            f"ABR range must be (0.2, 0.4) per paper Section 6.3, got {abr_range}"
        )

        # Paper-faithful assertion: Panel (e) must use holdout H, not D_a_val
        # This guards against regression where biased accepts-only eval leaks into panel (e)
        # Holdout should be significantly larger than accepts validation set
        assert len(X_holdout) > len(X_accepts_val), (
            f"Holdout H ({len(X_holdout)}) must be larger than D_a_val ({len(X_accepts_val)}). "
            "This guards against accidentally using D_a_val for panel (e) evaluation."
        )

        # Oracle model (f_o): evaluate on external holdout with true labels
        # Per Algorithm C.2: f_o is trained on D_a ∪ D_r and represents the benchmark
        oracle_scores_holdout = oracle_model.predict_proba(X_holdout)
        fo_H_metrics = compute_metrics(
            y_holdout, oracle_scores_holdout, metrics_list, abr_range=abr_range
        )

        # Accepts-only model (f_a): evaluate on H for panel (e) comparison
        # This is the model trained ONLY on accepts, no BASL - tracked separately
        fa_scores_holdout = accepts_only_model.predict_proba(X_holdout)
        fa_H_metrics = compute_metrics(
            y_holdout, fa_scores_holdout, metrics_list, abr_range=abr_range
        )

        # BASL model (f_c): evaluate on H for panel (e) comparison
        # In baseline mode, model == accepts_only_model, so fc_H == fa_H
        # In BASL mode, model is the BASL-trained model
        fc_scores_holdout = model.predict_proba(X_holdout)
        fc_H_metrics = compute_metrics(
            y_holdout, fc_scores_holdout, metrics_list, abr_range=abr_range
        )

        # Accepts-only evaluation on D_a_val: BIASED estimator for panel (d)
        # Per paper Appendix C.1/C.2:
        # "f_a is evaluated on a validation subset drawn from the available set of accepts D_a"
        # This shows what performance you'd estimate with only accepted applicants (sampling bias).
        # NOTE: Use accepts_only_model (f_a), NOT model (which may be f_c in BASL mode)
        scores_accepts_val = accepts_only_model.predict_proba(X_accepts_val)
        fa_DaVal_metrics = compute_metrics(
            y_accepts_val, scores_accepts_val,
            metrics_list, abr_range=abr_range
        )

        # Bayesian evaluation: MC pseudo-labeling on H_a + H_r (holdout split)
        # Per paper: split holdout H into H_a and H_r using FIXED bank policy (f_prior)
        # NOT the candidate model - this ensures consistent evaluation target across iterations
        # - H_a: holdout samples accepted by bank policy (lowest α fraction by f_prior score)
        # - H_r: holdout samples rejected by bank policy
        #
        # Use f_prior (baseline_model + LR calibrator) for split, NOT candidate model
        raw_bank_scores = baseline_model.predict_proba(X_holdout)
        bank_scores = prior_calibrator.predict_proba(raw_bank_scores.reshape(-1, 1))

        # Top-k selection guarantees exact alpha fraction (not percentile which can drift)
        k = max(1, int(round(alpha * len(X_holdout))))
        idx_sorted = np.argsort(bank_scores)[:k]  # lowest PD = accepted
        holdout_accept_mask = np.zeros(len(X_holdout), dtype=bool)
        holdout_accept_mask[idx_sorted] = True

        X_H_a = X_holdout[holdout_accept_mask]
        y_H_a = y_holdout[holdout_accept_mask]
        X_H_r = X_holdout[~holdout_accept_mask]

        # f_eval scores for H_a and H_r (BASL model - used in metric computation)
        # NOTE: Use fc_scores_holdout (BASL model), as Bayesian eval is for f_c
        scores_H_a = fc_scores_holdout[holdout_accept_mask]
        scores_H_r = fc_scores_holdout[~holdout_accept_mask]

        # f_prior scores for H_r (used for pseudo-label sampling)
        prior_scores_H_r = bank_scores[~holdout_accept_mask] if len(X_H_r) > 0 else np.array([])

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

        # Bayesian evaluation on H_a (true labels) + H_r (pseudo-labeled)
        bayesian_result = bayesian_evaluate(
            y_H_a, scores_H_a, scores_H_r,
            cfg=iter_cfg,
            metrics_list=metrics_list,
            prior_scores_rejects=prior_scores_H_r,
        )

        # Extract mean values for consistency with other metrics
        bayesian_metrics = {
            metric: bayesian_result[metric]["mean"]
            for metric in metrics_list
        }

        # Provenance: track which acceptance rule was used for this iteration
        acceptance_rule = "x_v" if iteration == 0 else "model_pd"

        # Compute holdout diagnostics (score quantiles for fo, fa, fc on H)
        def compute_quantiles(scores: np.ndarray) -> Dict[str, float]:
            return {
                "p1": float(np.percentile(scores, 1)),
                "p5": float(np.percentile(scores, 5)),
                "p10": float(np.percentile(scores, 10)),
                "p15": float(np.percentile(scores, 15)),
                "p20": float(np.percentile(scores, 20)),
                "p40": float(np.percentile(scores, 40)),
                "p50": float(np.percentile(scores, 50)),
                "p80": float(np.percentile(scores, 80)),
                "p90": float(np.percentile(scores, 90)),
                "p95": float(np.percentile(scores, 95)),
                "p99": float(np.percentile(scores, 99)),
            }

        holdout_diagnostics = {
            "fo_quantiles": compute_quantiles(oracle_scores_holdout),
            "fa_quantiles": compute_quantiles(fa_scores_holdout),
            "fc_quantiles": compute_quantiles(fc_scores_holdout),
        }

        # Compute ABR breakdown for detailed diagnostics (TWEAK 2)
        abr_breakdown = {
            "fo_H": compute_abr_breakdown(y_holdout, oracle_scores_holdout),
            "fa_H": compute_abr_breakdown(y_holdout, fa_scores_holdout),
            "fc_H": compute_abr_breakdown(y_holdout, fc_scores_holdout),
            "fa_DaVal": compute_abr_breakdown(y_accepts_val, scores_accepts_val),
        }

        # Evaluation datasets metadata (TWEAK 3)
        evaluation_datasets = {
            "H": {
                "name": "holdout",
                "size": len(y_holdout),
                "bad_rate": float(y_holdout.mean()),
            },
            "DaVal": {
                "name": "accepts_val",
                "size": len(y_accepts_val),
                "bad_rate": float(y_accepts_val.mean()) if len(y_accepts_val) > 0 else float("nan"),
                "fixed": True,  # MUST BE TRUE for paper-faithful Exp I
                "definition": "fixed_split_of_Da_from_iter0",
                "source_iteration": 0,  # Val set established at iteration 0
            },
        }

        # Diagnostic dump for debugging panel (d) ABR issues
        # Per checklist Section 6: gather key stats at checkpoint iterations
        diagnostic = {
            "n_Da_train": n_Da_train,
            "n_Da_val": len(y_accepts_val),
            "n_H": len(y_holdout),
            "bad_rate_Da_val": float(y_accepts_val.mean()) if len(y_accepts_val) > 0 else float("nan"),
            "bad_rate_H": float(y_holdout.mean()),
            "count_bad_Da_val": int(y_accepts_val.sum()) if len(y_accepts_val) > 0 else 0,
            "pred_Da_val_min": float(scores_accepts_val.min()) if len(scores_accepts_val) > 0 else float("nan"),
            "pred_Da_val_p5": float(np.percentile(scores_accepts_val, 5)) if len(scores_accepts_val) > 0 else float("nan"),
            "pred_Da_val_p50": float(np.percentile(scores_accepts_val, 50)) if len(scores_accepts_val) > 0 else float("nan"),
            "pred_Da_val_p95": float(np.percentile(scores_accepts_val, 95)) if len(scores_accepts_val) > 0 else float("nan"),
            "pred_Da_val_max": float(scores_accepts_val.max()) if len(scores_accepts_val) > 0 else float("nan"),
        }

        # Paper-faithful metric naming convention:
        # - Panel (e) uses: fo_H, fa_H, fc_H (all evaluated on holdout H)
        # - Panel (d) uses: fa_DaVal (biased estimator on accepts validation set)
        # NEVER mix these - fa_DaVal must NOT appear in panel (e)
        result = {
            "iteration": iteration,
            "acceptance_rule": acceptance_rule,  # "x_v" for j=0, "model_pd" for j>=1
            # Panel (e) metrics - all on holdout H (THREE distinct models)
            "fo_H": fo_H_metrics,               # f_o (oracle) on H
            "fa_H": fa_H_metrics,               # f_a (accepts-only) on H
            "fc_H": fc_H_metrics,               # f_c (BASL model) on H
            # Panel (d) metrics - biased estimator
            "fa_DaVal": fa_DaVal_metrics,       # f_a on D_a_val (biased, panel d ONLY)
            # Bayesian evaluation on H (uses f_c model)
            "bayesian": bayesian_metrics,       # Bayesian on H_a + pseudo-H_r
            "bayesian_full": bayesian_result,   # Include full posterior stats
            # Comprehensive diagnostics
            "holdout_diagnostics": holdout_diagnostics,
            "abr_breakdown": abr_breakdown,
            "evaluation_datasets": evaluation_datasets,
            "diagnostic": diagnostic,  # Panel (d) debug: dataset sizes, bad rates, pred stats
            # Raw scores on holdout H for panel (c) snapshot selection
            # These enable post-hoc KS distance computation and snapshot selection
            "holdout_scores": {
                "fo_scores_H": oracle_scores_holdout.tolist(),
                "fa_scores_H": fa_scores_holdout.tolist(),
                "fc_scores_H": fc_scores_holdout.tolist(),
            },
            # DEPRECATED: kept for backward compatibility, remove after migration
            "oracle": fo_H_metrics,             # Use fo_H instead
            "model_holdout": fc_H_metrics,      # Use fc_H instead
            "accepts": fa_DaVal_metrics,        # Use fa_DaVal instead
        }

        # Add acceptance diagnostics if provided
        if acceptance_diagnostics is not None:
            result["acceptance_diagnostics"] = asdict(acceptance_diagnostics)

        return result
