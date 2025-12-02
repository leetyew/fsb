"""
Acceptance loop simulation for creating selection-biased datasets.

Simulates lender behavior over multiple periods where applicants are
accepted/rejected based on model scores, creating D_a (accepts with labels),
D_r (rejects without labels), and H (unbiased holdout).

Key paper reference: Appendix C.1, page A8 describes x_v-based initial
acceptance before any model exists.

Supports two modes (for paper replication):
- Baseline mode: Retrain on D_a only each iteration (default)
- BASL mode: Apply BASL each iteration, retrain on augmented data (page A9)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.basl.trainer import BASLTrainer
from src.config import AcceptanceLoopConfig, BASLConfig, XGBoostConfig
from src.io.synthetic_generator import SyntheticGenerator
from src.models.xgboost_model import XGBoostModel

if TYPE_CHECKING:
    from src.evaluation.bayesian_eval import BayesianEvalConfig


class AcceptanceLoop:
    """Simulates lender acceptance decisions over multiple periods.

    Creates selection bias by accepting only top α percentile of applicants
    by model score (PD). Labels are only observed for accepted applicants.

    The initial acceptance (before any model) uses x_v - the feature with
    the largest mean difference between good and bad applicants - as a
    rule-based ranking proxy.

    Modes:
    - Baseline (basl_cfg=None): Retrain on D_a only each iteration
    - BASL (basl_cfg provided): Apply BASL to augment D_a with pseudo-labeled
      rejects, then retrain on augmented data (per paper page A9)
    """

    def __init__(
        self,
        generator: SyntheticGenerator,
        model_cfg: XGBoostConfig,
        cfg: AcceptanceLoopConfig,
        basl_cfg: Optional[BASLConfig] = None,
    ) -> None:
        self.generator = generator
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.basl_cfg = basl_cfg
        self.rng = np.random.default_rng(cfg.random_seed)

        # Initialize BASL trainer if config provided
        self.basl_trainer = BASLTrainer(basl_cfg) if basl_cfg else None

    def _accept_by_feature(
        self, df: pd.DataFrame, feature: str, accept_rate: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Accept top α percentile by feature value (lower = lower risk).

        Args:
            df: Applicant DataFrame with features and 'y'.
            feature: Feature name to rank by.
            accept_rate: Fraction to accept (α).

        Returns:
            (accepts_df, rejects_df) tuple.
        """
        threshold = np.percentile(df[feature], accept_rate * 100)
        accept_mask = df[feature] <= threshold
        return df[accept_mask].copy(), df[~accept_mask].copy()

    def _accept_by_score(
        self, df: pd.DataFrame, model: XGBoostModel, accept_rate: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Accept top α percentile by model score (lower PD = lower risk).

        Args:
            df: Applicant DataFrame with features and 'y'.
            model: Trained model to score applicants.
            accept_rate: Fraction to accept (α).

        Returns:
            (accepts_df, rejects_df) tuple.
        """
        feature_cols = [c for c in df.columns if c != "y"]
        X = df[feature_cols].values
        scores = model.predict_proba(X)

        threshold = np.percentile(scores, accept_rate * 100)
        accept_mask = scores <= threshold
        return df[accept_mask].copy(), df[~accept_mask].copy()

    def run(
        self,
        return_model: bool = False,
        holdout: Optional[pd.DataFrame] = None,
        eval_cfg: Optional["BayesianEvalConfig"] = None,
        track_every: int = 10,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBoostModel
    ] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBoostModel, list[dict]]:
        """Run the acceptance loop simulation with optional metric tracking.

        Args:
            return_model: If True, return the final trained model.
            holdout: Pre-generated holdout for tracking. If None, generates new one.
            eval_cfg: Bayesian eval config. If provided with holdout, enables tracking.
            track_every: Compute metrics every N iterations (default 10).

        Returns:
            Without tracking: (D_a, D_r, H) or (D_a, D_r, H, model)
            With tracking: (D_a, D_r, H, model, history) where history is:
                [{'iteration': int, 'oracle': {...}, 'accepts': {...}, 'bayesian': {...}}]
        """
        feature_cols = [f"x{i}" for i in range(self.generator.cfg.n_features)]
        alpha = self.cfg.target_accept_rate

        # Determine if tracking is enabled
        tracking_enabled = holdout is not None and eval_cfg is not None

        if tracking_enabled:
            from src.evaluation.bayesian_eval import bayesian_evaluate
            from src.evaluation.metrics import compute_metrics

            metrics = eval_cfg.metrics
            X_H = holdout[feature_cols].values
            y_H = holdout["y"].values
            H = holdout
        else:
            H = None  # Will generate later

        # Step 1: Generate initial batch
        initial_batch = self.generator.generate_population(self.cfg.initial_batch_size)

        # Step 2: Initial acceptance using x_v (no model yet)
        D_a, initial_rejects = self._accept_by_feature(
            initial_batch, self.cfg.x_v_feature, alpha
        )
        D_r = initial_rejects[feature_cols].copy()  # No labels for rejects

        # Step 3: Train initial model on D_a
        model = XGBoostModel(self.model_cfg)
        X_a = D_a[feature_cols].values
        y_a = D_a["y"].values
        model.fit(X_a, y_a)

        # Initialize tracking history
        history: list[dict] = []

        def compute_all_metrics(iteration: int) -> dict:
            """Compute oracle, accepts-based, and Bayesian metrics."""
            X_a_curr = D_a[feature_cols].values
            y_a_curr = D_a["y"].values
            X_r_curr = D_r.values

            scores_H = model.predict_proba(X_H)
            scores_a = model.predict_proba(X_a_curr)
            scores_r = model.predict_proba(X_r_curr) if len(X_r_curr) > 0 else np.array([])

            # Oracle: evaluate on holdout
            oracle_metrics = compute_metrics(y_H, scores_H, metrics, eval_cfg.accept_rate)

            # Accepts-based: evaluate on D_a only
            accepts_metrics = compute_metrics(y_a_curr, scores_a, metrics, eval_cfg.accept_rate)

            # Bayesian: posterior sampling on D_a + D_r
            if len(scores_r) > 0:
                bayes_result = bayesian_evaluate(y_a_curr, scores_a, scores_r, eval_cfg)
                bayesian_metrics = {m: bayes_result["metrics"][m]["mean"] for m in metrics}
            else:
                bayesian_metrics = accepts_metrics.copy()

            return {
                "iteration": iteration,
                "oracle": oracle_metrics,
                "accepts": accepts_metrics,
                "bayesian": bayesian_metrics,
            }

        # Record initial state if tracking
        if tracking_enabled:
            history.append(compute_all_metrics(0))

        # Step 4: Main loop - generate, score, accept, retrain
        mode_desc = "Acceptance loop (BASL)" if self.basl_trainer else "Acceptance loop"
        if tracking_enabled:
            mode_desc = f"Tracking {mode_desc}"

        for i in tqdm(range(1, self.cfg.n_periods + 1), desc=mode_desc):
            # Generate new batch
            batch = self.generator.generate_population(self.cfg.batch_size)

            # Accept by model score
            accepts, rejects = self._accept_by_score(batch, model, alpha)

            # Accumulate accepts (with labels) and rejects (features only)
            D_a = pd.concat([D_a, accepts], ignore_index=True)
            D_r = pd.concat([D_r, rejects[feature_cols]], ignore_index=True)

            # Retrain model
            X_a = D_a[feature_cols].values
            y_a = D_a["y"].values

            if self.basl_trainer:
                # BASL mode: augment D_a with pseudo-labeled rejects, then retrain
                X_r = D_r.values
                X_train, y_train = self.basl_trainer.run(X_a, y_a, X_r)
                model.fit(X_train, y_train)
            else:
                # Baseline mode: retrain on D_a only
                model.fit(X_a, y_a)

            # Track metrics at intervals
            if tracking_enabled and (i % track_every == 0 or i == self.cfg.n_periods):
                history.append(compute_all_metrics(i))

        # Step 5: Generate holdout if not provided
        if H is None:
            H = self.generator.generate_holdout()

        # Return based on options
        if tracking_enabled:
            return D_a, D_r, H, model, history
        elif return_model:
            return D_a, D_r, H, model
        return D_a, D_r, H
