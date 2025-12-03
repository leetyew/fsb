"""
Bayesian evaluation via stochastic pseudo-labeling.

Implements the paper's Algorithm 1: Bayesian Evaluation Framework.
Uses Monte Carlo sampling with convergence checking to estimate
performance metrics on the full population (accepts + rejects).

Two modes available:
1. Direct mode (paper-faithful): Each reject's pseudo-label sampled using
   model's predicted probability directly: y^r ~ Binomial(1, P(y^r|X^r))
2. Banded mode (variance reduction): Stratify by score bands and use
   Beta posteriors estimated from accepts in each band.

Key parameters from Table E.9:
- j_min = 100 (minimum MC samples before convergence check)
- j_max = 10^6 (maximum MC samples)
- ε = 10^-6 (convergence threshold)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from src.evaluation.metrics import compute_metrics

if TYPE_CHECKING:
    from src.config import BayesianEvalConfig


def _assign_score_bands(
    scores: np.ndarray,
    n_bands: int,
) -> np.ndarray:
    """Assign samples to score bands based on quantiles.

    Args:
        scores: Predicted scores (probability of bad).
        n_bands: Number of bands (K).

    Returns:
        Array of band assignments (0 to n_bands-1).
    """
    percentiles = np.linspace(0, 100, n_bands + 1)
    boundaries = np.percentile(scores, percentiles)
    bands = np.digitize(scores, boundaries[1:-1])  # 0 to n_bands-1
    return bands


def _pseudo_label_rejects_direct(
    scores_rejects: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Paper-faithful pseudo-labeling: use model predictions directly.

    Per Algorithm 1: y^r ~ Binomial(1, P(y^r | X^r))
    Each reject gets a coin flip using its own predicted probability.
    """
    return rng.binomial(1, scores_rejects).astype(int)


def _pseudo_label_rejects_banded(
    y_accepts: np.ndarray,
    bands_accepts: np.ndarray,
    bands_rejects: np.ndarray,
    n_bands: int,
    prior_alpha: float,
    prior_beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Banded pseudo-labeling with Beta posteriors (variance reduction).

    For each score band:
    1. Compute bad rate from accepts in that band
    2. Sample bad rate from Beta posterior
    3. Coin flip each reject using sampled bad rate
    """
    n_rejects = len(bands_rejects)
    y_rejects_pseudo = np.zeros(n_rejects, dtype=int)

    for k in range(n_bands):
        # Compute bad rate in this band from accepts
        mask_a = bands_accepts == k
        n_a_k = mask_a.sum()
        d_a_k = y_accepts[mask_a].sum() if n_a_k > 0 else 0

        # Sample bad rate from Beta posterior
        post_alpha = prior_alpha + d_a_k
        post_beta = prior_beta + n_a_k - d_a_k
        p_k = rng.beta(post_alpha, post_beta)

        # Coin flip for rejects in this band
        mask_r = bands_rejects == k
        n_r_k = mask_r.sum()
        if n_r_k > 0:
            y_rejects_pseudo[mask_r] = rng.binomial(1, p_k, n_r_k)

    return y_rejects_pseudo


def bayesian_evaluate(
    y_accepts: np.ndarray,
    scores_accepts: np.ndarray,
    scores_rejects: np.ndarray,
    cfg: BayesianEvalConfig,
    metrics_list: list[str] | None = None,
) -> dict[str, Any]:
    """Bayesian evaluation with MC convergence (Algorithm 1).

    Runs Monte Carlo iterations to estimate performance metrics on the full
    population. Each iteration pseudo-labels rejects using either:
    - Direct mode (cfg.use_banding=False, default): y^r ~ Binomial(1, P(y^r|X^r))
    - Banded mode (cfg.use_banding=True): stratify by score bands and use Beta posteriors

    Convergence: Stops when running mean changes by less than epsilon,
    or when j_max iterations are reached.

    Args:
        y_accepts: True labels for accepts (0=good, 1=bad).
        scores_accepts: Predicted scores for accepts.
        scores_rejects: Predicted scores for rejects.
        cfg: Bayesian evaluation configuration (includes use_banding, abr_range, etc.).
        metrics_list: Metrics to compute. Default: ["auc", "pauc", "brier", "abr"].

    Returns:
        Dictionary with posterior statistics for each metric:
        {
            "auc": {"mean": ..., "std": ..., "q2.5": ..., "q97.5": ...},
            "pauc": {...},
            ...
            "n_samples": <number of MC samples used>,
            "converged": <whether convergence was achieved>,
        }
    """
    if metrics_list is None:
        metrics_list = ["auc", "pauc", "brier", "abr"]

    rng = np.random.default_rng(cfg.random_seed)
    n_accepts = len(y_accepts)
    n_rejects = len(scores_rejects)

    # Handle edge case: no rejects
    if n_rejects == 0:
        base_metrics = compute_metrics(y_accepts, scores_accepts, metrics_list, cfg.abr_range)
        return {
            metric: {"mean": val, "std": 0.0, "q2.5": val, "q97.5": val}
            for metric, val in base_metrics.items()
        } | {"n_samples": 1, "converged": True}

    # Combine scores for evaluation
    all_scores = np.concatenate([scores_accepts, scores_rejects])

    # Precompute band assignments only if using banded mode
    if cfg.use_banding:
        all_bands = _assign_score_bands(all_scores, cfg.n_bands)
        bands_accepts = all_bands[:n_accepts]
        bands_rejects = all_bands[n_accepts:]

    # Storage for MC samples
    metric_samples: dict[str, list[float]] = {m: [] for m in metrics_list}
    running_means: dict[str, float] = {m: 0.0 for m in metrics_list}

    converged = False
    n_samples = 0

    for j in range(1, cfg.j_max + 1):
        # Pseudo-label rejects for this MC iteration
        if cfg.use_banding:
            y_rejects_pseudo = _pseudo_label_rejects_banded(
                y_accepts, bands_accepts, bands_rejects,
                cfg.n_bands, cfg.prior_alpha, cfg.prior_beta, rng
            )
        else:
            # Paper-faithful: use model predictions directly
            y_rejects_pseudo = _pseudo_label_rejects_direct(scores_rejects, rng)

        # Combine accepts and pseudo-labeled rejects
        y_combined = np.concatenate([y_accepts, y_rejects_pseudo])
        scores_combined = all_scores

        # Compute metrics for this MC sample
        sample_metrics = compute_metrics(y_combined, scores_combined, metrics_list, cfg.abr_range)

        # Store samples and update running means
        prev_means = running_means.copy()
        for metric in metrics_list:
            metric_samples[metric].append(sample_metrics[metric])
            # Incremental mean update
            running_means[metric] = running_means[metric] + (
                sample_metrics[metric] - running_means[metric]
            ) / j

        n_samples = j

        # Check convergence after j_min iterations
        if j >= cfg.j_min:
            max_change = max(
                abs(running_means[m] - prev_means[m]) for m in metrics_list
            )
            if max_change < cfg.epsilon:
                converged = True
                break

    # Compute posterior statistics
    result: dict[str, Any] = {}
    for metric in metrics_list:
        samples = np.array(metric_samples[metric])
        result[metric] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "q2.5": float(np.percentile(samples, 2.5)),
            "q97.5": float(np.percentile(samples, 97.5)),
        }

    result["n_samples"] = n_samples
    result["converged"] = converged

    return result


def pseudo_label_rejects_stochastic(
    y_accepts: np.ndarray,
    scores_accepts: np.ndarray,
    scores_rejects: np.ndarray,
    n_bands: int = 10,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    rng: np.random.Generator | None = None,
    use_banding: bool = False,
) -> np.ndarray:
    """Single stochastic pseudo-labeling of rejects.

    For proper Bayesian evaluation, use bayesian_evaluate() with MC sampling.

    Args:
        y_accepts: True labels for accepts (0=good, 1=bad).
        scores_accepts: Predicted scores for accepts.
        scores_rejects: Predicted scores for rejects (no labels).
        n_bands: Number of score bands for stratification (only used if use_banding=True).
        prior_alpha: Beta prior alpha (only used if use_banding=True).
        prior_beta: Beta prior beta (only used if use_banding=True).
        rng: Random number generator. If None, creates one.
        use_banding: If True, use banded mode. If False (default), use direct mode.

    Returns:
        Pseudo-labels for rejects (shape: n_rejects).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_rejects = len(scores_rejects)

    if n_rejects == 0:
        return np.array([], dtype=int)

    if not use_banding:
        # Paper-faithful: use model predictions directly
        return _pseudo_label_rejects_direct(scores_rejects, rng)

    # Banded mode: use Beta posteriors per band
    n_accepts = len(y_accepts)
    all_scores = np.concatenate([scores_accepts, scores_rejects])
    all_bands = _assign_score_bands(all_scores, n_bands)

    bands_accepts = all_bands[:n_accepts]
    bands_rejects = all_bands[n_accepts:]

    return _pseudo_label_rejects_banded(
        y_accepts, bands_accepts, bands_rejects,
        n_bands, prior_alpha, prior_beta, rng
    )
