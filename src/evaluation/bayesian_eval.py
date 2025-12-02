"""
Bayesian evaluation framework for credit scoring under sampling bias.

Uses Monte Carlo sampling to estimate posterior distributions of metrics
by pseudo-labeling rejects based on score-band specific bad rates.

Algorithm based on paper Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from src.evaluation.metrics import compute_metrics


@dataclass
class BayesianEvalConfig:
    """Configuration for Bayesian evaluation."""

    n_samples: int = 5000  # Number of Monte Carlo samples
    n_score_bands: int = 10  # K: number of score bands for stratified sampling
    prior_alpha: float = 1.0  # Beta prior alpha (uninformative)
    prior_beta: float = 1.0  # Beta prior beta (uninformative)
    metrics: List[str] = field(default_factory=lambda: ["auc", "pauc", "brier", "abr"])
    accept_rate: float = 0.15  # For ABR calculation
    random_seed: int = 42


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
    # Compute band boundaries as percentiles
    percentiles = np.linspace(0, 100, n_bands + 1)
    boundaries = np.percentile(scores, percentiles)

    # Assign each score to a band
    bands = np.digitize(scores, boundaries[1:-1])  # 0 to n_bands-1

    return bands


def bayesian_evaluate(
    y_accepts: np.ndarray,
    scores_accepts: np.ndarray,
    scores_rejects: np.ndarray,
    cfg: BayesianEvalConfig,
) -> Dict[str, Any]:
    """Run Bayesian evaluation using Monte Carlo sampling.

    For each MC sample:
    1. Sample band-specific bad rates from Beta posterior
    2. Pseudo-label rejects by sampling from Bernoulli(p_k)
    3. Combine accepts + pseudo-labeled rejects
    4. Compute metrics on combined data

    Args:
        y_accepts: True labels for accepts (0=good, 1=bad).
        scores_accepts: Predicted scores for accepts.
        scores_rejects: Predicted scores for rejects (no labels).
        cfg: Bayesian evaluation configuration.

    Returns:
        Dictionary with posterior statistics for each metric:
        {
            "metrics": {
                "auc": {"mean": ..., "median": ..., "q2.5": ..., "q97.5": ...},
                ...
            },
            "n_samples": ...,
            "n_accepts": ...,
            "n_rejects": ...,
        }
    """
    rng = np.random.default_rng(cfg.random_seed)

    n_accepts = len(y_accepts)
    n_rejects = len(scores_rejects)

    # Combine scores for band assignment
    all_scores = np.concatenate([scores_accepts, scores_rejects])
    all_bands = _assign_score_bands(all_scores, cfg.n_score_bands)

    bands_accepts = all_bands[:n_accepts]
    bands_rejects = all_bands[n_accepts:]

    # Compute observed bad counts per band (from accepts only)
    band_stats = []
    for k in range(cfg.n_score_bands):
        mask_a = bands_accepts == k
        n_a_k = mask_a.sum()
        d_a_k = y_accepts[mask_a].sum() if n_a_k > 0 else 0

        mask_r = bands_rejects == k
        n_r_k = mask_r.sum()

        band_stats.append({
            "n_accepts": int(n_a_k),
            "n_bads_accepts": int(d_a_k),
            "n_rejects": int(n_r_k),
        })

    # Monte Carlo sampling
    metric_samples = {m: [] for m in cfg.metrics}

    for _ in range(cfg.n_samples):
        # Sample pseudo-labels for rejects
        y_rejects_sampled = np.zeros(n_rejects, dtype=int)

        for k in range(cfg.n_score_bands):
            stats = band_stats[k]

            # Sample bad rate from Beta posterior
            # Posterior: Beta(alpha + d_a_k, beta + n_a_k - d_a_k)
            post_alpha = cfg.prior_alpha + stats["n_bads_accepts"]
            post_beta = cfg.prior_beta + stats["n_accepts"] - stats["n_bads_accepts"]
            p_k = rng.beta(post_alpha, post_beta)

            # Pseudo-label rejects in this band
            mask_r = bands_rejects == k
            n_r_k = mask_r.sum()
            if n_r_k > 0:
                y_rejects_sampled[mask_r] = rng.binomial(1, p_k, n_r_k)

        # Combine accepts and pseudo-labeled rejects
        y_combined = np.concatenate([y_accepts, y_rejects_sampled])
        scores_combined = np.concatenate([scores_accepts, scores_rejects])

        # Compute metrics on combined data
        sample_metrics = compute_metrics(
            y_combined,
            scores_combined,
            cfg.metrics,
            accept_rate=cfg.accept_rate,
        )

        for m in cfg.metrics:
            metric_samples[m].append(sample_metrics[m])

    # Aggregate results
    results = {
        "metrics": {},
        "n_samples": cfg.n_samples,
        "n_accepts": n_accepts,
        "n_rejects": n_rejects,
        "band_stats": band_stats,
    }

    for m in cfg.metrics:
        samples = np.array(metric_samples[m])
        results["metrics"][m] = {
            "mean": float(np.mean(samples)),
            "median": float(np.median(samples)),
            "std": float(np.std(samples)),
            "q2.5": float(np.percentile(samples, 2.5)),
            "q97.5": float(np.percentile(samples, 97.5)),
        }

    return results
