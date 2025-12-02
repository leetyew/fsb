"""
Profit and business value metrics for credit scoring.

Computes expected profit per account given:
- PD estimates
- LGD (Loss Given Default)
- Exposure (loan amount)
- Interest margin
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProfitConfig:
    """Configuration for profit calculation."""

    lgd: float = 0.5  # Loss Given Default (fraction of exposure lost)
    interest_margin: float = 0.05  # Annual interest margin (profit if no default)
    accept_rate: float = 0.15  # Fraction of applicants to accept


def compute_expected_profit(
    y_score: np.ndarray,
    exposure: np.ndarray | float = 1.0,
    lgd: float = 0.5,
    interest_margin: float = 0.05,
) -> np.ndarray:
    """Compute expected profit per account.

    Expected profit = (1 - PD) * margin * exposure - PD * LGD * exposure
                    = exposure * [(1 - PD) * margin - PD * LGD]
                    = exposure * [margin - PD * (margin + LGD)]

    Args:
        y_score: Predicted PD (probability of default/bad).
        exposure: Loan amount(s). Scalar or array.
        lgd: Loss Given Default (fraction of exposure lost on default).
        interest_margin: Profit margin if no default (fraction of exposure).

    Returns:
        Expected profit per account.
    """
    exposure = np.asarray(exposure)
    profit_if_good = interest_margin * exposure
    loss_if_bad = lgd * exposure

    expected_profit = (1 - y_score) * profit_if_good - y_score * loss_if_bad

    return expected_profit


def compute_portfolio_profit(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cfg: ProfitConfig,
    exposure: np.ndarray | float = 1.0,
) -> dict[str, float]:
    """Compute portfolio profit metrics.

    Simulates accepting top α fraction (lowest PD) and computes:
    - Expected profit (using predicted PD)
    - Realized profit (using true labels)
    - Profit per accepted account

    Args:
        y_true: True binary labels (0=good, 1=bad).
        y_score: Predicted PD.
        cfg: Profit configuration.
        exposure: Loan amount(s).

    Returns:
        Dictionary with profit metrics:
        - expected_total: Sum of expected profit across accepted
        - realized_total: Actual profit using true labels
        - expected_per_account: Average expected profit per accepted
        - realized_per_account: Average realized profit per accepted
        - n_accepted: Number of accepted applicants
    """
    exposure = np.asarray(exposure)
    if exposure.ndim == 0:
        exposure = np.full(len(y_true), float(exposure))

    n_accept = max(1, int(len(y_true) * cfg.accept_rate))

    # Accept applicants with lowest predicted PD
    accept_indices = np.argsort(y_score)[:n_accept]

    y_score_accepted = y_score[accept_indices]
    y_true_accepted = y_true[accept_indices]
    exposure_accepted = exposure[accept_indices]

    # Expected profit (using predicted PD)
    expected_profit = compute_expected_profit(
        y_score_accepted,
        exposure_accepted,
        cfg.lgd,
        cfg.interest_margin,
    )

    # Realized profit (using true labels)
    # Good loan (y=0): profit = margin * exposure
    # Bad loan (y=1): loss = LGD * exposure
    realized_profit = np.where(
        y_true_accepted == 0,
        cfg.interest_margin * exposure_accepted,
        -cfg.lgd * exposure_accepted,
    )

    return {
        "expected_total": float(expected_profit.sum()),
        "realized_total": float(realized_profit.sum()),
        "expected_per_account": float(expected_profit.mean()),
        "realized_per_account": float(realized_profit.mean()),
        "n_accepted": n_accept,
    }
