"""
Threshold specification for ABR and pAUC metrics.

This module provides an immutable ThresholdSpec that ensures identical
thresholds are used across all evaluation methods (biased, Bayesian)
and training comparisons (accepts-only, BASL).

Per paper Section 6.3: ABR integrates over acceptance rates [0.2, 0.4].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ThresholdSpec:
    """Immutable specification for ABR/pAUC thresholds.

    This dataclass enforces paper-faithful threshold values and ensures
    they are passed explicitly to all metric computations.

    Attributes:
        abr_range: Acceptance rate range for ABR integration, (0.2, 0.4) per paper.
        pauc_max_fnr: Maximum FNR for partial AUC (high-recall region), 0.2 per paper.
            FNR = 1 - TPR, so max_fnr=0.2 means TPR >= 0.8 (top 80% recall).
        policy: How thresholds are determined. Currently only "fixed_value".
    """

    abr_range: Tuple[float, float]
    pauc_max_fnr: float
    policy: str = "fixed_value"

    def __post_init__(self) -> None:
        """Validate paper-faithful threshold values."""
        if self.abr_range != (0.2, 0.4):
            raise ValueError(
                f"ABR range must be (0.2, 0.4) per paper Section 6.3, "
                f"got {self.abr_range}"
            )
        if self.pauc_max_fnr != 0.2:
            raise ValueError(
                f"pAUC max_fnr must be 0.2 per paper, got {self.pauc_max_fnr}"
            )
        if self.policy not in ("fixed_value", "fixed_quantile_on_H"):
            raise ValueError(
                f"policy must be 'fixed_value' or 'fixed_quantile_on_H', "
                f"got {self.policy}"
            )

    @classmethod
    def paper_default(cls) -> ThresholdSpec:
        """Create ThresholdSpec with paper-faithful defaults."""
        return cls(
            abr_range=(0.2, 0.4),
            pauc_max_fnr=0.2,
            policy="fixed_value",
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "abr_range": list(self.abr_range),
            "pauc_max_fnr": self.pauc_max_fnr,
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ThresholdSpec:
        """Create from dictionary (e.g., from YAML config)."""
        return cls(
            abr_range=tuple(d["abr_range"]),
            pauc_max_fnr=d["pauc_max_fnr"],
            policy=d.get("policy", "fixed_value"),
        )
