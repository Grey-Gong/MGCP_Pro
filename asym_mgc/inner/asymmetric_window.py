"""
Asymmetric Drift Window for Asym-MGC.

Defines the non-symmetric state space for the Viterbi decoder based on
the deletion-domination property of nanopore sequencing.
Reference: Theorem 1 in IMPROVEMENT_PLAN.md v2.0.

Phase 1.01: Section 3.1 of IMPROVEMENT_PLAN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AsymmetricWindow:
    """
    Asymmetric drift window bounds for the Viterbi trellis.

    In nanopore channels, deletion probability Pd >> insertion probability Pi,
    so the net drift is strongly negative. The asymmetric window allocates
    more budget for deletions and less for insertions.

    Parameters
    ----------
    D_max : int
        Maximum net deletion offset (negative direction).
        Negative drift beyond this is considered an error.
    I_max : int
        Maximum net insertion offset (positive direction).
        Positive drift beyond this is considered an error.
    Pd : float
        Deletion probability.
    Pi : float
        Insertion probability.

    Attributes
    ----------
    delta_range : range
        Range of all valid drift states.
    size : int
        Total number of drift states.
    drift_mean_per_symbol : float
        Expected drift per symbol (negative for nanopore).
    """

    D_max: int
    I_max: int
    Pd: float
    Pi: float

    @property
    def delta_range(self) -> range:
        return range(-self.D_max, self.I_max + 1)

    @property
    def size(self) -> int:
        return self.D_max + self.I_max + 1

    @property
    def drift_mean_per_symbol(self) -> float:
        """Expected net drift per symbol: E[Delta] = -(Pd - Pi)."""
        return -(self.Pd - self.Pi)

    @classmethod
    def from_hoeffding_bound(
        cls,
        Pd: float,
        Pi: float,
        t: int,
        delta: float = 1e-6,
        sigma_multiplier: float = 3.0,
    ) -> "AsymmetricWindow":
        """
        Compute window bounds from the Hoeffding/Chernoff bound in Theorem 1.

        This guarantees that the correct path stays within the window with
        probability at least 1 - delta.

        Parameters
        ----------
        Pd : float
            Deletion probability.
        Pi : float
            Insertion probability.
        t : int
            Sequence length (number of symbols).
        delta : float
            Failure probability (default 1e-6).
        sigma_multiplier : float
            Number of standard deviations for the bound (3 = ~3-sigma).

        Returns
        -------
        AsymmetricWindow
            Window with D_max, I_max computed from the bound.
        """
        if Pd <= Pi:
            raise ValueError(
                f"Deletion-domination required: Pd={Pd} must be > Pi={Pi}"
            )

        mu = Pd - Pi  # positive: expected drift per symbol
        # Variance bound: X_k = D_k - I_k, each bounded in [-1, 1]
        # Var[X_k] <= 1/4
        # For Hoeffding: we need (a - t*mu), where a is the bound distance
        # D_max = ceil(t * mu + sigma_multiplier * sqrt(t / 2 * 1/4 * 4))
        #       = ceil(t * mu + sigma_multiplier * sqrt(t))
        # Using the 3-sigma approximation from Chernoff
        safety = sigma_multiplier * math.sqrt(t)
        D_max = math.ceil(t * mu + safety)
        I_max = math.ceil(safety)  # insertions are rare
        return cls(D_max=D_max, I_max=I_max, Pd=Pd, Pi=Pi)

    @classmethod
    def from_practical_budget(
        cls,
        Pd: float,
        Pi: float,
        N: int,
        deletion_budget_fraction: float = 0.33,
    ) -> "AsymmetricWindow":
        """
        Compute window bounds from practical deletion budget.

        This is used for sliding-window decoding where each window
        (not the full sequence) needs its own budget.

        Parameters
        ----------
        Pd : float
            Deletion probability.
        Pi : float
            Insertion probability.
        N : int
            Total sequence length.
        deletion_budget_fraction : float
            Fraction of N * Pd to use as D_max (default 0.33).

        Returns
        -------
        AsymmetricWindow
            Window with practical budget bounds.
        """
        D_max = math.ceil(N * Pd * deletion_budget_fraction)
        I_max = math.ceil(N * Pi * 1.5)  # small buffer for insertions
        return cls(D_max=D_max, I_max=I_max, Pd=Pd, Pi=Pi)

    def capture_probability(self, t: int) -> float:
        """
        Lower bound on the probability that the correct path stays in window.

        Uses the simplified Hoeffding/Chernoff bound from Theorem 1.

        Parameters
        ----------
        t : int
            Sequence length.

        Returns
        -------
        float
            Lower bound on capture probability (clamped to [0, 1]).
        """
        mu = self.Pd - self.Pi
        var_bound = 0.25  # Var[X_k] <= 1/4

        # Chernoff-style bound for both directions
        # P(Δ_t <= -D_max) <= exp(-2 * (D_max - t*mu)^2 / t)  for D_max > t*mu
        # P(Δ_t >= +I_max) <= exp(-2 * (I_max + t*mu)^2 / t)   for I_max > -t*mu
        neg_bound = 1.0
        if self.D_max > t * mu:
            neg_bound = math.exp(-2 * (self.D_max - t * mu) ** 2 / t)

        pos_bound = 1.0
        if self.I_max > -t * mu:
            pos_bound = math.exp(-2 * (self.I_max + t * mu) ** 2 / t)

        return 1.0 - neg_bound - pos_bound

    def __repr__(self) -> str:
        return (
            f"AsymmetricWindow(D_max={self.D_max}, I_max={self.I_max}, "
            f"Pd={self.Pd}, Pi={self.Pi}, |Δ|={self.size})"
        )


def delta_to_index(delta: int, window: AsymmetricWindow) -> int:
    """Map a drift value delta to a zero-based array index."""
    idx = delta + window.D_max
    if idx < 0 or idx >= window.size:
        raise ValueError(
            f"Delta {delta} out of bounds for window with D_max={window.D_max}, "
            f"I_max={window.I_max}"
        )
    return idx


def index_to_delta(idx: int, window: AsymmetricWindow) -> int:
    """Map a zero-based array index to a drift value delta."""
    if idx < 0 or idx >= window.size:
        raise ValueError(f"Index {idx} out of bounds for window size {window.size}")
    return idx - window.D_max


def typical_window() -> AsymmetricWindow:
    """
    Return the typical asymmetric window for nanopore experiments.
    Based on Pd=0.5, Pi=0.03, N=120, D_max=20, I_max=4.
    """
    return AsymmetricWindow(D_max=20, I_max=4, Pd=0.5, Pi=0.03)
