"""
Soft Branch Metric Computation for Asym-MGC.

Implements LLR (log-likelihood ratio) computation from basecaller quality scores,
including homopolymer-aware adjustments.
Reference: Section 3.4 of IMPROVEMENT_PLAN.md v2.0.

Phase 1.11-1.12: Section 3.4 of IMPROVEMENT_PLAN.md.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def phred_to_prob_error(Q: int) -> float:
    """
    Convert Phred quality score to probability of error.

    P(error) = 10^{-Q/10}
    """
    return 10.0 ** (-Q / 10.0)


def phred_to_llr(Q: int, match: bool) -> float:
    """
    Convert Phred quality score to log-likelihood ratio.

    LLR = log [P(correct) / P(error)] = Q * ln(10) if match
    LLR = -Q * ln(10) if mismatch
    """
    return Q * math.log(10) if match else -Q * math.log(10)


def compute_llr(
    hypothesized_base: int,
    observed_base: int,
    phred_quality: int,
) -> float:
    """
    Compute the LLR for a base observation.

    Parameters
    ----------
    hypothesized_base : int
        Hypothesized base (0-3).
    observed_base : int
        Observed/received base (0-3).
    phred_quality : int
        Phred quality score.

    Returns
    -------
    float
        LLR: log [P(observed | hypothesized) / P(observed | other)]
    """
    is_match = (hypothesized_base == observed_base)
    return phred_quality * math.log(10) if is_match else -phred_quality * math.log(10)


def compute_match_llr(
    phred_quality: int,
    P_correct: float,
    P_substitution: float,
) -> float:
    """
    Compute the LLR for a MATCH transition with channel probabilities.

    LLR = log [P(y_t | x_t, MATCH) / P(y_t | x_t, DELETION)]
        = log [P_correct / P_deletion]
    """
    P_deletion = P_substitution  # Simplified: treat substitution as deletion for LLR
    return math.log(max(P_correct, 1e-12)) - math.log(max(P_deletion, 1e-12))


def compute_insertion_llr(
    P_insertion: float,
    P_correct: float,
) -> float:
    """
    Compute the LLR for an INSERTION transition.

    LLR_insertion = log [P(y_t | INSERTION) / P(y_t | MATCH)]
                  = log [P_insertion / P_correct]
    """
    return math.log(max(P_insertion, 1e-12)) - math.log(max(P_correct, 1e-12))


def homopolymer_aware_llr_adjustment(
    llr: float,
    in_homopolymer: bool,
    homopolymer_penalty: float = 2.0,
) -> dict:
    """
    Adjust LLR values based on homopolymer context.

    In nanopore, deletions are more likely at homopolymer boundaries
    and inside homopolymer runs. This function adjusts the branch
    metrics accordingly.

    Parameters
    ----------
    llr : float
        Base LLR from basecaller.
    in_homopolymer : bool
        Whether the current position is inside a homopolymer run.
    homopolymer_penalty : float
        Penalty factor for deletion probability (multiplicative).

    Returns
    -------
    dict
        Adjusted LLRs for MATCH, DELETION, INSERTION.
    """
    if in_homopolymer:
        deletion_boost = math.log(homopolymer_penalty)
        return {
            'MATCH': llr - deletion_boost,
            'DELETION': llr + deletion_boost,
            'INSERTION': llr,  # Insertions less affected by homopolymer
        }
    else:
        return {
            'MATCH': llr,
            'DELETION': llr,
            'INSERTION': llr,
        }


def compute_reliability_weight(phred_quality: int) -> float:
    """
    Compute reliability weight from Phred score.

    Weight = 10^{Q/10}, giving exponentially scaled weights.
    """
    return 10.0 ** (phred_quality / 10.0)


def simulate_basecaller_quality(
    observed_base: int,
    true_base: int,
    base_error_rate: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, int]:
    """
    Simulate basecaller quality for a noisy observation.

    Parameters
    ----------
    observed_base : int
        The observed base (0-3).
    true_base : int
        The true base (0-3).
    base_error_rate : float
        Approximate per-base error rate for quality calibration.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    (observed_base, phred_quality) : tuple
        The observed base and simulated quality score.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_error = (observed_base != true_base)
    if is_error:
        Q = max(1, int(rng.normal(10.0, 5.0)))
    else:
        Q = max(1, int(rng.normal(25.0, 5.0)))

    return observed_base, Q


def quality_array_to_llr_matrix(
    quality: np.ndarray,
    observed_bases: np.ndarray,
) -> np.ndarray:
    """
    Build a 4xT LLR matrix from basecaller quality scores.

    Parameters
    ----------
    quality : ndarray
        Per-position Phred quality scores (length T).
    observed_bases : ndarray
        Per-position observed base indices (length T).

    Returns
    -------
    llr_matrix : ndarray
        4xT matrix where llr_matrix[b, t] = LLR for base b at position t.
    """
    T = len(quality)
    llr_matrix = np.zeros((4, T))

    for t in range(T):
        Q = quality[t]
        obs = observed_bases[t]
        for b in range(4):
            is_match = (b == obs)
            llr_matrix[b, t] = Q * math.log(10) if is_match else -Q * math.log(10)

    return llr_matrix


def llr_to_probability(llr: float) -> float:
    """
    Convert LLR to probability.

    P = sigmoid(llr) = 1 / (1 + exp(-llr))
    """
    return 1.0 / (1.0 + math.exp(-llr))
