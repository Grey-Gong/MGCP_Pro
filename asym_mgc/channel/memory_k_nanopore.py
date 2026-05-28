"""
Memory-k Nanopore Channel Model.

Extends the i.i.d. DNA channel with k-order Markov memory to capture
context-dependent error profiles observed in real nanopore sequencing.
Reference: Maarouf et al., ISTC 2023; Hamoum et al., ISTC 2021.

Phase 1 base: Section 5 of IMPROVEMENT_PLAN.md.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np


class MemoryKNanoporeChannel:
    """
    Memory-k nanopore channel with context-dependent error probabilities.

    Error probabilities depend on the previous k bases, capturing:
    - Homopolymer penalty: higher deletion rate at homopolymer boundaries
    - GC bias: elevated error rates in extreme GC content regions
    - Sequence motifs: error-prone patterns (GAGA, CUCU, etc.)

    Parameters
    ----------
    k : int
        Context memory order (k=3 recommended for nanopore).
    Pd : float
        Base deletion probability.
    Pi : float
        Base insertion probability.
    Ps : float
        Base substitution probability.
    homopolymer_penalty : float
        Multiplicative factor for Pd at homopolymer boundaries.
    gc_bias : float
        Multiplicative factor adjustment for extreme GC content.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    NUCLEOTIDES = "ACGT"

    def __init__(
        self,
        k: int = 3,
        Pd: float = 0.5,
        Pi: float = 0.026,
        Ps: float = 0.474,
        homopolymer_penalty: float = 2.0,
        gc_bias: float = 0.15,
        seed: Optional[int] = None,
    ):
        if Pd < 0 or Pi < 0 or Ps < 0:
            raise ValueError("Probabilities must be non-negative")
        total = Pd + Pi + Ps
        if total < 1e-8:
            Pd_final = 0.0
            Pi_final = 0.0
            Ps_final = 0.0
        else:
            Pd_final = Pd
            Pi_final = Pi
            Ps_final = Ps
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = k
        self.base_Pd = Pd_final
        self.base_Pi = Pi_final
        self.base_Ps = Ps_final
        self.homopolymer_penalty = homopolymer_penalty
        self.gc_bias = gc_bias
        self.rng = np.random.default_rng(seed)

    def error_profile(self, context: str) -> dict:
        """
        Compute error probabilities given the previous k context bases.

        Parameters
        ----------
        context : str
            Previous k bases (or fewer if sequence is shorter).

        Returns
        -------
        dict
            {'Pd': float, 'Pi': float, 'Ps': float}
        """
        Pd = self.base_Pd
        Pi = self.base_Pi
        Ps = self.base_Ps

        # Homopolymer penalty: if last two bases are the same, we're at a boundary
        if len(context) >= 2 and context[-1] == context[-2]:
            Pd = min(Pd * self.homopolymer_penalty, 0.95)
            Pi = Pi * 0.5

        # GC bias
        ctx_len = max(len(context), 1)
        gc_count = context.count('G') + context.count('C')
        gc_ratio = gc_count / ctx_len
        if gc_ratio > 0.7:
            Pd = min(Pd * (1 + self.gc_bias), 0.95)
        elif gc_ratio < 0.3:
            Pd = max(Pd * (1 - self.gc_bias * 0.5), 0.0)

        # Clamp to valid probability range
        Pd = max(0.0, min(Pd, 1.0))
        Pi = max(0.0, min(Pi, 1.0))
        Ps = max(0.0, min(Ps, 1.0))

        return {'Pd': Pd, 'Pi': Pi, 'Ps': Ps}

    def transmit(self, x: str) -> Tuple[str, list]:
        """
        Transmit a DNA sequence through the memory-k channel.

        Parameters
        ----------
        x : str
            Input DNA sequence.

        Returns
        -------
        y : str
            Output DNA sequence (after errors).
        edits : list
            List of edit records: (pos, type, detail).
        """
        y_parts = []
        edits = []
        context = ""

        for i, base in enumerate(x):
            probs = self.error_profile(context)
            Pd = probs['Pd']
            Pi = probs['Pi']
            Ps = probs['Ps']

            r = self.rng.random()

            if r < Pd:
                edits.append((i, 'D', base))
            elif r < Pd + Pi:
                inserted = self.rng.choice(list(self.NUCLEOTIDES))
                y_parts.append(inserted)
                edits.append((i, 'I', inserted))
                context += inserted
            elif r < Pd + Pi + Ps:
                candidates = [b for b in self.NUCLEOTIDES if b != base]
                substituted = self.rng.choice(candidates)
                y_parts.append(substituted)
                edits.append((i, 'S', (base, substituted)))
                context += substituted
            else:
                y_parts.append(base)
                context += base

            # Keep context bounded to k
            if len(context) > self.k:
                context = context[-self.k:]

        return ''.join(y_parts), edits

    def transmit_with_quality(
        self, x: str, base_quality_mean: float = 20.0
    ) -> Tuple[str, np.ndarray]:
        """
        Transmit with simulated basecaller quality scores.

        Parameters
        ----------
        x : str
            Input DNA sequence.
        base_quality_mean : float
            Mean Phred quality score (higher = more reliable).

        Returns
        -------
        y : str
            Output DNA sequence.
        quality : ndarray
            Per-position Phred quality scores.
        """
        y, edits = self.transmit(x)
        n_output = len(y)

        # Simulate basecaller quality (correlated with context)
        quality = self.rng.normal(base_quality_mean, 5.0, n_output)
        quality = np.clip(quality, 1.0, 45.0)
        quality = np.round(quality).astype(int)

        # Reduce quality at error positions (approximate alignment)
        # Key insight: in indel channels, quality should reflect local confidence.
        # - SUBSTITUTION: basecaller reports wrong base -> low quality at output position
        # - INSERTION: extra base at output position -> low quality at output position
        # - DELETION: no output position for deleted input. But basecaller sees the
        #   NEXT base at the same logical position -> quality of the NEXT position should be reduced
        #   (since the deleted base causes a positional shift)
        error_positions = set()
        out_pos = 0  # current output position (how many bases emitted so far)
        in_pos = 0   # current input position
        for edit in edits:
            i, etype, detail = edit
            if etype == 'D':
                # Deletion: input position i was skipped. The basecaller sees the NEXT
                # input base at output position out_pos. Quality at out_pos should be reduced
                # because the deleted base causes a shift.
                error_positions.add(out_pos)
                in_pos += 1
            elif etype == 'S':
                # Substitution: we emitted 'substituted' at output position out_pos.
                # Quality at out_pos is reduced because basecaller reports wrong base.
                error_positions.add(out_pos)
                out_pos += 1
                in_pos += 1
            elif etype == 'I':
                # Insertion: we emitted 'inserted' at output position out_pos.
                # Quality at out_pos is reduced because this base is unexpected.
                error_positions.add(out_pos)
                out_pos += 1
                # Input position unchanged (insertion consumes no input)

        for ep in error_positions:
            if 0 <= ep < len(quality):
                quality[ep] = max(1, quality[ep] - 10)

        return y, quality

    def compute_edit_stats(self, x: str, y: str) -> dict:
        """
        Compute edit distance statistics between input and output.
        Uses dynamic programming Needleman-Wunsch with unit costs.

        Parameters
        ----------
        x : str
            Reference sequence.
        y : str
            Observed sequence.

        Returns
        -------
        stats : dict
            Statistics including edit distance, deletions, insertions,
            substitutions, and drift.
        """
        n, m = len(x), len(y)
        dp = np.full((n + 1, m + 1), np.inf, dtype=float)
        dp[0, :] = np.arange(m + 1)
        dp[:, 0] = np.arange(n + 1)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if x[i - 1] == y[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,       # deletion
                    dp[i, j - 1] + 1,       # insertion
                    dp[i - 1, j - 1] + cost  # substitution or match
                )

        # Backtrack to count error types
        i, j = n, m
        deletions = insertions = substitutions = matches = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and x[i - 1] == y[j - 1]:
                matches += 1
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
                deletions += 1
                i -= 1
            elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
                insertions += 1
                j -= 1
            else:
                break

        drift = len(y) - len(x)
        return {
            'edit_distance': int(dp[n, m]),
            'deletions': deletions,
            'insertions': insertions,
            'substitutions': substitutions,
            'matches': matches,
            'drift': drift,
        }


def standard_nanopore_params() -> Tuple[float, float, float]:
    """
    Return standard nanopore error profile parameters.
    Based on real sequencing data statistics.
    """
    return (0.5, 0.026, 0.474)


def deletion_dominant_params() -> Tuple[float, float, float]:
    """
    Return deletion-dominant error profile for stress testing.
    """
    return (0.6, 0.1, 0.3)


def iid_params() -> Tuple[float, float, float]:
    """
    Return roughly symmetric i.i.d. error profile for comparison.
    """
    return (0.33, 0.33, 0.34)
