"""
BCJR (MAP) Decoder for FSM-Trellis Joint Decoding.

Implements the forward-backward algorithm to compute per-position posterior
probabilities, enabling true symbol-level error correction.

Key differences from Viterbi:
- Viterbi: ML criterion → minimizes sequence error rate → outputs 1 path
- BCJR:   MAP criterion → minimizes per-symbol error rate → outputs P(x_t | Y)

Reference: Section 3.7.3 of IMPROVEMENT_PLAN.md v2.1.
Reference: ARCHITECTURE_REVISION_v2_1.md (CRITICAL-2 fix).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


LOG_ZERO = -1e100


# ---------------------------------------------------------------------------
# Log-domain utilities
# ---------------------------------------------------------------------------

def log_add(x: float, y: float) -> float:
    """
    Log-domain addition: log(exp(x) + exp(y)).

    Uses the identity: log(a + b) = max(x, y) + log(1 + exp(-|x - y|))
    """
    if x < LOG_ZERO + 50:
        return y
    if y < LOG_ZERO + 50:
        return x
    if x > y:
        return x + math.log1p(math.exp(y - x))
    return y + math.log1p(math.exp(x - y))


def log_sum_exp(vals: List[float]) -> float:
    """Log-sum-exp in a numerically stable way."""
    r = LOG_ZERO
    for v in vals:
        r = log_add(r, v)
    return r


# ---------------------------------------------------------------------------
# Trellis data structures
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """Edge from col t-1 to col t."""
    from_idx: int     # state index in col t-1 (the FROM column)
    to_idx: int       # state index in col t (the TO column)
    tt: int           # transition type: 0=MATCH, 1=DEL, 2=INS
    emitted: int      # 0-3 or -1 (deletion)
    log_gamma: float  # log P(obs | transition)


@dataclass
class TrellisCol:
    """One time step of the trellis."""
    states: List[Tuple]              # FSM state tuples
    edges: List[List[Edge]]           # edges[prev_state][next_state_idx] or flat
    log_alpha: np.ndarray             # forward message
    log_beta: np.ndarray = None       # backward message (filled later)
    log_gamma_matrix: np.ndarray = None  # branch metrics per state pair

    def __post_init__(self):
        if self.log_beta is None:
            self.log_beta = np.full(len(self.states), LOG_ZERO)


# ---------------------------------------------------------------------------
# Posterior utilities
# ---------------------------------------------------------------------------

def posterior_to_extrinsic_llr(
    posterior: np.ndarray,
    prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert posterior probabilities to extrinsic LLR for Turbo decoding.

    LLR_ext = LLR_post - LLR_prior
    """
    if prior is None:
        prior = np.ones(4) * 0.25

    eps = 1e-12
    p_post = np.clip(posterior, eps, 1 - eps)
    p_prior = np.clip(prior, eps, 1 - eps)

    # LLR = log P(b=1) - log P(b=0)
    # For multi-symbol: use mutual information as proxy
    extrinsic = np.log(p_post + eps) - np.log(p_prior + eps)
    return extrinsic


def posterior_to_reliability(posterior: np.ndarray) -> float:
    """
    Compute reliability metric from posterior distribution.

    Uses mutual information: I(X;Y) = 1 - H(posterior)/log(4)
    Higher is better (more certain).
    """
    eps = 1e-12
    p = np.clip(posterior, eps, 1 - eps)
    entropy = -np.sum(p * np.log(p))
    max_entropy = math.log(4)
    reliability = 1.0 - entropy / max_entropy
    return reliability


# ---------------------------------------------------------------------------
# Main BCJR Decoder
# ---------------------------------------------------------------------------

class FSMBCJRDecoder:
    """
    BCJR (MAP) decoder for FSM-Trellis joint decoding.

    Computes per-position posterior probabilities:
      P(state_t | observations_1..T)

    These posteriors can be used:
    1. As soft decisions for the next decoding pass
    2. As consensus reliability scores
    3. As extrinsic information for turbo decoding

    Parameters
    ----------
    N : int
        Number of RS codeword symbols.
    l : int
        Bits per RS symbol.
    c_crc : int
        CRC bits per block.
    D_max : int
        Max deletion offset (negative direction).
    I_max : int
        Max insertion offset (positive direction).
    Pd : float
        Deletion probability.
    Pi : float
        Insertion probability.
    Ps : float
        Substitution probability.
    crc_poly : int
        CRC polynomial.
    """

    def __init__(
        self,
        N: int = 120,
        l: int = 8,
        c_crc: int = 8,
        D_max: int = 20,
        I_max: int = 4,
        Pd: float = 0.5,
        Pi: float = 0.026,
        Ps: float = 0.474,
        crc_poly: int = 0x107,
    ):
        self.N = N
        self.l = l
        self.c_crc = c_crc
        self.D_max = D_max
        self.I_max = I_max
        self.Pd = Pd
        self.Pi = Pi
        self.Ps = Ps
        self.crc_poly = crc_poly
        self.crc_mask = (1 << c_crc) - 1

        self.log_P_CORR = np.log(1.0 - Pd - Pi - Ps + 1e-12)
        self.log_P_DEL = np.log(Pd + 1e-12)
        self.log_P_INS = np.log(Pi + 1e-12)
        self.log_P_SUB = np.log(Ps + 1e-12)

        self._crc_bit_buffer: List[int] = []

    def _reset_crc(self) -> None:
        """Reset CRC bit buffer."""
        self._crc_bit_buffer.clear()

    def _crc_update(self, syndrome: int, base_int: int) -> int:
        """
        Update CRC syndrome with one base (2 bits), MSB-first per symbol.

        CRITICAL-2 FIX (ARCHITECTURE_REVISION_v2_1.md):
        This must match the encoder's crc8_from_bases() exactly.
        Accumulates l/2 bases, then computes CRC on the complete symbol.
        prev_base is no longer needed here (CRC is symbol-level, not per-base).
        """
        self._crc_bit_buffer.append(base_int)
        if len(self._crc_bit_buffer) < self.l // 2:
            return syndrome

        # Build symbol from accumulated bases (MSB-first bit ordering)
        symbol = 0
        for b in self._crc_bit_buffer:
            symbol = (symbol << 2) | (b & 3)
        self._crc_bit_buffer.clear()

        # Compute CRC on the complete symbol (MSB-first)
        s = syndrome
        for bit_pos in range(self.l):
            feedback = (s >> (self.l - 1)) & 1
            s = ((s << 1) & self.crc_mask)
            data_bit = (symbol >> (self.l - 1 - bit_pos)) & 1
            if feedback ^ data_bit:
                s ^= self.crc_poly
        return s

    def _build_trellis(
        self,
        observed: np.ndarray,
        quality: Optional[np.ndarray],
    ) -> List[TrellisCol]:
        """
        Build the BCJR trellis for a given observation sequence.

        Returns list of TrellisCol, one per time step.
        """
        from .trellis import HomopolymerState

        cols: List[TrellisCol] = []
        self._reset_crc()

        # Initial state
        initial_state = (0, 0, 0, 0, HomopolymerState.NONE.value, -1)
        current_states = [initial_state]

        for t, obs in enumerate(observed):
            col = TrellisCol(states=current_states, edges=[], log_alpha=np.zeros(len(current_states)))
            cols.append(col)

            next_states_set: Dict[Tuple, int] = {}
            all_edges: List[Edge] = []

            for si, state in enumerate(current_states):
                i, delta, beta, gamma, s_hp, prev_base = state
                if i >= self.N:
                    continue

                can_delete = delta > -self.D_max
                can_insert = delta < self.I_max

                # Try all 4 bases for MATCH
                for emitted in range(4):
                    hp_state = HomopolymerState(s_hp)
                    if prev_base >= 0:
                        next_hp = HomopolymerState.next(hp_state, emitted, prev_base)
                        if next_hp is None:
                            continue
                    else:
                        next_hp = HomopolymerState.SINGLE

                    # Compute log_gamma (branch metric)
                    if quality is not None and t < len(quality):
                        q = quality[t]
                        if emitted == obs:
                            llr = q * math.log(10)
                            log_gamma = llr + self.log_P_CORR
                        else:
                            llr = -q * math.log(10)
                            log_gamma = llr + self.log_P_SUB
                    else:
                        if emitted == obs:
                            log_gamma = self.log_P_CORR
                        else:
                            log_gamma = self.log_P_SUB

                    next_beta = (beta + 1) % self.l
                    next_gamma = self._crc_update(gamma, emitted)
                    next_i = i + 1

                    ns = (next_i, delta, next_beta, next_gamma, next_hp.value, emitted)
                    ns_idx = next_states_set.get(ns)
                    if ns_idx is None:
                        ns_idx = len(current_states)
                        if len(cols) > 0:
                            cols[-1].states.append(ns)
                        next_states_set[ns] = ns_idx

                    edge = Edge(
                        from_idx=si, to_idx=ns_idx,
                        tt=0, emitted=emitted, log_gamma=log_gamma
                    )
                    all_edges.append(edge)

                # DELETION transitions
                if can_delete:
                    for emitted in range(4):
                        hp_state = HomopolymerState(s_hp)
                        if prev_base >= 0:
                            next_hp = HomopolymerState.next(hp_state, emitted, prev_base)
                            if next_hp is None:
                                continue
                        else:
                            next_hp = HomopolymerState.SINGLE

                        next_beta = (beta + 1) % self.l
                        next_gamma = self._crc_update(gamma, emitted)
                        next_i = i + 1
                        next_delta = delta - 1

                        ns = (next_i, next_delta, next_beta, next_gamma, next_hp.value, emitted)
                        ns_idx = next_states_set.get(ns)
                        if ns_idx is None:
                            ns_idx = len(current_states)
                            if len(cols) > 0:
                                cols[-1].states.append(ns)
                            next_states_set[ns] = ns_idx

                        edge = Edge(
                            from_idx=si, to_idx=ns_idx,
                            tt=1, emitted=emitted, log_gamma=self.log_P_DEL
                        )
                        all_edges.append(edge)

                # INSERTION transitions
                if can_insert:
                    next_delta = delta + 1
                    ns = (i, next_delta, beta, gamma, s_hp, prev_base)
                    ns_idx = next_states_set.get(ns)
                    if ns_idx is None:
                        ns_idx = len(current_states)
                        if len(cols) > 0:
                            cols[-1].states.append(ns)
                        next_states_set[ns] = ns_idx

                    edge = Edge(
                        from_idx=si, to_idx=ns_idx,
                        tt=2, emitted=-1, log_gamma=self.log_P_INS
                    )
                    all_edges.append(edge)

            # Add edges to column
            cols[-1].edges.append(all_edges)
            current_states = cols[-1].states.copy()

        return cols

    def decode(
        self,
        observed: np.ndarray,
        quality: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Run BCJR decoding on an observed sequence.

        Parameters
        ----------
        observed : np.ndarray
            Observed base integers (0-3).
        quality : Optional[np.ndarray]
            Phred quality scores.

        Returns
        -------
        posteriors : np.ndarray
            Per-position posterior probabilities [T, 4].
        info : List[dict]
            Per-step statistics.
        """
        cols = self._build_trellis(observed, quality)

        T = len(cols)
        if T == 0:
            return np.zeros((0, 4)), []

        # Forward pass
        for t in range(T):
            col = cols[t]
            if t == 0:
                col.log_alpha[:] = 0.0
            else:
                for edge in col.edges[0] if col.edges else []:
                    col.log_alpha[edge.to_idx] = log_add(
                        col.log_alpha[edge.to_idx],
                        cols[t - 1].log_alpha[edge.from_idx] + edge.log_gamma
                    )

        # Backward pass
        for t in range(T - 1, -1, -1):
            col = cols[t]
            if t == T - 1:
                col.log_beta[:] = 0.0
            else:
                for edge in col.edges[0] if col.edges else []:
                    if edge.to_idx < len(cols[t + 1].log_beta):
                        col.log_beta[edge.from_idx] = log_add(
                            col.log_beta[edge.from_idx],
                            cols[t + 1].log_beta[edge.to_idx] + edge.log_gamma
                        )

        # Compute posteriors per position
        posteriors = np.zeros((T, 4))
        info = []

        for t in range(T):
            col = cols[t]
            total_llr = log_sum_exp(
                col.log_alpha + col.log_beta
            )

            pos_probs = np.zeros(4)
            for base in range(4):
                llr = 0.0
                for edge in (col.edges[0] if col.edges else []):
                    if edge.tt == 0 and edge.emitted == base:  # MATCH only
                        alpha_beta = col.log_alpha[edge.from_idx] + edge.log_gamma + col.log_beta[edge.to_idx]
                        llr = log_add(llr, alpha_beta)
                pos_probs[base] = math.exp(llr - total_llr) if total_llr > -1e50 else 0.0

            posteriors[t] = pos_probs

            reliability = posterior_to_reliability(pos_probs)
            info.append({
                'reliability': reliability,
                'max_prob': float(np.max(pos_probs)),
                'alpha': float(np.max(col.log_alpha)),
                'beta': float(np.max(col.log_beta)),
            })

        return posteriors, info
