"""
FSM-Trellis Joint Decoder for Asym-MGC.

Implements Viterbi dynamic programming over the joint state space:
  s = (i, delta, beta, gamma, S_hp, prev_base)

Key features:
- Non-symmetric drift window (D_max for deletions, I_max for insertions)
- FSM homopolymer constraint embedded in state transitions
- List Viterbi (top-K candidates per state) for robust decoding
- CRC early-termination pruning at block boundaries
- Path metric threshold pruning
- CRITICAL-4 FIX: traceback_path uses prev_pm link (not _best_per_state dict)

Reference: Section 3 of IMPROVEMENT_PLAN.md v2.1.
Reference: ARCHITECTURE_REVISION_v2_1.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .trellis import HomopolymerState

# Re-export HPState for backward compatibility (tests import from fsm_joint)
HPState = HomopolymerState

# Re-export HPState for backward compatibility (tests import from fsm_joint)
HPState = HomopolymerState


# -----------------------------------------------------------------------
# Path metric structures
# -----------------------------------------------------------------------

@dataclass
class FSMPathMetric:
    """
    Path metric for a single path through the trellis.

    v2.1 FIX (CRITICAL-4): prev_pm links directly to the previous
    FSMPathMetric (not to a state). This ensures correct traceback
    even when list_k > 1 and multiple paths converge on the same state.
    """
    log_prob: float
    prev_state: Optional["FSMViterbiState"]
    prev_pm: Optional["FSMPathMetric"] = None  # Direct link to previous path metric
    transition: str = ''
    emitted_base: int = -1  # Base emitted by this transition (-1 = no emission)

    def __lt__(self, other: "FSMPathMetric") -> bool:
        return self.log_prob < other.log_prob


class FSMPathMetricTopK:
    """Maintain top-K path metrics, sorted by log_prob descending."""

    def __init__(self, list_k: int = 8):
        self.list_k = list_k
        self._metrics: List[FSMPathMetric] = []

    def __len__(self) -> int:
        return len(self._metrics)

    def add(self, pm: FSMPathMetric) -> None:
        self._metrics.append(pm)
        self._metrics.sort(key=lambda x: x.log_prob, reverse=True)
        if len(self._metrics) > self.list_k:
            self._metrics = self._metrics[:self.list_k]

    def extend(self, metrics: List[FSMPathMetric]) -> None:
        self._metrics.extend(metrics)
        self._metrics.sort(key=lambda x: x.log_prob, reverse=True)
        self._metrics = self._metrics[:self.list_k]

    def get_best(self) -> Optional[FSMPathMetric]:
        return self._metrics[0] if self._metrics else None

    def get_all(self) -> List[FSMPathMetric]:
        return self._metrics


# -----------------------------------------------------------------------
# Trellis state
# -----------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FSMViterbiState:
    """
    Immutable state in the FSM-Trellis joint decoder.

    State tuple: (i, delta, beta, gamma, S_hp, prev_base)
    """
    i: int           # Input symbol index (0 to N)
    delta: int        # Net deletion offset: -(deletions - insertions)
    beta: int        # Block-internal symbol offset (0 to l-1)
    gamma: int       # CRC syndrome for current block (0 to 2^c_crc-1)
    s_hp: HomopolymerState  # Homopolymer FSM state
    prev_base: int = -1  # Previously emitted base (0-3), -1 = start

    def __hash__(self) -> int:
        return hash((
            self.i, self.delta, self.beta, self.gamma,
            self.s_hp, self.prev_base
        ))

    def __repr__(self) -> str:
        pb = '?' if self.prev_base < 0 else 'ACGT'[self.prev_base]
        return (f"FSMViterbiState(i={self.i}, "
                f"Δ={self.delta}, β={self.beta}, "
                f"γ={self.gamma}, "
                f"s_hp={self.s_hp.name}, prev_base={pb})")


# -----------------------------------------------------------------------
# Main decoder
# -----------------------------------------------------------------------

class FSMJointDecoder:
    """
    FSM-Trellis Viterbi decoder with joint state space.

    Implements the complete decoding pipeline:
    1. init_states: initialize starting state
    2. decode_step: process one observed base → update states
    3. enumerate_transitions: generate valid transitions from a state
    4. branch_metric: compute log-likelihood for each transition
    5. prune_crc / prune_topk / prune_threshold: prune state space
    6. traceback_all: extract top-K decoded sequences
    7. traceback_path: trace back a single path via prev_pm links

    Parameters
    ----------
    N : int
        Number of RS codeword symbols.
    l : int
        Bits per RS symbol (default 8).
    c_crc : int
        CRC bits per block (default 8).
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
    crc_mask : int
        CRC mask.
    K_best : int
        Max states per (i, delta) group (Top-K pruning).
    T_threshold : float
        Path metric threshold for pruning.
    list_k : int
        Number of candidates per state (List Viterbi).
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
        K_best: int = 200,
        T_threshold: float = 15.0,
        list_k: int = 8,
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
        self.K_best = K_best
        self.T_threshold = T_threshold
        self.list_k = list_k

        # Log-domain probabilities
        self.log_P_CORR = np.log(1.0 - Pd - Pi - Ps + 1e-12)
        self.log_P_DEL = np.log(Pd + 1e-12)
        self.log_P_INS = np.log(Pi + 1e-12)
        self.log_P_SUB = np.log(Ps + 1e-12)

        # CRC bit buffer for decoding (accumulates l/2 bases per symbol)
        self._crc_bit_buffer: List[int] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_states(self) -> Dict[FSMViterbiState, FSMPathMetricTopK]:
        """Initialize starting state(s)."""
        start = FSMViterbiState(
            i=0, delta=0, beta=0, gamma=0,
            s_hp=HomopolymerState.NONE, prev_base=-1
        )
        pm = FSMPathMetric(log_prob=0.0, prev_state=None, prev_pm=None,
                           transition='', emitted_base=-1)
        topk = FSMPathMetricTopK(list_k=self.list_k)
        topk.add(pm)
        self._crc_bit_buffer.clear()
        self._best_per_state: Dict[FSMViterbiState, FSMPathMetric] = {}
        return {start: topk}

    # ------------------------------------------------------------------
    # Branch metric
    # ------------------------------------------------------------------

    def branch_metric(
        self,
        transition: str,
        emitted_base: int,
        observed_base: int,
        phred_quality: float = 0.0,
    ) -> float:
        """
        Compute log-likelihood for a transition.

        For MATCH: uses Phred quality and channel probabilities
        For DELETION/INSERTION: uses channel probabilities only
        """
        if transition == 'MATCH':
            # LLR from basecaller quality
            if emitted_base == observed_base:
                llr = phred_quality * np.log(10)
            else:
                llr = -phred_quality * np.log(10)
            return llr + self.log_P_CORR

        elif transition == 'DELETION':
            # Input symbol was deleted; received symbol is independent noise
            return self.log_P_DEL

        elif transition == 'INSERTION':
            # Extra symbol was inserted; consumes received symbol without emitting
            return self.log_P_INS

        return -np.inf

    # ------------------------------------------------------------------
    # Transition enumeration
    # ------------------------------------------------------------------

    def enumerate_transitions(
        self, state: FSMViterbiState
    ) -> List[Tuple[str, int, int]]:
        """
        Enumerate all valid transitions from a given state.

        Returns list of (transition_type, emitted_base, next_delta) tuples.
        """
        transitions = []

        # Check if we've processed all input symbols
        if state.i >= self.N:
            return transitions

        # Check drift bounds
        can_delete = state.delta > -self.D_max
        can_insert = state.delta < self.I_max

        # Try all 4 bases
        for emitted_base in range(4):
            # FSM constraint: check homopolymer
            if state.prev_base >= 0:
                next_hp = HomopolymerState.next(state.s_hp, emitted_base, state.prev_base)
                if next_hp is None:
                    continue  # FSM blocks this transition
            else:
                next_hp = HomopolymerState.SINGLE

            # MATCH transition
            transitions.append(('MATCH', state.delta, emitted_base))

            # DELETION transition (consumes input, doesn't consume received)
            if can_delete:
                next_delta = state.delta - 1
                next_beta = (state.beta + 1) % self.l
                transitions.append(('DELETION', next_delta, emitted_base))

            # INSERTION transition (consumes received, doesn't consume input)
            if can_insert:
                next_delta = state.delta + 1
                transitions.append(('INSERTION', next_delta, emitted_base))

        return transitions

    # ------------------------------------------------------------------
    # CRC utilities (consistent with encoder crc_utils.py)
    # ------------------------------------------------------------------

    def _crc_update(self, syndrome: int, base_int: int) -> int:
        """
        Update CRC syndrome with one base (2 bits), MSB-first per symbol.

        Accumulates l/2 bases, then computes CRC on the complete symbol.
        This matches the encoder's crc8_from_bases() function.
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

    # ------------------------------------------------------------------
    # Decode step (one observation)
    # ------------------------------------------------------------------

    def decode_step(
        self,
        states: Dict[FSMViterbiState, FSMPathMetricTopK],
        observed_base: int,
        phred_quality: float = 0.0,
        apply_crc_prune: bool = False,
    ) -> Tuple[Dict[FSMViterbiState, FSMPathMetricTopK], dict]:
        """
        Process one observed base and update the state trellis.

        Parameters
        ----------
        states : Dict[FSMViterbiState, FSMPathMetricTopK]
            Current active states with their path metric lists.
        observed_base : int
            Observed base (0-3).
        phred_quality : float
            Phred quality score.
        apply_crc_prune : bool
            If True, apply CRC early-termination pruning.

        Returns
        -------
        next_states, stats
        """
        next_states: Dict[FSMViterbiState, FSMPathMetricTopK] = {}
        stats = {'transitions': 0, 'pruned': 0}

        for state, pm_topk in states.items():
            for prev_pm in pm_topk.get_all():
                for trans_type, next_delta, emitted_base in self.enumerate_transitions(state):
                    stats['transitions'] += 1

                    bm = self.branch_metric(
                        trans_type, emitted_base, observed_base, phred_quality
                    )

                    # Build next state
                    if trans_type == 'MATCH':
                        next_beta = (state.beta + 1) % self.l
                        next_gamma = self._crc_update(state.gamma, emitted_base)
                        next_s_hp = HomopolymerState.next(
                            state.s_hp, emitted_base, state.prev_base
                        ) or state.s_hp
                        next_i = state.i + 1
                        next_prev_base = emitted_base
                        next_state = FSMViterbiState(
                            i=next_i, delta=next_delta, beta=next_beta,
                            gamma=next_gamma, s_hp=next_s_hp,
                            prev_base=next_prev_base
                        )
                    elif trans_type == 'DELETION':
                        next_beta = (state.beta + 1) % self.l
                        next_gamma = self._crc_update(state.gamma, emitted_base)
                        next_s_hp = HomopolymerState.next(
                            state.s_hp, emitted_base, state.prev_base
                        ) or state.s_hp
                        next_i = state.i + 1
                        next_prev_base = emitted_base
                        next_state = FSMViterbiState(
                            i=next_i, delta=next_delta, beta=next_beta,
                            gamma=next_gamma, s_hp=next_s_hp,
                            prev_base=next_prev_base
                        )
                    else:  # INSERTION
                        next_state = FSMViterbiState(
                            i=state.i, delta=next_delta, beta=state.beta,
                            gamma=state.gamma, s_hp=state.s_hp,
                            prev_base=state.prev_base
                        )

                    new_pm = FSMPathMetric(
                        log_prob=prev_pm.log_prob + bm,
                        prev_state=state,
                        prev_pm=prev_pm,
                        transition=trans_type,
                        emitted_base=emitted_base if trans_type != 'INSERTION' else -1,
                    )

                    if next_state not in next_states:
                        next_states[next_state] = FSMPathMetricTopK(list_k=self.list_k)
                    next_states[next_state].add(new_pm)

        # Pruning
        if apply_crc_prune:
            next_states = self.prune_crc(next_states)
        next_states = self.prune_topk(next_states)
        if self.T_threshold < np.inf:
            next_states = self.prune_threshold(next_states)

        stats['pruned'] = sum(1 for s in next_states.values() if len(s) == 0)

        # Track best PM per state (for traceback diagnostics)
        for state, topk in next_states.items():
            if topk.get_best() is not None:
                self._best_per_state[state] = topk.get_best()

        return next_states, stats

    # ------------------------------------------------------------------
    # Pruning strategies
    # ------------------------------------------------------------------

    def prune_crc(
        self, states: Dict[FSMViterbiState, FSMPathMetricTopK]
    ) -> Dict[FSMViterbiState, FSMPathMetricTopK]:
        """
        CRC early-termination: prune non-zero syndrome at block boundaries.

        Strategy A: Only prune when beta == 0 (at block boundary).
        """
        result = {}
        for state, topk in states.items():
            if state.beta == 0 and state.gamma != 0:
                continue  # Prune: CRC violation at block boundary
            result[state] = topk
        return result

    def prune_topk(
        self, states: Dict[FSMViterbiState, FSMPathMetricTopK]
    ) -> Dict[FSMViterbiState, FSMPathMetricTopK]:
        """
        Adaptive Top-K pruning: per (i, delta) group, keep at most K_best states.
        """
        from collections import defaultdict
        groups: Dict[Tuple[int, int], List[FSMViterbiState]] = defaultdict(list)
        for state in states:
            groups[(state.i, state.delta)].append(state)

        result = {}
        for (i, delta), group in groups.items():
            if len(group) <= self.K_best:
                for s in group:
                    result[s] = states[s]
            else:
                # Sort by best log_prob and keep top K_best
                sorted_group = sorted(
                    group,
                    key=lambda s: states[s].get_best().log_prob,
                    reverse=True
                )
                for s in sorted_group[:self.K_best]:
                    result[s] = states[s]
        return result

    def prune_threshold(
        self, states: Dict[FSMViterbiState, FSMPathMetricTopK]
    ) -> Dict[FSMViterbiState, FSMPathMetricTopK]:
        """
        Path metric threshold pruning: discard states much worse than best.
        """
        if not states:
            return states
        best_pm = max(
            states[s].get_best().log_prob
            for s in states
        )
        return {
            s: topk for s, topk in states.items()
            if topk.get_best() is not None
            and topk.get_best().log_prob >= best_pm - self.T_threshold
        }

    # ------------------------------------------------------------------
    # Traceback
    # ------------------------------------------------------------------

    def traceback_path(self, pm: FSMPathMetric) -> str:
        """
        Trace back a single path via prev_pm links.

        CRITICAL-4 FIX: Uses prev_pm direct link instead of _best_per_state dict.
        This ensures correct traceback even with multiple candidates per state.
        """
        emitted_bases = []
        current_pm = pm
        while current_pm is not None and current_pm.prev_state is not None:
            if current_pm.emitted_base >= 0:
                emitted_bases.append('ACGT'[current_pm.emitted_base])
            current_pm = current_pm.prev_pm
        emitted_bases.reverse()
        return ''.join(emitted_bases)

    def traceback_all(
        self,
        states: Dict[FSMViterbiState, FSMPathMetricTopK],
        top_k: int = 8,
    ) -> List[Tuple[str, float]]:
        """
        Extract top-K decoded DNA sequences.

        Returns list of (dna_sequence, log_prob) tuples, sorted descending by log_prob.
        """
        # Collect all terminal PMs (states at end of sequence)
        all_pms = []
        for state, topk in states.items():
            for pm in topk.get_all():
                all_pms.append(pm)

        # Sort by log_prob descending
        all_pms.sort(key=lambda x: x.log_prob, reverse=True)
        all_pms = all_pms[:top_k]

        # Trace back each
        candidates = []
        for pm in all_pms:
            dna = self.traceback_path(pm)
            candidates.append((dna, pm.log_prob))

        return candidates
