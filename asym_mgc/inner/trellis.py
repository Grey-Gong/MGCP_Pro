"""
Viterbi Trellis and State Definitions for Asym-MGC.

Implements the core dynamic programming structure for the asymmetric
trellis decoder with joint (drift, FSM) state space.
Reference: Section 3.2 of IMPROVEMENT_PLAN.md v2.0.

Phase 1.01-1.04: Sections 3.1-3.2 of IMPROVEMENT_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np


class HomopolymerState(IntEnum):
    """
    Homopolymer run-length FSM states.

    v2.1: max_run=4 (QUAD added), matching ConstrainedRSEncoder.max_run=4.

    S_hp = 0: no previous base (start state)
    S_hp = 1: previous base is different (run length = 1)
    S_hp = 2: run length = 2
    S_hp = 3: run length = 3
    S_hp = 4: run length = 4 (MAX, cannot extend further)
    """
    NONE = 0      # Start state, no previous base
    SINGLE = 1    # Previous base differs, run-length = 1
    DOUBLE = 2    # Homopolymer run-length = 2
    TRIPLE = 3   # Homopolymer run-length = 3
    QUAD = 4     # Homopolymer run-length = 4 (MAX in v2.1)
    MAX_RUN = 4  # Alias for convenience

    @classmethod
    def next(cls, current: "HomopolymerState", base_int: int, prev_base_int: int) -> Optional["HomopolymerState"]:
        """
        Compute the next homopolymer FSM state.

        Returns None if the transition is invalid (would exceed max run-length).
        """
        if current == cls.NONE:
            return cls.SINGLE
        if base_int == prev_base_int:
            if current.value < cls.MAX_RUN:
                return cls(current.value + 1)
            else:
                return None  # Cannot extend beyond MAX_RUN
        else:
            return cls.SINGLE


@dataclass(frozen=True, eq=False)
class TrellisState:
    """
    Immutable state in the Asym-MGC Viterbi trellis.

    State tuple: (i, delta, beta, gamma, S_hp, prev_base)
    Equality and hash include prev_base so that different paths (e.g. MATCH vs DELETION
    producing the same delta/gamma but with different prev_base) are kept separate.
    """
    i: int          # Number of input symbols processed (0 to N)
    delta: int      # Net deletion offset: -(deletions - insertions)
    beta: int       # Block-internal symbol offset (0 to l-1)
    gamma: int      # CRC syndrome for current block (0 to 2^c_crc - 1)
    S_hp: HomopolymerState  # Homopolymer run-length FSM state
    prev_base: int = -1  # Previously emitted base (0-3), -1 = unknown (start state)

    def __hash__(self) -> int:
        return hash((self.i, self.delta, self.beta, self.gamma, self.S_hp, self.prev_base))

    def __eq__(self, other) -> bool:
        if not isinstance(other, TrellisState):
            return NotImplemented
        return (
            self.i == other.i
            and self.delta == other.delta
            and self.beta == other.beta
            and self.gamma == other.gamma
            and self.S_hp == other.S_hp
            and self.prev_base == other.prev_base
        )

    def __repr__(self) -> str:
        return (
            f"TrellisState(i={self.i}, "
            f"Δ={self.delta:+d}, "
            f"β={self.beta}, "
            f"γ=0x{self.gamma:02x}, "
            f"S_hp={self.S_hp.name})"
        )


@dataclass
class PathMetric:
    """
    Viterbi path metric with accumulated log-probability and backpointer.
    """
    log_prob: float
    prev_state: Optional[TrellisState]
    best_transition: Optional[str] = None  # 'MATCH', 'DELETION', 'INSERTION'
    cum_deletions: int = 0
    cum_insertions: int = 0
    prev_base: str = 'N'

    def __repr__(self) -> str:
        return f"PathMetric(log_prob={self.log_prob:.2f}, Δ={self.cum_deletions - self.cum_insertions:+d})"


class ViterbiTrellis:
    """
    Asymmetric Viterbi decoder with joint (delta, FSM) state space.

    This implements the DP for decoding against the memory-k nanopore channel.
    Key features:
    - Asymmetric drift window: more states for deletions, fewer for insertions
    - Joint FSM-Drift state: homopolymer constraints integrated into state space
    - Three pruning strategies: CRC, metric threshold, Top-K

    Parameters
    ----------
    N : int
        Number of information symbols per strand.
    l : int
        Symbols per RS block.
    c_crc : int
        Number of CRC check bits.
    D_max : int
        Maximum deletion drift.
    I_max : int
        Maximum insertion drift.
    Pd : float
        Deletion probability.
    Pi : float
        Insertion probability.
    Ps : float
        Substitution probability.
    K_best : int
        Maximum states per (i, delta) group (Top-K pruning).
    T_threshold : float
        Path metric threshold for pruning.
    crc_polynomial : int
        CRC generator polynomial (default 0x107 for CRC-8).
    """

    DNA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    DNA_INV = {v: k for k, v in DNA_MAP.items()}

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
        K_best: int = 200,
        T_threshold: float = 15.0,
        crc_polynomial: int = 0x107,
    ):
        self.N = N
        self.l = l
        self.c_crc = c_crc
        self.D_max = D_max
        self.I_max = I_max
        self.Pd = Pd
        self.Pi = Pi
        self.Ps = Ps
        self.K_best = K_best
        self.T_threshold = T_threshold
        self.crc_polynomial = crc_polynomial

        # Derived
        self.num_blocks = N // l
        self.crc_mask = (1 << c_crc) - 1

        # Transition log-probabilities (precomputed)
        self.log_P_DEL = np.log(Pd * (1 - Pi - Ps) + 1e-12)
        self.log_P_INS = np.log(Pi * (1 - Pd - Ps) + 1e-12)
        self.log_P_CORR = np.log((1 - Pd - Pi - Ps) + 1e-12)
        self.log_P_SUB = np.log(Ps + 1e-12)

        # Valid delta range
        self.min_delta = -D_max
        self.max_delta = I_max
        self.delta_size = D_max + I_max + 1

    def init_trellis(self) -> Dict[TrellisState, PathMetric]:
        """Initialize the trellis with starting state."""
        start_state = TrellisState(
            i=0, delta=0, beta=0, gamma=0, S_hp=HomopolymerState.NONE, prev_base=-1
        )
        return {start_state: PathMetric(log_prob=0.0, prev_state=None)}

    def delta_to_idx(self, delta: int) -> int:
        """Map delta to array index."""
        return delta + self.D_max

    def idx_to_delta(self, idx: int) -> int:
        """Map array index to delta."""
        return idx - self.D_max

    def update_crc(self, syndrome: int, base: int) -> int:
        """Update CRC syndrome with a new data bit."""
        new_syndrome = ((syndrome << 1) | base) & self.crc_mask
        for _ in range(8):
            if new_syndrome & (1 << (crc_bits := self.c_crc - 1)):
                new_syndrome ^= self.crc_polynomial
            new_syndrome = (new_syndrome << 1) & self.crc_mask
        return new_syndrome

    def _crc_update_fast(self, syndrome: int, base_bits: int) -> int:
        """
        Update CRC syndrome with multiple bits at once.

        For efficiency, we update per base (2 bits in DNA encoding).
        """
        s = syndrome
        for _ in range(2):  # 2 bits per DNA base
            bit = base_bits & 1
            s = ((s << 1) | bit) & self.crc_mask
            if s & (1 << (self.c_crc - 1)):
                s ^= self.crc_polynomial
            base_bits >>= 1
        return s

    def get_valid_transitions(
        self, state: TrellisState
    ) -> List[Tuple[str, int, HomopolymerState, int, int, int, int]]:
        """
        Enumerate valid transitions from the current state.

        Returns list of (trans_type, next_delta, next_S_hp, next_i, next_beta, next_gamma, base_idx).

        Transition types:
        - MATCH: consume one input symbol, output matches
        - DELETION: consume one input symbol, output skips it
        - INSERTION: insert a spurious symbol without consuming input
        """
        transitions = []

        # --- MATCH ---
        for base_idx in range(4):  # A=0, C=1, G=2, T=3
            next_beta = (state.beta + 1) % self.l
            next_gamma = self._crc_update_fast(state.gamma, base_idx)

            # Block boundary: reset beta, increment i
            if next_beta == 0:
                next_i = state.i + 1
            else:
                next_i = state.i

            # Delta unchanged for MATCH
            next_delta = state.delta

            # Homopolymer FSM: check if this transition is valid
            # prev_base is now stored as int in TrellisState (-1 for unknown)
            prev_base_char = self.DNA_INV.get(state.prev_base, 'N')
            next_S_hp = HomopolymerState.next_state(state.S_hp, self.DNA_INV.get(base_idx, base_idx), prev_base_char)
            if next_S_hp is not None:
                transitions.append(('MATCH', next_delta, next_S_hp, next_i, next_beta, next_gamma, base_idx))

        # --- DELETION ---
        # Consume one input symbol, but output has nothing for it
        for base_idx in range(4):
            next_delta = state.delta - 1  # Net deletion
            if self.min_delta <= next_delta <= self.max_delta:
                next_beta = (state.beta + 1) % self.l
                next_gamma = self._crc_update_fast(state.gamma, base_idx)
                if next_beta == 0:
                    next_i = state.i + 1
                else:
                    next_i = state.i
                # Deletion consumes a base, so FSM state advances
                prev_base_char = self.DNA_INV.get(state.prev_base, 'N')
                next_S_hp = HomopolymerState.next_state(state.S_hp, self.DNA_INV.get(base_idx, base_idx), prev_base_char)
                if next_S_hp is not None:
                    transitions.append(('DELETION', next_delta, next_S_hp, next_i, next_beta, next_gamma, base_idx))

        # --- INSERTION ---
        # Insert a spurious base (does not consume input)
        for base_idx in range(4):
            next_delta = state.delta + 1  # Net insertion
            if self.min_delta <= next_delta <= self.max_delta:
                # beta, i, gamma unchanged (no input consumed)
                # Inserted base goes through FSM, so S_hp may advance
                prev_base_char = self.DNA_INV.get(state.prev_base, 'N')
                next_S_hp = HomopolymerState.next_state(state.S_hp, self.DNA_INV.get(base_idx, base_idx), prev_base_char)
                if next_S_hp is not None:
                    transitions.append(('INSERTION', next_delta, next_S_hp, state.i, state.beta, state.gamma, base_idx))

        return transitions

    def viterbi_step(
        self,
        current_states: Dict[TrellisState, PathMetric],
        observed_base: int,
        observed_log_prob: float,
    ) -> Tuple[Dict[TrellisState, PathMetric], dict]:
        """
        Execute one Viterbi step.

        Parameters
        ----------
        current_states : dict
            Active states from the previous step.
        observed_base : int
            Received base as integer 0-3.
        observed_log_prob : float
            Log-probability of this observation (from basecaller LLR).

        Returns
        -------
        next_states : dict
            Updated active states.
        stats : dict
            Statistics for this step.
        """
        candidates: Dict[TrellisState, PathMetric] = {}

        for state, pm in current_states.items():
            for trans in self.get_valid_transitions(state):
                trans_type, next_delta, next_S_hp, next_i, next_beta, next_gamma, base_idx = trans

                # Compute branch metric
                if trans_type == 'MATCH':
                    if observed_base == base_idx:
                        branch_metric = observed_log_prob + self.log_P_CORR
                    else:
                        branch_metric = -observed_log_prob + self.log_P_SUB
                elif trans_type == 'DELETION':
                    branch_metric = self.log_P_DEL
                elif trans_type == 'INSERTION':
                    branch_metric = self.log_P_INS

                new_pm = PathMetric(
                    log_prob=pm.log_prob + branch_metric,
                    prev_state=state,
                    best_transition=trans_type,
                    cum_deletions=pm.cum_deletions + (1 if trans_type == 'DELETION' else 0),
                    cum_insertions=pm.cum_insertions + (1 if trans_type == 'INSERTION' else 0),
                    prev_base=self.DNA_INV.get(base_idx, 'N'),
                )

                next_state = TrellisState(
                    i=next_i,
                    delta=next_delta,
                    beta=next_beta,
                    gamma=next_gamma,
                    S_hp=next_S_hp,
                    prev_base=base_idx,
                )
                if next_state in candidates:
                    if new_pm.log_prob > candidates[next_state].log_prob:
                        candidates[next_state] = new_pm
                else:
                    candidates[next_state] = new_pm

        # --- Pruning ---
        # Strategy A: Top-K per (i, delta) group
        grouped: Dict[Tuple[int, int], List[Tuple[TrellisState, PathMetric]]] = {}
        for st, pm in candidates.items():
            key = (st.i, st.delta)
            grouped.setdefault(key, []).append((st, pm))

        next_states = {}
        for key, candidates_list in grouped.items():
            if len(candidates_list) > self.K_best:
                sorted_list = sorted(candidates_list, key=lambda x: x[1].log_prob, reverse=True)
                for st, pm in sorted_list[:self.K_best]:
                    next_states[st] = pm
            else:
                for st, pm in candidates_list:
                    next_states[st] = pm

        # Strategy B: Metric threshold pruning
        if next_states:
            best_pm = max(pm.log_prob for pm in next_states.values())
            next_states = {
                st: pm for st, pm in next_states.items()
                if pm.log_prob >= best_pm - self.T_threshold
            }

        stats = {
            'num_candidates': len(candidates),
            'num_active': len(next_states),
            'num_groups': len(grouped),
        }
        return next_states, stats


class TrellisTransitions:
    """
    Precomputed transition table for the Viterbi trellis.

    This class precomputes all valid state transitions for efficiency,
    since the transition rules are purely deterministic.
    """

    def __init__(
        self,
        N: int,
        l: int,
        c_crc: int,
        D_max: int,
        I_max: int,
        crc_polynomial: int = 0x107,
    ):
        self.N = N
        self.l = l
        self.c_crc = c_crc
        self.D_max = D_max
        self.I_max = I_max
        self.crc_polynomial = crc_polynomial
        self.crc_mask = (1 << c_crc) - 1

        # We don't precompute all transitions here because they depend
        # on the base being processed (for CRC update). Instead, we
        # provide efficient transition enumeration.

    def enumerate_match_transitions(
        self, state: TrellisState, base_idx: int
    ) -> TrellisState:
        """Create next state for a MATCH transition."""
        next_beta = (state.beta + 1) % self.l
        next_gamma = self._crc_update(state.gamma, base_idx)
        next_i = state.i if next_beta != 0 else state.i + 1
        return TrellisState(
            i=min(next_i, self.N),
            delta=state.delta,
            beta=next_beta,
            gamma=next_gamma,
            S_hp=state.S_hp,
        )

    def enumerate_deletion_transitions(
        self, state: TrellisState, base_idx: int
    ) -> Optional[TrellisState]:
        """Create next state for a DELETION transition, or None if out of bounds."""
        next_delta = state.delta - 1
        if not (-self.D_max <= next_delta <= self.I_max):
            return None
        next_beta = (state.beta + 1) % self.l
        next_gamma = self._crc_update(state.gamma, base_idx)
        next_i = state.i if next_beta != 0 else state.i + 1
        return TrellisState(
            i=min(next_i, self.N),
            delta=next_delta,
            beta=next_beta,
            gamma=next_gamma,
            S_hp=state.S_hp,
        )

    def enumerate_insertion_transitions(
        self, state: TrellisState
    ) -> List[TrellisState]:
        """Create next states for INSERTION transitions (one per possible base)."""
        next_delta = state.delta + 1
        if not (-self.D_max <= next_delta <= self.I_max):
            return []
        return [
            TrellisState(
                i=state.i,
                delta=next_delta,
                beta=state.beta,
                gamma=state.gamma,
                S_hp=state.S_hp,
            )
        ]

    def _crc_update(self, syndrome: int, base_idx: int) -> int:
        """Update CRC syndrome with a DNA base (2 bits)."""
        s = syndrome
        base_bits = base_idx  # A=0, C=1, G=2, T=3 maps to 00, 01, 10, 11
        for _ in range(2):
            bit = base_bits & 1
            s = ((s << 1) | bit) & self.crc_mask
            if s & (1 << (self.c_crc - 1)):
                s ^= self.crc_polynomial
            base_bits >>= 1
        return s
