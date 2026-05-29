"""
Unit tests for the FSM-Trellis Joint Decoder.
Tests: state transitions, FSM constraints, pruning, traceback.
"""

import pytest
import numpy as np

from asym_mgc.inner.fsm_joint import (
    HPState,
    FSMViterbiState,
    FSMPathMetric,
    FSMPathMetricTopK,
    FSMJointDecoder,
)


class TestHPState:
    """Test homopolymer FSM state machine."""

    def test_none_to_single(self):
        """Test transition from NONE to SINGLE."""
        next_state = HPState.next(HPState.NONE, 0, None)
        assert next_state == HPState.SINGLE

    def test_extend_homopolymer(self):
        """Test that homopolymer can extend up to length 3."""
        s = HPState.SINGLE
        s = HPState.next(s, 0, 0)  # Same base, run -> 2
        assert s == HPState.DOUBLE

        s = HPState.next(s, 0, 0)  # Same base, run -> 3
        assert s == HPState.TRIPLE

    def test_max_homopolymer_blocks_extension(self):
        """Test that run-length 4 (QUAD) cannot be extended."""
        # s_hp=TRIPLE (3) extending same base gives QUAD (4)
        s = HPState.TRIPLE
        next_state = HPState.next(s, 0, 0)
        assert next_state == HPState.QUAD  # Now allowed

        # s_hp=QUAD (4) extending same base is blocked
        s = HPState.QUAD
        next_state = HPState.next(s, 0, 0)
        assert next_state is None  # Blocked by FSM

    def test_base_change_resets(self):
        """Test that changing base resets run-length to 1."""
        s = HPState.DOUBLE
        next_state = HPState.next(s, 1, 0)  # Different base
        assert next_state == HPState.SINGLE


class TestFSMViterbiState:
    """Test the joint FSM-Drift state."""

    def test_state_creation(self):
        """Test that FSMViterbiState can be created."""
        s = FSMViterbiState(
            i=0, delta=0, beta=0, gamma=0,
            s_hp=HPState.NONE, prev_base=-1
        )
        assert s.i == 0
        assert s.delta == 0
        assert s.s_hp == HPState.NONE

    def test_state_immutability_concept(self):
        """Test that state repr is informative."""
        s = FSMViterbiState(
            i=5, delta=-3, beta=2, gamma=42,
            s_hp=HPState.DOUBLE, prev_base=0
        )
        r = repr(s)
        assert 'i=5' in r
        assert 'Δ=-3' in r
        assert 'DOUBLE' in r


class TestFSMJointDecoder:
    """Test the FSM-Viterbi joint decoder."""

    def test_init_states(self):
        """Test decoder initialization."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=20, I_max=4)
        states = dec.init_states()

        assert len(states) == 1
        start_state = list(states.keys())[0]
        assert start_state.i == 0
        assert start_state.delta == 0
        assert start_state.s_hp == HPState.NONE

    def test_enumerate_transitions_no_pruning(self):
        """Test that transitions are enumerated without errors."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=20, I_max=4)
        start = FSMViterbiState(
            i=0, delta=0, beta=0, gamma=0,
            s_hp=HPState.NONE, prev_base=-1
        )

        transitions = dec.enumerate_transitions(start)

        assert len(transitions) > 0
        trans_types = set(t[0] for t in transitions)
        assert 'MATCH' in trans_types
        assert 'DELETION' in trans_types
        assert 'INSERTION' in trans_types

    def test_fsm_constrains_transitions(self):
        """Test that FSM prevents excessive homopolymer runs."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=20, I_max=4)

        # Create a state at max homopolymer run (QUAD=4)
        s = FSMViterbiState(
            i=4, delta=0, beta=4, gamma=0,
            s_hp=HPState.QUAD, prev_base=0  # At run-length 4, base=0 (A)
        )

        transitions = dec.enumerate_transitions(s)

        # MATCH with same base (A) should be blocked by FSM
        same_base_match = [
            t for t in transitions
            if t[0] == 'MATCH' and t[2] == 0  # t[2]=emitted_base
        ]
        assert len(same_base_match) == 0, f"FSM should block same-base MATCH: {same_base_match}"

        # MATCH with different base should be allowed (emitted != prev_base)
        diff_base_match = [
            t for t in transitions
            if t[0] == 'MATCH' and t[2] != 0  # t[2]=emitted_base != prev_base=0
        ]
        assert len(diff_base_match) > 0, f"FSM should allow different-base MATCH: {diff_base_match}"

    def test_delta_bounds_enforced(self):
        """Test that delta bounds are enforced in transitions."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2)

        # Start with max negative delta
        s = FSMViterbiState(
            i=0, delta=-5, beta=0, gamma=0,
            s_hp=HPState.NONE, prev_base=-1
        )

        transitions = dec.enumerate_transitions(s)

        # DELETION should be blocked (would go beyond -D_max)
        del_transitions = [t for t in transitions if t[0] == 'DELETION']
        assert len(del_transitions) == 0

    def test_topk_pruning(self):
        """Test Top-K pruning reduces state count."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=20, I_max=4, K_best=5)

        # Create many candidate states
        candidates = {}
        for i in range(20):
            for delta in range(-5, 6):
                s = FSMViterbiState(
                    i=i, delta=delta, beta=0, gamma=i * 17 + delta,
                    s_hp=HPState.SINGLE, prev_base=0
                )
                pm = FSMPathMetric(log_prob=np.random.random(), prev_state=None, transition='MATCH')
                topk = FSMPathMetricTopK(list_k=dec.list_k)
                topk.add(pm)
                candidates[s] = topk

        pruned = dec.prune_topk(candidates)

        # Each (i, delta) group should have at most K_best states
        from collections import Counter
        group_counts = Counter((s.i, s.delta) for s in pruned)
        assert all(c <= 5 for c in group_counts.values())

    def test_threshold_pruning(self):
        """Test metric threshold pruning."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=20, I_max=4, T_threshold=10.0)

        candidates = {}
        for i in range(10):
            s = FSMViterbiState(
                i=i, delta=0, beta=0, gamma=0,
                s_hp=HPState.NONE, prev_base=-1
            )
            pm = FSMPathMetric(log_prob=-i * 5.0, prev_state=None, transition='MATCH')
            topk = FSMPathMetricTopK(list_k=dec.list_k)
            topk.add(pm)
            candidates[s] = topk

        pruned = dec.prune_threshold(candidates)

        if len(pruned) > 0:
            best_prob = max(pm_topk.get_best().log_prob for pm_topk in pruned.values())
            for pm_topk in pruned.values():
                for pm in pm_topk.get_all():
                    assert pm.log_prob >= best_prob - 10.0

    def test_crc_pruning(self):
        """Test CRC early termination pruning."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=20, I_max=4)

        # CRC boundary: beta==0 AND i >= l (first l=8 steps complete).
        # After exactly l steps, beta wraps back to 0. So boundaries at i=8,16,24,...
        # Note: i=0,beta=0 is the initial state (before any step) - not a boundary.
        states = {}
        for i in [8, 16]:  # CRC block boundaries
            for gamma in [0, 17, 42]:  # 0 = valid CRC, others invalid
                s = FSMViterbiState(
                    i=i, delta=0, beta=0, gamma=gamma,
                    s_hp=HPState.NONE, prev_base=-1
                )
                pm = FSMPathMetric(log_prob=float(-i), prev_state=None, transition='MATCH')
                topk = FSMPathMetricTopK(list_k=dec.list_k)
                topk.add(pm)
                states[s] = topk

        # Also add non-boundary states (should NOT be pruned)
        for i, beta in [(7, 7), (4, 4), (3, 3), (0, 0)]:
            s = FSMViterbiState(
                i=i, delta=0, beta=beta, gamma=42,
                s_hp=HPState.NONE, prev_base=-1
            )
            pm = FSMPathMetric(log_prob=float(-i), prev_state=None, transition='MATCH')
            topk = FSMPathMetricTopK(list_k=dec.list_k)
            topk.add(pm)
            states[s] = topk

        pruned = dec.prune_crc(states)

        # Only gamma=0 states at CRC block boundaries survive.
        # All non-boundary states (even with gamma!=0) must survive.
        for s in pruned:
            at_boundary = (s.beta == 0) and (s.i >= dec.l)
            if at_boundary:
                assert s.gamma == 0, f"gamma={s.gamma} should be 0 at boundary i={s.i}"

        # All boundary states with gamma!=0 must be pruned (not present in pruned)
        boundary_bad = sum(1 for s in pruned
                          if (s.beta == 0) and (s.i >= dec.l) and (s.gamma != 0))
        assert boundary_bad == 0, f"Bad CRC states survived at boundaries: {boundary_bad}"

        # Boundary states with gamma==0 must survive
        boundary_good = sum(1 for s in pruned if (s.beta == 0) and (s.i >= dec.l) and (s.gamma == 0))
        assert boundary_good == 2, f"Expected 2 boundary states with good CRC, got {boundary_good}"

    def test_branch_metric_match(self):
        """Test branch metric for MATCH transitions."""
        # Use moderate error rates where correct matches are possible
        dec = FSMJointDecoder(Pd=0.1, Pi=0.02, Ps=0.05)

        # Match: correct base, positive LLR should boost the metric
        bm = dec.branch_metric('MATCH', emitted_base=0, observed_base=0, phred_quality=5.0)
        # Should be positive: LLR boost > 0 and P_CORR > 0
        assert bm > 0

        # Match: wrong base, negative LLR should penalize
        bm_wrong = dec.branch_metric('MATCH', emitted_base=0, observed_base=1, phred_quality=5.0)
        assert bm_wrong < 0

        # The wrong-base metric should be lower than correct-base metric
        assert bm_wrong < bm

    def test_branch_metric_deletion(self):
        """Test branch metric for DELETION transitions."""
        dec = FSMJointDecoder(Pd=0.5, Pi=0.03, Ps=0.47)

        bm = dec.branch_metric('DELETION', -1, 0, 0.0)
        assert bm < 0  # Log probability should be negative

    def test_branch_metric_insertion(self):
        """Test branch metric for INSERTION transitions."""
        dec = FSMJointDecoder(Pd=0.5, Pi=0.03, Ps=0.47)

        bm = dec.branch_metric('INSERTION', 0, 0, 0.0)
        assert bm < 0  # Log probability should be negative
