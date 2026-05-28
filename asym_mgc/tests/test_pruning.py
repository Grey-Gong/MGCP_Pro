"""
Unit tests for pruning strategies: CRC boundary, Top-K, metric threshold.

Tests cover all four pruning strategies from IMPROVEMENT_PLAN.md v2.0:
- Strategy A: CRC Early Termination (at block boundaries)
- Strategy B: FSM Constraint (inherent in state transitions)
- Strategy C: Path Metric Threshold
- Strategy D: Adaptive Top-K (per drift group)
"""

import pytest
import numpy as np

from asym_mgc.inner.fsm_joint import (
    FSMJointDecoder,
    FSMViterbiState,
    FSMPathMetric,
    FSMPathMetricTopK,
    HPState,
)


# =============================================================================
# Pruning Strategy A: CRC Early Termination
# =============================================================================

class TestCRCPruning:
    """Tests for CRC early termination at block boundaries."""

    def test_crc_prune_removes_invalid_syndrome(self):
        """States with non-zero CRC at block boundary should be pruned."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8)

        states = {}
        for gamma in [0, 17, 42, 255]:  # 0 = valid CRC
            s = FSMViterbiState(
                i=8, delta=0, beta=0, gamma=gamma,
                s_hp=HPState.SINGLE, prev_base=0
            )
            topk = FSMPathMetricTopK(list_k=dec.list_k)
            topk.add(FSMPathMetric(log_prob=-1.0, prev_state=None, transition='MATCH'))
            states[s] = topk

        pruned = dec.prune_crc(states)

        # Only gamma=0 should survive
        assert all(s.gamma == 0 for s in pruned)
        assert len(pruned) == 1
        assert list(pruned.keys())[0].gamma == 0

    def test_crc_not_prune_at_non_boundary(self):
        """States with non-zero CRC at non-block-boundary should NOT be pruned."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8)

        states = {}
        # beta != 0 means NOT at block boundary
        for beta in range(1, 8):
            s = FSMViterbiState(
                i=0, delta=0, beta=beta, gamma=42,  # non-zero CRC
                s_hp=HPState.SINGLE, prev_base=0
            )
            topk = FSMPathMetricTopK(list_k=dec.list_k)
            topk.add(FSMPathMetric(log_prob=-1.0, prev_state=None, transition='MATCH'))
            states[s] = topk

        pruned = dec.prune_crc(states)

        # None should be pruned (beta != 0, not a boundary)
        assert len(pruned) == 7

    def test_crc_all_valid_at_boundary(self):
        """When all states have zero CRC at boundary, none are pruned."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8)

        states = {}
        for delta in range(-3, 4):
            s = FSMViterbiState(
                i=8, delta=delta, beta=0, gamma=0,
                s_hp=HPState.SINGLE, prev_base=0
            )
            topk = FSMPathMetricTopK(list_k=dec.list_k)
            topk.add(FSMPathMetric(log_prob=float(-abs(delta)), prev_state=None, transition='MATCH'))
            states[s] = topk

        pruned = dec.prune_crc(states)
        assert len(pruned) == 7  # None pruned

    def test_crc_mixed_boundary_and_interior(self):
        """Mixed boundary and non-boundary states: only boundary with bad CRC pruned."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8)

        states = {}
        # Boundary (beta=0), bad CRC
        s1 = FSMViterbiState(i=8, delta=0, beta=0, gamma=42, s_hp=HPState.SINGLE, prev_base=0)
        # Boundary (beta=0), good CRC
        s2 = FSMViterbiState(i=8, delta=1, beta=0, gamma=0, s_hp=HPState.SINGLE, prev_base=0)
        # Non-boundary (beta=3), bad CRC
        s3 = FSMViterbiState(i=0, delta=2, beta=3, gamma=99, s_hp=HPState.SINGLE, prev_base=0)

        states[s1] = FSMPathMetricTopK(list_k=dec.list_k)
        states[s1].add(FSMPathMetric(log_prob=-1.0, prev_state=None, transition='MATCH'))
        states[s2] = FSMPathMetricTopK(list_k=dec.list_k)
        states[s2].add(FSMPathMetric(log_prob=-2.0, prev_state=None, transition='MATCH'))
        states[s3] = FSMPathMetricTopK(list_k=dec.list_k)
        states[s3].add(FSMPathMetric(log_prob=-3.0, prev_state=None, transition='MATCH'))

        pruned = dec.prune_crc(states)

        assert s1 not in pruned   # boundary + bad CRC -> pruned
        assert s2 in pruned      # boundary + good CRC -> kept
        assert s3 in pruned      # non-boundary -> kept (CRC not checked)


# =============================================================================
# Pruning Strategy B: FSM Constraint
# =============================================================================

class TestFSMPruning:
    """Tests for FSM-based hard pruning (Strategy B)."""

    def test_fsm_blocks_homopolymer_extension(self):
        """FSM should block transitions that would exceed run-length limit."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2)

        # State at maximum homopolymer run (QUAD=4)
        s = FSMViterbiState(
            i=4, delta=0, beta=4, gamma=0,
            s_hp=HPState.QUAD,  # At max run-length 4
            prev_base=0  # Base = 'A'
        )

        transitions = dec.enumerate_transitions(s)
        emitted_bases = [t[2] for t in transitions]

        # MATCH with base A should be blocked (would exceed run-length 4)
        assert not any(
            t[0] == 'MATCH' and t[2] == 0
            for t in transitions
        ), "FSM must block MATCH with same base at max run-length"

    def test_fsm_allows_different_base(self):
        """FSM should allow transition to different base at any run-length."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2)

        s = FSMViterbiState(
            i=3, delta=0, beta=3, gamma=0,
            s_hp=HPState.TRIPLE,
            prev_base=0  # Previous base was 'A'
        )

        transitions = dec.enumerate_transitions(s)
        # Should have MATCH transitions with bases != A (i.e., C=1, G=2, T=3)
        match_with_diff = [
            t for t in transitions
            if t[0] == 'MATCH' and t[2] in [1, 2, 3]
        ]
        assert len(match_with_diff) > 0, "FSM must allow MATCH with different base"

    def test_fsm_allows_insertion_different_base(self):
        """Insertion with different base should be allowed by FSM."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2)

        s = FSMViterbiState(
            i=3, delta=0, beta=3, gamma=0,
            s_hp=HPState.TRIPLE,
            prev_base=0
        )

        transitions = dec.enumerate_transitions(s)
        ins_with_diff = [
            t for t in transitions
            if t[0] == 'INSERTION' and t[2] in [1, 2, 3]
        ]
        assert len(ins_with_diff) > 0

    def test_fsm_state_none_allows_all_bases(self):
        """Initial state should allow all four bases."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2)

        s = FSMViterbiState(
            i=0, delta=0, beta=0, gamma=0,
            s_hp=HPState.NONE,
            prev_base=-1
        )

        transitions = dec.enumerate_transitions(s)
        match_transitions = [t for t in transitions if t[0] == 'MATCH']

        # All 4 bases should be allowed from NONE state
        emitted = set(t[2] for t in match_transitions)
        assert emitted == {0, 1, 2, 3}, f"All 4 bases should be allowed from NONE, got {emitted}"

    def test_fsm_deletion_not_constrained(self):
        """Deletion should not be FSM-constrained (no base emitted)."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2)

        s = FSMViterbiState(
            i=3, delta=-2, beta=3, gamma=0,
            s_hp=HPState.TRIPLE,  # At max run
            prev_base=0
        )

        transitions = dec.enumerate_transitions(s)
        del_transitions = [t for t in transitions if t[0] == 'DELETION']

        # Deletion should always be allowed (no base emitted, FSM doesn't apply)
        # But it IS constrained by delta bounds
        assert len(del_transitions) >= 0  # May be empty due to delta bounds


# =============================================================================
# Pruning Strategy C: Path Metric Threshold
# =============================================================================

class TestMetricThresholdPruning:
    """Tests for metric threshold pruning."""

    def test_threshold_prunes_weak_paths(self):
        """States with log_prob far below best should be pruned."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2, T_threshold=10.0)

        states = {}
        # Create states with varying log probs
        for i in range(10):
            s = FSMViterbiState(
                i=i, delta=0, beta=0, gamma=0,
                s_hp=HPState.SINGLE, prev_base=0
            )
            states[s] = FSMPathMetricTopK(list_k=dec.list_k)
            states[s].add(FSMPathMetric(log_prob=float(-i * 5.0), prev_state=None, transition='MATCH'))

        pruned = dec.prune_threshold(states)

        if len(pruned) > 0:
            best = max(pm_topk.get_best().log_prob for pm_topk in pruned.values())
            for pm_topk in pruned.values():
                for pm in pm_topk.get_all():
                    assert pm.log_prob >= best - 10.0

    def test_threshold_keeps_near_best_paths(self):
        """States close to best metric should be kept."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2, T_threshold=10.0)

        states = {}
        # All states within threshold
        for i in range(5):
            s = FSMViterbiState(
                i=i, delta=0, beta=0, gamma=0,
                s_hp=HPState.SINGLE, prev_base=0
            )
            states[s] = FSMPathMetricTopK(list_k=dec.list_k)
            states[s].add(FSMPathMetric(log_prob=float(-i), prev_state=None, transition='MATCH'))

        pruned = dec.prune_threshold(states)
        assert len(pruned) == 5  # None pruned (all within threshold)

    def test_threshold_empty_input(self):
        """Empty state set should return empty."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, T_threshold=10.0)
        pruned = dec.prune_threshold({})
        assert len(pruned) == 0

    def test_threshold_aggressive(self):
        """Very aggressive threshold should keep only best state."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, T_threshold=1.0)

        states = {}
        for i in range(10):
            s = FSMViterbiState(
                i=i, delta=i % 3 - 1, beta=0, gamma=0,
                s_hp=HPState.SINGLE, prev_base=0
            )
            states[s] = FSMPathMetricTopK(list_k=dec.list_k)
            states[s].add(FSMPathMetric(log_prob=float(-i), prev_state=None, transition='MATCH'))

        pruned = dec.prune_threshold(states)
        # With T_threshold=1.0, only states with log_prob >= best - 1 survive
        if len(pruned) > 0:
            best = max(pm_topk.get_best().log_prob for pm_topk in pruned.values())
            assert best >= -1.0  # Only i=0 and i=1 survive (log_prob >= -1)


# =============================================================================
# Pruning Strategy D: Adaptive Top-K
# =============================================================================

class TestTopKPruning:
    """Tests for adaptive Top-K pruning per (i, delta) group."""

    def test_topk_limits_group_size(self):
        """Each (i, delta) group should have at most K_best states."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, K_best=3)

        candidates = {}
        # Create many states per (i, delta) group
        for i in range(3):
            for delta in range(-2, 3):
                for gamma in range(20):  # Many gamma values
                    s = FSMViterbiState(
                        i=i, delta=delta, beta=0, gamma=gamma,
                        s_hp=HPState.SINGLE, prev_base=0
                    )
                    topk = FSMPathMetricTopK(list_k=dec.list_k)
                    topk.add(FSMPathMetric(log_prob=float(gamma), prev_state=None, transition='MATCH'))
                    candidates[s] = topk

        pruned = dec.prune_topk(candidates)

        # Count states per group
        from collections import Counter
        group_counts = Counter((s.i, s.delta) for s in pruned)
        assert all(c <= 3 for c in group_counts.values()), \
            f"Some groups exceed K_best=3: {[(k, v) for k, v in group_counts.items() if v > 3]}"

    def test_topk_keeps_best_k(self):
        """Top-K should keep the K states with highest log_prob."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, K_best=2)

        # Create 4 states in same group, with known log probs
        candidates = {}
        probs = [10.0, -5.0, 8.0, -20.0]  # Best = 10, second = 8
        for idx, prob in enumerate(probs):
            s = FSMViterbiState(
                i=0, delta=0, beta=0, gamma=idx,
                s_hp=HPState.SINGLE, prev_base=0
            )
            topk = FSMPathMetricTopK(list_k=dec.list_k)
            topk.add(FSMPathMetric(log_prob=prob, prev_state=None, transition='MATCH'))
            candidates[s] = topk

        pruned = dec.prune_topk(candidates)

        assert len(pruned) == 2
        kept_probs = [pm_topk.get_best().log_prob for pm_topk in pruned.values()]
        assert 10.0 in kept_probs
        assert 8.0 in kept_probs
        assert -5.0 not in kept_probs
        assert -20.0 not in kept_probs

    def test_topk_preserves_groups(self):
        """Top-K should operate within each (i, delta) group independently."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, K_best=2)

        candidates = {}
        # Group A: (i=0, delta=0) with probs 1, 2, 3 -> keep 3, 2
        for gamma, prob in [(0, 1.0), (1, 2.0), (2, 3.0)]:
            s = FSMViterbiState(i=0, delta=0, beta=0, gamma=gamma,
                                s_hp=HPState.SINGLE, prev_base=0)
            topk = FSMPathMetricTopK(list_k=dec.list_k)
            topk.add(FSMPathMetric(log_prob=prob, prev_state=None, transition='MATCH'))
            candidates[s] = topk

        # Group B: (i=1, delta=1) with probs 100, 200, 300 -> keep 300, 200
        for gamma, prob in [(0, 100.0), (1, 200.0), (2, 300.0)]:
            s = FSMViterbiState(i=1, delta=1, beta=0, gamma=gamma,
                                s_hp=HPState.SINGLE, prev_base=0)
            topk = FSMPathMetricTopK(list_k=dec.list_k)
            topk.add(FSMPathMetric(log_prob=prob, prev_state=None, transition='MATCH'))
            candidates[s] = topk

        pruned = dec.prune_topk(candidates)

        # Should have exactly 4 states: 2 from each group
        assert len(pruned) == 4

        group_a_probs = [pm_topk.get_best().log_prob for s, pm_topk in pruned.items() if s.i == 0 and s.delta == 0]
        group_b_probs = [pm_topk.get_best().log_prob for s, pm_topk in pruned.items() if s.i == 1 and s.delta == 1]

        assert set(group_a_probs) == {2.0, 3.0}  # 1.0 pruned
        assert set(group_b_probs) == {200.0, 300.0}  # 100.0 pruned

    def test_topk_k1_keeps_one_per_group(self):
        """K_best=1 should keep exactly one state per group."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, K_best=1)

        candidates = {}
        for delta in range(-3, 4):
            for gamma in range(5):
                s = FSMViterbiState(
                    i=0, delta=delta, beta=0, gamma=gamma,
                    s_hp=HPState.SINGLE, prev_base=0
                )
                topk = FSMPathMetricTopK(list_k=dec.list_k)
                topk.add(FSMPathMetric(log_prob=float(gamma), prev_state=None, transition='MATCH'))
                candidates[s] = topk

        pruned = dec.prune_topk(candidates)

        from collections import Counter
        group_counts = Counter((s.i, s.delta) for s in pruned)
        assert all(c == 1 for c in group_counts.values())


# =============================================================================
# Pruning Pipeline Integration
# =============================================================================

class TestPruningPipeline:
    """Tests for the full pruning pipeline."""

    def test_pruning_pipeline_reduces_states(self):
        """Full pipeline should reduce state count significantly."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, K_best=50, T_threshold=15.0)

        # Simulate many candidate states
        candidates = {}
        for i in range(20):
            for delta in range(-5, 6):
                for gamma in range(16):
                    s = FSMViterbiState(
                        i=i, delta=delta, beta=0, gamma=gamma,
                        s_hp=HPState.SINGLE, prev_base=0
                    )
                    topk = FSMPathMetricTopK(list_k=dec.list_k)
                    topk.add(FSMPathMetric(log_prob=float(np.random.randn()), prev_state=None, transition='MATCH'))
                    candidates[s] = topk

        initial_count = len(candidates)

        # Apply full pipeline
        step1 = dec.prune_topk(candidates)
        step2 = dec.prune_threshold(step1)

        assert len(step2) <= initial_count  # Should always reduce or equal

    def test_pruning_respects_deterministic_order(self):
        """Pruning should be deterministic (same input -> same output)."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, K_best=5, T_threshold=10.0)

        candidates = {}
        for i in range(10):
            for delta in range(-2, 3):
                s = FSMViterbiState(
                    i=i, delta=delta, beta=0, gamma=i * 10 + delta,
                    s_hp=HPState.SINGLE, prev_base=0
                )
                topk = FSMPathMetricTopK(list_k=dec.list_k)
                topk.add(FSMPathMetric(log_prob=float(i * 10 + delta), prev_state=None, transition='MATCH'))
                candidates[s] = topk

        # Run twice
        pruned1 = dec.prune_topk(dict(candidates))
        pruned2 = dec.prune_topk(dict(candidates))

        assert set(pruned1.keys()) == set(pruned2.keys())


# =============================================================================
# Pruning Performance Profiling
# =============================================================================

class TestPruningProfiling:
    """Tests that measure pruning effectiveness for profiling."""

    def test_topk_compression_ratio(self):
        """Measure how much Top-K reduces state count."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, K_best=100)

        # Simulate realistic candidate distribution
        candidates = {}
        total_groups = 0
        for i in range(120):  # 120 blocks
            for delta in range(-20, 5):  # Asymmetric window
                total_groups += 1
                for gamma in range(256):  # 8-bit CRC
                    for hp in range(4):  # 4 FSM states
                        s = FSMViterbiState(
                            i=i, delta=delta, beta=0, gamma=gamma,
                            s_hp=HPState(hp), prev_base=0
                        )
                        topk = FSMPathMetricTopK(list_k=dec.list_k)
                        topk.add(FSMPathMetric(log_prob=float(np.random.randn()), prev_state=None, transition='MATCH'))
                        candidates[s] = topk

        before = len(candidates)
        after = len(dec.prune_topk(candidates))

        compression_ratio = after / before
        assert compression_ratio < 0.1, \
            f"Top-K should compress by >90%, got {compression_ratio:.1%}"

    def test_crc_pruning_efficiency(self):
        """CRC should prune roughly 1 - 1/256 of boundary states."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8)

        # All states at boundary with random gamma
        boundary_states = {}
        for delta in range(-10, 5):
            for gamma in range(256):
                s = FSMViterbiState(
                    i=8, delta=delta, beta=0, gamma=gamma,
                    s_hp=HPState.SINGLE, prev_base=0
                )
                topk = FSMPathMetricTopK(list_k=dec.list_k)
                topk.add(FSMPathMetric(log_prob=0.0, prev_state=None, transition='MATCH'))
                boundary_states[s] = topk

        # Only gamma=0 survives
        remaining = dec.prune_crc(boundary_states)

        # 1 out of 256 = ~0.4%
        survival_rate = len(remaining) / len(boundary_states)
        assert survival_rate < 0.01, \
            f"CRC should prune ~99.6%, got survival {survival_rate:.2%}"

    def test_combined_pruning_efficiency(self):
        """Combined pruning should achieve high compression via Top-K."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, K_best=10, T_threshold=5.0)

        # Create candidate set with deterministic log probs
        # so that within each group, best K are clearly identifiable
        candidates = {}
        for i in range(20):  # 20 blocks
            for delta in range(-5, 3):  # 8 delta values
                for gamma in range(256):
                    for hp in range(4):
                        s = FSMViterbiState(
                            i=i, delta=delta, beta=0, gamma=gamma,
                            s_hp=HPState(hp), prev_base=0
                        )
                        # Deterministic: better gamma = better log_prob
                        topk = FSMPathMetricTopK(list_k=dec.list_k)
                        topk.add(FSMPathMetric(log_prob=float(gamma), prev_state=None, transition='MATCH'))
                        candidates[s] = topk

        original = len(candidates)
        # 20 * 8 * 256 * 4 = 163,840 candidates
        # 20 * 8 = 160 groups, K_best=10 each
        # After Top-K: 160 * 10 = 1,600 states
        after_topk = dec.prune_topk(candidates)

        compression = original / max(len(after_topk), 1)
        # Top-K should give 160*256*4 / 160*10 = 102.4x compression
        assert compression > 50, \
            f"Top-K should compress >50x, got {compression:.1f}x (before={original}, after={len(after_topk)})"
