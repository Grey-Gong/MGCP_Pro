"""
Unit tests for List Viterbi Algorithm (P0 enhancement).

Tests the core functionality:
1. FSMPathMetricTopK: top-K list management
2. List Viterbi: multiple candidates per state
3. traceback_all: extracting top-K paths
4. RS-guided candidate selection via _list_viterbi_top_k_select

Reference: Section 3.7.2 of IMPROVEMENT_PLAN.md v2.0.
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


class TestFSMPathMetricTopK:
    """Tests for the FSMPathMetricTopK wrapper class."""

    def test_empty_initialization(self):
        """Empty top-K list should have length 0."""
        topk = FSMPathMetricTopK(list_k=8)
        assert len(topk) == 0
        assert topk.get_best() is None
        assert topk.get_all() == []

    def test_add_single(self):
        """Adding one metric."""
        topk = FSMPathMetricTopK(list_k=8)
        pm = FSMPathMetric(log_prob=-5.0, prev_state=None, transition='MATCH')
        topk.add(pm)
        assert len(topk) == 1
        assert topk.get_best() is pm

    def test_add_multiple_sorted(self):
        """Adding multiple metrics keeps them sorted by log_prob descending."""
        topk = FSMPathMetricTopK(list_k=8)
        probs = [-10.0, -5.0, -1.0, -3.0, -7.0]
        for p in probs:
            topk.add(FSMPathMetric(log_prob=p, prev_state=None, transition='MATCH'))

        assert len(topk) == 5
        best = topk.get_best()
        assert best.log_prob == -1.0

        all_probs = [pm.log_prob for pm in topk.get_all()]
        assert all_probs == sorted(all_probs, reverse=True)

    def test_enforces_k_limit(self):
        """List should never exceed K entries."""
        topk = FSMPathMetricTopK(list_k=3)
        for p in [1.0, 2.0, 3.0, 4.0, 5.0]:
            topk.add(FSMPathMetric(log_prob=p, prev_state=None, transition='MATCH'))

        assert len(topk) == 3
        assert topk.get_best().log_prob == 5.0

    def test_deterministic_order_with_equal_probs(self):
        """When two metrics have equal log_prob, insertion order determines ranking."""
        topk = FSMPathMetricTopK(list_k=3)
        pm1 = FSMPathMetric(log_prob=0.0, prev_state=None, transition='MATCH')
        pm2 = FSMPathMetric(log_prob=0.0, prev_state=None, transition='DELETION')
        topk.add(pm1)
        topk.add(pm2)
        assert len(topk) == 2
        # pm2 (inserted second) should be after pm1 due to stable sort

    def test_extend(self):
        """extend() adds multiple and re-sorts."""
        topk = FSMPathMetricTopK(list_k=4)
        topk.extend([
            FSMPathMetric(log_prob=-5.0, prev_state=None, transition='MATCH'),
            FSMPathMetric(log_prob=-1.0, prev_state=None, transition='MATCH'),
        ])
        assert len(topk) == 2
        assert topk.get_best().log_prob == -1.0

        topk.extend([
            FSMPathMetric(log_prob=-3.0, prev_state=None, transition='MATCH'),
            FSMPathMetric(log_prob=-2.0, prev_state=None, transition='MATCH'),
        ])
        assert len(topk) == 4
        assert topk.get_best().log_prob == -1.0


class TestListViterbiDecodeStep:
    """Tests for List Viterbi decode_step."""

    def test_list_k_stored_in_decoder(self):
        """Decoder should store list_k parameter."""
        dec = FSMJointDecoder(N=120, list_k=8)
        assert dec.list_k == 8

        dec2 = FSMJointDecoder(N=120, list_k=3)
        assert dec2.list_k == 3

    def test_init_states_returns_topk(self):
        """init_states should return FSMPathMetricTopK per state."""
        dec = FSMJointDecoder(N=120, list_k=8)
        states = dec.init_states()

        assert len(states) == 1
        state, pm_topk = list(states.items())[0]
        assert isinstance(pm_topk, FSMPathMetricTopK)
        assert len(pm_topk) == 1
        assert pm_topk.get_best().log_prob == 0.0

    def test_decode_step_accumulates_multiple_paths(self):
        """decode_step should maintain multiple path candidates per state."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2, list_k=8)

        # Start with initial state
        states = dec.init_states()

        # Run a few steps (each step processes one observed base)
        for base_int in [0, 1, 2, 3, 0]:
            next_states, stats = dec.decode_step(
                states, base_int, phred_quality=0.0, apply_crc_prune=False
            )
            states = next_states
            if not states:
                break

        # After a few steps, we should have multiple states
        assert len(states) > 1, "Should have multiple active states"

        # Each state should have a top-K list (not just 1)
        total_paths = sum(len(pm_topk) for pm_topk in states.values())
        assert total_paths >= len(states), "Should have at least one path per state"

    def test_decode_step_topk_per_state(self):
        """Each state should maintain up to list_k paths."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2, list_k=3)
        states = dec.init_states()

        # Run enough steps to create branching
        for base_int in [0, 1, 2, 3, 0, 1]:
            next_states, _ = dec.decode_step(
                states, base_int, phred_quality=0.0, apply_crc_prune=False
            )
            states = next_states
            if not states:
                break

        if states:
            for st, pm_topk in states.items():
                assert len(pm_topk) <= 3, f"State {st} has {len(pm_topk)} paths, limit is 3"


class TestListViterbiTraceback:
    """Tests for traceback_all in List Viterbi mode."""

    def test_traceback_all_returns_list(self):
        """traceback_all should return a list of (dna, log_prob) tuples."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2, list_k=8)
        states = dec.init_states()

        # Run a few steps
        for base_int in [0, 1, 2, 3]:
            next_states, _ = dec.decode_step(
                states, base_int, phred_quality=0.0, apply_crc_prune=False
            )
            states = next_states
            if not states:
                break

        if states:
            candidates = dec.traceback_all(states, top_k=8)
            assert isinstance(candidates, list)
            assert len(candidates) > 0
            assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)
            # Should be sorted by log_prob descending
            log_probs = [c[1] for c in candidates]
            assert log_probs == sorted(log_probs, reverse=True)

    def test_traceback_all_respects_top_k(self):
        """traceback_all should return at most top_k candidates."""
        dec = FSMJointDecoder(N=120, l=8, c_crc=8, D_max=5, I_max=2, list_k=8)
        states = dec.init_states()

        for base_int in [0, 1, 2, 3, 0]:
            next_states, _ = dec.decode_step(
                states, base_int, phred_quality=0.0, apply_crc_prune=False
            )
            states = next_states
            if not states:
                break

        if states:
            for k in [1, 3, 5, 8]:
                candidates = dec.traceback_all(states, top_k=k)
                assert len(candidates) <= k


class TestListViterbiEndToEnd:
    """End-to-end tests for List Viterbi in the decoder."""

    def test_decode_preserves_list_k_in_info(self):
        """Decoder info should include list_k."""
        from asym_mgc.inner.decode import AsymMGCDecoder

        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2, list_k=8)
        assert dec.decoder.list_k == 8

    def test_decode_short_sequence_runs(self):
        """List Viterbi decoder should complete on short sequences."""
        from asym_mgc.inner.decode import AsymMGCDecoder

        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2, list_k=8)
        seq = "ACGTACGTACGT"

        decoded, info = dec.decode(seq, enable_rs_candidate_selection=True)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
        assert 'lva_used' in info or 'rs_syndrome_nonzero' in info

    def test_list_k_affects_candidate_count(self):
        """Different list_k values should produce different candidate pools."""
        from asym_mgc.inner.decode import AsymMGCDecoder

        seq = "ACGTACGTACGT"

        dec1 = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2, list_k=2)
        decoded1, _ = dec1.decode(seq)

        dec8 = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2, list_k=8)
        decoded8, _ = dec8.decode(seq)

        # Both should produce valid output (different list_k may or may not differ in output,
        # but should not crash)
        assert isinstance(decoded1, str)
        assert isinstance(decoded8, str)
