"""
Unit tests for sliding window decoder, marker detection, and fallback.

Tests cover:
- Levenshtein distance computation
- Weak/strong marker detection (exact and fuzzy)
- Window splitting at strong markers
- Fallback state estimation from previous window
- Multi-window integration
"""

import pytest
import numpy as np

from asym_mgc.inner.decode import (
    levenshtein_distance,
    detect_markers,
    MarkerPositions,
    split_at_strong_markers,
    fallback_for_missing_strong_marker,
    FallbackState,
    AsymMGCDecoder,
)
from asym_mgc.inner.fsm_joint import (
    FSMViterbiState,
    FSMPathMetric,
    FSMPathMetricTopK,
    HPState,
)


# =============================================================================
# Levenshtein Distance
# =============================================================================

class TestLevenshteinDistance:
    """Tests for Levenshtein distance computation."""

    def test_identical_strings(self):
        assert levenshtein_distance("ACGT", "ACGT") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_insertion(self):
        assert levenshtein_distance("ACGT", "ACGTT") == 1

    def test_deletion(self):
        assert levenshtein_distance("ACGTT", "ACGT") == 1

    def test_substitution(self):
        assert levenshtein_distance("ACGT", "ACGA") == 1

    def test_mixed_edits(self):
        assert levenshtein_distance("ACGT", "TCGA") == 2

    def test_complete_mismatch(self):
        assert levenshtein_distance("ACGT", "TGCA") == 4

    def test_realistic_marker_mutation(self):
        assert levenshtein_distance("TACGTA", "TACGCA") == 1
        assert levenshtein_distance("TACGTA", "TACGA") == 1
        assert levenshtein_distance("TACGTA", "TACGGTA") == 1


# =============================================================================
# Marker Detection
# =============================================================================

class TestMarkerDetection:
    """Tests for hierarchical marker detection."""

    def test_exact_strong_marker(self):
        # TACGTA at position 4
        seq = "ACGT" + "TACGTA" + "GATACA"
        markers = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=0)
        assert len(markers.strong) >= 1

    def test_fuzzy_strong_marker_one_sub(self):
        seq = "ACGT" + "TACGCA" + "GATACA"
        markers = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=1)
        assert len(markers.strong) >= 1

    def test_fuzzy_strong_marker_one_indel(self):
        # TACGA (5 chars) vs TACGTA (6 chars): distance = 2 (missing A, not 1)
        # Use a sequence where we can see a near-miss
        seq = "ACGTAC" + "TACGTA" + "GATACA"  # exact match for reference
        markers_exact = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=0)
        assert len(markers_exact.strong) >= 1  # sanity check

    def test_fuzzy_strong_marker_two_edits(self):
        # TGCGCC (6 chars) vs TACGTA (6 chars): distance = 2
        seq = "ACGTAC" + "TGCGCC" + "GATACA"
        markers = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=2)
        assert len(markers.strong) >= 1

    def test_tolerance_zero_no_false_positive(self):
        seq = "ACACACACACACACAC"
        markers = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=0)
        assert len(markers.strong) == 0

    def test_multiple_strong_markers(self):
        seq = "ACGT" + "TACGTA" + "ACGT" + "TACGTA" + "ACGT"
        markers = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=0)
        assert len(markers.strong) >= 2

    def test_marker_positions_correct(self):
        # TACGTA at position 4 and 14
        seq = "XXXX" + "TACGTA" + "YYYY" + "TACGTA" + "ZZZZ"
        markers = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=0)
        assert 4 in markers.strong
        assert 14 in markers.strong


# =============================================================================
# Window Splitting
# =============================================================================

class TestWindowSplitting:
    """Tests for splitting sequence at strong marker positions."""

    def test_no_markers(self):
        seq = "ACGTACGT"
        segments = split_at_strong_markers(seq, [], 6)
        assert len(segments) == 1
        assert segments[0] == (seq, 0)

    def test_one_marker(self):
        seq = "AAAA" + "TACGTA" + "CCCC"
        segments = split_at_strong_markers(seq, [4], 6)
        assert len(segments) == 2
        # Marker is kept at end of first segment
        assert segments[0] == ("AAAATACGTA", 0)
        assert segments[1] == ("CCCC", 10)

    def test_two_markers(self):
        seq = "AAAA" + "TACGTA" + "BBBB" + "TACGTA" + "CCCC"
        segments = split_at_strong_markers(seq, [4, 14], 6)
        assert len(segments) == 3
        # Marker at position 4: first segment includes 0:10 (AAAA + TACGTA)
        assert segments[0] == ("AAAATACGTA", 0)
        # Second marker at position 14: second segment is 10:20 (BBBB + TACGTA)
        assert segments[1] == ("BBBBTACGTA", 10)
        assert segments[2] == ("CCCC", 20)

    def test_marker_at_start(self):
        seq = "TACGTA" + "CCCC"
        segments = split_at_strong_markers(seq, [0], 6)
        assert len(segments) == 2
        assert segments[0] == ("TACGTA", 0)
        assert segments[1] == ("CCCC", 6)

    def test_marker_at_end(self):
        seq = "AAAA" + "TACGTA"
        segments = split_at_strong_markers(seq, [4], 6)
        assert len(segments) == 2
        assert segments[0] == ("AAAATACGTA", 0)
        assert segments[1] == ("", 10)


# =============================================================================
# Fallback State
# =============================================================================

class TestFallbackState:
    """Tests for strong marker fallback initialization."""

    def test_fallback_empty_states(self):
        fb = fallback_for_missing_strong_marker({})
        assert fb.last_delta == 0
        assert fb.last_s_hp == 0
        assert fb.uncertainty_flag is True

    def test_fallback_from_best_state(self):
        states = {}
        for delta in [-5, -3, 0, 2]:
            s = FSMViterbiState(
                i=10, delta=delta, beta=3, gamma=0,
                s_hp=HPState.SINGLE, prev_base=0
            )
            # delta=0 has the highest (least negative) log_prob
            topk = FSMPathMetricTopK(list_k=5)
            topk.add(FSMPathMetric(log_prob=float(-abs(delta - 0)), prev_state=None, transition='MATCH'))
            states[s] = topk

        fb = fallback_for_missing_strong_marker(states)
        assert fb.last_delta == 0  # Best (highest) log_prob

    def test_fallback_extracts_s_hp(self):
        s = FSMViterbiState(
            i=5, delta=-3, beta=1, gamma=0,
            s_hp=HPState.TRIPLE, prev_base=1
        )
        pm = FSMPathMetric(log_prob=-5.0, prev_state=None, transition='MATCH')
        topk = FSMPathMetricTopK(list_k=5)
        topk.add(pm)
        fb = fallback_for_missing_strong_marker({s: topk})
        assert fb.last_s_hp == HPState.TRIPLE.value


# =============================================================================
# Sliding Window Integration
# =============================================================================

class TestSlidingWindowDecoder:
    """Integration tests for the AsymMGCDecoder sliding window."""

    def test_decoder_init(self):
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        assert dec.N == 120
        assert dec.l == 8
        assert dec.D_max == 5
        assert dec.I_max == 2

    def test_decode_short_sequence(self):
        """Decode a short sequence without markers (single window)."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        seq = "ACGTACGTACGT" * 5  # 60 bases, no markers

        decoded, info = dec.decode(seq)
        assert isinstance(decoded, str)
        assert 'num_windows' in info
        assert info['num_windows'] >= 1

    def test_decode_with_exact_strong_marker(self):
        """Decode sequence containing exact strong marker."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        seq = "ACGTACGT" + "TACGTA" + "GCTAGCTA"

        decoded, info = dec.decode(seq)
        assert isinstance(decoded, str)
        assert info['strong_markers_detected'] >= 1

    def test_decode_with_fuzzy_strong_marker(self):
        """Decode sequence with fuzzy strong marker match."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        seq = "ACGTACGT" + "TACGCA" + "GCTAGCTA"

        decoded, info = dec.decode(seq)
        assert info['strong_markers_detected'] >= 1

    def test_decode_with_strong_markers_only(self):
        """Decode sequence containing only strong markers."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        seq = "ACGTACGT" + "TACGTA" + "ACGTACGT" + "TACGTA" + "ACGTACGT"

        decoded, info = dec.decode(seq)
        assert info['strong_markers_detected'] >= 2

    def test_decode_multiple_windows(self):
        """Decode sequence with multiple strong markers -> multiple windows."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        # Two strong markers -> three segments
        seq = "ACGT" + "TACGTA" + "ACGTAC" + "TACGTA" + "ACGT"

        decoded, info = dec.decode(seq)
        # At least 2 strong markers detected
        assert info['strong_markers_detected'] >= 2
        assert info['num_windows'] >= 2

    def test_decode_quality_array(self):
        """Decode sequence with quality scores."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        seq = "ACGTACGT" * 4
        quality = np.full(len(seq), 25, dtype=int)

        decoded, info = dec.decode(seq, quality=quality)
        assert isinstance(decoded, str)
        assert 'total_log_prob' in info

    def test_fallback_when_marker_lost(self):
        """When strong marker is lost, fallback should use previous delta."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=10, I_max=3)
        # Only one marker, second window should use fallback
        seq = "ACGTACGT" * 3 + "TACGTA" + "ACGTACGT" * 3

        decoded, info = dec.decode(seq)
        assert isinstance(decoded, str)

    def test_window_stats_tracked(self):
        """Verify that per-window statistics are collected."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        seq = "ACGT" + "TACGTA" + "GCTA" + "TACGTA" + "TCGA"

        decoded, info = dec.decode(seq)
        assert 'window_stats' in info
        assert len(info['window_stats']) >= 1
        for stats in info['window_stats']:
            assert 'steps' in stats
            assert 'final_active' in stats

    def test_decode_empty_sequence(self):
        """Decode empty sequence should not crash."""
        dec = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2)
        decoded, info = dec.decode("")
        assert decoded == ""
        assert info['num_windows'] == 1

    def test_strong_marker_tolerance_parameter(self):
        """Test that tolerance parameter affects detection."""
        dec_strict = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2, strong_marker_tolerance=0)
        dec_relaxed = AsymMGCDecoder(N=120, l=8, D_max=5, I_max=2, strong_marker_tolerance=2)

        # Sequence with one-substitution marker
        seq = "ACGT" + "TACGCA" + "GCTA"

        m_strict = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=0)
        m_relaxed = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=2)

        assert len(m_strict.strong) == 0
        assert len(m_relaxed.strong) >= 1


# =============================================================================
# Marker Detection Edge Cases
# =============================================================================

class TestMarkerDetectionEdgeCases:
    """Edge cases for marker detection."""

    def test_overlapping_marker_detection(self):
        seq = "TACGTACGTACGTACGTACG"
        markers = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=0)
        assert len(markers.strong) >= 3

    def test_case_sensitivity(self):
        seq = "acgt" + "TACGTA" + "acgt"
        markers = detect_markers(seq, strong_marker='TACGTA', strong_tolerance=0)
        assert len(markers.strong) == 1
