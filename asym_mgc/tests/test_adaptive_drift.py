"""
Unit tests for P1: Adaptive Drift Estimation.

Tests the adaptive_drift_estimator() and rolling_mean() functions,
and their integration into AsymMGCDecoder.

Reference: Section 3.7.5 of IMPROVEMENT_PLAN.md v2.0.
"""

import pytest
import numpy as np

from asym_mgc.inner.decode import (
    adaptive_drift_estimator,
    rolling_mean,
    AsymMGCDecoder,
)


class TestRollingMean:
    """Tests for rolling_mean helper function."""

    def test_empty_input(self):
        """Empty array returns empty array."""
        result = rolling_mean(np.array([]), window=5)
        assert len(result) == 0

    def test_short_input(self):
        """Input shorter than window returns mean of all values."""
        vals = np.array([10.0, 20.0])
        result = rolling_mean(vals, window=5)
        np.testing.assert_allclose(result, [15.0, 15.0])

    def test_reflect_padding(self):
        """Reflect padding should extend values symmetrically."""
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = rolling_mean(vals, window=3)
        # With reflect padding: padded = [10, 10,20,30,40,50, 50]
        # result[0] = mean([10,10,20]) = 13.333
        # result[1] = mean([10,20,30]) = 20.0
        # result[2] = mean([20,30,40]) = 30.0
        np.testing.assert_allclose(result[0], 40.0 / 3, rtol=1e-6)
        assert result[2] == pytest.approx(30.0)

    def test_correct_length(self):
        """Output length should match input length."""
        vals = np.random.rand(100)
        result = rolling_mean(vals, window=20)
        assert len(result) == len(vals)

    def test_even_window(self):
        """Even window size should work."""
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = rolling_mean(vals, window=4)
        assert len(result) == 6


class TestAdaptiveDriftEstimator:
    """Tests for adaptive_drift_estimator()."""

    def test_empty_quality(self):
        """Empty quality returns base values."""
        D, I = adaptive_drift_estimator(
            np.array([]), base_D_max=20, base_I_max=4
        )
        assert D == 20
        assert I == 4

    def test_low_quality_increases_budget(self):
        """Low quality (Q < Q_low) increases D_max."""
        low_q = np.array([5.0, 6.0, 7.0, 8.0])  # all below Q_low=10
        D, I = adaptive_drift_estimator(
            low_q, base_D_max=20, base_I_max=4, Q_low=10.0, Q_high=25.0
        )
        assert D > 20, f"Low quality should increase D_max, got {D}"

    def test_high_quality_decreases_budget(self):
        """High quality (Q > Q_high) decreases D_max."""
        high_q = np.array([28.0, 30.0, 32.0, 35.0])  # all above Q_high=25
        D, I = adaptive_drift_estimator(
            high_q, base_D_max=20, base_I_max=4, Q_low=10.0, Q_high=25.0
        )
        assert D < 20, f"High quality should decrease D_max, got {D}"

    def test_intermediate_quality(self):
        """Intermediate quality uses linear interpolation."""
        mid_q = np.array([17.5] * 20)  # midpoint between Q_low=10 and Q_high=25
        D, I = adaptive_drift_estimator(
            mid_q, base_D_max=20, base_I_max=4, Q_low=10.0, Q_high=25.0
        )
        # scale = 1.5 - 0.9 * 0.5 = 1.05
        # D = round(20 * 1.05) = 21
        assert 15 <= D <= 25

    def test_D_adaptive_respects_minimum(self):
        """D_adaptive should not go below D_min."""
        D, I = adaptive_drift_estimator(
            np.array([100.0] * 20), base_D_max=20, base_I_max=4,
            Q_low=10.0, Q_high=25.0, D_min=10
        )
        assert D >= 10

    def test_I_adaptive_respects_minimum(self):
        """I_adaptive should not go below I_min."""
        _, I = adaptive_drift_estimator(
            np.array([100.0] * 20), base_D_max=20, base_I_max=4,
            Q_low=10.0, Q_high=25.0, I_min=5
        )
        assert I >= 5

    def test_D_adaptive_respects_maximum(self):
        """D_adaptive should not exceed 2 * base_D_max."""
        D, _ = adaptive_drift_estimator(
            np.array([1.0] * 20), base_D_max=20, base_I_max=4,
            Q_low=10.0, Q_high=25.0
        )
        assert D <= 40  # 2 * 20

    def test_I_adaptive_respects_maximum(self):
        """I_adaptive should not exceed 2 * base_I_max."""
        _, I = adaptive_drift_estimator(
            np.array([1.0] * 20), base_D_max=20, base_I_max=4,
            Q_low=10.0, Q_high=25.0
        )
        assert I <= 8  # 2 * 4

    def test_scale_at_Q_low(self):
        """Scale should be 1.5 at Q_low boundary."""
        q = np.array([10.0] * 20)
        D, _ = adaptive_drift_estimator(
            q, base_D_max=20, base_I_max=4, Q_low=10.0, Q_high=25.0
        )
        # scale = 1.5, D = round(20 * 1.5) = 30
        assert D == 30

    def test_scale_at_Q_high(self):
        """Scale should be 0.6 at Q_high boundary."""
        q = np.array([25.0] * 20)
        D, _ = adaptive_drift_estimator(
            q, base_D_max=20, base_I_max=4, Q_low=10.0, Q_high=25.0
        )
        # scale = 0.6, D = round(20 * 0.6) = 12
        assert D == 12


class TestAdaptiveDriftIntegration:
    """Tests for adaptive drift integration in AsymMGCDecoder."""

    def test_decoder_accepts_adaptive_params(self):
        """Decoder should accept all adaptive_drift parameters."""
        dec = AsymMGCDecoder(
            N=120,
            D_max=20,
            I_max=4,
            list_k=8,
            adaptive_drift=True,
            adaptive_drift_window=20,
            adaptive_Q_low=10.0,
            adaptive_Q_high=25.0,
        )
        assert dec.adaptive_drift is True
        assert dec.adaptive_drift_window == 20
        assert dec.adaptive_Q_low == 10.0
        assert dec.adaptive_Q_high == 25.0

    def test_decoder_defaults_to_non_adaptive(self):
        """Decoder should default to adaptive_drift=False."""
        dec = AsymMGCDecoder(N=120, D_max=20, I_max=4)
        assert dec.adaptive_drift is False

    def test_adaptive_decoder_decode_runs(self):
        """Decoder with adaptive_drift=True should complete decoding."""
        import numpy as np
        dec = AsymMGCDecoder(
            N=120,
            D_max=20,
            I_max=4,
            list_k=8,
            adaptive_drift=True,
        )
        seq = "ACGTACGTACGTACGTACGTACGT"
        quality = np.array([15.0] * len(seq))
        decoded, info = dec.decode(seq, quality=quality)
        assert isinstance(decoded, str)

    def test_non_adaptive_decoder_decode_runs(self):
        """Decoder with adaptive_drift=False should complete decoding."""
        import numpy as np
        dec = AsymMGCDecoder(
            N=120,
            D_max=20,
            I_max=4,
            list_k=8,
            adaptive_drift=False,
        )
        seq = "ACGTACGTACGTACGTACGTACGT"
        quality = np.array([15.0] * len(seq))
        decoded, info = dec.decode(seq, quality=quality)
        assert isinstance(decoded, str)

    def test_update_decoder_params_changes_D_I(self):
        """_update_decoder_params should change decoder's D_max/I_max."""
        dec = AsymMGCDecoder(N=120, D_max=20, I_max=4, list_k=8)
        assert dec.decoder.D_max == 20
        assert dec.decoder.I_max == 4

        dec._update_decoder_params(15, 2)
        assert dec.decoder.D_max == 15
        assert dec.decoder.I_max == 2

        # Restore
        dec._update_decoder_params(20, 4)
        assert dec.decoder.D_max == 20
        assert dec.decoder.I_max == 4

    def test_adaptive_sets_params_before_each_window(self):
        """When adaptive_drift=True, params should be updated per window."""
        import numpy as np
        from asym_mgc.inner.decode import split_at_strong_markers, detect_markers

        dec = AsymMGCDecoder(
            N=120,
            D_max=20,
            I_max=4,
            list_k=8,
            adaptive_drift=True,
        )

        # Sequence with markers to create multiple windows
        seq = "ACGT" + "TACGTA" + "ACGTACGTACGT" + "TACGTA" + "ACGTACGT"
        quality = np.array([5.0] * len(seq))  # All low quality

        markers = detect_markers(seq, strong_marker='TACGTA', weak_marker='AC',
                               strong_tolerance=0, weak_tolerance=0)
        windows = split_at_strong_markers(seq, markers.strong, 6)
        assert len(windows) >= 2, "Need at least 2 windows to test per-window adaptation"

        # Decode and check that params were updated
        decoded, info = dec.decode(seq, quality=quality)

        # The first window should have increased D_max (low quality → scale=1.5)
        assert dec.decoder.D_max > 20 or dec.decoder.I_max != 4, \
            "Low quality window should trigger adaptive budget increase"
