"""
Unit tests for the Asymmetric Drift Window.
Tests: bound computation, capture probability, parameter consistency.
"""

import pytest
import math

from asym_mgc.inner.asymmetric_window import (
    AsymmetricWindow,
    delta_to_index,
    index_to_delta,
    typical_window,
)


class TestAsymmetricWindow:
    """Test suite for the asymmetric drift window."""

    def test_basic_properties(self):
        """Test basic window properties."""
        w = AsymmetricWindow(D_max=20, I_max=4, Pd=0.5, Pi=0.03)
        assert w.size == 25  # -20 to +4 = 25 states
        assert w.delta_range == range(-20, 5)
        assert w.D_max == 20
        assert w.I_max == 4

    def test_drift_mean(self):
        """Test drift mean computation."""
        w = AsymmetricWindow(D_max=20, I_max=4, Pd=0.5, Pi=0.03)
        expected_mean = -(0.5 - 0.03)  # -(Pd - Pi)
        assert w.drift_mean_per_symbol == pytest.approx(expected_mean, abs=1e-6)

    def test_from_hoeffding_bound(self):
        """Test window creation from Hoeffding bound."""
        w = AsymmetricWindow.from_hoeffding_bound(
            Pd=0.5, Pi=0.03, t=120, delta=1e-6
        )
        assert w.D_max > 0
        assert w.I_max >= 0
        assert w.Pd == 0.5
        assert w.Pi == 0.03

    def test_from_hoeffding_bound_deletion_required(self):
        """Test that deletion domination is required."""
        with pytest.raises(ValueError, match="Deletion-domination required"):
            AsymmetricWindow.from_hoeffding_bound(Pd=0.03, Pi=0.5, t=120)

    def test_from_practical_budget(self):
        """Test window creation from practical budget."""
        w = AsymmetricWindow.from_practical_budget(
            Pd=0.5, Pi=0.03, N=120, deletion_budget_fraction=0.33
        )
        assert w.D_max == math.ceil(120 * 0.5 * 0.33)  # 20
        assert w.I_max == math.ceil(120 * 0.03 * 1.5)  # 6
        assert w.size == w.D_max + w.I_max + 1

    def test_capture_probability_positive(self):
        """Test that capture probability is positive for reasonable bounds."""
        w = AsymmetricWindow.from_hoeffding_bound(
            Pd=0.5, Pi=0.03, t=120, delta=1e-6
        )
        prob = w.capture_probability(120)
        assert prob >= 0.0
        assert prob <= 1.0

    def test_capture_probability_increases_with_larger_bounds(self):
        """Test that larger window bounds increase capture probability."""
        w_small = AsymmetricWindow(D_max=10, I_max=2, Pd=0.5, Pi=0.03)
        w_large = AsymmetricWindow(D_max=30, I_max=6, Pd=0.5, Pi=0.03)
        p_small = w_small.capture_probability(50)
        p_large = w_large.capture_probability(50)
        assert p_large >= p_small

    def test_delta_index_mapping(self):
        """Test bidirectional delta <-> index mapping."""
        w = AsymmetricWindow(D_max=20, I_max=4, Pd=0.5, Pi=0.03)
        for delta in range(-20, 5):
            idx = delta_to_index(delta, w)
            assert 0 <= idx < w.size
            assert index_to_delta(idx, w) == delta

    def test_delta_index_out_of_bounds(self):
        """Test that out-of-bounds deltas raise errors."""
        w = AsymmetricWindow(D_max=20, I_max=4, Pd=0.5, Pi=0.03)
        with pytest.raises(ValueError):
            delta_to_index(25, w)
        with pytest.raises(ValueError):
            delta_to_index(-25, w)
        with pytest.raises(ValueError):
            index_to_delta(30, w)

    def test_typical_window(self):
        """Test the typical window factory."""
        w = typical_window()
        assert w.D_max == 20
        assert w.I_max == 4
        assert w.Pd == 0.5
        assert w.Pi == 0.03
        assert w.size == 25

    def test_asymmetry_ratio(self):
        """Test that window is asymmetric (D_max >> I_max for nanopore)."""
        w = typical_window()
        assert w.D_max / w.I_max > 3  # Deletion budget much larger than insertion

    def test_repr(self):
        """Test string representation."""
        w = AsymmetricWindow(D_max=20, I_max=4, Pd=0.5, Pi=0.03)
        r = repr(w)
        assert 'D_max=20' in r
        assert 'I_max=4' in r
        assert '|Δ|=25' in r

    def test_window_scaling(self):
        """Test that window scales correctly with sequence length."""
        w1 = AsymmetricWindow.from_practical_budget(
            Pd=0.5, Pi=0.03, N=100
        )
        w2 = AsymmetricWindow.from_practical_budget(
            Pd=0.5, Pi=0.03, N=200
        )
        assert w2.D_max > w1.D_max
        assert w2.D_max / w1.D_max == pytest.approx(2.0, rel=0.1)

    def test_hoeffding_vs_practical(self):
        """Test that Hoeffding bounds are more conservative than practical."""
        Pd, Pi = 0.5, 0.03
        t = 120
        w_hoeffding = AsymmetricWindow.from_hoeffding_bound(
            Pd=Pd, Pi=Pi, t=t, delta=1e-6
        )
        w_practical = AsymmetricWindow.from_practical_budget(
            Pd=Pd, Pi=Pi, N=t
        )
        assert w_hoeffding.D_max >= w_practical.D_max
        assert w_hoeffding.capture_probability(t) >= w_practical.capture_probability(t)


class TestTheorem1Consistency:
    """Test the theoretical bounds from Theorem 1."""

    def test_capture_probability_at_10_minus_6(self):
        """Test that window designed for delta=1e-6 achieves ~1-1e-6 capture."""
        w = AsymmetricWindow.from_hoeffding_bound(
            Pd=0.5, Pi=0.03, t=120, delta=1e-6
        )
        prob = w.capture_probability(120)
        # Should be close to 1 - delta = ~1
        assert prob >= 1.0 - 1e-4

    def test_state_reduction_vs_symmetric(self):
        """Test that asymmetric window reduces state count vs symmetric."""
        Pd, Pi = 0.5, 0.03
        t = 120

        w_asym = AsymmetricWindow.from_practical_budget(
            Pd=Pd, Pi=Pi, N=t
        )

        # Symmetric window with same |D_max|
        w_sym = AsymmetricWindow(
            D_max=w_asym.D_max, I_max=w_asym.D_max, Pd=Pd, Pi=Pi
        )

        # Asymmetric should have fewer states
        assert w_asym.size < w_sym.size
        reduction = 1 - w_asym.size / w_sym.size
        assert reduction > 0.1  # At least 10% reduction
