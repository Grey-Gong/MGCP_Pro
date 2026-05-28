"""
Unit tests for the Memory-k Nanopore Channel.
Tests: error profile computation, transmission, statistics, edge cases.
"""

import pytest
import numpy as np

from asym_mgc.channel.memory_k_nanopore import (
    MemoryKNanoporeChannel,
    standard_nanopore_params,
    deletion_dominant_params,
    iid_params,
)


class TestMemoryKNanoporeChannel:
    """Test suite for the memory-k nanopore channel."""

    def test_error_profile_base(self):
        """Test that error_profile returns valid probabilities."""
        channel = MemoryKNanoporeChannel(Pd=0.5, Pi=0.026, Ps=0.474)
        profile = channel.error_profile("")

        assert 'Pd' in profile
        assert 'Pi' in profile
        assert 'Ps' in profile
        assert profile['Pd'] + profile['Pi'] + profile['Ps'] == pytest.approx(1.0, abs=0.05)

    def test_error_profile_homopolymer_penalty(self):
        """Test that homopolymer regions have elevated deletion probability."""
        channel = MemoryKNanoporeChannel(
            Pd=0.5, Pi=0.026, Ps=0.474, homopolymer_penalty=2.0
        )
        normal_profile = channel.error_profile("ACG")
        hp_profile = channel.error_profile("ACC")  # Same base repeated

        assert hp_profile['Pd'] > normal_profile['Pd']

    def test_error_profile_gc_bias(self):
        """Test that extreme GC content affects deletion probability."""
        channel = MemoryKNanoporeChannel(
            Pd=0.5, Pi=0.026, Ps=0.474, gc_bias=0.15
        )
        normal_profile = channel.error_profile("ACGT")
        high_gc_profile = channel.error_profile("GGGG")

        assert high_gc_profile['Pd'] != normal_profile['Pd']

    def test_transmit_no_errors(self):
        """Test that zero-error transmission returns the same sequence."""
        # Use seed with channel that has near-zero error rates
        channel = MemoryKNanoporeChannel(
            Pd=1e-9, Pi=1e-9, Ps=1e-9, seed=42
        )
        x = "ACGTACGT"
        y, edits = channel.transmit(x)

        assert y == x
        assert len(edits) == 0

    def test_transmit_with_errors(self):
        """Test that transmission with non-zero error rate produces edits."""
        # Use probabilities that sum to 1
        channel = MemoryKNanoporeChannel(
            Pd=0.3, Pi=0.1, Ps=0.2, seed=42
        )
        x = "ACGTACGT" * 10
        y, edits = channel.transmit(x)

        assert len(edits) > 0
        edit_types = set(e[1] for e in edits)
        assert edit_types.issubset({'D', 'I', 'S'})

    def test_transmit_deletion_domination(self):
        """Test that deletion-dominated channel produces more deletions than insertions."""
        channel = MemoryKNanoporeChannel(
            Pd=0.5, Pi=0.03, Ps=0.47, seed=123
        )
        x = "ACGT" * 50
        y, edits = channel.transmit(x)

        deletions = sum(1 for e in edits if e[1] == 'D')
        insertions = sum(1 for e in edits if e[1] == 'I')

        assert deletions > insertions, "Deletion-dominated channel should produce more deletions"

    def test_transmit_reproducibility(self):
        """Test that same seed produces same output."""
        params = dict(k=3, Pd=0.5, Pi=0.026, Ps=0.474, seed=42)
        x = "ACGT" * 20

        ch1 = MemoryKNanoporeChannel(**params)
        y1, _ = ch1.transmit(x)

        ch2 = MemoryKNanoporeChannel(**params)
        y2, _ = ch2.transmit(x)

        assert y1 == y2

    def test_transmit_different_seeds_different_output(self):
        """Test that different seeds produce different outputs."""
        x = "ACGT" * 20
        ch1 = MemoryKNanoporeChannel(Pd=0.5, Pi=0.026, Ps=0.474, seed=1)
        ch2 = MemoryKNanoporeChannel(Pd=0.5, Pi=0.026, Ps=0.474, seed=2)

        y1, _ = ch1.transmit(x)
        y2, _ = ch2.transmit(x)

        assert y1 != y2

    def test_edit_stats(self):
        """Test that edit statistics are computed correctly."""
        channel = MemoryKNanoporeChannel(
            Pd=0.0, Pi=0.0, Ps=0.0, seed=42
        )
        x = "ACGTACGT"
        y, _ = channel.transmit(x)

        stats = channel.compute_edit_stats(x, y)
        assert stats['edit_distance'] == 0
        assert stats['drift'] == 0

    def test_edit_stats_with_errors(self):
        """Test edit stats with simulated errors."""
        channel = MemoryKNanoporeChannel(
            Pd=0.3, Pi=0.1, Ps=0.2, seed=42
        )
        x = "ACGT" * 20
        y, edits = channel.transmit(x)

        stats = channel.compute_edit_stats(x, y)
        assert stats['edit_distance'] >= 0
        assert stats['drift'] == len(y) - len(x)

    def test_parameter_validation(self):
        """Test that invalid parameters are rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            MemoryKNanoporeChannel(Pd=-0.1, Pi=0.5, Ps=0.6)  # Negative probability

        with pytest.raises(ValueError):
            MemoryKNanoporeChannel(k=0)  # k must be >= 1

    def test_transmit_with_quality(self):
        """Test transmission with quality scores."""
        channel = MemoryKNanoporeChannel(Pd=0.1, Pi=0.05, Ps=0.2, seed=42)
        x = "ACGT" * 20

        y, quality = channel.transmit_with_quality(x, base_quality_mean=20.0)

        assert len(y) > 0
        assert len(quality) == len(y)
        assert all(1 <= q <= 45 for q in quality)

    def test_standard_params(self):
        """Test standard nanopore parameters."""
        Pd, Pi, Ps = standard_nanopore_params()
        assert Pd > Pi  # Deletion domination
        assert Pd + Pi + Ps == pytest.approx(1.0, abs=1e-6)

    def test_deletion_dominant_params(self):
        """Test deletion-dominant parameters."""
        Pd, Pi, Ps = deletion_dominant_params()
        assert Pd > Pi
        assert Pd > Ps
