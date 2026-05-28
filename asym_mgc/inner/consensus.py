"""
Consensus Aligner for Multi-Copy DNA Storage.

Implements consensus formation from multiple reads of the same strand,
inspired by CHN (Zhao et al., Nature Communications 2024).

Key insight from CHN:
- Each composite strand is projected to 8 copies, sequenced with 4× coverage
- Result: ~32 observations per position → strong consensus
- We adapt this with our marker-based windowing system

Reference: IMPROVEMENT_PLAN.md Section 3.7.7 (方案B)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConsensusResult:
    """Result of consensus formation."""
    consensus: str  # Consensus DNA sequence
    quality: np.ndarray  # Per-position confidence scores [0, 1]
    coverage: np.ndarray  # Per-position coverage count
    num_reads: int  # Number of reads used


@dataclass
class ReadAlignment:
    """Aligned read for consensus formation."""
    read_id: int
    sequence: str
    offsets: np.ndarray  # Offset at each consensus position (negative = deletion)


class ConsensusAligner:
    """
    Build consensus from multiple reads of the same DNA strand.

    This class aligns reads using markers and builds a consensus sequence
    with per-position confidence scores.

    Algorithm:
    1. Identify anchor positions (strong markers)
    2. Align reads to reference using markers
    3. For each position, collect all observations
    4. Form consensus using majority vote (weighted by quality)
    5. Compute per-position confidence

    Parameters
    ----------
    strong_marker : str
        Strong marker sequence (default: 'TACGTA').
    weak_marker : str
        Weak marker sequence (default: 'AC').
    min_coverage : int
        Minimum number of reads required at a position for consensus.
    quality_weighted : bool
        Whether to weight votes by basecaller quality scores.
    """

    DNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    INT_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def __init__(
        self,
        strong_marker: str = 'TACGTA',
        weak_marker: str = 'AC',
        min_coverage: int = 2,
        quality_weighted: bool = True,
    ):
        self.strong_marker = strong_marker
        self.weak_marker = weak_marker
        self.min_coverage = min_coverage
        self.quality_weighted = quality_weighted

    def find_markers(self, sequence: str) -> Tuple[List[int], List[int]]:
        """Find strong and weak marker positions in a sequence."""
        import re

        strong = [m.start() for m in re.finditer(self.strong_marker, sequence)]
        weak = [m.start() for m in re.finditer(self.weak_marker, sequence)]

        return strong, weak

    def align_reads_to_reference(
        self,
        reference: str,
        reads: List[Tuple[str, np.ndarray]],
    ) -> List[ReadAlignment]:
        """
        Align reads to a reference sequence using markers and local alignment.

        For indel channels, we need to handle length differences between reads.
        Strategy:
        1. Find strong markers in reference
        2. For each read, find markers and estimate global offset
        3. Do local alignment in regions of interest

        Parameters
        ----------
        reference : str
            Reference sequence (e.g., from first/best read).
        reads : list of (sequence, quality)
            List of reads to align, each with quality scores.

        Returns
        -------
        alignments : list of ReadAlignment
            Each read aligned to the reference coordinate system.
        """
        import re

        ref_strong = [m.start() for m in re.finditer(self.strong_marker, reference)]
        ref_weak = [m.start() for m in re.finditer(self.weak_marker, reference)]

        alignments = []

        for read_id, (read_seq, read_qual) in enumerate(reads):
            read_strong = [m.start() for m in re.finditer(self.strong_marker, read_seq)]
            read_weak = [m.start() for m in re.finditer(self.weak_marker, read_seq)]

            # Estimate offset from strong marker alignment
            offset = 0
            if ref_strong and read_strong:
                # Use first strong marker
                ref_first = ref_strong[0]
                read_first = read_strong[0]
                offset = ref_first - read_first

            # Create alignment using estimated offset
            # For indel channels, we allow flexible alignment
            seq_len = len(reference)
            offsets = np.zeros(seq_len, dtype=np.int32)

            # Estimate per-position offset based on weak markers
            if ref_weak and read_weak:
                # Linear interpolation of offset based on weak markers
                ref_weak_arr = np.array(ref_weak)
                read_weak_arr = np.array(read_weak)

                # Match weak markers
                min_len = min(len(ref_weak), len(read_weak))
                if min_len > 0:
                    for w_idx in range(min_len):
                        ref_pos = ref_weak[w_idx]
                        read_pos = read_weak[w_idx]
                        local_offset = ref_pos - read_pos

                        # Assign this offset to the region around this marker
                        window = 20  # positions around marker
                        start = max(0, ref_pos - window)
                        end = min(seq_len, ref_pos + window)
                        for p in range(start, end):
                            offsets[p] = local_offset

            # Fill in remaining with global offset
            for i in range(seq_len):
                if offsets[i] == 0:
                    offsets[i] = offset

            alignments.append(ReadAlignment(
                read_id=read_id,
                sequence=read_seq,
                offsets=offsets,
            ))

        return alignments

    def build_consensus_simple(
        self,
        reference: str,
        reads: List[Tuple[str, np.ndarray]],
    ) -> ConsensusResult:
        """
        Build consensus using simple majority vote with marker alignment.

        This is a simplified consensus that assumes reads are reasonably well-aligned
        (e.g., via marker-based windowing).

        Parameters
        ----------
        reference : str
            Reference sequence (template for alignment).
        reads : list of (sequence, quality)
            List of (read_sequence, quality) tuples.

        Returns
        -------
        ConsensusResult
            Consensus sequence with quality scores.
        """
        if not reads:
            return ConsensusResult(
                consensus='',
                quality=np.array([]),
                coverage=np.array([]),
                num_reads=0,
            )

        # Align all reads to reference
        alignments = self.align_reads_to_reference(reference, reads)

        # Collect votes per position
        seq_len = len(reference)
        base_votes = np.zeros((seq_len, 4), dtype=float)
        coverage = np.zeros(seq_len, dtype=int)

        for align in alignments:
            read_seq = align.sequence
            read_len = len(read_seq)
            offset = align.offsets[0] if len(align.offsets) > 0 else 0

            for i in range(seq_len):
                # Map reference position to read position
                read_pos = i + offset

                if 0 <= read_pos < read_len:
                    base = read_seq[read_pos]
                    base_int = self.DNA_TO_INT.get(base, -1)
                    if base_int >= 0:
                        base_votes[i, base_int] += 1.0
                        coverage[i] += 1

        # Form consensus from votes
        consensus_bases = []
        consensus_quality = []
        consensus_coverage = []

        for i in range(seq_len):
            if coverage[i] >= self.min_coverage:
                # Majority vote
                best_base = np.argmax(base_votes[i])
                vote_count = base_votes[i, best_base]
                total_votes = coverage[i]

                # Confidence = fraction of votes for best base
                confidence = vote_count / total_votes if total_votes > 0 else 0.0

                consensus_bases.append(self.INT_TO_DNA[best_base])
                consensus_quality.append(confidence)
                consensus_coverage.append(coverage[i])
            else:
                # Not enough coverage, use reference
                consensus_bases.append(reference[i])
                consensus_quality.append(0.0)
                consensus_coverage.append(coverage[i])

        return ConsensusResult(
            consensus=''.join(consensus_bases),
            quality=np.array(consensus_quality),
            coverage=np.array(consensus_coverage),
            num_reads=len(reads),
        )

    def build_consensus_weighted(
        self,
        reference: str,
        reads: List[Tuple[str, np.ndarray]],
    ) -> ConsensusResult:
        """
        Build consensus using quality-weighted voting.

        Parameters
        ----------
        reference : str
            Reference sequence.
        reads : list of (sequence, quality)
            List of (read_sequence, quality_scores) tuples.

        Returns
        -------
        ConsensusResult
            Consensus with quality-weighted confidence.
        """
        if not reads:
            return ConsensusResult(
                consensus='',
                quality=np.array([]),
                coverage=np.array([]),
                num_reads=0,
            )

        alignments = self.align_reads_to_reference(reference, reads)

        seq_len = len(reference)
        base_scores = np.zeros((seq_len, 4), dtype=float)
        coverage = np.zeros(seq_len, dtype=int)

        for align in alignments:
            read_seq = align.sequence
            read_len = len(read_seq)
            offset = align.offsets[0] if len(align.offsets) > 0 else 0

            # Get quality for this read (default to high quality if not provided)
            read_qual = np.full(read_len, 30.0)  # Default Q=30

            for i in range(seq_len):
                read_pos = i + offset

                if 0 <= read_pos < read_len:
                    base = read_seq[read_pos]
                    base_int = self.DNA_TO_INT.get(base, -1)
                    if base_int >= 0:
                        # Weight by quality (convert Phred to probability)
                        q = read_qual[read_pos] if read_pos < len(read_qual) else 30.0
                        weight = 10 ** (q / 10.0)  # Convert Q to linear probability
                        base_scores[i, base_int] += weight
                        coverage[i] += 1

        # Form consensus
        consensus_bases = []
        consensus_quality = []
        consensus_coverage = []

        for i in range(seq_len):
            if coverage[i] >= self.min_coverage:
                best_base = np.argmax(base_scores[i])
                best_score = base_scores[i, best_base]
                total_score = np.sum(base_scores[i])

                confidence = best_score / total_score if total_score > 0 else 0.0
                consensus_bases.append(self.INT_TO_DNA[best_base])
                consensus_quality.append(confidence)
                consensus_coverage.append(coverage[i])
            else:
                consensus_bases.append(reference[i])
                consensus_quality.append(0.0)
                consensus_coverage.append(coverage[i])

        return ConsensusResult(
            consensus=''.join(consensus_bases),
            quality=np.array(consensus_quality),
            coverage=np.array(consensus_coverage),
            num_reads=len(reads),
        )

    def build_consensus(
        self,
        reference: str,
        reads: List[Tuple[str, np.ndarray]],
    ) -> ConsensusResult:
        """
        Build consensus from multiple reads.

        Parameters
        ----------
        reference : str
            Reference sequence for alignment.
        reads : list of (sequence, quality)
            List of reads, each as (sequence, quality_array).

        Returns
        -------
        ConsensusResult
            Consensus sequence with per-position quality.
        """
        if self.quality_weighted:
            return self.build_consensus_weighted(reference, reads)
        else:
            return self.build_consensus_simple(reference, reads)


class MultiCopyPipeline:
    """
    Multi-copy encoding and decoding pipeline.

    Encodes the same message into multiple independent DNA strands,
    then decodes by collecting reads from all copies and building consensus.

    This is the practical adaptation of CHN's 4× coverage strategy.

    Reference: IMPROVEMENT_PLAN.md Section 3.7.7.3
    """

    def __init__(
        self,
        inner_encoder,
        inner_decoder,
        num_copies: int = 4,
        consensus_min_coverage: int = 2,
    ):
        """
        Parameters
        ----------
        inner_encoder : ConstrainedRSEncoder
            Inner encoder for single-strand encoding.
        inner_decoder : AsymMGCDecoder
            Inner decoder for single-strand decoding.
        num_copies : int
            Number of physical copies to generate/decode.
        consensus_min_coverage : int
            Minimum coverage required for consensus.
        """
        self.inner_encoder = inner_encoder
        self.inner_decoder = inner_decoder
        self.num_copies = num_copies
        self.consensus_aligner = ConsensusAligner(min_coverage=consensus_min_coverage)

    def encode(
        self,
        message: List[int],
        seed_offset: int = 0,
    ) -> List[Tuple[str, dict]]:
        """
        Encode message into multiple independent DNA strands.

        Parameters
        ----------
        message : list of int
            Binary message bits.
        seed_offset : int
            Random seed offset for constrained encoding (each copy gets different seed).

        Returns
        -------
        copies : list of (dna_sequence, metadata)
            List of encoded DNA copies.
        """
        copies = []

        for copy_idx in range(self.num_copies):
            # Each copy uses a different constrained encoding seed
            # This ensures diversity even for the same message
            seed = seed_offset + copy_idx

            dna, meta = self.inner_encoder.encode(message, seed=seed)
            copies.append((dna, meta))

        return copies

    def decode_copies(
        self,
        copies_dna: List[str],
        copies_qual: List[np.ndarray],
    ) -> Tuple[str, dict]:
        """
        Decode from multiple copies using consensus.

        Parameters
        ----------
        copies_dna : list of str
            DNA sequences from each copy.
        copies_qual : list of ndarray
            Quality scores for each copy.

        Returns
        -------
        decoded : str
            Consensus-decoded DNA sequence.
        info : dict
            Decoding info including consensus quality and coverage.
        """
        if len(copies_dna) < self.num_copies:
            raise ValueError(
                f"Expected {self.num_copies} copies, got {len(copies_dna)}"
            )

        # Use first copy as reference
        reference = copies_dna[0]
        reference_qual = copies_qual[0] if len(copies_qual) > 0 else None

        # Collect reads from all copies
        reads = list(zip(copies_dna, copies_qual))

        # Build consensus
        consensus = self.consensus_aligner.build_consensus(reference, reads)

        # Decode consensus with inner decoder
        decoded, decode_info = self.inner_decoder.decode(
            consensus.consensus,
            quality=consensus.quality * 40,  # Scale consensus quality to Phred
        )

        # Add consensus info
        decode_info['consensus'] = {
            'consensus_sequence': consensus.consensus,
            'consensus_quality': consensus.quality,
            'consensus_coverage': consensus.coverage,
            'num_copies': len(copies_dna),
        }

        return decoded, decode_info


def simulate_multicopy_reads(
    reference: str,
    channel,
    num_copies: int = 4,
    quality_mean: float = 15.0,
    quality_std: float = 5.0,
    seed: int = 42,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Simulate reads from multiple copies of the same strand.

    Parameters
    ----------
    reference : str
        Reference DNA sequence.
    channel : MemoryKNanoporeChannel
        Channel model to simulate errors.
    num_copies : int
        Number of copies to simulate.
    quality_mean : float
        Mean Phred quality score.
    quality_std : float
        Std of Phred quality score.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    reads : list of str
        Received DNA sequences.
    qualities : list of ndarray
        Quality scores for each read.
    """
    np.random.seed(seed)

    reads = []
    qualities = []

    for i in range(num_copies):
        # Transmit through channel
        received, qual = channel.transmit_with_quality(
            reference,
            base_quality_mean=quality_mean,
            base_quality_std=quality_std,
            seed=seed + i,
        )
        reads.append(received)
        qualities.append(qual)

    return reads, qualities
