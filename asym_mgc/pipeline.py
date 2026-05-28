"""
DNA Storage Pipeline: Encode → Channel → Decode for Asym-MGC.

Integrates all components:
- ConstrainedRSEncoder: RS + CRC + homopolymer + GC + markers
- MemoryKNanoporeChannel: realistic nanopore errors
- FSMJointDecoder + AsymMGCDecoder: Viterbi + List Viterbi + sliding window
- Outer soft code: consensus, GMD/OSD, extrinsic IT (optional)

Reference: Section 3 of IMPROVEMENT_PLAN.md v2.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .inner.encode import (
    ConstrainedRSEncoder,
    create_test_message,
    dna_to_binary,
    dna_to_base_ints,
    binary_to_dna,
)
from .inner.decode import AsymMGCDecoder
from .inner.fsm_joint import FSMJointDecoder
from .inner.soft_branch_metric import (
    compute_llr,
    compute_reliability_weight,
    quality_array_to_llr_matrix,
)
from .channel.memory_k_nanopore import MemoryKNanoporeChannel
from .outer.outer_soft import (
    soft_consensus,
    gmd_osd_rs_decode,
    extrinsic_information_transfer,
)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class StrandResult:
    """Result of decoding a single DNA strand."""
    dna_decoded: str
    quality: np.ndarray  # Per-position quality scores
    log_prob: float = 0.0
    num_windows: int = 0
    strong_markers: int = 0
    rs_syndrome_nonzero: bool = False
    fallback_used: int = 0


# =============================================================================
# Soft branch metric helpers
# =============================================================================

def compute_soft_branch_metric(
    emitted_base: int,
    observed_base: int,
    phred_quality: float,
    channel_probs: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute soft branch metric (log-likelihood ratio) for a transition.

    Parameters
    ----------
    emitted_base : int
        Hypothesized base (0-3).
    observed_base : int
        Observed base (0-3).
    phred_quality : float
        Phred quality score.
    channel_probs : Optional[Dict]
        Channel probabilities (P_CORR, P_DEL, P_INS, P_SUB).

    Returns
    -------
    float
        Log-likelihood ratio.
    """
    if channel_probs is None:
        channel_probs = {
            'P_CORR': 0.5,
            'P_DEL': 0.26,
            'P_INS': 0.026,
            'P_SUB': 0.214,
        }

    if emitted_base == observed_base:
        llr = phred_quality * np.log(10)
        log_p_corr = np.log(channel_probs.get('P_CORR', 0.5) + 1e-12)
        return llr + log_p_corr
    else:
        llr = -phred_quality * np.log(10)
        log_p_sub = np.log(channel_probs.get('P_SUB', 0.2) + 1e-12)
        return llr + log_p_sub


# =============================================================================
# Strand helpers
# =============================================================================

def build_strand_copies(
    dna_template: str,
    coverage: int,
    channel: MemoryKNanoporeChannel,
    base_quality_mean: float = 25.0,
    seed_start: int = 0,
) -> List[Tuple[str, np.ndarray]]:
    """
    Simulate multiple sequencing copies of a DNA template.

    Parameters
    ----------
    dna_template : str
        Reference DNA sequence.
    coverage : int
        Number of copies to generate.
    channel : MemoryKNanoporeChannel
        Channel model for simulating errors.
    base_quality_mean : float
        Mean base quality score.
    seed_start : int
        Starting random seed for reproducibility.

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        List of (received_sequence, quality_array) tuples.
    """
    copies = []
    for cov in range(coverage):
        y, qual = channel.transmit_with_quality(
            dna_template,
            base_quality_mean=base_quality_mean,
            seed=seed_start + cov,
        )
        copies.append((y, qual))
    return copies


def decode_single_strand(
    seq: str,
    quality: Optional[np.ndarray],
    decoder_params: Optional[Dict] = None,
) -> StrandResult:
    """
    Decode a single strand using the inner FSM-Viterbi decoder.

    Parameters
    ----------
    seq : str
        Received DNA sequence.
    quality : Optional[np.ndarray]
        Phred quality scores.
    decoder_params : Optional[Dict]
        Decoder configuration overrides.

    Returns
    -------
    StrandResult
        Decoding result with consensus and metadata.
    """
    if decoder_params is None:
        decoder_params = {}

    decoder = AsymMGCDecoder(**decoder_params)
    decoded, info = decoder.decode(seq, quality=quality)

    return StrandResult(
        dna_decoded=decoded,
        quality=quality if quality is not None else np.array([]),
        log_prob=info.get('total_log_prob', 0.0),
        num_windows=info.get('num_windows', 0),
        strong_markers=info.get('strong_markers_detected', 0),
        rs_syndrome_nonzero=info.get('rs_syndrome_nonzero', 0) > 0,
        fallback_used=info.get('fallback_used', 0),
    )


# =============================================================================
# Main Pipeline
# =============================================================================

class DNAPipeline:
    """
    Complete Asym-MGC DNA storage pipeline.

    Integrates encoding, channel simulation, and decoding into a single
    high-level interface.

    Usage:
        pipe = DNAPipeline()
        dna, meta = pipe.encode(message_bits)
        strands = build_strand_copies(dna, coverage=3, channel=...)
        decoded, info = pipe.full_decode(strands, use_outer=True)

    Parameters (encoder)
    ----------
    l : int
        Bits per RS symbol (default 8).
    c_rs : int
        RS parity symbols (default 8).
    c_crc : int
        CRC bits per block (default 8).
    max_run : int
        Max homopolymer run length (default 4).
    gc_low : float
        GC fraction lower bound (default 0.40).
    gc_high : float
        GC fraction upper bound (default 0.60).

    Parameters (decoder)
    ----------
    N : int
        RS codeword symbols per window (default 120).
    D_max : int
        Max deletion offset (default 20).
    I_max : int
        Max insertion offset (default 4).
    K_best : int
        Max states per (i, delta) group (Top-K pruning, default 200).
    T_threshold : float
        Path metric threshold (default 15.0).
    list_k : int
        Candidates per state (List Viterbi, default 8).
    Pd : float
        Deletion probability (default 0.5).
    Pi : float
        Insertion probability (default 0.026).
    Ps : float
        Substitution probability (default 0.474).
    """

    DNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __init__(
        self,
        l: int = 8,
        c_rs: int = 8,
        c_crc: int = 8,
        max_run: int = 4,
        gc_low: float = 0.40,
        gc_high: float = 0.60,
        N: int = 120,
        D_max: int = 20,
        I_max: int = 4,
        K_best: int = 200,
        T_threshold: float = 15.0,
        list_k: int = 8,
        Pd: float = 0.5,
        Pi: float = 0.026,
        Ps: float = 0.474,
    ):
        self.l = l
        self.c_rs = c_rs
        self.c_crc = c_crc
        self.max_run = max_run
        self.gc_low = gc_low
        self.gc_high = gc_high
        self.N = N
        self.D_max = D_max
        self.I_max = I_max
        self.K_best = K_best
        self.T_threshold = T_threshold
        self.list_k = list_k
        self.Pd = Pd
        self.Pi = Pi
        self.Ps = Ps

        self.encoder = ConstrainedRSEncoder(
            l=l, c_rs=c_rs, c_crc=c_crc,
            max_run=max_run, gc_low=gc_low, gc_high=gc_high,
        )

        self.decoder = AsymMGCDecoder(
            N=N, l=l, c_crc=c_crc,
            D_max=D_max, I_max=I_max,
            Pd=Pd, Pi=Pi, Ps=Ps,
            list_k=list_k,
        )

        self._fsm_decoder = FSMJointDecoder(
            N=N, l=l, c_crc=c_crc,
            D_max=D_max, I_max=I_max,
            Pd=Pd, Pi=Pi, Ps=Ps,
            K_best=K_best, T_threshold=T_threshold,
            list_k=list_k,
        )

    def encode(self, message_bits: List[int]) -> Tuple[str, "EncoderMetadata"]:
        """
        Encode a binary message into a DNA strand.

        Returns
        -------
        dna : str
            Encoded DNA sequence.
        meta : EncoderMetadata
            Encoding metadata (N, K, etc.).
        """
        dna, meta_dict = self.encoder.encode(message_bits)

        # Wrap metadata
        meta = EncoderMetadata(**meta_dict)
        return dna, meta

    def inner_decode_strand(
        self,
        seq: str,
        quality: Optional[np.ndarray] = None,
    ) -> StrandResult:
        """
        Decode a single strand using the inner FSM-Viterbi decoder.

        Parameters
        ----------
        seq : str
            Received DNA sequence.
        quality : Optional[np.ndarray]
            Phred quality scores.

        Returns
        -------
        StrandResult
            Decoding result.
        """
        decoded, info = self.decoder.decode(seq, quality=quality)

        return StrandResult(
            dna_decoded=decoded,
            quality=quality if quality is not None else np.array([]),
            log_prob=info.get('total_log_prob', 0.0),
            num_windows=info.get('num_windows', 0),
            strong_markers=info.get('strong_markers_detected', 0),
            rs_syndrome_nonzero=info.get('rs_syndrome_nonzero', 0) > 0,
            fallback_used=info.get('fallback_used', 0),
        )

    def full_decode(
        self,
        strands: List[Tuple[str, np.ndarray]],
        use_outer: bool = False,
        outer_iterations: int = 3,
    ) -> Tuple[str, dict]:
        """
        Full decode of multiple strands.

        Parameters
        ----------
        strands : List[Tuple[str, np.ndarray]]
            List of (sequence, quality) tuples from multiple copies.
        use_outer : bool
            If True, apply outer soft-decision RS decoding.
        outer_iterations : int
            Number of extrinsic IT iterations.

        Returns
        -------
        decoded : str
            Final decoded DNA sequence.
        info : dict
            Decoding statistics.
        """
        info = {
            'num_strands': len(strands),
            'strand_stats': [],
        }

        if not strands:
            return "", info

        # Inner decode each strand
        results = []
        for seq, qual in strands:
            result = self.inner_decode_strand(seq, qual)
            results.append(result)
            info['strand_stats'].append({
                'log_prob': result.log_prob,
                'num_windows': result.num_windows,
                'strong_markers': result.strong_markers,
                'rs_syndrome_nonzero': result.rs_syndrome_nonzero,
            })

        # Consensus formation
        copies = [
            (r.dna_decoded, r.quality)
            for r in results
            if len(r.dna_decoded) > 0
        ]

        if not copies:
            return "", info

        consensus_seq, consensus_weights = soft_consensus(copies)
        info['consensus_length'] = len(consensus_seq)

        if not use_outer:
            return consensus_seq, info

        # Outer soft decoding: extrinsic IT + GMD/OSD
        final_seq = consensus_seq
        for it in range(outer_iterations):
            symbols = []
            for base in final_seq:
                symbols.append(self.DNA_TO_INT.get(base, 0))

            confidence = consensus_weights.tolist() if len(consensus_weights) > 0 else [50.0] * len(final_seq)
            decoded_outer, status = gmd_osd_rs_decode(
                symbols,
                np.array(confidence),
                error_probs=None,
            )

            if status != 'failed':
                final_seq = ''.join(chr(b) for b in decoded_outer)
            info[f'outer_iter_{it}'] = status

        return final_seq, info

    def run(
        self,
        message_bits: List[int],
        channel: Optional[MemoryKNanoporeChannel] = None,
        coverage: int = 1,
        use_outer: bool = False,
        base_quality_mean: float = 25.0,
        seed: int = 42,
    ) -> Tuple[str, dict]:
        """
        Full encode → channel → decode pipeline.

        Parameters
        ----------
        message_bits : List[int]
            Binary message.
        channel : Optional[MemoryKNanoporeChannel]
            Channel model (if None, uses no-error channel).
        coverage : int
            Number of sequencing copies.
        use_outer : bool
            Enable outer soft decoding.
        base_quality_mean : float
            Mean Phred quality score.
        seed : int
            Random seed.

        Returns
        -------
        decoded : str
            Decoded DNA sequence.
        info : dict
            Pipeline statistics.
        """
        # Encode
        dna, meta = self.encode(message_bits)

        # Transmit
        if channel is None:
            channel = MemoryKNanoporeChannel(Pd=1e-9, Pi=1e-9, Ps=1e-9, seed=seed)

        strands = []
        for cov in range(coverage):
            y, qual = channel.transmit_with_quality(
                dna, base_quality_mean=base_quality_mean, seed=seed + cov
            )
            strands.append((y, qual))

        # Decode
        decoded, info = self.full_decode(strands, use_outer=use_outer)
        info['message_bits'] = len(message_bits)
        info['encoded_dna_len'] = len(dna)

        return decoded, info

    def benchmark_fer_no_rs(
        self,
        n_bits: int = 256,
        n_trials: int = 10,
        coverage: int = 1,
        Pd: float = 0.1,
        Pi: float = 0.03,
        Ps: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """
        Benchmark frame error rate (FER) without RS outer code.

        Parameters
        ----------
        n_bits : int
            Message size in bits.
        n_trials : int
            Number of trials.
        coverage : int
            Sequencing coverage.
        Pd, Pi, Ps : float
            Channel error probabilities.
        seed : int
            Starting random seed.

        Returns
        -------
        dict
            FER and per-trial statistics.
        """
        channel = MemoryKNanoporeChannel(
            Pd=Pd, Pi=Pi, Ps=Ps, seed=seed,
        )

        results = []
        errors = 0

        for trial in range(n_trials):
            message = create_test_message(n_bits, seed=seed + trial)
            dna, _ = self.encode(message)

            # Simulate multiple strands
            strands = []
            for cov in range(coverage):
                y, qual = channel.transmit_with_quality(
                    dna, base_quality_mean=25.0, seed=seed + trial * 100 + cov
                )
                strands.append((y, qual))

            # Decode
            decoded, _ = self.full_decode(strands, use_outer=False)

            # Compare (allow for length differences due to indel errors)
            dec_bits = dna_to_binary(decoded)
            msg_bits = message[:len(dec_bits)] if len(dec_bits) <= len(message) else message
            if dec_bits != msg_bits:
                errors += 1

            results.append({
                'trial': trial,
                'error': dec_bits != message[:len(dec_bits)],
                'decoded_len': len(decoded),
                'expected_len': len(dna),
            })

        fer = errors / n_trials if n_trials > 0 else 0.0

        return {
            'fer': fer,
            'errors': errors,
            'n_trials': n_trials,
            'trial_results': results,
        }


@dataclass
class EncoderMetadata:
    """Metadata from the encoder."""
    k_bits: int = 0
    K: int = 0
    N: int = 0
    l: int = 8
    c_rs: int = 8
    c_crc: int = 8
    crc_values: List[int] = field(default_factory=list)
    max_run: int = 4
    gc_low: float = 0.40
    gc_high: float = 0.60
    strong_marker: str = 'TACGTA'
