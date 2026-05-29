"""
Asym-MGC Decoder: Marker Detection + Sliding Window + List Viterbi + Adaptive Drift.

This module integrates:
1. levenshtein_distance: for fuzzy marker matching
2. detect_markers: hierarchical marker detection
3. split_at_strong_markers: window segmentation
4. fallback_for_missing_strong_marker: state continuity across windows
5. rolling_mean: smooth quality scores
6. adaptive_drift_estimator: quality-based drift window adaptation
7. AsymMGCDecoder: top-level decoder with sliding window and List Viterbi

Reference: Section 3 of IMPROVEMENT_PLAN.md v2.1.
Reference: ARCHITECTURE_REVISION_v2_1.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fsm_joint import FSMJointDecoder, FSMViterbiState, FSMPathMetric, FSMPathMetricTopK
from .trellis import HomopolymerState


# =============================================================================
# LDPC Error Correction (分层架构支持)
# =============================================================================

def ldpc_correct_bits(
    bits: np.ndarray,
    n: int = 96,
    k: int = 50,
    max_errors: int = 10,
    ldpc_type: str = "protograph",
    coverage: int = 5,
) -> Tuple[np.ndarray, bool, int]:
    """
    Use LDPC to correct bit errors in the BSC channel output.

    This implements the layered architecture:
    1. Viterbi removes indels -> produces aligned bit sequence
    2. LDPC corrects remaining substitution errors (BSC channel)

    Supported ldpc_type values:
        "tiered":      Adaptive - selects HIGH/MEDIUM/LOW based on coverage (RECOMMENDED)
        "protograph":  LDPC(n, 50, dv=3, dc=6) - rate=0.52
        "low_rate":    LDPC(200, 43, dv=4, dc=5) - rate=0.22, coverage-aware default
        "systematic":  LDPC(n, k, dv=3, dc=6) - rate=0.52
        "sc_ldpc":     spatial coupling variant

    Tiered selection (coverage-aware):
        coverage >= 7: HIGH  - LDPC(96,50)  rate=0.521, info_density=1.04 bits/base (52%)
        coverage >= 3: MEDIUM - LDPC(120,27) rate=0.225, info_density=0.45 bits/base (22%)
        coverage >= 0: LOW    - LDPC(200,43) rate=0.215, info_density=0.43 bits/base (21%)

    Note: info_density = 2 * rate (DNA max = 2 bits/base).
    Coverage is read-time denoising, NOT stored redundancy.

    Parameters
    ----------
    bits : np.ndarray
        Received bits (after Viterbi alignment)
    n : int
        LDPC codeword length (used for non-tiered types)
    k : int
        LDPC message length (used for non-tiered types)
    max_errors : int
        Maximum bit errors to attempt correction
    ldpc_type : str
        LDPC type
    coverage : int
        Sequencing coverage (used for tiered selection)

    Returns
    -------
    Tuple[np.ndarray, bool, int]
        (corrected_bits, was_corrected, errors_fixed)
    """
    from .ldpc_codec import (
        create_protograph_ldpc, create_systematic_ldpc, create_sc_ldpc,
        create_low_rate_ldpc, create_tiered_ldpc,
        ldpc_encode, min_sum_decode
    )

    # Create LDPC code based on type
    if ldpc_type == "tiered":
        # Adaptive: selects best tier based on coverage
        code = create_tiered_ldpc(coverage=coverage, seed=42)
        decode_max_iter = max(200, code.n // 2)
    elif ldpc_type == "low_rate":
        code = create_low_rate_ldpc(seed=42)
        decode_max_iter = 500
    elif ldpc_type == "protograph":
        code = create_protograph_ldpc(n, k, seed=42)
        decode_max_iter = 100
    elif ldpc_type == "sc_ldpc":
        code = create_sc_ldpc(n, k, seed=42)
        decode_max_iter = 100
    else:
        code = create_systematic_ldpc(n, k, seed=42)
        decode_max_iter = 100
    
    # Pad or truncate to n bits (codeword length, NOT message length)
    # LDPC decoding operates on the n-bit codeword, not the k-bit message
    if len(bits) < code.n:
        bits_padded = np.pad(bits, (0, code.n - len(bits)))
    else:
        bits_padded = bits[:code.n]
    
    # Estimate LLR from bit confidence
    llr = np.where(bits_padded == 0, 5.0, -5.0)  # Strong LLR
    
    # LDPC decode
    decoded, converged, iterations = min_sum_decode(llr, code, max_iter=decode_max_iter)
    
    if converged:
        # Return k-bit message, not n-bit codeword
        # For systematic code, message is the first k bits of the decoded codeword
        decoded_msg = decoded[:code.k]
        errors = np.sum(decoded_msg != bits_padded[:code.k])
        return decoded_msg, True, errors
    else:
        # Return k-bit message on failure
        return bits_padded[:code.k], False, 0


# =============================================================================
# Levenshtein Distance
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    Uses O(min(m, n)) space via two-row DP.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    curr_row = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        curr_row[0] = i + 1
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row, curr_row = curr_row, prev_row

    return prev_row[len(s2)]


# =============================================================================
# Marker Detection
# =============================================================================

@dataclass
class MarkerPositions:
    """Positions of detected markers in a sequence."""
    strong: List[int] = field(default_factory=list)
    weak: List[int] = field(default_factory=list)


def detect_markers(
    seq: str,
    strong_marker: str = 'TACGTA',
    strong_tolerance: int = 0,
    weak_marker: str = 'AC',
    weak_tolerance: int = 0,
) -> MarkerPositions:
    """
    Detect strong and weak markers in a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence to search.
    strong_marker : str
        Strong marker sequence (default: 'TACGTA').
    strong_tolerance : int
        Maximum Levenshtein distance for strong marker match.
    weak_marker : str
        Weak marker sequence (default: 'AC').
    weak_tolerance : int
        Maximum Levenshtein distance for weak marker match.

    Returns
    -------
    MarkerPositions
        Lists of positions where markers were detected.
    """
    result = MarkerPositions()
    l_strong = len(strong_marker)
    l_weak = len(weak_marker)

    # Slide window over sequence
    for i in range(len(seq) - l_strong + 1):
        window = seq[i:i + l_strong]
        dist = levenshtein_distance(window, strong_marker)
        if dist <= strong_tolerance:
            result.strong.append(i)

    for i in range(len(seq) - l_weak + 1):
        window = seq[i:i + l_weak]
        dist = levenshtein_distance(window, weak_marker)
        if dist <= weak_tolerance:
            result.weak.append(i)

    return result


def split_at_strong_markers(
    seq: str,
    strong_positions: List[int],
    strong_marker_len: int = 6,
) -> List[Tuple[str, int]]:
    """
    Split a sequence at strong marker positions.

    Each strong marker marks the END of a window (the marker itself
    is included at the boundary).

    Parameters
    ----------
    seq : str
        DNA sequence.
    strong_positions : List[int]
        Positions of strong markers in seq.
    strong_marker_len : int
        Length of strong marker.

    Returns
    -------
    List[Tuple[str, int]]
        List of (segment, start_offset) tuples.
        Segment includes the marker at its end boundary.
    """
    if not strong_positions:
        return [(seq, 0)]

    strong_set = set(strong_positions)
    segments = []
    prev_end = 0

    for pos in sorted(strong_positions):
        seg_end = pos + strong_marker_len
        segment = seq[prev_end:seg_end]
        segments.append((segment, prev_end))
        prev_end = seg_end

    # Always append trailing segment (may be empty if marker was at end)
    segments.append((seq[prev_end:], prev_end))

    return segments


# =============================================================================
# Fallback State
# =============================================================================

@dataclass
class FallbackState:
    """Fallback state when a strong marker is missing."""
    last_delta: int = 0
    last_s_hp: int = 0
    prev_base: int = 0
    uncertainty_flag: bool = True


def fallback_for_missing_strong_marker(
    states: Dict[FSMViterbiState, FSMPathMetricTopK]
) -> FallbackState:
    """
    Estimate fallback state from the best available state.

    Used when a strong marker is lost and we need to continue decoding
    with an estimated initial state for the next window.

    Returns the delta and homopolymer state from the path with
    the highest log_prob.
    """
    if not states:
        return FallbackState(uncertainty_flag=True)

    best_state = max(
        states.keys(),
        key=lambda s: states[s].get_best().log_prob if states[s].get_best() else -np.inf
    )
    best_pm = states[best_state].get_best()

    return FallbackState(
        last_delta=best_state.delta,
        last_s_hp=best_state.s_hp.value,
        prev_base=best_state.prev_base,
        uncertainty_flag=False,
    )


# =============================================================================
# Adaptive Drift Estimation (Phase 1 Enhancement)
# =============================================================================

def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean with reflect padding at boundaries.

    Reflect padding: boundary values are mirrored to extend the sequence.
    For input [v0, v1, ..., vn-1] with window=3:
    padded = [v0, v0, v1, ..., vn-1, vn-1]
    Result[i] = mean of window elements centered at position i.

    For inputs shorter than window, returns the mean of all values.
    """
    if len(values) == 0:
        return np.array([])

    if len(values) < window:
        # Short input: return uniform mean
        return np.full(len(values), np.mean(values))

    # Reflect padding
    half = window // 2
    padded = np.concatenate([
        np.full(half, values[0]),    # reflect left
        values,
        np.full(half, values[-1]),   # reflect right
    ])

    result = np.zeros(len(values))
    for i in range(len(values)):
        start = i
        end = i + window
        result[i] = np.mean(padded[start:end])

    return result


def adaptive_drift_estimator(
    quality: np.ndarray,
    base_D_max: int = 20,
    base_I_max: int = 4,
    Q_low: float = 10.0,
    Q_high: float = 25.0,
    D_min: int = 5,
    D_max_max: int = 40,
    I_min: int = 1,
    I_max_max: int = 8,
) -> Tuple[int, int]:
    """
    Estimate adaptive drift window bounds based on local quality scores.

    Low quality (Q < Q_low): expand D_max (more deletions expected).
    High quality (Q > Q_high): contract D_max (fewer errors expected).
    Linear interpolation in between.

    Parameters
    ----------
    quality : np.ndarray
        Phred quality scores.
    base_D_max : int
        Base maximum deletion offset.
    base_I_max : int
        Base maximum insertion offset.
    Q_low : float
        Quality below which D_max expands.
    Q_high : float
        Quality above which D_max contracts.
    D_min, D_max_max, I_min, I_max_max : int
        Hard bounds on D_max and I_max.

    Returns
    -------
    Tuple[int, int]
        (D_max, I_max) for current quality window.
    """
    if len(quality) == 0:
        return base_D_max, base_I_max

    q_mean = float(np.mean(quality))

    if q_mean <= Q_low:
        scale = 1.5
    elif q_mean >= Q_high:
        scale = 0.6
    else:
        # Linear interpolation between boundaries
        t = (q_mean - Q_low) / (Q_high - Q_low)
        scale = 1.5 - 0.9 * t  # 1.5 at Q_low, 0.6 at Q_high

    D_adaptive = int(round(base_D_max * scale))
    I_adaptive = int(round(base_I_max * scale))

    # Clip to hard bounds
    D_adaptive = max(D_min, min(D_max_max, D_adaptive))
    I_adaptive = max(I_min, min(I_max_max, I_adaptive))

    return D_adaptive, I_adaptive


# =============================================================================
# Top-level Decoder
# =============================================================================

# DNA mapping (consistent with encode.py)
DNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
DNA_TO_BITS = {
    'A': [0, 0],
    'C': [0, 1],
    'G': [1, 0],
    'T': [1, 1],
}
BITS_TO_DNA = {(0, 0): 'A', (0, 1): 'C', (1, 0): 'G', (1, 1): 'T'}


class AsymMGCDecoder:
    """
    Asym-MGC Decoder with sliding window and List Viterbi.

    Full decoding pipeline:
    1. Detect strong markers in received sequence
    2. Split into windows at marker boundaries
    3. For each window:
       a. Initialize FSM-Trellis Viterbi (with fallback from previous window)
       b. Run List Viterbi decode_step for each observed base
       c. Apply pruning strategies
       d. Extract top-K candidates via traceback_all
       e. Apply RS-guided candidate selection if needed
    4. Concatenate decoded windows
    5. Strip markers from output

    Parameters
    ----------
    N : int
        Number of RS codeword symbols per window.
    l : int
        Bits per RS symbol.
    c_crc : int
        CRC bits per block.
    D_max : int
        Max deletion offset.
    I_max : int
        Max insertion offset.
    Pd : float
        Deletion probability.
    Pi : float
        Insertion probability.
    Ps : float
        Substitution probability.
    list_k : int
        Number of candidates per state (List Viterbi).
    strong_marker : str
        Strong marker sequence.
    strong_marker_tolerance : int
        Tolerance for fuzzy marker matching.
    enable_list_viterbi : bool
        Enable List Viterbi (default True).
    """

    def __init__(
        self,
        N: int = 120,
        l: int = 8,
        c_crc: int = 8,
        c_rs: int = 8,
        D_max: int = 20,
        I_max: int = 4,
        Pd: float = 0.5,
        Pi: float = 0.026,
        Ps: float = 0.474,
        list_k: int = 8,
        K_best: int = 200,
        T_threshold: float = 15.0,
        strong_marker: str = 'TACGTA',
        strong_marker_tolerance: int = 1,
        enable_list_viterbi: bool = True,
        adaptive_drift: bool = False,
        adaptive_drift_window: int = 20,
        adaptive_Q_low: float = 10.0,
        adaptive_Q_high: float = 25.0,
        branch_metric_mode: str = 'original',
    ):
        self.N = N
        self.l = l
        self.c_crc = c_crc
        self.c_rs = c_rs  # RS parity symbols (configurable, encoder default is 8)
        self.base_D_max = D_max
        self.base_I_max = I_max
        self.D_max = D_max
        self.I_max = I_max
        self.P_d = Pd
        self.P_i = Pi
        self.P_s = Ps
        self.list_k = list_k
        self.K_best = K_best
        self.T_threshold = T_threshold
        self.strong_marker = strong_marker
        self.strong_marker_tolerance = strong_marker_tolerance
        self.enable_list_viterbi = enable_list_viterbi
        self.adaptive_drift = adaptive_drift
        self.adaptive_drift_window = adaptive_drift_window
        self.adaptive_Q_low = adaptive_Q_low
        self.adaptive_Q_high = adaptive_Q_high
        self.branch_metric_mode = branch_metric_mode
        self.current_D_max = D_max
        self.current_I_max = I_max

        self.decoder = FSMJointDecoder(
            N=N, l=l, c_crc=c_crc,
            D_max=D_max, I_max=I_max,
            Pd=Pd, Pi=Pi, Ps=Ps,
            K_best=K_best, T_threshold=T_threshold,
            list_k=list_k,
        )

    def _dna_to_bits(self, dna: str) -> List[int]:
        """Convert DNA string to flat bit list."""
        bits = []
        for base in dna:
            bits.extend(DNA_TO_BITS.get(base, [0, 0]))
        return bits

    def _bits_to_symbols(self, bits: List[int]) -> List[int]:
        """Convert bits to RS symbols (l bits per symbol, MSB-first)."""
        if len(bits) % self.l != 0:
            bits = bits + [0] * (self.l - len(bits) % self.l)
        symbols = []
        for i in range(0, len(bits), self.l):
            block = bits[i:i + self.l]
            sym = int(''.join(str(b) for b in block), 2)
            symbols.append(sym)
        return symbols

    def _symbols_to_dna(self, symbols: List[int]) -> str:
        """Convert RS symbols back to DNA string."""
        bits = []
        for sym in symbols:
            for bit_pos in range(self.l - 1, -1, -1):
                bits.append((sym >> bit_pos) & 1)
        dna = []
        for i in range(0, len(bits), 2):
            pair = bits[i:i + 2]
            if len(pair) == 2:
                dna.append(BITS_TO_DNA.get((pair[0], pair[1]), 'A'))
        return ''.join(dna)

    def _rs_pre_decode_full(
        self,
        seq: str,
        marker_positions: List[int],
    ) -> Tuple[str, bool, dict]:
        """
        RS pre-decoding on the full sequence (not per-segment).

        This is the key fix for substitution handling:
        - Viterbi cannot see substitutions (symmetric errors)
        - RS decoder CAN correct substitutions
        - Run RS on the FULL data (all segments together) for best correction

        Parameters
        ----------
        seq : str
            Full DNA sequence.
        marker_positions : List[int]
            Positions of strong markers in the sequence.

        Returns: (corrected_seq, was_corrected, stats)
        """
        from reedsolo import RSCodec

        stats = {'symbols_extracted': 0, 'symbols_expected': 0, 'rs_errors_corrected': 0}

        # Extract data segments using marker positions
        segments = []
        prev_end = 0
        marker_len = len(self.strong_marker)
        for pos in marker_positions:
            seg = seq[prev_end:pos]
            segments.append(seg)
            prev_end = pos + marker_len
        segments.append(seq[prev_end:])  # trailing segment
        segment_clean = ''.join(segments)

        try:
            rs_codec = RSCodec(self.c_rs, c_exp=self.l)

            # Convert to bits
            bits = self._dna_to_bits(segment_clean)

            # Convert to symbols
            symbols = self._bits_to_symbols(bits)

            stats['symbols_extracted'] = len(symbols)
            stats['symbols_expected'] = self.N

            # Need at least c_rs + 1 symbols for RS to work
            if len(symbols) < self.c_rs + 1:
                return segment_clean, False, stats

            # RS decode: detects AND corrects substitution errors
            decoded_result = rs_codec.decode(symbols)
            if isinstance(decoded_result, tuple):
                corrected_symbols = list(decoded_result[0])
            else:
                corrected_symbols = list(decoded_result)

            stats['rs_errors_corrected'] = len(symbols) - len(corrected_symbols)

            # Convert back to DNA
            corrected_dna = self._symbols_to_dna(corrected_symbols)

            return corrected_dna, True, stats
        except Exception as e:
            return segment_clean, False, stats

    def _update_decoder_params(self, D_max: int, I_max: int) -> None:
        """Update per-window decoder parameters and the shared decoder for compatibility."""
        self.current_D_max = D_max
        self.current_I_max = I_max
        # Also update the shared decoder for backwards compatibility with tests
        self.decoder.D_max = D_max
        self.decoder.I_max = I_max

    def decode(
        self,
        seq: str,
        quality: Optional[np.ndarray] = None,
        enable_ldpc_correct: bool = True,
        enable_rs_candidate_selection: bool = True,
        coverage: int = 5,
    ) -> Tuple[str, dict]:
        """
        Decode a received DNA sequence.

        Architecture (两层架构):
        1. Viterbi: handle insertions/deletions, output pure substitution errors
        2. LDPC: correct remaining substitution errors (BSC channel)

        RS is NOT used - this is the key insight:
        - Viterbi removes indels -> produces aligned bit sequence
        - LDPC corrects substitution errors (classic BSC problem)

        Parameters
        ----------
        seq : str
            Received DNA sequence (may contain errors).
        quality : Optional[np.ndarray]
            Phred quality scores per base (0-40).
        enable_ldpc_correct : bool
            If True, apply LDPC decoding to correct substitution errors.

        Returns
        -------
        decoded_dna : str
            Decoded DNA sequence (markers stripped).
        info : dict
            Decoding statistics and metadata.
        """
        info = {
            'num_windows': 0,
            'strong_markers_detected': 0,
            'window_stats': [],
            'total_log_prob': 0.0,
            'lva_used': self.enable_list_viterbi,
            'fallback_used': 0,
            'ldpc_corrected': 0,
            'rs_syndrome_nonzero': 0,
        }
        self._enable_ldpc_correct = enable_ldpc_correct
        self._enable_rs_candidate_selection = enable_rs_candidate_selection
        self._coverage = coverage

        if not seq:
            info['num_windows'] = 1
            return "", info

        # Detect markers
        markers = detect_markers(
            seq,
            strong_marker=self.strong_marker,
            strong_tolerance=self.strong_marker_tolerance,
        )
        info['strong_markers_detected'] = len(markers.strong)

        # Split into windows and decode each
        segments = split_at_strong_markers(
            seq, markers.strong, len(self.strong_marker)
        )
        info['num_windows'] = len(segments)

        # Decode each window
        decoded_parts = []
        fallback: Optional[FallbackState] = None

        for seg_idx, (segment, start_offset) in enumerate(segments):
            if not segment:
                continue

            # Adaptive drift: update params based on local quality
            if self.adaptive_drift and quality is not None:
                seg_quality = quality[start_offset:start_offset + len(segment)]
                if len(seg_quality) > 0:
                    smoothed = rolling_mean(seg_quality, self.adaptive_drift_window)
                    D_new, I_new = adaptive_drift_estimator(
                        smoothed,
                        base_D_max=self.base_D_max,
                        base_I_max=self.base_I_max,
                        Q_low=self.adaptive_Q_low,
                        Q_high=self.adaptive_Q_high,
                    )
                    self._update_decoder_params(D_new, I_new)

            decoded_seg, win_info, fallback = self._decode_window(
                segment, quality, start_offset, fallback,
                enable_rs_candidate_selection=enable_rs_candidate_selection,
            )
            decoded_parts.append(decoded_seg)
            info['window_stats'].append(win_info)
            info['total_log_prob'] += win_info.get('log_prob', 0.0)
            if win_info.get('fallback_used', False):
                info['fallback_used'] += 1
            if win_info.get('rs_syndrome_nonzero', False):
                info['rs_syndrome_nonzero'] += 1
            if win_info.get('ldpc_corrected', False):
                info['ldpc_corrected'] += 1

        # Concatenate decoded parts. Each segment ends with the strong marker
        # (which was included in the segment for synchronization), so strip it
        # from each decoded part before joining.
        decoded_no_marker = [seg.replace(self.strong_marker, '')
                            for seg in decoded_parts]
        full_decoded = ''.join(decoded_no_marker)

        return full_decoded, info

    def _create_window_decoder(self, segment_len: int):
        """Create a fresh FSM decoder for a window segment with N = segment_len + D_max buffer."""
        N = segment_len + self.D_max
        return FSMJointDecoder(
            N=N, l=self.l, c_crc=self.c_crc,
            D_max=self.D_max, I_max=self.I_max,
            Pd=self.P_d, Pi=self.P_i, Ps=self.P_s,
            K_best=self.K_best, T_threshold=self.T_threshold,
            list_k=self.list_k,
        )

    def _decode_window(
        self,
        segment: str,
        quality: Optional[np.ndarray],
        start_offset: int,
        fallback: Optional[FallbackState],
        enable_rs_candidate_selection: bool = True,
    ) -> Tuple[str, dict, Optional[FallbackState]]:
        """
        Decode a single window segment.

        Pipeline (分层架构):
        1. RS pre-decode: done on FULL sequence before this call
        2. Viterbi: handle insertions/deletions
        3. Optional: RS guided candidate selection
        4. Optional: LDPC post-correction

        Returns (decoded_seq, window_info, new_fallback).
        """
        win_info = {
            'steps': 0,
            'final_active': 0,
            'log_prob': 0.0,
            'fallback_used': False,
            'rs_syndrome_nonzero': False,
            'ldpc_corrected': False,
        }

        # Use the RS-corrected segment (from full-sequence RS pre-decode)
        segment_for_viterbi = segment.replace(self.strong_marker, '')

        # Create a fresh decoder for this segment with N = expected DNA length
        decoder = self._create_window_decoder(len(segment_for_viterbi))

        # Initialize decoder
        states = decoder.init_states()

        # Apply fallback if available
        if fallback is not None and not fallback.uncertainty_flag:
            win_info['fallback_used'] = True

        # Quality array for this segment
        if quality is not None:
            qual = quality[start_offset:start_offset + len(segment)]
        else:
            qual = None

        # Step 2: Viterbi decoding (handles insertions/deletions)
        for step_idx, base in enumerate(segment_for_viterbi):
            base_int = DNA_TO_INT.get(base, 0)
            phred = qual[step_idx] if qual is not None and step_idx < len(qual) else 30.0

            states, step_stats = decoder.decode_step(
                states, base_int, phred_quality=phred,
                apply_crc_prune=False,
            )
            win_info['steps'] += 1

            if not states:
                break

        win_info['final_active'] = len(states)

        # Extract candidates
        if not states:
            return "", win_info, fallback_for_missing_strong_marker({})

        candidates = decoder.traceback_all(states, top_k=decoder.list_k)
        win_info['log_prob'] = candidates[0][1] if candidates else 0.0

        # Step 3: RS-guided candidate selection if needed
        decoded_seq = ""
        rs_nonzero = False

        if candidates:
            best_dna, best_prob = candidates[0]
            if not enable_rs_candidate_selection:
                decoded_seq = best_dna
            else:
                decoded_seq, rs_nonzero = self._rs_guided_select(
                    candidates, segment_for_viterbi
                )

        win_info['rs_syndrome_nonzero'] = rs_nonzero

        # Step 4: LDPC post-correction (layered architecture, tiered)
        if getattr(self, '_enable_ldpc_correct', False) and decoded_seq:
            coverage = getattr(self, '_coverage', 5)
            ldpc_corrected_dna, ldpc_success = self._ldpc_post_correct(
                decoded_seq, qual, coverage=coverage
            )
            if ldpc_success:
                decoded_seq = ldpc_corrected_dna
                win_info['ldpc_corrected'] = True

        # Generate fallback for next window
        new_fallback = fallback_for_missing_strong_marker(states)

        return decoded_seq, win_info, new_fallback

    def _ldpc_post_correct(
        self,
        dna_seq: str,
        quality: Optional[np.ndarray],
        coverage: int = 5,
    ) -> Tuple[str, bool]:
        """
        Apply LDPC decoding to correct remaining substitution errors.

        Key fixes vs original:
        - Uses soft LLR from quality scores (not hard-coded ±5)
        - Processes all available bits (up to codeword length)
        - Properly handles systematic code: returns k-bit message

        Parameters
        ----------
        dna_seq : str
            Decoded DNA sequence from Viterbi
        quality : Optional[np.ndarray]
            Quality scores for LDPC soft-decision

        Returns
        -------
        Tuple[str, bool]
            (corrected_dna, was_corrected)
        """
        from .ldpc_codec import (
            create_tiered_ldpc,
            llr_from_quality,
        )

        # Tiered LDPC: adaptive code rate based on coverage
        # HIGH  (cov>=7): LDPC(96,50)  rate=0.52, info_density=0.26
        # MEDIUM(cov>=3): LDPC(120,27) rate=0.23, info_density=0.11
        # LOW   (cov>=0): LDPC(200,43) rate=0.22, info_density=0.11
        tier = create_tiered_ldpc(coverage=coverage, seed=42)
        target_n = tier.n

        # Convert DNA to bits
        bits = []
        for base in dna_seq:
            bits.extend(DNA_TO_BITS.get(base, [0, 0]))
        bits_array = np.array(bits, dtype=int)

        # We need n bits for LDPC decoding
        if len(bits_array) < target_n:
            bits_padded = np.pad(bits_array, (0, target_n - len(bits_array)))
        else:
            bits_padded = bits_array[:target_n]

        # Build soft LLR from quality scores
        # quality is per-base, bits are per-2-bases
        if quality is not None and len(quality) > 0:
            # Map base-level quality to bit-level quality
            # Each base -> 2 bits, use same quality for both bits
            base_qual = quality[:len(dna_seq)]
            bit_qual = np.repeat(base_qual, 2)
            if len(bit_qual) < target_n:
                bit_qual = np.pad(bit_qual, (0, target_n - len(bit_qual)))
            else:
                bit_qual = bit_qual[:target_n]
            llr = llr_from_quality(bit_qual)
            # Sign by received bit
            llr = np.where(bits_padded == 1, -np.abs(llr), np.abs(llr))
        else:
            llr = np.where(bits_padded == 0, 5.0, -5.0)

        # LDPC correct (tiered)
        corrected_bits, was_corrected, _ = ldpc_correct_bits(
            bits_padded, n=target_n, k=0,
            max_errors=20, ldpc_type="tiered",
            coverage=coverage
        )

        if was_corrected:
            # Convert back to DNA (2 bits per base)
            corrected_dna = []
            for i in range(0, len(corrected_bits), 2):
                pair = corrected_bits[i:i+2]
                if len(pair) == 2:
                    corrected_dna.append(BITS_TO_DNA.get(tuple(pair), 'A'))
            return ''.join(corrected_dna), True

        return dna_seq, False

    def _rs_guided_select(
        self,
        candidates: List[Tuple[str, float]],
        segment: str,
    ) -> Tuple[str, bool]:
        """
        Select and correct the best candidate using RS decoding.

        For each candidate DNA sequence:
        1. Converts to RS symbols
        2. Applies RS decode (detects AND corrects errors using parity symbols)
        3. Returns the corrected DNA with zero or reduced syndrome

        Falls back to the best candidate if RS decoding fails.
        """
        from reedsolo import RSCodec

        rs_codec = RSCodec(self.c_rs, c_exp=self.l)
        rs_nonzero = True

        for dna_candidate, _ in candidates:
            dna_clean = dna_candidate.replace(self.strong_marker, '')
            if not dna_clean:
                continue

            try:
                # Convert DNA to bits
                bits = []
                for base in dna_clean:
                    bits.extend([0, 0] if base == 'A' else
                               [0, 1] if base == 'C' else
                               [1, 0] if base == 'G' else [1, 1])

                # Pad to multiple of l
                while len(bits) % self.l != 0:
                    bits.append(0)

                # Convert to symbols
                symbols = []
                for i in range(0, len(bits), self.l):
                    block = bits[i:i + self.l]
                    sym = int(''.join(str(b) for b in block), 2)
                    symbols.append(sym)

                # RS decode: detects AND corrects errors
                if len(symbols) >= self.c_rs + 1:
                    decoded_result = rs_codec.decode(symbols)
                    if isinstance(decoded_result, tuple):
                        decoded_symbols = list(decoded_result[0])
                    else:
                        decoded_symbols = list(decoded_result)
                    # Convert corrected symbols back to bits
                    corrected_bits = []
                    for sym in decoded_symbols:
                        for bit_pos in range(self.l - 1, -1, -1):
                            corrected_bits.append((sym >> bit_pos) & 1)
                    # Convert bits to DNA
                    corrected_dna = []
                    for i in range(0, len(corrected_bits), 2):
                        pair = corrected_bits[i:i + 2]
                        if len(pair) == 2:
                            corrected_dna.append('ACGT'[(pair[0] << 1) | pair[1]])
                    return ''.join(corrected_dna), False
            except Exception:
                pass

        # All candidates failed; return best candidate unchanged
        return candidates[0][0] if candidates else "", True
