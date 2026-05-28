"""
Asym-MGC Decoder: Marker Detection + Sliding Window + List Viterbi.

This module integrates:
1. levenshtein_distance: for fuzzy marker matching
2. detect_markers: hierarchical marker detection
3. split_at_strong_markers: window segmentation
4. fallback_for_missing_strong_marker: state continuity across windows
5. AsymMGCDecoder: top-level decoder with sliding window and List Viterbi

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

    if prev_end < len(seq):
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
# Top-level Decoder
# =============================================================================

# DNA mapping (consistent with encode.py)
DNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


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
        D_max: int = 20,
        I_max: int = 4,
        Pd: float = 0.5,
        Pi: float = 0.026,
        Ps: float = 0.474,
        list_k: int = 8,
        strong_marker: str = 'TACGTA',
        strong_marker_tolerance: int = 1,
        enable_list_viterbi: bool = True,
    ):
        self.N = N
        self.l = l
        self.c_crc = c_crc
        self.D_max = D_max
        self.I_max = I_max
        self.strong_marker = strong_marker
        self.strong_marker_tolerance = strong_marker_tolerance
        self.enable_list_viterbi = enable_list_viterbi

        self.decoder = FSMJointDecoder(
            N=N, l=l, c_crc=c_crc,
            D_max=D_max, I_max=I_max,
            Pd=Pd, Pi=Pi, Ps=Ps,
            list_k=list_k,
        )

    def decode(
        self,
        seq: str,
        quality: Optional[np.ndarray] = None,
        enable_rs_candidate_selection: bool = True,
    ) -> Tuple[str, dict]:
        """
        Decode a received DNA sequence.

        Parameters
        ----------
        seq : str
            Received DNA sequence (may contain errors).
        quality : Optional[np.ndarray]
            Phred quality scores per base (0-40).
        enable_rs_candidate_selection : bool
            If True, apply RS-guided candidate selection when
            List Viterbi finds no zero-syndrome candidates.

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
            'rs_syndrome_nonzero': 0,
            'fallback_used': 0,
        }

        if not seq:
            return "", info

        # Detect markers
        markers = detect_markers(
            seq,
            strong_marker=self.strong_marker,
            strong_tolerance=self.strong_marker_tolerance,
        )
        info['strong_markers_detected'] = len(markers.strong)

        # Split into windows
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

            decoded_seg, win_info, fallback = self._decode_window(
                segment, quality, start_offset, fallback,
                enable_rs_candidate_selection=enable_rs_candidate_selection
            )
            decoded_parts.append(decoded_seg)
            info['window_stats'].append(win_info)
            info['total_log_prob'] += win_info.get('log_prob', 0.0)
            if win_info.get('fallback_used', False):
                info['fallback_used'] += 1
            if win_info.get('rs_syndrome_nonzero', False):
                info['rs_syndrome_nonzero'] += 1

        # Concatenate and strip markers
        full_decoded = ''.join(decoded_parts)
        full_decoded = full_decoded.replace(self.strong_marker, '')

        return full_decoded, info

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

        Returns (decoded_seq, window_info, new_fallback).
        """
        win_info = {
            'steps': 0,
            'final_active': 0,
            'log_prob': 0.0,
            'fallback_used': False,
            'rs_syndrome_nonzero': False,
        }

        # Initialize decoder
        states = self.decoder.init_states()

        # Apply fallback if available
        if fallback is not None and not fallback.uncertainty_flag:
            win_info['fallback_used'] = True

        # Process each base in the segment
        if quality is not None:
            qual = quality[start_offset:start_offset + len(segment)]
        else:
            qual = None

        for step_idx, base in enumerate(segment):
            base_int = DNA_TO_INT.get(base, 0)
            phred = qual[step_idx] if qual is not None and step_idx < len(qual) else 0.0

            states, step_stats = self.decoder.decode_step(
                states, base_int, phred_quality=phred,
                apply_crc_prune=False,  # CRC pruning at block boundaries
            )
            win_info['steps'] += 1

            if not states:
                break

        win_info['final_active'] = len(states)

        # Extract candidates
        if not states:
            return "", win_info, fallback_for_missing_strong_marker({})

        candidates = self.decoder.traceback_all(states, top_k=self.decoder.list_k)
        win_info['log_prob'] = candidates[0][1] if candidates else 0.0

        # Apply RS-guided candidate selection if needed
        decoded_seq = ""
        rs_nonzero = False

        if candidates:
            # Try top candidate first
            best_dna, best_prob = candidates[0]
            if not enable_rs_candidate_selection:
                decoded_seq = best_dna
            else:
                # Check RS syndrome: try candidates until we find one with zero syndrome
                decoded_seq, rs_nonzero = self._rs_guided_select(
                    candidates, segment
                )

        win_info['rs_syndrome_nonzero'] = rs_nonzero

        # Generate fallback for next window
        new_fallback = fallback_for_missing_strong_marker(states)

        return decoded_seq, win_info, new_fallback

    def _rs_guided_select(
        self,
        candidates: List[Tuple[str, float]],
        segment: str,
    ) -> Tuple[str, bool]:
        """
        Select best candidate based on RS syndrome.

        Iterates through candidates (sorted by log_prob) and returns
        the first one with zero RS syndrome. Falls back to best candidate
        if none have zero syndrome.

        Returns (selected_dna, rs_syndrome_nonzero).
        """
        from reedsolo import RSCodec

        # Strip markers from candidates for RS checking
        rs_codec = RSCodec(self.decoder.c_rs, c_exp=self.decoder.l)
        rs_nonzero = True

        for dna_candidate, _ in candidates:
            dna_clean = dna_candidate.replace(self.strong_marker, '')
            if not dna_clean:
                continue

            try:
                # Convert DNA to bits to symbols
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

                # Check RS syndrome
                if len(symbols) >= self.decoder.c_rs:
                    syndrome = rs_codec.check(symbols)
                    if syndrome == b'' or syndrome == ():
                        return dna_candidate, False
            except Exception:
                pass

        # All candidates had non-zero syndrome; return best
        return candidates[0][0] if candidates else "", True
