"""
Robust Anchor System for DNA Data Storage.

Inspired by Composite Hedges Nanopores (CHN, Zhao et al. 2024):
- Select k-mers with lowest error rates from channel model statistics
- Insert robust anchors at regular intervals during encoding
- Cross-validate detected anchors against expected positions and identities
- Only segments bounded by two consecutive valid anchors are kept

Key insight: by knowing EXACTLY where each anchor should be and what it should
be (from encoder metadata), we can filter out false positives with very high
precision. This is far more reliable than hoping a random 5-mer in the data
happens to be an anchor.

Reference: https://www.nature.com/articles/s41467-024-53455-3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Top 3 most error-resistant 5-mers for our channel model (Pd=0.10, Pi=0.03, Ps=0.20)
# Selected from 50k-sample k-mer error rate analysis
DEFAULT_ANCHORS = ['TAGCG', 'TATCC', 'TGACA']
ANCHOR_LEN = 5
DNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# =============================================================================
# Core Utilities
# =============================================================================

def hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    m, n = len(s1), len(s2)
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
    # two rows
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s2[i - 1] == s1[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def fuzzy_match_anchor(
    window: str,
    anchors: List[str],
    tolerance: int = 1,
) -> Optional[Tuple[str, int]]:
    """
    Find the best matching anchor in a window by Hamming distance.

    Returns (matched_anchor, distance) if distance <= tolerance, else None.
    """
    if len(window) < ANCHOR_LEN:
        return None

    best_match = None
    best_dist = tolerance + 1

    for anchor in anchors:
        for start in range(len(window) - ANCHOR_LEN + 1):
            w = window[start:start + ANCHOR_LEN]
            dist = hamming_distance(w, anchor)
            if dist < best_dist:
                best_dist = dist
                best_match = (anchor, dist)

    if best_dist <= tolerance:
        return best_match
    return None


def find_all_robust_anchors(
    seq: str,
    anchors: List[str] = None,
    tolerance: int = 1,
    min_gap: int = 1,
) -> List[Dict]:
    """
    Scan a sequence for all anchors using fuzzy matching.

    Returns list of dicts with keys: anchor, position, distance.
    """
    if anchors is None:
        anchors = DEFAULT_ANCHORS

    detections = []
    i = 0
    while i <= len(seq) - ANCHOR_LEN:
        window = seq[i:i + ANCHOR_LEN]
        match = fuzzy_match_anchor(window, anchors, tolerance)
        if match is not None:
            anchor_seq, dist = match
            detections.append({
                'anchor': anchor_seq,
                'position': i,
                'distance': dist,
            })
            i += ANCHOR_LEN
        else:
            i += 1

    return detections


# =============================================================================
# Cross-Validation
# =============================================================================

@dataclass
class CrossValidatedAnchor:
    """A validated anchor detection."""
    position: int
    matched_anchor: str
    expected_anchor: str
    distance: int
    is_valid: bool


def validate_anchors_at_expected_positions(
    seq: str,
    expected_positions: List[int],
    expected_identities: List[str],
    tolerance: int = 1,
    anchor_len: int = 5,
    position_tolerance: int = 30,
) -> List[CrossValidatedAnchor]:
    """
    Validate anchors by scanning around known expected positions.

    For each expected position, we look for the expected anchor identity
    within a window. This is the key to cross-validation: we know exactly
    what to look for and where.

    Parameters
    ----------
    seq : str
        Received DNA sequence.
    expected_positions : List[int]
        Exact anchor positions from encoder metadata.
    expected_identities : List[str]
        Expected anchor sequences (same length as expected_positions).
    tolerance : int
        Max Hamming distance for a match.
    anchor_len : int
        Length of each anchor.
    position_tolerance : int
        Max position deviation from expected position.

    Returns
    -------
    List[CrossValidatedAnchor]
        Per-position validation results.
    """
    results = []

    for exp_pos, exp_identity in zip(expected_positions, expected_identities):
        search_start = max(0, exp_pos - position_tolerance)
        search_end = min(len(seq) - anchor_len + 1, exp_pos + position_tolerance + anchor_len)

        best_dist = anchor_len + 1
        best_offset = exp_pos

        for offset in range(search_start, search_end):
            window = seq[offset:offset + anchor_len]
            dist = hamming_distance(window, exp_identity)
            if dist < best_dist:
                best_dist = dist
                best_offset = offset

        found = (best_dist <= tolerance) and (abs(best_offset - exp_pos) <= position_tolerance)
        results.append(CrossValidatedAnchor(
            position=best_offset,
            matched_anchor=exp_identity if found else '',
            expected_anchor=exp_identity,
            distance=best_dist if found else best_dist,
            is_valid=found,
        ))

    return results


def extract_segments_between_valid_anchors(
    seq: str,
    validated: List[CrossValidatedAnchor],
    anchor_len: int = 5,
) -> List[Tuple[str, int, int, int]]:
    """
    Extract segments bounded by consecutive valid anchors.

    Parameters
    ----------
    seq : str
        DNA sequence.
    validated : List[CrossValidatedAnchor]
        Validated anchor results.
    anchor_len : int
        Length of each anchor.

    Returns
    -------
    List[Tuple[str, int, int, int]]
        List of (segment, start_pos, end_pos, num_valid_anchors).
    """
    valid_anchors = [v for v in validated if v.is_valid]

    if len(valid_anchors) < 2:
        return []

    segments = []
    for i in range(len(valid_anchors) - 1):
        a1 = valid_anchors[i]
        a2 = valid_anchors[i + 1]

        seg_start = a1.position + anchor_len
        seg_end = a2.position
        segment = seq[seg_start:seg_end]

        if len(segment) >= 4:
            segments.append((segment, seg_start, seg_end, i + 2))

    return segments


# =============================================================================
# Main Decoder API
# =============================================================================

def decode_with_robust_anchors(
    seq: str,
    metadata: dict,
    tolerance: int = 1,
    position_tolerance: int = 30,
) -> Tuple[List[Tuple[str, int, int, int]], dict]:
    """
    Detect segments using multi-anchor cross-validation.

    Parameters
    ----------
    seq : str
        Received DNA sequence (may contain errors).
    metadata : dict
        Encoding metadata (must contain 'anchor_positions' and 'strong_marker_cycle').
    tolerance : int
        Hamming distance tolerance for anchor matching.
    position_tolerance : int
        Maximum deviation from expected anchor position.

    Returns
    -------
    Tuple[List[Tuple], dict]
        (segments, info) where segments = [(segment, start, end, num_valid), ...]
    """
    anchors = metadata.get('strong_marker_cycle', DEFAULT_ANCHORS)
    anchor_positions = metadata.get('anchor_positions', [])
    anchor_identities = [anchors[i % len(anchors)] for i in range(len(anchor_positions))]

    validated = validate_anchors_at_expected_positions(
        seq,
        expected_positions=anchor_positions,
        expected_identities=anchor_identities,
        tolerance=tolerance,
        anchor_len=ANCHOR_LEN,
        position_tolerance=position_tolerance,
    )

    segments = extract_segments_between_valid_anchors(seq, validated, ANCHOR_LEN)
    valid_count = sum(1 for v in validated if v.is_valid)

    info = {
        'valid_anchors': valid_count,
        'total_anchors_expected': len(anchor_positions),
        'segments_found': len(segments),
        'anchors_used': anchors,
        'tolerance': tolerance,
        'position_tolerance': position_tolerance,
        'validated': validated,
    }

    return segments, info


def decode_strand_with_robust_anchors(
    seq: str,
    metadata: dict,
    fsm_decoder_factory,
    quality: Optional[np.ndarray] = None,
    tolerance: int = 1,
    position_tolerance: int = 30,
) -> Tuple[str, dict]:
    """
    Full decode pipeline for a single strand using robust anchors.

    Parameters
    ----------
    seq : str
        Received DNA.
    metadata : dict
        Encoding metadata.
    fsm_decoder_factory : callable
        Factory: int -> FSM decoder instance.
    quality : np.ndarray
        Phred quality scores per base.
    tolerance : int
        Hamming distance tolerance.
    position_tolerance : int
        Position deviation tolerance.

    Returns
    -------
    Tuple[str, dict]
        (decoded_dna, info_dict).
    """
    from .encode import dna_to_base_ints

    anchors = metadata.get('strong_marker_cycle', DEFAULT_ANCHORS)
    anchor_positions = metadata.get('anchor_positions', [])
    anchor_identities = [anchors[i % len(anchors)] for i in range(len(anchor_positions))]

    validated = validate_anchors_at_expected_positions(
        seq,
        expected_positions=anchor_positions,
        expected_identities=anchor_identities,
        tolerance=tolerance,
        anchor_len=ANCHOR_LEN,
        position_tolerance=position_tolerance,
    )

    segments = extract_segments_between_valid_anchors(seq, validated, ANCHOR_LEN)

    decoded_parts = []
    for seg, start_pos, end_pos, num_valid in segments:
        if len(seg) < 4:
            continue

        bi = dna_to_base_ints(seg)
        N_s = len(seg) + 8
        dec = fsm_decoder_factory(N_s)
        st = dec.init_states()
        q = quality[start_pos:start_pos + len(seg)] if quality is not None else None
        q_val = float(np.mean(q)) if q is not None and len(q) > 0 else 30.0

        for b in bi:
            st, _ = dec.decode_step(st, b, phred_quality=q_val, apply_crc_prune=False)
            if not st:
                break
        if st:
            cands = dec.traceback_all(st, top_k=1)
            if cands:
                decoded_parts.append(cands[0][0])

    valid_count = sum(1 for v in validated if v.is_valid)

    info = {
        'valid_anchors': valid_count,
        'total_anchors_expected': len(anchor_positions),
        'segments_found': len(segments),
        'segments_decoded': len(decoded_parts),
        'decoded_dna_len': len(''.join(decoded_parts)),
        'anchors_used': anchors,
        'tolerance': tolerance,
        'position_tolerance': position_tolerance,
    }

    return ''.join(decoded_parts), info


# =============================================================================
# Read Alignment Using Anchors
# =============================================================================

def anchor_align_reads(
    reads: List[str],
    metadata: dict,
    tolerance: int = 1,
    position_tolerance: int = 30,
) -> Tuple[List[np.ndarray], dict]:
    """
    Align multiple noisy reads to the reference anchor framework.

    Uses the known anchor positions and identities from metadata to align
    each read to a common reference frame. This enables reliable consensus
    even when reads have different lengths due to indels.

    Algorithm:
    1. For each read, find valid anchors near expected positions
    2. Compute global scale + shift using least-squares fit
    3. Apply affine transform to build aligned read (same length as reference)

    Parameters
    ----------
    reads : List[str]
        List of noisy DNA reads.
    metadata : dict
        Encoding metadata with anchor_positions and strong_marker_cycle.
    tolerance : int
        Hamming distance tolerance for anchor matching.
    position_tolerance : int
        Maximum deviation from expected anchor position.

    Returns
    -------
    Tuple[List[np.ndarray], dict]
        (aligned_reads, info) where aligned_reads is a list of aligned
        base-int arrays (same length), info contains per-read statistics.
    """
    anchors = metadata.get('strong_marker_cycle', DEFAULT_ANCHORS)
    ref_positions = metadata.get('anchor_positions', [])
    ref_identities = [anchors[i % len(anchors)] for i in range(len(ref_positions))]

    # Reference length: last anchor + its length + trailing data
    ref_len = ref_positions[-1] + ANCHOR_LEN + 200 if ref_positions else max(len(r) for r in reads) + 200

    aligned = []
    info_list = []

    for read in reads:
        validated = validate_anchors_at_expected_positions(
            read,
            expected_positions=ref_positions,
            expected_identities=ref_identities,
            tolerance=tolerance,
            anchor_len=ANCHOR_LEN,
            position_tolerance=position_tolerance,
        )

        # Build (read_pos, ref_pos) pairs by SEQUENCE position in the list
        # The i-th VALID anchor in the read corresponds to the i-th reference anchor
        # (since they appear in the same order)
        pts = []
        for i, v in enumerate(validated):
            if v.is_valid and i < len(ref_positions):
                pts.append((v.position, ref_positions[i]))

        valid_count = sum(1 for v in validated if v.is_valid)

        if len(pts) < 2:
            aligned.append(np.full(ref_len, -1, dtype=np.int32))
            info_list.append({'valid_anchors': valid_count, 'aligned': False})
            continue

        # Compute global scale and shift from first and last matching pairs
        r0, f0 = pts[0]
        r1, f1 = pts[-1]
        dr = r1 - r0
        df = f1 - f0

        if abs(df) > 1 and abs(dr) > 1:
            scale = dr / df
            shift = r0 - f0 * scale
        else:
            scale = len(read) / ref_len
            shift = 0.0

        # Build aligned read
        aligned_read = _align_read_linear(read, ref_len, scale, shift)
        aligned.append(aligned_read)
        info_list.append({
            'valid_anchors': valid_count,
            'aligned': True,
            'scale': scale,
            'shift': shift,
            'read_len': len(read),
            'ref_len': ref_len,
        })

    info = {
        'num_reads': len(reads),
        'aligned_count': sum(1 for i in info_list if i['aligned']),
        'per_read': info_list,
        'ref_length': ref_len,
    }

    return aligned, info


def _align_read_linear(
    read: str,
    ref_len: int,
    scale: float,
    shift: float,
) -> np.ndarray:
    """
    Align a read to reference using a linear (scale + shift) transform.

    For each reference position, computes read_pos = ref_pos * scale + shift.
    """
    _DNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    aligned = np.full(ref_len, -1, dtype=np.int32)

    for rp in range(ref_len):
        read_pos = int(round(rp * scale + shift))
        if 0 <= read_pos < len(read):
            base = read[read_pos]
            if base in _DNA_TO_INT:
                aligned[rp] = _DNA_TO_INT[base]

    return aligned


def majority_vote_consensus(
    aligned_reads: List[np.ndarray],
    min_votes: int = 2,
) -> Tuple[str, np.ndarray]:
    """
    Majority vote consensus from aligned reads.

    Parameters
    ----------
    aligned_reads : List[np.ndarray]
        List of aligned reads (base-int arrays, -1 = missing).
    min_votes : int
        Minimum number of votes for a call.

    Returns
    -------
    Tuple[str, np.ndarray]
        (consensus, per_position_vote_counts).
    """
    if not aligned_reads:
        return '', np.array([])

    INT_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    ref_len = max(len(r) for r in aligned_reads)
    consensus = []
    vote_counts = []

    for pos in range(ref_len):
        counts = [0, 0, 0, 0]  # A, C, G, T
        for read in aligned_reads:
            if pos < len(read):
                val = read[pos]
                if 0 <= val <= 3:
                    counts[val] += 1

        total_votes = sum(counts)
        best_idx = max(range(4), key=counts.__getitem__)
        best_count = counts[best_idx]

        if best_count < min_votes:
            consensus.append('N')
            vote_counts.append(0)
        else:
            consensus.append(INT_TO_BASE[best_idx])
            vote_counts.append(best_count)

    return ''.join(consensus), np.array(vote_counts)


def build_consensus_from_reads(
    reads: List[str],
    metadata: dict,
    tolerance: int = 1,
    position_tolerance: int = 30,
    min_votes: int = 2,
) -> Tuple[str, dict]:
    """
    End-to-end consensus from raw noisy reads.

    Parameters
    ----------
    reads : List[str]
        Raw noisy DNA reads.
    metadata : dict
        Encoding metadata.
    tolerance : int
        Hamming distance tolerance.
    position_tolerance : int
        Position deviation tolerance.
    min_votes : int
        Minimum votes for a consensus call.

    Returns
    -------
    Tuple[str, dict]
        (consensus_seq, info_dict).
    """
    if not reads:
        return '', {'error': 'no reads'}

    aligned_reads, align_info = anchor_align_reads(
        reads, metadata, tolerance=tolerance, position_tolerance=position_tolerance
    )
    consensus, vote_counts = majority_vote_consensus(aligned_reads, min_votes=min_votes)

    # Strip anchor positions to get data-only consensus
    anchors = metadata.get('strong_marker_cycle', DEFAULT_ANCHORS)
    ref_positions = metadata.get('anchor_positions', [])
    anchor_set = set()
    for i, pos in enumerate(ref_positions):
        for k in range(ANCHOR_LEN):
            anchor_set.add(pos + k)

    data_parts = []
    for pos, base in enumerate(consensus):
        if base != 'N' and pos not in anchor_set:
            data_parts.append(base)

    data_consensus = ''.join(data_parts)

    info = {
        'align_info': align_info,
        'consensus_length': len(consensus),
        'data_length': len(data_consensus),
        'n_count': consensus.count('N'),
    }

    return data_consensus, info


# =============================================================================
# Segmented Consensus: Anchor-Defined Boundaries
# =============================================================================

def decode_read_segments(
    read: str,
    metadata: dict,
    fsm_decoder_factory,
    quality: Optional[np.ndarray] = None,
    tolerance: int = 1,
    position_tolerance: int = 30,
) -> List[Tuple[str, int, int, int, int]]:
    """
    Extract segments from a read and decode each segment.

    Uses anchor-defined boundaries from metadata to extract segments,
    then decodes each segment independently.

    Parameters
    ----------
    read : str
        Noisy DNA read.
    metadata : dict
        Encoding metadata.
    fsm_decoder_factory : callable
        Factory: int -> FSM decoder.
    quality : Optional[np.ndarray]
        Phred quality scores.
    tolerance : int
        Hamming distance tolerance.
    position_tolerance : int
        Position deviation tolerance.

    Returns
    -------
    List[Tuple[str, int, int, int, int]]
        List of (decoded_dna, segment_idx, start_pos, end_pos, num_valid_anchors)
        for successfully decoded segments.
    """
    from .encode import dna_to_base_ints

    anchors = metadata.get('strong_marker_cycle', DEFAULT_ANCHORS)
    anchor_positions = metadata.get('anchor_positions', [])
    anchor_identities = [anchors[i % len(anchors)] for i in range(len(anchor_positions))]

    validated = validate_anchors_at_expected_positions(
        read,
        expected_positions=anchor_positions,
        expected_identities=anchor_identities,
        tolerance=tolerance,
        anchor_len=ANCHOR_LEN,
        position_tolerance=position_tolerance,
    )

    valid_anchors = [v for v in validated if v.is_valid]

    decoded_segments = []
    for seg_idx in range(len(valid_anchors) - 1):
        a1 = valid_anchors[seg_idx]
        a2 = valid_anchors[seg_idx + 1]

        seg_start = a1.position + ANCHOR_LEN
        seg_end = a2.position
        segment = read[seg_start:seg_end]

        if len(segment) < 4:
            continue

        bi = dna_to_base_ints(segment)
        N_s = len(segment) + 8
        dec = fsm_decoder_factory(N_s)
        st = dec.init_states()

        q = quality[seg_start:seg_end] if quality is not None and len(quality) >= seg_end else None
        q_val = float(np.mean(q)) if q is not None and len(q) > 0 else 30.0

        for b in bi:
            st, _ = dec.decode_step(st, b, phred_quality=q_val, apply_crc_prune=False)
            if not st:
                break

        if st:
            cands = dec.traceback_all(st, top_k=1)
            if cands:
                decoded_segments.append((
                    cands[0][0],  # decoded DNA string
                    seg_idx,        # segment index
                    seg_start,      # start position
                    seg_end,        # end position
                    seg_idx + 2,   # number of valid anchors bounding this segment
                ))

    return decoded_segments


def segmented_consensus_from_reads(
    reads: List[str],
    metadata: dict,
    fsm_decoder_factory,
    qualities: Optional[List[np.ndarray]] = None,
    tolerance: int = 1,
    position_tolerance: int = 30,
    min_reads_per_segment: int = 2,
) -> Tuple[str, dict]:
    """
    Segmented consensus: decode each segment from each read, then vote per segment.

    Strategy:
    1. For each read, extract segments using anchor-defined boundaries
    2. Decode each segment independently
    3. Group decoded segments by index across all reads
    4. Apply majority voting within each segment
    5. Concatenate voted segments

    This is the approach validated in our experiments: anchor-defined boundaries
    eliminate indel-induced misalignments that plague global alignment methods.

    Parameters
    ----------
    reads : List[str]
        Raw noisy DNA reads.
    metadata : dict
        Encoding metadata.
    fsm_decoder_factory : callable
        Factory: int -> FSM decoder.
    qualities : Optional[List[np.ndarray]]
        Phred quality scores per read.
    tolerance : int
        Hamming distance tolerance.
    position_tolerance : int
        Position deviation tolerance.
    min_reads_per_segment : int
        Minimum reads required to vote on a segment.

    Returns
    -------
    Tuple[str, dict]
        (consensus_seq, info_dict).
    """
    if not reads:
        return '', {'error': 'no reads'}

    if qualities is None:
        qualities = [np.array([]) for _ in reads]

    # Step 1: Decode all segments from all reads
    all_segments = []  # list of per-read decoded segments
    for i, read in enumerate(reads):
        qual = qualities[i] if i < len(qualities) else np.array([])
        segs = decode_read_segments(
            read, metadata, fsm_decoder_factory,
            quality=qual if len(qual) > 0 else None,
            tolerance=tolerance, position_tolerance=position_tolerance,
        )
        all_segments.append(segs)

    # Step 2: Find max segment index across all reads
    max_seg_idx = 0
    for segs in all_segments:
        for decoded, seg_idx, _, _, _ in segs:
            if seg_idx > max_seg_idx:
                max_seg_idx = seg_idx

    if max_seg_idx < 0:
        return '', {'error': 'no segments decoded', 'segments_per_read': [len(s) for s in all_segments]}

    # Step 3: Per-segment majority voting
    consensus_parts = []
    segment_info = []

    for seg_idx in range(max_seg_idx + 1):
        # Collect all decoded strings for this segment index
        seg_variants = []
        for read_idx, segs in enumerate(all_segments):
            for decoded, si, start, end, num_valid in segs:
                if si == seg_idx:
                    seg_variants.append((decoded, read_idx))

        if len(seg_variants) < min_reads_per_segment:
            segment_info.append({
                'seg_idx': seg_idx,
                'num_variants': len(seg_variants),
                'voted': False,
            })
            continue

        # Majority vote within this segment
        voted, vote_count = _majority_vote_string([s[0] for s in seg_variants])
        if voted:
            consensus_parts.append(voted)
            segment_info.append({
                'seg_idx': seg_idx,
                'num_variants': len(seg_variants),
                'voted': True,
                'vote_count': vote_count,
                'length': len(voted),
            })
        else:
            segment_info.append({
                'seg_idx': seg_idx,
                'num_variants': len(seg_variants),
                'voted': False,
            })

    consensus = ''.join(consensus_parts)
    total_voted = sum(1 for s in segment_info if s['voted'])

    info = {
        'num_reads': len(reads),
        'segments_per_read': [len(s) for s in all_segments],
        'total_segments': max_seg_idx + 1,
        'segments_decoded': total_voted,
        'per_segment': segment_info,
        'consensus_length': len(consensus),
    }

    return consensus, info


def _majority_vote_string(strings: List[str]) -> Tuple[Optional[str], int]:
    """
    Majority vote across multiple strings of equal length.

    Parameters
    ----------
    strings : List[str]
        List of strings (should all have the same length).

    Returns
    -------
    Tuple[Optional[str], int]
        (voted_string, max_vote_count). Returns (None, 0) if strings is empty.
    """
    if not strings:
        return None, 0

    seg_len = max(len(s) for s in strings)
    if seg_len == 0:
        return '', 0

    # Pad all strings to same length
    padded = [s.ljust(seg_len, '-') for s in strings]
    result = []
    max_votes = 0

    for pos in range(seg_len):
        counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        for s in padded:
            base = s[pos]
            if base in counts:
                counts[base] += 1

        best_base = max(counts, key=counts.get)
        best_count = counts[best_base]
        max_votes = max(max_votes, best_count)
        result.append(best_base)

    return ''.join(result), max_votes


# =============================================================================
# Raw Base Consensus (Consensus BEFORE Decoding)
# =============================================================================

def raw_base_consensus_from_reads(
    reads: List[str],
    metadata: dict,
    qualities: List[np.ndarray] = None,
    tolerance: int = 1,
    position_tolerance: int = 30,
    min_reads_per_segment: int = 2,
) -> Tuple[str, dict]:
    """
    Consensus from raw reads BEFORE Viterbi decoding.

    Correct pipeline order:
      Raw Reads → Anchor segmentation → NW alignment → Vote → Consensus → Viterbi

    Parameters
    ----------
    reads : List[str]
        Raw noisy DNA reads.
    metadata : dict
        Encoding metadata with anchor_positions and strong_marker_cycle.
    qualities : List[np.ndarray], optional
        Per-read quality arrays. If provided, enables quality-weighted voting.
    tolerance : int
        Hamming distance tolerance for anchor matching.
    position_tolerance : int
        Maximum deviation from expected anchor position.
    min_reads_per_segment : int
        Minimum reads required to produce a voted segment.

    Returns
    -------
    Tuple[str, dict]
        (consensus_seq, info_dict).
    """
    anchors = metadata.get('strong_marker_cycle', DEFAULT_ANCHORS)
    ref_positions = metadata.get('anchor_positions', [])
    ref_identities = [anchors[i % len(anchors)] for i in range(len(ref_positions))]

    if len(ref_positions) < 2:
        return '', {'error': 'need at least 2 anchors for segmentation'}

    # Step 1: Extract segment data from each read
    #   For each read, find valid anchors and map them to reference anchor indices
    #   Segment si is bounded by ref_anchor[si] and ref_anchor[si+1]
    #   We extract the region between these anchors in the read
    seg_data: Dict[int, List[str]] = {}
    seg_quals: Dict[int, List[np.ndarray]] = {}

    for ri, read in enumerate(reads):
        validated = validate_anchors_at_expected_positions(
            read,
            expected_positions=ref_positions,
            expected_identities=ref_identities,
            tolerance=tolerance,
            anchor_len=ANCHOR_LEN,
            position_tolerance=position_tolerance,
        )

        # Build map: for each reference anchor index i, find read position of that anchor
        # valid_anchors[i] = read position of ref_anchor[i] (or None if not found)
        valid_anchors = [None] * len(ref_positions)
        for i, v in enumerate(validated):
            if v.is_valid and i < len(ref_positions):
                valid_anchors[i] = v.position

        # Extract quality for this read if available
        qual = None
        if qualities is not None and ri < len(qualities):
            qual = qualities[ri]

        # Extract each segment: bounded by consecutive valid reference anchors
        for si in range(len(ref_positions) - 1):
            r_pos1 = valid_anchors[si]
            r_pos2 = valid_anchors[si + 1]

            if r_pos1 is None or r_pos2 is None:
                continue

            # Data in read: after anchor si, before anchor si+1
            seg_start = r_pos1 + ANCHOR_LEN
            seg_end = r_pos2

            if seg_end <= seg_start or seg_end - seg_start < 4:
                continue

            seg_seq = read[seg_start:seg_end]
            if len(seg_seq) >= 4:
                if si not in seg_data:
                    seg_data[si] = []
                    seg_quals[si] = []
                seg_data[si].append(seg_seq)
                if qual is not None:
                    seg_quals[si].append(qual[seg_start:seg_end])

    if not seg_data:
        return '', {'error': 'no segments extracted from any read'}

    # Step 2: Per-segment NW alignment + majority vote
    consensus_parts = []
    segment_info = []
    total_voted_bases = 0

    for si in sorted(seg_data.keys()):
        variants = seg_data[si]
        if len(variants) < min_reads_per_segment:
            segment_info.append({
                'seg_idx': si, 'num_reads': len(variants),
                'voted': False, 'reason': 'insufficient_reads',
            })
            continue

        # Progressive alignment: align variants to each other, building consensus
        # Start with first variant, progressively align remaining variants
        aligned = [variants[0]]
        consensus = variants[0]
        qual_aligned = []
        if si in seg_quals and len(seg_quals[si]) > 0:
            qual_aligned = [seg_quals[si][0]]

        for vi, variant in enumerate(variants[1:]):
            a_ref, a_seq = _nw_align(consensus, variant)
            # Merge: for each position, keep base if both have it, else gap
            merged = []
            for ra, rb in zip(a_ref, a_seq):
                if ra == rb:
                    merged.append(ra)
                elif ra == '-':
                    merged.append(rb)
                elif rb == '-':
                    merged.append(ra)
                else:
                    merged.append(ra)
            consensus = ''.join(merged)
            aligned.append(consensus)

            # Also align quality
            if si in seg_quals and vi + 1 < len(seg_quals[si]):
                src_qual = seg_quals[si][vi + 1]
                qual_merged = []
                qi = 0
                for ra, rb in zip(a_ref, a_seq):
                    if ra == rb:
                        qual_merged.append(src_qual[qi] if qi < len(src_qual) else 20.0)
                        qi += 1
                    elif ra == '-':
                        qual_merged.append(src_qual[qi] if qi < len(src_qual) else 20.0)
                        qi += 1
                    elif rb == '-':
                        qual_merged.append(20.0)
                    else:
                        qual_merged.append(src_qual[qi] if qi < len(src_qual) else 20.0)
                        qi += 1
                qual_aligned.append(np.array(qual_merged[:len(consensus)]))

        # Quality-weighted vote column-wise
        max_len = max(len(a) for a in aligned)
        padded = [a.ljust(max_len, '-') for a in aligned]

        use_quality = (qual_aligned is not None and
                        len(qual_aligned) == len(padded) and
                        all(len(q) >= max_len for q in qual_aligned))

        votes = []
        for pos in range(max_len):
            if use_quality:
                # Quality-weighted scoring
                scores = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
                for pi, p in enumerate(padded):
                    b = p[pos]
                    if b in scores:
                        q = qual_aligned[pi][pos] if pos < len(qual_aligned[pi]) else 20.0
                        scores[b] += 10.0 ** (q / 10.0)
                best = max(scores, key=scores.get)
                best_score = scores[best]
                # Require at least 50% of weighted sum from best base
                if best_score < sum(scores.values()) * 0.5:
                    best = 'N'
            else:
                # Simple majority
                counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
                for p in padded:
                    b = p[pos]
                    if b in counts:
                        counts[b] += 1
                best = max(counts, key=counts.get)
                best_n = counts[best]
                if best_n < min_reads_per_segment:
                    best = 'N'
            votes.append(best)

        voted_seg = ''.join(votes).replace('-', '')
        consensus_parts.append(voted_seg)
        total_voted_bases += len(voted_seg)

        segment_info.append({
            'seg_idx': si,
            'num_reads': len(variants),
            'voted': True,
            'voted_length': len(voted_seg),
        })

    consensus = ''.join(consensus_parts)

    info = {
        'num_reads': len(reads),
        'segments_extracted': len(seg_data),
        'segments_voted': sum(1 for s in segment_info if s.get('voted')),
        'consensus_length': len(consensus),
        'total_voted_bases': total_voted_bases,
        'per_segment': segment_info,
    }

    return consensus, info


def _nw_align(ref: str, seq: str, gap_penalty: float = 2.0) -> Tuple[str, str]:
    """
    Needleman-Wunsch global alignment between ref and seq.

    Returns (aligned_ref, aligned_seq).
    """
    m, n = len(ref), len(seq)
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for j in range(1, n + 1):
        dp[0][j] = j * gap_penalty
    for i in range(1, m + 1):
        dp[i][0] = i * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == seq[j - 1]:
                match = dp[i - 1][j - 1]
            else:
                match = dp[i - 1][j - 1] + 1
            ins = dp[i][j - 1] + gap_penalty
            dele = dp[i - 1][j] + gap_penalty
            dp[i][j] = min(match, ins, dele)

    # Backtrack
    aligned_ref, aligned_seq = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == seq[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned_ref.append(ref[i - 1])
                aligned_seq.append(seq[j - 1])
                i -= 1
                j -= 1
                continue
        if j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + gap_penalty):
            aligned_ref.append('-')
            aligned_seq.append(seq[j - 1])
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + gap_penalty):
            aligned_ref.append(ref[i - 1])
            aligned_seq.append('-')
            i -= 1

    return ''.join(reversed(aligned_ref)), ''.join(reversed(aligned_seq))


# =============================================================================
# Utility: Insert Anchors Into DNA
# =============================================================================

def insert_anchors_into_dna(
    dna: str,
    anchors: List[str],
    every: int = 32,
) -> Tuple[str, List[int]]:
    """
    Insert robust anchors into a DNA sequence at regular intervals.

    Parameters
    ----------
    dna : str
        DNA sequence.
    anchors : List[str]
        Anchor sequences to cycle through.
    every : int
        Insert anchor every N bases.

    Returns
    -------
    Tuple[str, List[int]]
        (dna_with_anchors, anchor_positions).
    """
    result = []
    positions = []
    anchor_idx = 0
    pos = 0

    while pos < len(dna):
        chunk_end = min(pos + every, len(dna))
        result.append(dna[pos:chunk_end])
        pos = chunk_end

        if pos < len(dna):
            anchor = anchors[anchor_idx % len(anchors)]
            positions.append(sum(len(p) for p in result))
            result.append(anchor)
            anchor_idx += 1

    return ''.join(result), positions
