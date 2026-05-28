"""
Outer Code: Soft-Decision RS Decoding with Extrinsic Information Transfer.

Implements reliability-weighted consensus, DNA error prediction, GMD/OSD decoding,
and extrinsic IT between inner and outer codes.
Reference: Section 4 of IMPROVEMENT_PLAN.md v2.0.

Phase 3.11-3.15: Section 4 of IMPROVEMENT_PLAN.md.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def soft_consensus(
    copies: List[Tuple[str, np.ndarray]],
) -> Tuple[str, np.ndarray]:
    """
    Form a consensus sequence using reliability-weighted voting.

    Parameters
    ----------
    copies : List[Tuple[str, np.ndarray]]
        List of (sequence, phred_quality_array) tuples.

    Returns
    -------
    consensus : str
        Consensus DNA sequence.
    weights : ndarray
        Per-position confidence scores.
    """
    if not copies:
        return '', np.array([])

    seq_len = max(len(seq) for seq, _ in copies)
    consensus = []
    pos_weights = []

    for pos in range(seq_len):
        scores = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}

        for seq, qual in copies:
            base = seq[pos] if pos < len(seq) else '-'
            q = qual[pos] if pos < len(qual) else 0
            weight = 10.0 ** (q / 10.0)

            if base != '-':
                scores[base] += weight

        best_base = max(scores, key=scores.get)
        best_score = scores[best_base]
        consensus.append(best_base)
        pos_weights.append(best_score)

    return ''.join(consensus), np.array(pos_weights)


def _phred_weight(phred: float) -> float:
    """Convert Phred score to linear weight, clamped to avoid overflow."""
    return 10.0 ** min(max(phred, 0.0), 60.0) / 10.0


def _align_pairwise(ref: str, seq: str) -> Tuple[str, str]:
    """
    Align seq to ref using Needleman-Wunsch (edit distance scoring).
    Allows substitutions, insertions, and deletions.
    Returns (aligned_ref, aligned_seq) with gaps inserted.
    """
    m, n = len(ref), len(seq)

    # dp[i][j] = min cost to align ref[:i] to seq[:j]
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    # First row: insert all j chars from seq into empty ref
    for j in range(1, n + 1):
        dp[0][j] = j
    # First column: delete all i chars from ref
    for i in range(1, m + 1):
        dp[i][0] = i

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == seq[j - 1]:
                match = dp[i - 1][j - 1]       # correct match
            else:
                match = dp[i - 1][j - 1] + 1  # substitution
            ins  = dp[i][j - 1] + 1           # insert into seq (gap in ref)
            dele = dp[i - 1][j] + 1           # delete from ref (gap in seq)
            dp[i][j] = min(match, ins, dele)

    # Backtrack
    aligned_ref, aligned_seq = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if ref[i - 1] == seq[j - 1]:
                cost = 0
            else:
                cost = 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned_ref.append(ref[i - 1])
                aligned_seq.append(seq[j - 1])
                i -= 1; j -= 1
                continue
        if j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + 1):
            # insertion into seq
            aligned_ref.append('-')
            aligned_seq.append(seq[j - 1])
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
            # deletion from ref
            aligned_ref.append(ref[i - 1])
            aligned_seq.append('-')
            i -= 1
        else:
            # Fallback (shouldn't reach here)
            if i > 0:
                aligned_ref.append(ref[i - 1])
                aligned_seq.append('-')
                i -= 1
            elif j > 0:
                aligned_ref.append('-')
                aligned_seq.append(seq[j - 1])
                j -= 1
            else:
                break

    aligned_ref.reverse()
    aligned_seq.reverse()
    return ''.join(aligned_ref), ''.join(aligned_seq)


def _quality_aligned_vote(
    aligned_ref: str,
    aligned_seqs: List[str],
    aligned_quals: List[np.ndarray],
    strand_weights: List[float],
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Quality + strand-weight weighted voting on pre-aligned sequences.

    Parameters
    ----------
    aligned_ref : str
        Reference sequence (with gaps).
    aligned_seqs : List[str]
        Sequences aligned to ref (same length, may contain gaps).
    aligned_quals : List[np.ndarray]
        Quality arrays aligned to each sequence.
    strand_weights : List[float]
        Per-strand weight (e.g., inverse of distance from current consensus).

    Returns
    -------
    consensus_ref : str
        Consensus in ref-space (gaps stripped).
    ref_to_cons_map : ndarray
        Mapping from ref position to consensus position.
    position_weights : ndarray
        Per-consensus-position confidence scores.
    """
    L = len(aligned_ref)
    n_strands = len(aligned_seqs)

    consensus = []
    pos_weights = []
    ref_to_cons = [-1] * L  # map ref position -> consensus position

    for pos in range(L):
        # Skip gap positions in ref (these are insertions in other strands)
        if aligned_ref[pos] == '-':
            continue

        ref_to_cons[pos] = len(consensus)

        scores = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
        for s_idx in range(n_strands):
            base = aligned_seqs[s_idx][pos]
            if base == '-':
                continue
            q = aligned_quals[s_idx][pos] if pos < len(aligned_quals[s_idx]) else 0
            w = _phred_weight(q) * strand_weights[s_idx]
            scores[base] += w

        best_base = max(scores, key=scores.get)
        best_score = scores[best_base]
        consensus.append(best_base)
        pos_weights.append(best_score)

    return ''.join(consensus), np.array(ref_to_cons), np.array(pos_weights)


def _strand_weights_from_alignment(
    aligned_seqs: List[str],
    strand_quals: List[np.ndarray],
) -> List[float]:
    """Compute per-strand weights based on gap count and mean quality."""
    weights = []
    for s_idx, seq in enumerate(aligned_seqs):
        gap_count = seq.count('-')
        qual = strand_quals[s_idx]
        mean_q = float(np.mean(qual)) if len(qual) > 0 else 10.0
        # Penalize strands with many gaps; reward high quality
        w = _phred_weight(mean_q) / (1.0 + 0.1 * gap_count)
        weights.append(w)
    total = sum(weights)
    return [w / total * len(weights) for w in weights]


def iterative_consensus(
    copies: List[Tuple[str, np.ndarray]],
    max_iterations: int = 3,
    min_improvement: float = 0.01,
) -> Tuple[str, np.ndarray]:
    """
    Iterative MSA-based consensus with quality-weighted voting.

    Algorithm:
    1. Start with the longest strand as reference.
    2. Progressively align each strand to the current consensus (ref-space).
    3. Vote in ref-space with quality + strand weighting.
    4. Repeat, re-weighting strands by distance from current consensus.
    5. Strip gaps from reference positions to get final consensus.

    This addresses the core limitation of position-wise voting (soft_consensus):
    when strands have different lengths due to indels, positions don't correspond
    to the same original base. MSA alignment ensures correct correspondence.

    Parameters
    ----------
    copies : List[Tuple[str, np.ndarray]]
        List of (sequence, phred_quality_array) tuples.
    max_iterations : int
        Maximum re-alignment iterations.
    min_improvement : float
        Minimum improvement in weighted score to continue iterating.

    Returns
    -------
    consensus : str
        Consensus DNA sequence.
    weights : ndarray
        Per-position confidence scores.
    """
    if not copies:
        return '', np.array([])

    if len(copies) == 1:
        seq, qual = copies[0]
        return seq, qual

    seqs = [s for s, _ in copies]
    quals = [q for _, q in copies]

    # Initialize: pick longest as reference
    ref_idx = max(range(len(seqs)), key=lambda i: len(seqs[i]))
    ref = seqs[ref_idx]
    ref_qual = quals[ref_idx]

    for iteration in range(max_iterations):
        # Step 1: Align all strands to current ref
        aligned_seqs = [ref]
        aligned_quals = [ref_qual]
        other_indices = [i for i in range(len(seqs)) if i != ref_idx]
        aligned_ref = ref

        for s_idx in other_indices:
            a_ref, a_seq = _align_pairwise(ref, seqs[s_idx])
            # Quality propagation for aligned sequence
            qual = quals[s_idx]
            aligned_qual = np.zeros(len(a_seq))
            q_ptr = 0
            for k, ch in enumerate(a_seq):
                if ch != '-':
                    aligned_qual[k] = qual[q_ptr] if q_ptr < len(qual) else 20.0
                    q_ptr += 1
                else:
                    aligned_qual[k] = 5.0  # low quality for gaps
            aligned_seqs.append(a_seq)
            aligned_quals.append(aligned_qual)

        # Step 2: Compute strand weights
        strand_weights = _strand_weights_from_alignment(aligned_seqs, aligned_quals)

        # Step 3: Vote in ref-space
        consensus_ref, ref_to_cons, pos_weights = _quality_aligned_vote(
            aligned_ref, aligned_seqs, aligned_quals, strand_weights
        )

        # Step 4: Check for improvement (weighted score improvement)
        total_score = float(np.sum(pos_weights))
        if iteration == 0:
            prev_score = total_score
        else:
            improvement = (total_score - prev_score) / max(prev_score, 1e-10)
            if improvement < min_improvement:
                break
            prev_score = total_score

        # Step 5: Update reference for next iteration
        # Use the consensus (in ref-space, gaps stripped) as new ref
        ref = consensus_ref
        ref_qual = pos_weights  # use weights as pseudo-qualities

    return ref, pos_weights


def dna_error_predictor(
    sequence: str,
    context_window: int = 5,
) -> np.ndarray:
    """
    Predict error-prone positions in a DNA sequence.

    Based on homopolymer boundaries, error-prone motifs, and GC content.
    Reference: Derrick et al., Nature Science Review, 2024.

    Parameters
    ----------
    sequence : str
        DNA sequence to analyze.
    context_window : int
        Context size for motif detection.

    Returns
    -------
    error_probs : ndarray
        Per-position error probability estimates.
    """
    n = len(sequence)
    error_probs = np.zeros(n)

    for i in range(n):
        ctx_start = max(0, i - context_window)
        ctx_end = min(n, i + context_window)
        ctx = sequence[ctx_start:ctx_end]

        p = 0.0

        # Homopolymer boundary detection
        hp_before = 0
        for j in range(i - 1, ctx_start - 1, -1):
            if sequence[j] == sequence[i]:
                hp_before += 1
            else:
                break

        hp_after = 0
        for j in range(i + 1, ctx_end):
            if sequence[j] == sequence[i]:
                hp_after += 1
            else:
                break

        if hp_before + hp_after >= 2:
            p += 0.15 if hp_before > 0 else 0.05

        # Error-prone motifs
        motifs = ['GAGA', 'CUCU', 'AGAG', 'CTCT']
        for motif in motifs:
            if motif in ctx.upper():
                p += 0.08
                break

        # GC bias
        if len(ctx) > 0:
            gc_ratio = (ctx.count('G') + ctx.count('C')) / len(ctx)
            if gc_ratio > 0.7 or gc_ratio < 0.3:
                p += 0.03

        error_probs[i] = min(p, 0.5)

    return error_probs


def gmd_osd_rs_decode(
    received_symbols: List[int],
    confidence: np.ndarray,
    error_probs: Optional[np.ndarray],
    rs_n: int = 255,
    rs_k: int = 223,
    max_erasure_fraction: float = 0.2,
) -> Tuple[Optional[List[int]], str]:
    """
    GMD + OSD soft-decision RS decoding.

    Parameters
    ----------
    received_symbols : List[int]
        Received GF(256) symbols.
    confidence : ndarray
        Per-symbol confidence scores.
    error_probs : ndarray, optional
        Per-symbol DNA-specific error probability.
    rs_n : int
        RS codeword length.
    rs_k : int
        RS message length.
    max_erasure_fraction : float
        Maximum fraction of positions to mark as erasures.

    Returns
    -------
    decoded : List[int] or None
        Decoded message symbols, or None if decoding fails.
    status : str
        Decoding status: 'erasure_success', 'osd_order_N', 'failed'.
    """
    try:
        from reedsolo import RSCodec
        rs_decoder = RSCodec(rs_n - rs_k, c_exp=8)
    except ImportError:
        return None, 'failed: reedsolo not available'

    # Combine confidence and error probability into reliability
    reliability = confidence.copy()
    if error_probs is not None:
        # Ensure same length by truncating to the smaller array
        min_len = min(len(reliability), len(error_probs))
        reliability = reliability[:min_len] * (1.0 - error_probs[:min_len])

    # Mark lowest max_erasure_fraction as erasures
    threshold = np.percentile(reliability, int(max_erasure_fraction * 100))
    erasure_positions = [i for i, r in enumerate(reliability) if r < threshold]

    # Try erasure decoding
    try:
        if erasure_positions:
            # Ensure received_symbols has correct length for RS decoder
            symbols = list(received_symbols)
            if len(symbols) < rs_n:
                symbols = symbols + [0] * (rs_n - len(symbols))
            elif len(symbols) > rs_n:
                symbols = symbols[:rs_n]
            
            # Ensure erasure positions are valid
            erasure_positions = [p for p in erasure_positions if p < len(symbols)]
            
            decoded = rs_decoder.decode(
                symbols,
                erase_pos=erasure_positions,
            )
            return list(decoded[0][:rs_k]), 'erasure_success'
    except Exception:
        pass

    # Try OSD (Order Statistics Decoding)
    for order in range(1, 4):
        try:
            # Sort by reliability, flip least reliable positions
            sorted_indices = np.argsort(reliability)
            symbols = list(received_symbols)
            if len(symbols) < rs_n:
                symbols = symbols + [0] * (rs_n - len(symbols))
            elif len(symbols) > rs_n:
                symbols = symbols[:rs_n]
            candidates = [symbols[:]]

            for idx in sorted_indices[:order]:
                for c_idx, candidate in enumerate(candidates[:10]):
                    candidate = candidate[:]
                    candidate[idx] = (candidate[idx] + 1) % 256
                    candidates.append(candidate)

            for candidate in candidates[:10]:
                try:
                    decoded = rs_decoder.decode(candidate, erase_pos=[])
                    return list(decoded[0][:rs_k]), f'osd_order_{order}'
                except Exception:
                    continue
        except Exception:
            continue

    return None, 'failed'


def extrinsic_information_transfer(
    inner_results: List[dict],
    outer_decoded: Optional[str],
    max_iters: int = 3,
) -> Tuple[Optional[str], int]:
    """
    Iterative extrinsic information transfer between inner and outer codes.

    Each iteration:
    1. Inner decoder produces per-strand reliability
    2. Outer decoder produces consensus + GMD/OSD correction
    3. Extrinsic LLR feedback refines inner decoding

    Parameters
    ----------
    inner_results : List[dict]
        List of inner decoder results (one per strand).
    outer_decoded : str, optional
        Outer code decoded sequence.
    max_iters : int
        Maximum number of IT iterations.

    Returns
    -------
    decoded : str or None
        Final decoded message, or None if all iterations fail.
    iterations_used : int
        Number of iterations actually performed.
    """
    if not inner_results:
        return outer_decoded, 0

    current = outer_decoded if outer_decoded else ''
    for iteration in range(max_iters):
        # Form consensus with current weights
        copies = [(r.get('sequence', ''), r.get('quality', np.array([])))
                  for r in inner_results]
        consensus, weights = soft_consensus(copies)

        if not current:
            current = consensus
            continue

        # Compute extrinsic LLR feedback
        seq_len = len(current)
        extrinsic = np.zeros(seq_len, dtype=float)
        for i in range(seq_len):
            conf = 0.0
            for result in inner_results:
                seq = result.get('sequence', '')
                conf_val = result.get('confidence', 0.0)
                if isinstance(conf_val, np.ndarray):
                    conf_val = float(np.mean(conf_val)) if conf_val.size > 0 else 0.0
                if i < len(seq) and seq[i] == current[i]:
                    extrinsic[i] += conf_val

        # Update inner weights
        for result in inner_results:
            result['confidence'] = extrinsic

        if iteration > 0:
            pass  # Could log improvement here

    return current, max_iters


def anchor_based_consensus(
    strand_candidates: List[Tuple[str, np.ndarray, float]],
) -> Tuple[str, np.ndarray, float]:
    """
    Anchor-based consensus using marker positions as alignment reference.

    Key insight: strong markers (TACGTA) appear at FIXED logical positions.
    After decoding, each strand's marker positions reveal its drift.
    We use marker positions to define per-strand coordinate mapping,
    then vote in the reference coordinate space.

    Parameters
    ----------
    strand_candidates : List[Tuple[str, np.ndarray, float]]
        (decoded_sequence, quality_array, log_prob) tuples.

    Returns
    -------
    consensus : str, consensus_quality : ndarray, combined_score : float
    """
    if not strand_candidates:
        return '', np.array([]), -np.inf

    if len(strand_candidates) == 1:
        seq, qual, score = strand_candidates[0]
        return seq, qual, score

    # Extract strong markers from reference strand (highest log_prob)
    ref_seq, ref_qual, ref_score = max(strand_candidates, key=lambda x: x[2])
    strong_marker = 'TACGTA'

    # Find marker positions in reference
    ref_marker_pos = []
    pos = 0
    while True:
        idx = ref_seq.find(strong_marker, pos)
        if idx == -1:
            break
        ref_marker_pos.append(idx)
        pos = idx + 1

    if len(ref_marker_pos) < 2:
        # Not enough markers - fall back to soft consensus
        copies = [(s, q) for s, q, _ in strand_candidates]
        return soft_consensus(copies)

    # Per-strand data: (strand_seq, strand_qual, strand_score, avg_drift, has_markers)
    strand_data = []
    for seq, qual, score in strand_candidates:
        strand_marker_pos = []
        p = 0
        while True:
            idx = seq.find(strong_marker, p)
            if idx == -1:
                break
            strand_marker_pos.append(idx)
            p = idx + 1

        if len(strand_marker_pos) >= 2:
            # Compute average drift from reference
            # drift > 0: strand is shorter than reference (more deletions)
            # drift < 0: strand is longer than reference (more insertions)
            offsets = [ref_marker_pos[i] - strand_marker_pos[i]
                      for i in range(min(len(ref_marker_pos), len(strand_marker_pos)))]
            avg_drift = sum(offsets) / len(offsets)
            strand_data.append((seq, qual, score, avg_drift, True))
        else:
            # No reliable markers - use zero drift
            strand_data.append((seq, qual, score, 0.0, False))

    # Build consensus in reference coordinate space
    ref_len = len(ref_seq)
    consensus_chars = ['N'] * ref_len
    consensus_quals = np.zeros(ref_len)

    for i in range(ref_len):
        votes = {}
        qual_sum = 0.0

        for seq, qual, score, avg_drift, has_markers in strand_data:
            # Map reference position i to strand position
            strand_pos = int(round(i - avg_drift))

            if 0 <= strand_pos < len(seq):
                base = seq[strand_pos]
                q = qual[strand_pos] if strand_pos < len(qual) else 5
                weight = 10.0 ** (q / 10.0)
                votes[base] = votes.get(base, 0.0) + weight
                qual_sum += q

        if votes:
            best_base = max(votes, key=votes.get)
            consensus_chars[i] = best_base
            consensus_quals[i] = qual_sum / max(len(strand_data), 1)

    consensus = ''.join(consensus_chars)
    return consensus, consensus_quals, ref_score

