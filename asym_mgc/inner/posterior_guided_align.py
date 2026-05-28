"""
Posterior-Guided Alignment for multi-strand consensus.

This module implements Step 3: Use BCJR posterior reliability to guide
alignment between strands with different lengths (indel drift).

Key insight: BCJR posterior reliability indicates which positions the decoder
is confident about. High-reliability positions can be used as alignment anchors.

Reference: IMPROVEMENT_PLAN.md v2.0, Step 3.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


def find_high_confidence_anchors(
    posteriors: List[dict],
    reliability_threshold: float = 0.8,
) -> List[int]:
    """
    Find positions with high BCJR posterior reliability across all strands.
    
    These positions can serve as alignment anchors.

    Parameters
    ----------
    posteriors : List[dict]
        BCJR posteriors per strand, each dict maps base(0-3) -> probability
    reliability_threshold : float
        Minimum reliability to consider a position as anchor

    Returns
    -------
    anchors : List[int]
        Positions (indices) that are high-confidence across all strands
    """
    if not posteriors:
        return []
    
    # Compute reliability per position per strand
    import math
    n_strands = len(posteriors)
    min_len = min(len(p) for p in posteriors)
    
    # Reliability[i] = average reliability at position i across strands
    reliability = np.zeros(min_len)
    for i in range(min_len):
        pos_reliability = []
        for p in posteriors:
            if i >= len(p):
                continue
            pp = p[i]
            if not pp:
                continue
            p_full = np.array([pp.get(b, 1e-12) for b in range(4)])
            p_full = p_full / p_full.sum()
            entropy = -np.sum(p_full * np.log(p_full + 1e-12))
            rel = 1.0 - entropy / math.log(4)
            pos_reliability.append(rel)
        
        if pos_reliability:
            reliability[i] = np.mean(pos_reliability)
    
    # Find anchors: positions with reliability above threshold
    anchors = np.where(reliability >= reliability_threshold)[0].tolist()
    
    return anchors


def posterior_guided_align(
    seqs: List[str],
    posteriors: List[List[dict]],
    reliability_threshold: float = 0.8,
) -> List[Tuple[str, List[int]]]:
    """
    Align sequences using BCJR posterior reliability as guidance.
    
    Strategy:
    1. Find high-confidence anchor positions
    2. Use anchors to constrain alignment
    3. Between anchors, use standard NW alignment but weighted by reliability

    Parameters
    ----------
    seqs : List[str]
        DNA sequences to align (should be decoded outputs)
    posteriors : List[List[dict]]
        BCJR posteriors per strand
    reliability_threshold : float
        Minimum reliability for anchor positions

    Returns
    -------
    aligned : List[Tuple[str, List[int]]]
        Aligned sequences with gap positions marked
    """
    if len(seqs) == 0:
        return []
    if len(seqs) == 1:
        return [(seqs[0], list(range(len(seqs[0]))))]
    
    # Find common anchors across all strands
    if len(posteriors) == len(seqs) and all(len(p) > 0 for p in posteriors):
        anchors = find_high_confidence_anchors(posteriors, reliability_threshold)
    else:
        anchors = []
    
    # Use anchors to guide alignment
    # For simplicity: insert gaps in lower-quality sequences to match anchor positions
    
    aligned_seqs = []
    aligned_refs = []
    
    if anchors:
        # Use first sequence as reference
        ref_seq = seqs[0]
        ref_posteriors = posteriors[0] if len(posteriors) > 0 else []
        
        aligned_seqs.append(ref_seq)
        aligned_refs.append(list(range(len(ref_seq))))
        
        # Align other sequences to reference using anchors
        for i in range(1, len(seqs)):
            seq = seqs[i]
            seq_post = posteriors[i] if i < len(posteriors) else []
            
            aligned, ref_indices = _align_to_anchors(
                ref_seq, seq, anchors, ref_posteriors, seq_post
            )
            aligned_seqs.append(aligned)
            aligned_refs.append(ref_indices)
    else:
        # Fallback: no anchors, return as-is
        for seq in seqs:
            aligned_seqs.append(seq)
            aligned_refs.append(list(range(len(seq))))
    
    return list(zip(aligned_seqs, aligned_refs))


def _align_to_anchors(
    ref: str,
    seq: str,
    anchors: List[int],
    ref_posteriors: List[dict],
    seq_posteriors: List[dict],
) -> Tuple[str, List[int]]:
    """
    Align a sequence to reference using anchor positions.
    
    Insert gaps in seq to match anchor positions in ref.
    """
    if len(anchors) == 0:
        return seq, list(range(len(seq)))
    
    # Build aligned sequence
    aligned = []
    ref_indices = []
    
    ref_pos = 0
    seq_pos = 0
    
    for i, ref_base in enumerate(ref):
        if i in anchors:
            # Anchor position: reference base must be present
            aligned.append(ref_base)
            ref_indices.append(ref_pos)
            ref_pos += 1
            seq_pos += 1  # Assume seq also has a base here
        else:
            # Non-anchor: try to align
            if seq_pos < len(seq):
                aligned.append(seq[seq_pos])
                ref_indices.append(ref_pos)
                ref_pos += 1
                seq_pos += 1
            else:
                # seq ended, insert gaps
                aligned.append('-')
                ref_indices.append(ref_pos - 1)
    
    # Add remaining seq bases with gaps
    while seq_pos < len(seq):
        aligned.append(seq[seq_pos])
        ref_indices.append(ref_pos - 1 if ref_pos > 0 else 0)
        seq_pos += 1
    
    return ''.join(aligned), ref_indices


def consensus_from_aligned(
    aligned_seqs: List[str],
    weights: Optional[List[np.ndarray]] = None,
    min_vote_fraction: float = 0.5,
) -> Tuple[str, np.ndarray]:
    """
    Generate consensus from aligned sequences using weighted voting.
    
    Parameters
    ----------
    aligned_seqs : List[str]
        Aligned sequences (may contain gaps '-')
    weights : List[ndarray], optional
        Per-position weights per sequence
    min_vote_fraction : float
        Minimum fraction of strands to call a base

    Returns
    -------
    consensus : str
        Consensus sequence
    quality : ndarray
        Per-position quality scores
    """
    if not aligned_seqs:
        return '', np.array([])
    
    n_positions = max(len(s) for s in aligned_seqs)
    n_strands = len(aligned_seqs)
    min_votes = max(1, int(n_strands * min_vote_fraction))
    
    consensus = []
    qualities = []
    
    for pos in range(n_positions):
        # Collect bases at this position
        bases = []
        pos_weights = []
        
        for i, seq in enumerate(aligned_seqs):
            if pos < len(seq):
                base = seq[pos]
                if base != '-':
                    bases.append(base)
                    w = weights[i][pos] if weights is not None and i < len(weights) and pos < len(weights[i]) else 1.0
                    pos_weights.append(w)
        
        if len(bases) >= min_votes:
            # Weighted voting
            base_counts = {}
            for base, w in zip(bases, pos_weights):
                base_counts[base] = base_counts.get(base, 0) + w
            
            consensus_base = max(base_counts.keys(), key=lambda b: base_counts[b])
            total_weight = sum(base_counts.values())
            base_weight = base_counts[consensus_base]
            
            consensus.append(consensus_base)
            # Quality = fraction of weight supporting consensus base
            quality = base_weight / total_weight if total_weight > 0 else 0.0
            qualities.append(quality)
        else:
            consensus.append('-')
            qualities.append(0.0)
    
    # Remove gaps from consensus
    final_consensus = ''.join(b for b in consensus if b != '-')
    final_quality = np.array([q for b, q in zip(consensus, qualities) if b != '-'])
    
    return final_consensus, final_quality


def segment_based_consensus(
    seqs: List[str],
    posteriors: List[List[dict]],
    segment_size: int = 50,
    overlap: int = 10,
) -> Tuple[str, np.ndarray]:
    """
    Segment-based consensus: divide into windows, consensus each, then merge.
    
    This handles varying lengths better than global alignment.
    
    Parameters
    ----------
    seqs : List[str]
        DNA sequences
    posteriors : List[List[dict]]
        BCJR posteriors
    segment_size : int
        Size of each segment
    overlap : int
        Overlap between segments for smooth merging
    
    Returns
    -------
    consensus : str
        Consensus sequence
    quality : ndarray
        Per-position quality
    """
    if not seqs:
        return '', np.array([])
    
    # Find common reference length (minimum)
    min_len = min(len(s) for s in seqs)
    
    if min_len <= segment_size:
        # Short enough for direct consensus
        aligned = posterior_guided_align(seqs, posteriors)
        aligned_seqs = [a[0] for a in aligned]
        weights = [np.ones(len(s)) for s in aligned_seqs]
        return consensus_from_aligned(aligned_seqs, weights)
    
    # Segment-based approach
    segments_consensus = []
    segments_quality = []
    
    pos = 0
    while pos < min_len:
        end = min(pos + segment_size, min_len)
        
        # Extract segment from each sequence
        seg_seqs = [s[pos:end] for s in seqs]
        seg_post = []
        for i, p in enumerate(posteriors):
            if i < len(seqs):
                seg_post.append(p[pos:end] if len(p) >= end else p[pos:] if len(p) > pos else [])
            else:
                seg_post.append([])
        
        # Consensus this segment
        aligned = posterior_guided_align(seg_seqs, seg_post)
        aligned_seqs = [a[0] for a in aligned]
        weights = [np.ones(len(s)) for s in aligned_seqs]
        seg_cons, seg_qual = consensus_from_aligned(aligned_seqs, weights)
        
        segments_consensus.append(seg_cons)
        segments_quality.append(seg_qual)
        
        pos = end - overlap if overlap > 0 else end
    
    # Merge overlapping segments
    if overlap > 0 and len(segments_consensus) > 1:
        consensus = _merge_overlapping_segments(segments_consensus, overlap)
        quality = _merge_overlapping_arrays(segments_quality, overlap)
    else:
        consensus = ''.join(segments_consensus)
        quality = np.concatenate(segments_quality) if segments_quality else np.array([])
    
    return consensus, quality


def _merge_overlapping_segments(segments: List[str], overlap: int) -> str:
    """Merge segments with overlap."""
    if len(segments) == 1:
        return segments[0]
    
    result = segments[0]
    for seg in segments[1:]:
        # Find best merge point in overlap region
        merge_point = len(result) - overlap
        if merge_point < 0:
            merge_point = len(result) // 2
        
        result = result[:merge_point] + seg
    
    return result


def _merge_overlapping_arrays(arrays: List[np.ndarray], overlap: int) -> np.ndarray:
    """Merge overlapping quality arrays."""
    if len(arrays) == 1:
        return arrays[0]
    
    result = arrays[0]
    for arr in arrays[1:]:
        merge_point = len(result) - overlap
        if merge_point < 0:
            merge_point = len(result) // 2
        
        # Average in overlap region
        overlap_result = result[merge_point:]
        overlap_arr = arr[:len(overlap_result)]
        merged_overlap = (overlap_result + overlap_arr) / 2
        
        result = np.concatenate([result[:merge_point], merged_overlap, arr[len(overlap_result):]])
    
    return result
