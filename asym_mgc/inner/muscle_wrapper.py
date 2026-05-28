"""
MUSCLE MSA Integration for Consensus Formation.

Wrapper for MUSCLE5 multiple sequence alignment to enable consensus
formation from reads with indels.

Reference: IMPROVEMENT_PLAN.md Section 3.7.8 (方案C)
"""

from __future__ import annotations

import subprocess
import tempfile
import os
from typing import List, Tuple, Optional
import numpy as np


class MuscleAligner:
    """
    Interface to MUSCLE5 for multiple sequence alignment.

    MUSCLE5 is used for consensus formation because it handles indels
    correctly, unlike simple position-based alignment.

    Parameters
    ----------
    muscle_path : str
        Path to MUSCLE5 executable. If None, looks in PATH.
    max_iters : int
        Maximum refinement iterations (default 2 for speed).
    """

    def __init__(
        self,
        muscle_path: Optional[str] = None,
        max_iters: int = 2,
    ):
        self.muscle_path = muscle_path or self._find_muscle()
        self.max_iters = max_iters

    def _find_muscle(self) -> str:
        """Find MUSCLE5 in PATH or common locations."""
        # Check PATH
        import shutil
        path = shutil.which('muscle')
        if path:
            return path

        # Check common locations
        common_paths = [
            '/usr/local/bin/muscle',
            '/usr/bin/muscle',
            '~/muscle',
        ]
        for p in common_paths:
            expanded = os.path.expanduser(p)
            if os.path.exists(expanded):
                return expanded

        return 'muscle'  # Fallback to assuming it's in PATH

    def align(self, sequences: List[str], names: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Perform multiple sequence alignment.

        Parameters
        ----------
        sequences : list of str
            DNA sequences to align.
        names : list of str, optional
            Names for each sequence.

        Returns
        -------
        aligned : list of str
            Aligned sequences (same length).
        order : list of str
            Order of sequences in output.
        """
        if not sequences:
            return [], []

        if len(sequences) < 2:
            # Single sequence, no alignment needed
            return sequences, names or ['seq0']

        if names is None:
            names = [f's{i}' for i in range(len(sequences))]

        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f_in:
            for name, seq in zip(names, sequences):
                f_in.write(f'>{name}\n{seq}\n')
            input_path = f_in.name

        output_path = input_path + '.aligned'

        try:
            # Run MUSCLE5
            cmd = [
                self.muscle_path,
                '-align', input_path,
                '-output', output_path,
                '-refine', str(self.max_iters),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
            )

            if result.returncode != 0:
                # MUSCLE failed, fall back to simple alignment
                return self._fallback_align(sequences, names)

            # Parse aligned output
            aligned, order = self._parse_fasta(output_path)

            return aligned, order

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # MUSCLE not available or timeout, use fallback
            return self._fallback_align(sequences, names)
        finally:
            # Clean up temp files
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass

    def _parse_fasta(self, path: str) -> Tuple[List[str], List[str]]:
        """Parse aligned sequences from FASTA file."""
        aligned = []
        order = []

        with open(path, 'r') as f:
            name = None
            seq = []

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    if name is not None:
                        aligned.append(''.join(seq))
                    name = line[1:].strip()
                    order.append(name)
                    seq = []
                else:
                    seq.append(line.strip())

            if name is not None:
                aligned.append(''.join(seq))

        return aligned, order

    def _fallback_align(self, sequences: List[str], names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Fallback alignment when MUSCLE is not available.

        Uses simple prefix alignment - not ideal but better than nothing.
        """
        if len(sequences) == 0:
            return [], []

        if len(sequences) == 1:
            return sequences, names

        # Use first sequence as reference
        ref = sequences[0]
        aligned = [ref]
        order = [names[0]]

        for i in range(1, len(sequences)):
            seq = sequences[i]
            # Simple left-alignment
            aligned_seq = self._left_align(ref, seq)
            aligned.append(aligned_seq)
            order.append(names[i])

        return aligned, order

    def _left_align(self, ref: str, seq: str) -> str:
        """
        Left-align a sequence to a reference, handling insertions/deletions.

        This is a simplified version - proper alignment requires MSA.
        """
        # For now, just pad to match reference length
        if len(seq) >= len(ref):
            return seq[:len(ref)]
        else:
            return seq + '-' * (len(ref) - len(seq))


def build_consensus_from_msa(aligned_sequences: List[str]) -> Tuple[str, np.ndarray]:
    """
    Build consensus from MUSCLE-aligned sequences.

    Parameters
    ----------
    aligned_sequences : list of str
        Aligned sequences (all same length).

    Returns
    -------
    consensus : str
        Consensus DNA sequence.
    quality : ndarray
        Per-position confidence scores [0, 1].
    """
    if not aligned_sequences:
        return '', np.array([])

    seq_len = len(aligned_sequences[0])
    num_seqs = len(aligned_sequences)

    base_votes = {'A': 0, 'C': 0, 'G': 0, 'T': 0, '-': 0}
    consensus_bases = []
    quality_scores = []

    for pos in range(seq_len):
        # Count votes at this position
        votes = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        gap_count = 0

        for seq in aligned_sequences:
            if pos < len(seq):
                base = seq[pos].upper()
                if base in votes:
                    votes[base] += 1
                elif base == '-' or base == '.':
                    gap_count += 1

        # Majority vote (excluding gaps)
        total_non_gap = sum(votes.values())
        if total_non_gap == 0:
            # All gaps, use reference or N
            consensus_bases.append('N')
            quality_scores.append(0.0)
        else:
            best_base = max(votes.keys(), key=lambda b: votes[b])
            best_count = votes[best_base]
            confidence = best_count / total_non_gap

            consensus_bases.append(best_base)
            quality_scores.append(confidence)

    return ''.join(consensus_bases), np.array(quality_scores)


class ConsensusPipeline:
    """
    Complete consensus pipeline: reads → MUSCLE → consensus.

    Reference: IMPROVEMENT_PLAN.md Section 3.7.8.3
    """

    def __init__(self, muscle_path: Optional[str] = None):
        self.aligner = MuscleAligner(muscle_path=muscle_path)

    def run(
        self,
        sequences: List[str],
        names: Optional[List[str]] = None,
    ) -> Tuple[str, np.ndarray, List[str]]:
        """
        Run complete consensus pipeline.

        Parameters
        ----------
        sequences : list of str
            Input DNA sequences (reads of the same strand).
        names : list of str, optional
            Names for each sequence.

        Returns
        -------
        consensus : str
            Consensus DNA sequence.
        quality : ndarray
            Per-position confidence scores.
        aligned : list of str
            Aligned sequences from MUSCLE.
        """
        if len(sequences) < 2:
            # Single sequence, return as-is
            return sequences[0] if sequences else '', np.array([]), sequences

        # Align with MUSCLE
        aligned, order = self.aligner.align(sequences, names)

        # Build consensus
        consensus, quality = build_consensus_from_msa(aligned)

        return consensus, quality, aligned


def is_muscle_available() -> bool:
    """Check if MUSCLE5 is available."""
    import shutil
    return shutil.which('muscle') is not None or os.path.exists('/usr/bin/muscle')
