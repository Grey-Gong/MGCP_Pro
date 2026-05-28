"""
Squigulator wrapper: realistic nanopore signal simulation from DNA sequences.

Uses squigulator (Hadfield et al.) to generate raw electrical signals and
leverages the PAF alignment output to recover the ground-truth CIGAR string
for every simulated read — giving us both the realistic noise AND the exact
edit operations applied.

Key insight: squigulator's PAF output includes a CIGAR string (tag `cs:Z`)
that precisely describes insertions, deletions, and substitutions applied to
each read. We use this CIGAR to reconstruct the "basecalled" sequence without
needing to parse the binary blow5 signal files.

Squigulator path: DNA -> raw signal (with physical noise) -> PAF alignment
                  -> CIGAR -> reconstructed basecalled DNA
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SQUIGULATOR_BIN = "/home/ubuntu/gongrui/HEDGES_pro/squigulator/squigulator-v0.5.0/squigulator"
ONT_PROFILES = [
    "dna-r9-prom",   # DNA R9.4.1 (promethion) — most common
    "dna-r9-min",    # DNA R9.4.1 (minion)
    "dna-r10-min",   # DNA R10.4.1 (minion)
    "dna-r10-prom",  # DNA R10.4.1 (promethion)
    "rna-r9-min",    # RNA R9.4.1
    "rna004-min",    # RNA R10.4.1
]


def _is_squigulator_available() -> bool:
    return Path(SQUIGULATOR_BIN).exists()


def _run_squigulator(
    dna_sequences: List[str],
    output_dir: str,
    profile: str = "dna-r9-prom",
    read_length_hint: int = 200,
    seed: int = 42,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, str, Dict]:
    """
    Run squigulator and return paths to output files + metadata.
    """
    if not _is_squigulator_available():
        raise RuntimeError(
            f"squigulator not found at {SQUIGULATOR_BIN}. "
            "Install from: git clone https://github.com/nanoporetech/squigulator"
        )

    fasta_path = os.path.join(output_dir, "input.fasta")
    signal_path = os.path.join(output_dir, "signals.blow5")
    paf_path = os.path.join(output_dir, "alignments.paf")

    with open(fasta_path, "w") as f:
        for i, seq in enumerate(dna_sequences):
            f.write(f">ref_{i}\n{seq}\n")

    cmd = [
        SQUIGULATOR_BIN,
        fasta_path,
        "-o", signal_path,
        "-x", profile,
        "--seed", str(seed),
        "-c", paf_path,
        "--full-contigs",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )

    meta = {
        "profile": profile,
        "seed": seed,
        "read_length_hint": read_length_hint,
        "num_sequences": len(dna_sequences),
        "signal_file": signal_path,
        "paf_file": paf_path,
    }

    if result.returncode != 0:
        raise RuntimeError(
            f"squigulator failed (rc={result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    return signal_path, paf_path, meta


def _parse_paf_with_cigar(paf_path: str) -> List[Dict]:
    """
    Parse PAF file, extracting CIGAR string (cs:Z tag) for each read.

    PAF format (11 mandatory + optional fields):
      qname  qlen  qstart  qend  strand  tname  tlen  tstart  tend
      nmatch  alnlen  mapq  [cg  cs:Z:...]

    The cs:Z tag contains the CIGAR-like edit string:
      :<len><op>   where op is one of:
        :<N>       (match N bases)
        +<N><base> (insertion of N copies of <base>)
        -<N><base> (deletion of N copies of <base>)
        ~<N><base> (substitution: N copies of <base> replacing previous bases)
        *<base><base> (single-base substitution)
    """
    reads = []
    if not os.path.exists(paf_path):
        return reads

    with open(paf_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue

            qname = parts[0]
            qlen = int(parts[1])
            qstart = int(parts[2])
            qend = int(parts[3])
            strand = parts[4]
            tname = parts[5]
            tlen = int(parts[6])
            tstart = int(parts[7])
            tend = int(parts[8])
            nmatch = int(parts[9])
            alnlen = int(parts[10])
            mapq = int(parts[11]) if len(parts) > 11 else 0

            cigar = ""
            for tag_field in parts[12:]:
                if tag_field.startswith("cs:Z:"):
                    cigar = tag_field[5:]
                    break

            reads.append({
                "qname": qname,
                "query_len": qlen,
                "query_start": qstart,
                "query_end": qend,
                "strand": strand,
                "target_name": tname,
                "target_len": tlen,
                "target_start": tstart,
                "target_end": tend,
                "n_match": nmatch,
                "aln_len": alnlen,
                "mapq": mapq,
                "cigar": cigar,
            })

    return reads


def _apply_cigar_to_dna(ref_dna: str, cigar: str) -> Tuple[str, str, Dict]:
    """
    Apply CIGAR string to reference DNA to produce the basecalled sequence.

    Returns (basecalled_dna, edits_summary, detailed_edits).
    """
    if not cigar:
        return ref_dna, {}, {"status": "no_cigar"}

    # Parse CIGAR operations
    # Format: :<N> (match), +<N><BASE> (ins), -<N><BASE> (del), *<B><B> (sub)
    result = []
    deletions = 0
    insertions = 0
    substitutions = 0
    matches = 0
    detailed_edits = []

    # Tokenize: find all operations
    ops = re.findall(r'([-+*:~:])(\d*)([ACGT]*)', cigar)
    ref_pos = 0

    for op, length_str, bases in ops:
        length = int(length_str) if length_str else 1

        if op == ':':  # Match
            # length is the number of matching bases
            end_pos = ref_pos + length
            result.append(ref_dna[ref_pos:end_pos])
            ref_pos = end_pos
            matches += length

        elif op == '+':  # Insertion (extra bases in read, not in reference)
            # bases are inserted; they're already in the read
            result.append(bases[:length])
            insertions += length
            detailed_edits.append(f"I@{len(''.join(result))}:{bases[:length]}")

        elif op == '-':  # Deletion (reference bases absent in read)
            ref_pos += length
            deletions += length
            detailed_edits.append(f"D@{ref_pos - length}:{bases[:length]}")

        elif op == '*':  # Single-base substitution
            # Format: *<ref><sub> (length is implicit=1, but handle multi-char)
            if length_str:
                # Multi-char substitution
                sub_str = length_str + bases
                ref_base = sub_str[0]
                sub_bases = sub_str[1:]
                for i, sb in enumerate(sub_bases):
                    if ref_pos < len(ref_dna):
                        detailed_edits.append(
                            f"S@{ref_pos}:{ref_dna[ref_pos]}->{sb}"
                        )
                        result.append(sb)
                        ref_pos += 1
                        substitutions += 1
            else:
                if ref_pos < len(ref_dna):
                    ref_base = ref_dna[ref_pos]
                    detailed_edits.append(f"S@{ref_pos}:{ref_base}->{bases}")
                    result.append(bases)
                    ref_pos += 1
                    substitutions += 1

        elif op == '~':  # Multi-base substitution
            # ~<N><BASE> means N reference bases replaced by BASE
            ref_pos += length
            result.append(bases[:length])
            substitutions += length
            detailed_edits.append(
                f"Sub@{ref_pos - length}:{bases[:length]}({length}bp)"
            )

    basecalled = ''.join(result)

    edits_summary = {
        "matches": matches,
        "deletions": deletions,
        "insertions": insertions,
        "substitutions": substitutions,
        "total_errors": deletions + insertions + substitutions,
        "edit_rate": (deletions + insertions + substitutions) / max(len(ref_dna), 1),
        "detailed": detailed_edits,
        "cigar_length": len(cigar),
        "output_length": len(basecalled),
        "input_length": len(ref_dna),
        "drift": len(basecalled) - len(ref_dna),
    }

    return basecalled, edits_summary, detailed_edits


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def simulate_basecalled_dna(
    dna_sequences: List[str],
    profile: str = "dna-r9-prom",
    seed: int = 42,
    read_length_hint: int = 200,
    extra_args: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Full pipeline: DNA -> squigulator signal (with noise) -> CIGAR -> basecalled DNA.

    This is the recommended entry point. It runs squigulator and reconstructs
    the basecalled sequences from the PAF CIGAR, bypassing the need for
    slow5kit and dorado GPU inference.

    Parameters
    ----------
    dna_sequences : List[str]
        Input DNA sequences (one per strand).
    profile : str
        ONT profile. Options: dna-r9-prom (default), dna-r9-min,
        dna-r10-prom, dna-r10-min, rna-r9-min, rna004-min.
    seed : int
        Random seed for reproducibility.
    read_length_hint : int
        Minimum read length hint for squigulator (default 200).
    extra_args : List[str], optional
        Extra squigulator flags.

    Returns
    -------
    results : List[Dict]
        One entry per input sequence, each containing:
          - 'basecalled_dna': str — the reconstructed basecalled sequence
          - 'quality': ndarray — simulated Phred quality scores
          - 'cigar': str — the CIGAR edit string
          - 'edits': dict — edit statistics
          - 'signal_meta': dict — squigulator metadata
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        signal_path, paf_path, meta = _run_squigulator(
            dna_sequences,
            output_dir=tmpdir,
            profile=profile,
            read_length_hint=read_length_hint,
            seed=seed,
            extra_args=extra_args,
        )

        reads = _parse_paf_with_cigar(paf_path)

        results = []
        for i, (dna, read_info) in enumerate(zip(dna_sequences, reads)):
            basecalled, edits, details = _apply_cigar_to_dna(dna, read_info["cigar"])

            # Simulate quality scores: lower near edit positions
            quality = _simulate_quality_scores(
                basecalled, edits, mean_quality=15.0, seed=seed + i
            )

            results.append({
                "basecalled_dna": basecalled,
                "quality": quality,
                "cigar": read_info["cigar"],
                "edits": edits,
                "detailed_edits": details,
                "signal_meta": {
                    **meta,
                    **read_info,
                },
            })

        return results


def _simulate_quality_scores(
    sequence: str,
    edits: Dict,
    mean_quality: float = 15.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate Phred quality scores correlated with local error density.

    Positions near actual edit locations get systematically lower scores.
    Quality follows a truncated normal distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(sequence)

    # Base quality (truncated normal centered at mean_quality)
    quality = rng.normal(mean_quality, 4.0, n)
    quality = np.clip(np.round(quality), 1, 45).astype(int)

    # Degrade quality near edit positions
    edit_positions = set()
    for edit_desc in edits.get("detailed", []):
        if edit_desc.startswith("S@"):
            pos = int(edit_desc.split("@")[1].split(":")[0])
            edit_positions.add(pos)
        elif edit_desc.startswith("I@"):
            pos = int(edit_desc.split("@")[1].split(":")[0])
            for p in range(max(0, pos - 2), min(n, pos + 3)):
                edit_positions.add(p)
        elif edit_desc.startswith("D@"):
            pos = int(edit_desc.split("@")[1].split(":")[0])
            for p in range(max(0, pos - 2), min(n, pos + 3)):
                edit_positions.add(p)

    for ep in edit_positions:
        if 0 <= ep < n:
            quality[ep] = max(1, quality[ep] - rng.integers(5, 15))

    return quality


def simulate_single_strand(
    dna: str,
    profile: str = "dna-r9-prom",
    seed: int = 42,
    read_length_hint: int = 200,
) -> Tuple[str, np.ndarray, Dict]:
    """
    Convenience wrapper for a single DNA strand.

    Returns (basecalled_dna, quality_scores, metadata).
    """
    results = simulate_basecalled_dna(
        [dna],
        profile=profile,
        seed=seed,
        read_length_hint=read_length_hint,
    )
    result = results[0]
    return result["basecalled_dna"], result["quality"], result
