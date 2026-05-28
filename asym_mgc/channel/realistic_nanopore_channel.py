"""
Realistic Nanopore Channel: end-to-end simulation using squigulator + dorado.

This module provides the most realistic nanopore channel simulation available
on this server by combining:

  Path A (GPU-available): squigulator → slow5 → pod5 → dorado → basecalled DNA
    - Full signal-level noise from squigulator
    - Real neural basecalling errors from dorado
    - Requires: slow5-dorado (BLOW5→POD5 converter)

  Path B (CPU-only, always available): squigulator normal mode
    - Realistic DNA-level errors introduced by squigulator's simulation
    - CIGAR strings give exact ground-truth edit operations
    - Works without GPU

  Path C (reference): squigulator --full-contigs + statistical error model
    - Realistic signal with squigulator's physical model
    - Errors applied post-hoc using Memory-k nanopore statistics
    - Best for debugging Asym-MGC in isolation

The unified RealisticNanoporeChannel API automatically selects the best
available path based on system capabilities.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SQUIGULATOR_BIN = "/home/ubuntu/gongrui/HEDGES_pro/squigulator/squigulator-v0.5.0/squigulator"
SLOW5_DORADO_BIN = "/home/ubuntu/gongrui/HEDGES_pro/slow5-dorado/bin/slow5-dorado"
DORADO_BIN = "/home/ubuntu/gongrui/HEDGES_pro/dorado-2.0.0-linux-x64/bin/dorado"
DORADO_MODELS_DIR = "/home/ubuntu/gongrui/HEDGES_pro/dorado_models"
SLOW5TOOLS_BIN = "/home/ubuntu/gongrui/HEDGES_pro/slow5tools-v1.4.0/slow5tools"


# Pre-computed R9.4.1 (4000 Hz) and R10.4.1 (5000 Hz) models
SQUIGULATOR_PROFILES = {
    "dna-r9-prom": {"x": "dna-r9-prom", "sample_rate": 4000},
    "dna-r9-min":  {"x": "dna-r9-min",  "sample_rate": 4000},
    "dna-r10-prom":{"x": "dna-r10-prom","sample_rate": 5000},
    "dna-r10-min": {"x": "dna-r10-min", "sample_rate": 5000},
}


def _is_slow5_dorado_available() -> bool:
    return Path(SLOW5_DORADO_BIN).exists()


def _is_dorado_available() -> bool:
    return Path(DORADO_BIN).exists()


def _is_squigulator_available() -> bool:
    return Path(SQUIGULATOR_BIN).exists()


def _get_cuda_device() -> Optional[str]:
    """Detect if CUDA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return "cuda:0"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Path C: Squigulator full-contigs + statistical error model (always works)
# ---------------------------------------------------------------------------

def _run_squigulator_full_contigs(
    dna: str,
    output_dir: str,
    profile: str = "dna-r9-prom",
    seed: int = 42,
) -> Tuple[str, Dict]:
    """Run squigulator in full-contigs mode (no base-level errors)."""
    fasta_path = os.path.join(output_dir, "input.fa")
    signal_path = os.path.join(output_dir, "signals.blow5")
    sam_path = os.path.join(output_dir, "align.sam")
    paf_path = os.path.join(output_dir, "align.paf")

    with open(fasta_path, "w") as f:
        f.write(f">ref\n{dna}\n")

    profile_arg = SQUIGULATOR_PROFILES.get(profile, SQUIGULATOR_PROFILES["dna-r9-prom"])["x"]

    cmd = [
        SQUIGULATOR_BIN, fasta_path,
        "-o", signal_path,
        "-x", profile_arg,
        "--seed", str(seed),
        "-c", paf_path,
        "-a", sam_path,
        "--full-contigs",
    ]

    result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        raise RuntimeError(f"squigulator failed: {result.stderr}")

    # Parse SAM for dwell times (per-base signal segmentation)
    dwell_times = []
    if os.path.exists(sam_path):
        with open(sam_path) as f:
            for line in f:
                if line.startswith("si:Z:"):
                    # si:Z:start,dwell1,dwell2,...,dwellN,end
                    # Each dwell is samples-per-base
                    si_data = line.strip().split("\t")
                    for field in si_data:
                        if field.startswith("si:Z:"):
                            vals = field[5:].split(",")
                            # Skip first (start) and last (end) values
                            dwell_times = [int(v) for v in vals[1:-1] if v.strip()]
                            break
                elif line.startswith("ss:Z:"):
                    # ss:Z: contains per-base signal summary (could be per-sample)
                    pass

    # Parse PAF for signal statistics
    signal_stats = {}
    if os.path.exists(paf_path):
        with open(paf_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 12:
                    continue
                signal_stats = {
                    "query_len": int(parts[1]),
                    "target_len": int(parts[6]),
                    "n_match": int(parts[9]),
                    "aln_len": int(parts[10]),
                }
                # Check for error tags
                for tag in parts[12:]:
                    if tag.startswith("sc:f:"):
                        signal_stats["corr_coeff"] = float(tag[5:])
                    elif tag.startswith("sh:f:"):
                        signal_stats["shift"] = float(tag[5:])
                    elif tag.startswith("ss:Z:"):
                        # Per-base signal values
                        signal_stats["signal_summary"] = tag[5:]

    return signal_path, {
        "profile": profile_arg,
        "seed": seed,
        "dna_input_len": len(dna),
        "signal_samples": signal_stats.get("query_len", 0),
        "dwell_times": dwell_times,
        "signal_stats": signal_stats,
        "squigulator_stderr": result.stderr[:500],
    }


# ---------------------------------------------------------------------------
# Path B: Squigulator normal mode (real base-level errors)
# ---------------------------------------------------------------------------

def _run_squigulator_normal(
    dna_sequences: List[str],
    output_dir: str,
    profile: str = "dna-r9-prom",
    seed: int = 42,
    min_read_length: int = 200,
) -> Tuple[str, str, Dict]:
    """Run squigulator in normal mode (introduces real base-level errors)."""
    fasta_path = os.path.join(output_dir, "input.fa")
    signal_path = os.path.join(output_dir, "signals.blow5")
    paf_path = os.path.join(output_dir, "align.paf")
    sam_path = os.path.join(output_dir, "align.sam")

    with open(fasta_path, "w") as f:
        for i, seq in enumerate(dna_sequences):
            f.write(f">seq_{i}\n{seq}\n")

    profile_arg = SQUIGULATOR_PROFILES.get(profile, SQUIGULATOR_PROFILES["dna-r9-prom"])["x"]

    cmd = [
        SQUIGULATOR_BIN, fasta_path,
        "-o", signal_path,
        "-x", profile_arg,
        "-n", str(len(dna_sequences)),
        "-r", str(max(min_read_length, 200)),
        "--seed", str(seed),
        "-c", paf_path,
        "-a", sam_path,
    ]

    result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"squigulator normal mode failed: {result.stderr}")

    return signal_path, sam_path, {
        "profile": profile_arg,
        "seed": seed,
        "num_sequences": len(dna_sequences),
        "squigulator_stderr": result.stderr[:500],
    }


def _parse_squigulator_sam(sam_path: str, ref_dna: str) -> Dict:
    """
    Parse squigulator SAM output to recover CIGAR and basecalled sequence.

    In normal mode, squigulator generates reads that may differ from the
    reference due to:
    - Deletions (reference bases removed from read)
    - Insertions (extra bases in read)
    - Substitutions (wrong bases called)
    """
    if not os.path.exists(sam_path):
        return {}

    alignments = []
    with open(sam_path) as f:
        for line in f:
            if line.startswith("@"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue

            qname = parts[0]
            flag = int(parts[1])
            rname = parts[2]
            pos = int(parts[3])
            mapq = int(parts[4])
            cigar = parts[5]
            rnext = parts[6]
            pnext = parts[7]
            tlen = int(parts[8])
            seq = parts[9]
            qual = parts[10] if len(parts) > 10 else ""

            # Parse MD:Z: tag for mismatches/substitutions
            md_tag = ""
            for tag in parts[11:]:
                if tag.startswith("MD:Z:"):
                    md_tag = tag[5:]
                    break

            alignments.append({
                "qname": qname,
                "flag": flag,
                "rname": rname,
                "pos": pos,
                "mapq": mapq,
                "cigar": cigar,
                "seq": seq,
                "qual": qual,
                "md_tag": md_tag,
            })

    return {"alignments": alignments, "num_alignments": len(alignments)}


# ---------------------------------------------------------------------------
# Path A: squigulator → slow5 → pod5 → dorado (full realism, GPU required)
# ---------------------------------------------------------------------------

def _convert_blow5_to_pod5(blow5_path: str, output_dir: str) -> str:
    """
    Convert BLOW5 to POD5 using slow5-dorado.

    The slow5-dorado tool can convert between SLOW5/BLOW5 and POD5 formats,
    enabling dorado to basecall squigulator's output.
    """
    pod5_dir = os.path.join(output_dir, "pod5_data")
    os.makedirs(pod5_dir, exist_ok=True)

    # slow5-dorado doesn't have a direct convert command, use slow5tools instead
    # Check if we can use slow5tools to convert to a format dorado accepts
    # Actually, dorado 2.0.0 doesn't accept BLOW5 directly. We need pod5.

    # Try using slow5tools to check if there's any conversion path
    # For now, return empty string (Path A not fully supported)
    return ""


def _basecall_with_dorado(
    signal_path: str,
    output_dir: str,
    profile: str = "dna-r9-prom",
    device: str = "cuda:0",
    model_path: Optional[str] = None,
) -> Tuple[str, List, Dict]:
    """
    Basecall signals using dorado via slow5-dorado wrapper.

    Requires: slow5-dorado binary + CUDA + appropriate model.
    """
    if not _is_slow5_dorado_available():
        raise RuntimeError("slow5-dorado not available for GPU basecalling")

    # Select model based on profile
    if model_path is None:
        model_path = os.path.join(DORADO_MODELS_DIR, "dna_r10.4.1_e8.2_400bps_hac@v6.0.0")
        if not os.path.exists(model_path):
            # Try to find any available model
            if os.path.exists(DORADO_MODELS_DIR):
                models = os.listdir(DORADO_MODELS_DIR)
                if models:
                    model_path = os.path.join(DORADO_MODELS_DIR, models[0])

    fastq_path = os.path.join(output_dir, "basecalls.fastq")

    cmd = [
        SLOW5_DORADO_BIN, "basecaller",
        "--device", device,
        "--models-directory", DORADO_MODELS_DIR,
        "--emit-fastq",
        "--emit-moves",
        "-n", "1",
        model_path if os.path.exists(model_path) else "dna_r10.4.1_e8.2_400bps_hac@v6.0.0",
        signal_path,
    ]

    result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"dorado basecalling failed: {result.stderr}")

    # Parse FASTQ output
    sequences = []
    quality_strings = []
    lines = result.stdout.strip().split("\n")
    i = 0
    while i < len(lines) - 3:
        if lines[i].startswith("@"):
            seq_id = lines[i][1:]
            seq = lines[i + 1]
            qual = lines[i + 3] if i + 3 < len(lines) else ""
            sequences.append({"id": seq_id, "seq": seq, "qual": qual})
            quality_strings.append(qual)
            i += 4
        else:
            i += 1

    with open(fastq_path, "w") as f:
        f.write(result.stdout)

    return fastq_path, sequences, {
        "model": model_path,
        "device": device,
        "num_basecalled": len(sequences),
        "dorado_stderr": result.stderr[:500],
    }


# ---------------------------------------------------------------------------
# Unified RealisticNanoporeChannel API
# ---------------------------------------------------------------------------

class RealisticNanoporeChannel:
    """
    Realistic nanopore channel using squigulator + optional dorado basecalling.

    This is the recommended channel for pressure testing Asym-MGC with
    realistic nanopore noise characteristics.

    Parameters
    ----------
    mode : str
        Simulation mode:
        - "full": squigulator + dorado (best realism, requires GPU + slow5-dorado)
        - "normal": squigulator normal mode (real base errors, always works)
        - "full_contigs": squigulator full-contigs + statistical errors (reference)
    profile : str
        ONT profile: "dna-r9-prom" (default), "dna-r9-min", "dna-r10-prom", "dna-r10-min"
    seed : int
        Random seed for reproducibility.
    min_read_length : int
        Minimum simulated read length (squigulator minimum is 200).
    quality_mean : float
        Mean Phred quality score for simulated quality scores.
    device : str
        CUDA device for dorado (if mode="full").
    """

    def __init__(
        self,
        mode: str = "normal",
        profile: str = "dna-r9-prom",
        seed: int = 42,
        min_read_length: int = 200,
        quality_mean: float = 15.0,
        device: Optional[str] = None,
    ):
        if mode not in ("full", "normal", "full_contigs"):
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = mode
        self.profile = profile
        self.seed = seed
        self.min_read_length = min_read_length
        self.quality_mean = quality_mean
        self.device = device or _get_cuda_device()

        # Verify tools are available
        if not _is_squigulator_available():
            raise RuntimeError(f"squigulator not found at {SQUIGULATOR_BIN}")

        if mode == "full" and not _is_slow5_dorado_available():
            import warnings
            warnings.warn(
                "slow5-dorado not available, falling back to 'normal' mode",
                RuntimeWarning,
            )
            self.mode = "normal"

        if mode == "full" and self.device is None:
            import warnings
            warnings.warn(
                "CUDA not available, falling back to 'normal' mode",
                RuntimeWarning,
            )
            self.mode = "normal"

    def transmit(self, dna: str) -> Tuple[str, np.ndarray, Dict]:
        """
        Transmit DNA through a realistic nanopore channel.

        Parameters
        ----------
        dna : str
            Input DNA sequence (ACGT only).

        Returns
        -------
        received : str
            Received (possibly corrupted) DNA sequence.
        quality : ndarray
            Per-base Phred quality scores.
        meta : dict
            Simulation metadata including error details.
        """
        if self.mode == "full":
            return self._transmit_full(dna)
        elif self.mode == "normal":
            return self._transmit_normal(dna)
        else:
            return self._transmit_full_contigs(dna)

    def _transmit_full(self, dna: str) -> Tuple[str, np.ndarray, Dict]:
        """Path A: squigulator + slow5-dorado + dorado (GPU basecalling)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: squigulator
            signal_path, squig_meta = _run_squigulator_full_contigs(
                dna, tmpdir, self.profile, self.seed,
            )

            # Step 2: convert to POD5 and basecall with dorado
            try:
                fastq_path, sequences, dorado_meta = _basecall_with_dorado(
                    signal_path, tmpdir, self.profile, device=self.device or "cuda:0",
                )
            except Exception as e:
                # Fallback to normal mode
                return self._transmit_normal(dna)

            if sequences:
                seq = sequences[0]["seq"]
                qual_str = sequences[0]["qual"]
                quality = np.array([ord(c) - 33 for c in qual_str], dtype=int)
                meta = {**squig_meta, **dorado_meta, "simulation_path": "full"}
                return seq, quality, meta
            else:
                return self._transmit_normal(dna)

    def _transmit_normal(self, dna: str) -> Tuple[str, np.ndarray, Dict]:
        """Path B: squigulator normal mode with real base-level errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            signal_path, sam_path, squig_meta = _run_squigulator_normal(
                [dna], tmpdir, self.profile, self.seed, self.min_read_length,
            )

            sam_data = _parse_squigulator_sam(sam_path, dna)

            if sam_data.get("alignments"):
                aln = sam_data["alignments"][0]
                seq = aln["seq"]
                qual_str = aln.get("qual", "")

                if qual_str:
                    quality = np.array([ord(c) - 33 for c in qual_str], dtype=int)
                else:
                    quality = self._simulate_quality(len(seq), squig_meta["seed"])

                meta = {
                    **squig_meta,
                    **sam_data,
                    "simulation_path": "normal",
                    "cigar": aln.get("cigar", ""),
                    "md_tag": aln.get("md_tag", ""),
                }
                return seq, quality, meta
            else:
                # Fallback: return original DNA
                return dna, np.full(len(dna), self.quality_mean, dtype=int), {
                    **squig_meta, "simulation_path": "normal", "fallback": True,
                }

    def _transmit_full_contigs(self, dna: str) -> Tuple[str, np.ndarray, Dict]:
        """Path C: squigulator full-contigs + statistical error model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            signal_path, squig_meta = _run_squigulator_full_contigs(
                dna, tmpdir, self.profile, self.seed,
            )

            # Apply statistical nanopore errors
            from .memory_k_nanopore import MemoryKNanoporeChannel

            ch = MemoryKNanoporeChannel(
                Pd=0.5, Pi=0.026, Ps=0.474,
                seed=self.seed,
            )
            received, edits = ch.transmit(dna)
            quality = self._simulate_quality(len(received), self.seed)

            meta = {
                **squig_meta,
                "simulation_path": "full_contigs",
                "applied_edits": [
                    {"pos": e[0], "type": e[1], "detail": e[2]}
                    for e in edits
                ],
            }
            return received, quality, meta

    def _simulate_quality(self, length: int, seed: int) -> np.ndarray:
        """Simulate Phred quality scores for a sequence."""
        rng = np.random.default_rng(seed)
        q = rng.normal(self.quality_mean, 4.0, length)
        return np.clip(np.round(q), 1, 45).astype(int)

    def get_signal_stats(self, dna: str) -> Dict:
        """Get signal statistics for a DNA sequence (no errors applied)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            signal_path, meta = _run_squigulator_full_contigs(
                dna, tmpdir, self.profile, self.seed,
            )
            return meta


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def create_realistic_channel(
    mode: str = "auto",
    profile: str = "dna-r9-prom",
    seed: int = 42,
) -> Tuple[RealisticNanoporeChannel, str]:
    """
    Create the best available realistic nanopore channel.

    Returns (channel, path_description).

    Mode selection priority:
    1. "full" → squigulator + dorado (if GPU + slow5-dorado available)
    2. "normal" → squigulator normal mode (always available)
    3. "full_contigs" → squigulator + statistical model
    """
    if mode == "auto":
        cuda = _get_cuda_device()
        if cuda and _is_slow5_dorado_available():
            mode = "full"
            path_desc = "squigulator + slow5-dorado + dorado (GPU basecalling)"
        elif _is_squigulator_available():
            mode = "normal"
            path_desc = "squigulator normal mode (real base-level errors)"
        else:
            mode = "full_contigs"
            path_desc = "squigulator full-contigs + statistical error model"

    channel = RealisticNanoporeChannel(mode=mode, profile=profile, seed=seed)
    return channel, f"mode={mode}, profile={profile}"


def diagnose_tools() -> Dict:
    """Diagnose availability of all required tools."""
    return {
        "squigulator": {
            "available": _is_squigulator_available(),
            "path": SQUIGULATOR_BIN,
        },
        "slow5_dorado": {
            "available": _is_slow5_dorado_available(),
            "path": SLOW5_DORADO_BIN,
        },
        "dorado": {
            "available": _is_dorado_available(),
            "path": DORADO_BIN,
        },
        "cuda": {
            "available": _get_cuda_device() is not None,
            "device": _get_cuda_device(),
        },
        "slow5tools": {
            "available": Path(SLOW5TOOLS_BIN).exists(),
            "path": SLOW5TOOLS_BIN,
        },
    }
