"""
Dorado basecaller wrapper: convert raw nanopore signals back to basecalled DNA.

Dorado (ONT) is the official basecaller. It takes raw signal (.blow5) as input
and outputs:
  1. A FASTA/FASTQ of basecalled sequences.
  2. Per-base quality scores (Phred scores).

Integration: this module wraps the dorado binary to:
  1. Basecall squigulator-generated signals back to DNA.
  2. Extract Phred quality scores for soft-decision decoding in Asym-MGC.

Using dorado's built-in simplex basecaller (no GPU needed for simulation).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DORADO_BIN = "/home/ubuntu/gongrui/HEDGES_pro/dorado-2.0.0-linux-x64/bin/dorado"

DORADO_MODELS = {
    "dna-r9-prom": "dna_r9.4.1_e8_fast@v3.3",
    "dna-r9-min": "dna_r9.4.1_e8_fast@v3.3",
    "dna-r10-prom": "dna_r10.4.1_e8_fast@v3.3",
    "dna-r10-min": "dna_r10.4.1_e8_fast@v3.3",
    "rna-r9-min": "rna_r9.4.1_e8_fast@v3.3",
}


def _is_dorado_available() -> bool:
    return Path(DORADO_BIN).exists()


def basecall_blow5(
    blow5_path: str,
    model: Optional[str] = None,
    profile: str = "dna-r9-prom",
    output_format: str = "fastq",
    temperature: float = 0.0,
    seed: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> Tuple[List[str], List[np.ndarray], Dict]:
    """
    Basecall a blow5 file using dorado.

    Parameters
    ----------
    blow5_path : str
        Path to the blow5/SLOW5 file with raw signal data.
    model : str, optional
        Dorado model name. If None, inferred from profile.
    profile : str
        ONT profile (used to select model if not specified).
    output_format : str
        Output format: 'fasta', 'fastq', or 'json'.
    temperature : float
        Sampling temperature (0 = argmax, >0 = stochastic).
    seed : int, optional
        Random seed for reproducible basecalling.
    extra_args : List[str], optional
        Additional dorado CLI flags.

    Returns
    -------
    sequences : List[str]
        Basecalled sequences (FASTA/FASTQ sequences).
    quality_arrays : List[ndarray]
        Per-read array of Phred quality scores (int).
    meta : Dict
        Basecalling metadata.
    """
    if not _is_dorado_available():
        raise RuntimeError(
            f"dorado not found at {DORADO_BIN}. "
            "Please install from https://github.com/nanoporetech/dorado"
        )

    if model is None:
        model = DORADO_MODELS.get(profile, "dna_r9.4.1_e8_fast@v3.3")

    with tempfile.TemporaryDirectory() as tmpdir:
        if output_format == "json":
            out_path = os.path.join(tmpdir, "output.json")
        elif output_format == "fasta":
            out_path = os.path.join(tmpdir, "output.fasta")
        else:
            out_path = os.path.join(tmpdir, "output.fastq")

        cmd = [
            DORADO_BIN,
            "basecaller",
            model,
            blow5_path,
        ]
        if output_format == "json":
            cmd.append("--json")
        if temperature > 0:
            cmd.extend(["--temperature", str(temperature)])
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"dorado basecaller failed with return code {result.returncode}:\n"
                f"STDOUT: {result.stdout[:2000]}\nSTDERR: {result.stderr[:2000]}"
            )

        output = result.stdout

        if output_format == "json":
            try:
                records = json.loads(output)
            except json.JSONDecodeError:
                records = []
            sequences = [r.get("sequence", "") for r in records]
            quality_arrays = [np.array(r.get("quality", []), dtype=int) for r in records]
        else:
            sequences, quality_arrays = _parse_fasta_fastq(output, output_format)

        meta = {
            "model": model,
            "profile": profile,
            "temperature": temperature,
            "seed": seed,
            "num_reads": len(sequences),
            "dorado_version": _get_dorado_version(),
            "dorado_stdout": result.stdout[:500],
            "dorado_stderr": result.stderr[:500],
        }

        return sequences, quality_arrays, meta


def _parse_fasta_fastq(content: str, fmt: str) -> Tuple[List[str], List[np.ndarray]]:
    """Parse FASTA or FASTQ output from dorado."""
    sequences = []
    quality_arrays = []

    if fmt == "fasta":
        for line in content.splitlines():
            if line.startswith(">"):
                continue
            if line.strip():
                sequences.append(line.strip())
                quality_arrays.append(np.array([], dtype=int))
    else:
        # FASTQ: entries are groups of 4 lines
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("@"):
                header = lines[i]
                seq = lines[i + 1] if i + 1 < len(lines) else ""
                qual_line = ""
                if i + 3 < len(lines):
                    if lines[i + 2].startswith("+"):
                        qual_line = lines[i + 3]
                sequences.append(seq)
                if qual_line:
                    # Convert Sanger Phred+33 to integer scores
                    q_scores = np.array(
                        [ord(c) - 33 for c in qual_line],
                        dtype=int,
                    )
                    quality_arrays.append(q_scores)
                else:
                    quality_arrays.append(np.array([], dtype=int))
                i += 4
            else:
                i += 1

    return sequences, quality_arrays


def _get_dorado_version() -> str:
    """Get dorado version string."""
    try:
        result = subprocess.run(
            [DORADO_BIN, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def dorado_simplex_basecall(
    dna_sequence: str,
    profile: str = "dna-r9-prom",
    seed: int = 42,
    signal_length_hint: int = 1200,
) -> Tuple[str, np.ndarray, Dict]:
    """
    End-to-end: DNA sequence → squigulator signal → dorado basecall → (DNA, quality).

    This is the full simulation pipeline that mimics the physical process:
    1. Encode message to DNA.
    2. Squigulator converts DNA to raw electrical signals (with realistic noise).
    3. Dorado basecalls signals back to DNA + quality scores.

    Parameters
    ----------
    dna_sequence : str
        Input DNA sequence.
    profile : str
        ONT profile (dna-r9-prom, dna-r10-prom, etc.).
    seed : int
        Seed for reproducibility.
    signal_length_hint : int
        Mean read length hint for squigulator.

    Returns
    -------
    basecalled : str
        Dorado's basecalled sequence (may differ from input due to basecalling errors).
    quality : ndarray
        Per-base Phred quality scores.
    meta : Dict
        Full simulation metadata including squigulator/dorado info.
    """
    from .squigulator_wrapper import simulate_signals, SQUIGULATOR_BIN

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: squigulator — DNA → signal
        signals, sim_meta = simulate_signals(
            [dna_sequence],
            output_dir=tmpdir,
            profile=profile,
            read_length_hint=signal_length_hint,
            num_reads=1,
            seed=seed,
        )

        blow5_path = sim_meta["signal_file"]

        if not signals:
            raise RuntimeError(
                "Signal simulation produced no signals. "
                "Install slow5kit to read blow5 files, or use dorado's "
                "built-in simulation via dorado_simplex_basecall_builtin()."
            )

        # Step 2: dorado — signal → basecall
        model = DORADO_MODELS.get(profile, "dna_r9.4.1_e8_fast@v3.3")
        basecalled_list, quality_list, bc_meta = basecall_blow5(
            blow5_path,
            model=model,
            profile=profile,
            seed=seed,
        )

        if not basecalled_list:
            raise RuntimeError(
                f"dorado basecaller returned no sequences. "
                f"stderr: {bc_meta.get('dorado_stderr', 'N/A')}"
            )

        # Merge metadata
        sim_meta.update(bc_meta)
        sim_meta["simulation_method"] = "squigulator+dorado"

        return basecalled_list[0], quality_list[0], sim_meta


def dorado_simplex_basecall_builtin(
    dna_sequence: str,
    profile: str = "dna-r9-prom",
    seed: int = 42,
    signal_length_hint: int = 1200,
) -> Tuple[str, np.ndarray, Dict]:
    """
    Pure-dorado simulation path: uses dorado's own simplex model.

    This avoids the need for slow5kit by relying entirely on dorado's
    built-in simulation capability.
    """
    if not _is_dorado_available():
        raise RuntimeError(f"dorado not found at {DORADO_BIN}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write DNA to FASTA
        fasta_path = os.path.join(tmpdir, "input.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">ref\n{dna_sequence}\n")

        model = DORADO_MODELS.get(profile, "dna_r9.4.1_e8_fast@v3.3")

        cmd = [
            DORADO_BIN,
            "simplex",
            "--ref", fasta_path,
            "--kit-name", model,
            "--seed", str(seed),
            "--json",
        ]

        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"dorado simplex failed:\n"
                f"STDOUT: {result.stdout[:1000]}\n"
                f"STDERR: {result.stderr[:1000]}"
            )

        try:
            records = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"dorado simplex returned invalid JSON:\n{result.stdout[:1000]}"
            )

        if not records:
            raise RuntimeError("dorado simplex returned no records")

        rec = records[0]
        sequence = rec.get("sequence", "")
        quality = np.array(rec.get("quality", []), dtype=int)

        meta = {
            "model": model,
            "profile": profile,
            "seed": seed,
            "simulation_method": "dorado_simplex",
            "num_simulated": len(records),
            "dorado_version": _get_dorado_version(),
        }

        return sequence, quality, meta
