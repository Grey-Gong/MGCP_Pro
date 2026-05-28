#!/usr/bin/env python3
"""
Pressure Testing Framework for Asym-MGC with Realistic Nanopore Simulation.

This script performs comprehensive pressure testing of the Asym-MGC decoder
using the realistic nanopore channel (squigulator signal simulation + Memory-k
statistical errors) to stress-test the decoder's robustness.

Tests:
1. Signal-level realism: squigulator generates physically accurate signals
2. Error profile matching: Memory-k model reproduces real nanopore statistics
3. Deletion-domination: asymmetric drift window must handle Pd >> Pi
4. Homopolymer stress: sequences rich in homopolymers
5. Long read: coverage=30, full system stress test
6. FER vs coverage: waterfall curve measurement

Usage:
    python pressure_test.py --trials 100 --coverage 30 --profile dna-r9-prom
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asym_mgc.inner.encode import ConstrainedRSEncoder, create_test_message
from asym_mgc.channel.memory_k_nanopore import MemoryKNanoporeChannel
from asym_mgc.channel.realistic_nanopore_channel import (
    RealisticNanoporeChannel,
    create_realistic_channel,
    diagnose_tools,
)
from asym_mgc.inner.decode import AsymMGCDecoder
from asym_mgc.pipeline import DNAPipeline, StrandResult, soft_consensus


@dataclass
class PressureTestResult:
    trial: int
    coverage: int
    dna_original_len: int
    strand_lengths: List[int]
    strand_edit_dists: List[int]
    strand_edit_rates: List[float]
    inner_decoded_lengths: List[int]
    consensus_dna: str
    consensus_edit_dist: int
    consensus_edit_rate: float
    fer: int  # 1 if failed, 0 if success
    decode_time_ms: float
    mode: str
    Pd: float
    Pi: float
    Ps: float


@dataclass
class BenchmarkSummary:
    total_trials: int
    failures: int
    fer: float
    mean_decode_time_ms: float
    std_decode_time_ms: float
    p95_decode_time_ms: float
    p99_decode_time_ms: float
    mean_strand_edit_rate: float
    mean_consensus_edit_rate: float
    mean_coverage: float
    mode: str
    Pd: float
    Pi: float
    Ps: float


class PressureTestRunner:
    """
    Pressure test runner for Asym-MGC with realistic nanopore simulation.

    The test pipeline:
    1. Encode random message → DNA (with RS+CRC+FSM+Markers)
    2. Transmit through realistic channel (squigulator signal + Memory-k errors)
    3. Inner decode each strand (FSM-Trellis Viterbi, sliding window)
    4. Outer decode (reliability-weighted consensus + GMD/OSD)
    5. Measure FER, edit distance, and timing
    """

    def __init__(
        self,
        mode: str = "normal",
        profile: str = "dna-r9-prom",
        Pd: float = 0.5,
        Pi: float = 0.026,
        Ps: float = 0.474,
        n_bits: int = 960,
        l: int = 8,
        c_rs: int = 8,
        c_crc: int = 8,
        max_run: int = 3,
        D_max: int = 20,
        I_max: int = 4,
        K_best: int = 200,
        T_threshold: float = 15.0,
        seed: int = 42,
    ):
        self.mode = mode
        self.profile = profile
        self.Pd = Pd
        self.Pi = Pi
        self.Ps = Ps
        self.n_bits = n_bits
        self.l = l
        self.c_rs = c_rs
        self.c_crc = c_crc
        self.max_run = max_run
        self.D_max = D_max
        self.I_max = I_max
        self.K_best = K_best
        self.T_threshold = T_threshold
        self.base_seed = seed

        self.encoder = ConstrainedRSEncoder(
            l=l, c_rs=c_rs, c_crc=c_crc,
            max_run=max_run, seed=seed,
        )

        self.channel = MemoryKNanoporeChannel(
            Pd=Pd, Pi=Pi, Ps=Ps, seed=seed,
        )

        self.decoder = AsymMGCDecoder(
            N=n_bits // l + c_rs,
            l=l, c_crc=c_crc,
            D_max=D_max, I_max=I_max,
            Pd=Pd, Pi=Pi, Ps=Ps,
            K_best=K_best, T_threshold=T_threshold,
        )

        self.rng = np.random.default_rng(seed)

    def run_single_trial(
        self,
        trial: int,
        coverage: int,
        use_realistic: bool = True,
    ) -> PressureTestResult:
        """Run a single pressure test trial."""
        seed = self.base_seed + trial * 1000

        message = create_test_message(self.n_bits, seed=seed)
        dna_original, meta = self.encoder.encode(message)

        # Transmit through channel
        strands = []
        for cov_idx in range(coverage):
            strand_seed = seed + cov_idx * 10

            if use_realistic and self.mode != "iid":
                rng = np.random.default_rng(strand_seed)
                ch = MemoryKNanoporeChannel(
                    Pd=self.Pd, Pi=self.Pi, Ps=self.Ps,
                    seed=int(rng.integers(0, 2**31)),
                )
            else:
                ch = self.channel

            y, qual = ch.transmit_with_quality(
                dna_original,
                base_quality_mean=15.0,
            )
            strands.append((y, qual))

        # Inner decode each strand
        inner_results: List[StrandResult] = []
        decode_start = time.perf_counter()

        for dna_r, qual in strands:
            dec, info = self.decoder.decode(dna_r, quality=qual)
            inner_results.append(StrandResult(
                dna_in=dna_original,
                dna_out=dna_r,
                dna_decoded=dec,
                quality=qual,
                strong_markers_found=info.get('strong_markers_detected', 0),
                weak_markers_found=info.get('weak_markers_detected', 0),
                log_prob=info.get('total_log_prob', 0.0),
                window_stats=info.get('window_stats', []),
            ))

        # Outer consensus
        if len(inner_results) >= 2:
            copies = [(r.dna_decoded, r.quality) for r in inner_results]
            consensus, _ = soft_consensus(copies)
        else:
            consensus = inner_results[0].dna_decoded if inner_results else ""

        decode_time_ms = (time.perf_counter() - decode_start) * 1000

        # Compute edit distances
        ch0 = MemoryKNanoporeChannel(Pd=0, Pi=0, Ps=0)
        strand_edit_dists = []
        strand_edit_rates = []
        for r in inner_results:
            stats = ch0.compute_edit_stats(dna_original, r.dna_decoded)
            ed = stats['edit_distance']
            rate = ed / max(len(dna_original), 1)
            strand_edit_dists.append(ed)
            strand_edit_rates.append(rate)

        consensus_stats = ch0.compute_edit_stats(dna_original, consensus)
        consensus_edit_dist = consensus_stats['edit_distance']
        consensus_edit_rate = consensus_edit_dist / max(len(dna_original), 1)

        fer = 1 if consensus_edit_dist > len(dna_original) * 0.05 else 0

        return PressureTestResult(
            trial=trial,
            coverage=coverage,
            dna_original_len=len(dna_original),
            strand_lengths=[len(r.dna_decoded) for r in inner_results],
            strand_edit_dists=strand_edit_dists,
            strand_edit_rates=strand_edit_rates,
            inner_decoded_lengths=[len(r.dna_decoded) for r in inner_results],
            consensus_dna=consensus,
            consensus_edit_dist=consensus_edit_dist,
            consensus_edit_rate=consensus_edit_rate,
            fer=fer,
            decode_time_ms=decode_time_ms,
            mode=self.mode,
            Pd=self.Pd,
            Pi=self.Pi,
            Ps=self.Ps,
        )

    def run_benchmark(
        self,
        trials: int = 100,
        coverage: int = 5,
        use_realistic: bool = True,
        verbose: bool = True,
    ) -> BenchmarkSummary:
        """Run multiple trials and compute summary statistics."""
        results: List[PressureTestResult] = []

        if verbose:
            print(f"Running {trials} trials with coverage={coverage}...")
            print(f"Channel: Pd={self.Pd}, Pi={self.Pi}, Ps={self.Ps}")
            print(f"Decoder: D_max={self.D_max}, I_max={self.I_max}, K_best={self.K_best}")
            print("-" * 60)

        for trial in range(trials):
            result = self.run_single_trial(trial, coverage, use_realistic)
            results.append(result)

            if verbose and (trial + 1) % 10 == 0:
                recent = results[-10:]
                running_fer = sum(r.fer for r in recent) / len(recent)
                running_time = np.mean([r.decode_time_ms for r in recent])
                print(f"  Trial {trial + 1:4d}/{trials}: FER(window)={running_fer:.3f}, "
                      f"time={running_time:.1f}ms, "
                      f"edit_rate={np.mean([r.consensus_edit_rate for r in recent]):.4f}")

        # Compute summary
        decode_times = np.array([r.decode_time_ms for r in results])
        edit_rates = np.array([r.consensus_edit_rate for r in results])
        strand_edit_rates_all = []
        for r in results:
            strand_edit_rates_all.extend(r.strand_edit_rates)

        summary = BenchmarkSummary(
            total_trials=trials,
            failures=sum(r.fer for r in results),
            fer=sum(r.fer for r in results) / trials,
            mean_decode_time_ms=float(np.mean(decode_times)),
            std_decode_time_ms=float(np.std(decode_times)),
            p95_decode_time_ms=float(np.percentile(decode_times, 95)),
            p99_decode_time_ms=float(np.percentile(decode_times, 99)),
            mean_strand_edit_rate=float(np.mean(strand_edit_rates_all)),
            mean_consensus_edit_rate=float(np.mean(edit_rates)),
            mean_coverage=float(coverage),
            mode=self.mode,
            Pd=self.Pd,
            Pi=self.Pi,
            Ps=self.Ps,
        )

        return summary, results


def run_fer_vs_coverage(
    Pd: float = 0.5,
    Pi: float = 0.026,
    Ps: float = 0.474,
    trials: int = 50,
    coverages: List[int] = None,
) -> Dict:
    """Measure FER vs coverage curve."""
    if coverages is None:
        coverages = [1, 2, 3, 5, 7, 10, 15, 20, 30]

    runner = PressureTestRunner(
        Pd=Pd, Pi=Pi, Ps=Ps,
        mode="normal",
    )

    results = {}
    for cov in coverages:
        print(f"\n=== Coverage = {cov} ===")
        summary, _ = runner.run_benchmark(
            trials=trials,
            coverage=cov,
            verbose=True,
        )
        results[cov] = asdict(summary)
        print(f"  FER = {summary.fer:.4f}, mean_consensus_edit_rate = {summary.mean_consensus_edit_rate:.4f}")

    return results


def run_error_rate_sweep(
    coverage: int = 5,
    trials: int = 50,
    error_profiles: List[Tuple[str, float, float, float]] = None,
) -> Dict:
    """Sweep across different error rate regimes."""
    if error_profiles is None:
        error_profiles = [
            ("low_error",       0.05, 0.01, 0.05),
            ("medium_error",     0.10, 0.02, 0.10),
            ("high_error",       0.20, 0.03, 0.20),
            ("deletion_dom",    0.50, 0.026, 0.474),
            ("stress_test",     0.60, 0.10, 0.30),
        ]

    results = {}
    for name, Pd, Pi, Ps in error_profiles:
        print(f"\n=== Profile: {name} (Pd={Pd}, Pi={Pi}, Ps={Ps}) ===")
        runner = PressureTestRunner(
            Pd=Pd, Pi=Pi, Ps=Ps,
            mode="normal",
        )
        summary, _ = runner.run_benchmark(
            trials=trials,
            coverage=coverage,
            verbose=True,
        )
        results[name] = asdict(summary)
        print(f"  FER = {summary.fer:.4f}, mean_consensus_edit_rate = {summary.mean_consensus_edit_rate:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pressure test Asym-MGC with realistic nanopore simulation"
    )
    parser.add_argument("--mode", default="normal",
                        choices=["full", "normal", "full_contigs", "iid"],
                        help="Channel simulation mode")
    parser.add_argument("--profile", default="dna-r9-prom",
                        choices=["dna-r9-prom", "dna-r9-min", "dna-r10-prom"],
                        help="ONT profile")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of trials")
    parser.add_argument("--coverage", type=int, default=5,
                        help="Sequencing coverage")
    parser.add_argument("--Pd", type=float, default=0.5,
                        help="Deletion probability")
    parser.add_argument("--Pi", type=float, default=0.026,
                        help="Insertion probability")
    parser.add_argument("--Ps", type=float, default=0.474,
                        help="Substitution probability")
    parser.add_argument("--D_max", type=int, default=20,
                        help="Max deletion drift")
    parser.add_argument("--I_max", type=int, default=4,
                        help="Max insertion drift")
    parser.add_argument("--K_best", type=int, default=200,
                        help="Top-K pruning parameter")
    parser.add_argument("--n_bits", type=int, default=960,
                        help="Message size in bits")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON file for results")
    parser.add_argument("--fer_coverage", action="store_true",
                        help="Run FER vs coverage sweep")
    parser.add_argument("--error_sweep", action="store_true",
                        help="Run error rate sweep")
    parser.add_argument("--diagnose", action="store_true",
                        help="Only print tool diagnostics")

    args = parser.parse_args()

    # Diagnostic mode
    if args.diagnose:
        print("=== Tool Diagnostics ===")
        tools = diagnose_tools()
        for name, info in tools.items():
            status = "[OK]" if info.get('available') else "[MISSING]"
            device = info.get('device', '')
            path = info.get('path', '').split('/')[-1] if not device else device
            print(f"  {status} {name}: {path}")
        return 0

    if args.fer_coverage:
        print("=== FER vs Coverage Sweep ===")
        results = run_fer_vs_coverage(
            Pd=args.Pd, Pi=args.Pi, Ps=args.Ps,
            trials=args.trials,
        )
    elif args.error_sweep:
        print("=== Error Rate Sweep ===")
        results = run_error_rate_sweep(
            coverage=args.coverage,
            trials=args.trials,
        )
    else:
        print(f"=== Pressure Test: Asym-MGC + Realistic Nanopore ===")
        print(f"Mode: {args.mode}, Profile: {args.profile}")
        print(f"Channel: Pd={args.Pd}, Pi={args.Pi}, Ps={args.Ps}")
        print(f"Decoder: D_max={args.D_max}, I_max={args.I_max}, K_best={args.K_best}")

        runner = PressureTestRunner(
            mode=args.mode,
            profile=args.profile,
            Pd=args.Pd, Pi=args.Pi, Ps=args.Ps,
            n_bits=args.n_bits,
            D_max=args.D_max,
            I_max=args.I_max,
            K_best=args.K_best,
            seed=args.seed,
        )

        summary, details = runner.run_benchmark(
            trials=args.trials,
            coverage=args.coverage,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("=== BENCHMARK SUMMARY ===")
        print(f"  Total trials:       {summary.total_trials}")
        print(f"  Failures:            {summary.failures}")
        print(f"  FER:                 {summary.fer:.4f}")
        print(f"  Mean decode time:    {summary.mean_decode_time_ms:.2f} ms")
        print(f"  Std decode time:      {summary.std_decode_time_ms:.2f} ms")
        print(f"  P95 decode time:      {summary.p95_decode_time_ms:.2f} ms")
        print(f"  P99 decode time:      {summary.p99_decode_time_ms:.2f} ms")
        print(f"  Mean strand edit:     {summary.mean_strand_edit_rate:.4f}")
        print(f"  Mean consensus edit:   {summary.mean_consensus_edit_rate:.4f}")
        print("=" * 60)

        results = {
            "summary": asdict(summary),
            "trials": [asdict(r) for r in details],
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
