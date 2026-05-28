"""
Benchmarks: Unified Experiment Framework for Asym-MGC.

Provides standardized experiments for FER measurement, scalability testing,
and SOTA comparison.
Reference: Section 6 of IMPROVEMENT_PLAN.md v2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class FERExperimentConfig:
    """Configuration for a FER experiment."""
    file_size_bits: int = 5000
    max_strand_length: int = 120
    inner_redundancy: List[int] = field(default_factory=lambda: [4, 6, 8])
    outer_redundancy: List[int] = field(default_factory=lambda: [100, 200, 500])
    error_rate_range: List[float] = field(
        default_factory=lambda: [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
    )
    Pd_Pi_Ps_ratios: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.447, 0.026, 0.527),  # Real nanopore
            (0.6, 0.1, 0.3),         # Deletion-dominant
            (0.33, 0.33, 0.34),      # I.I.D.
        ]
    )
    iterations: int = 1000
    coverages: List[int] = field(default_factory=lambda: [5, 10, 30])


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    fer: float  # Frame error rate
    bit_error_rate: float
    mean_decoding_time_ms: float
    std_decoding_time_ms: float
    p95_decoding_time_ms: float
    num_trials: int
    num_failures: int
    config: dict


@dataclass
class ScalabilityResult:
    """Result of a scalability experiment."""
    lengths: List[int]
    mean_times: List[float]
    std_times: List[float]
    p95_times: List[float]
    fer_per_length: List[float]
    config: dict


class ExperimentRunner:
    """
    Unified experiment runner for Asym-MGC benchmarks.

    Supports:
    - FER vs. error rate curves
    - FER vs. redundancy trade-off
    - Scalability analysis
    - Homopolymer ablation
    - Real nanopore data evaluation
    """

    def __init__(self, encoder=None, decoder=None):
        self.encoder = encoder
        self.decoder = decoder
        self.results: List[ExperimentResult] = []

    def run_fer_experiment(
        self,
        config: FERExperimentConfig,
        channel_factory: Callable,
        encode_fn: Callable,
        decode_fn: Callable,
    ) -> List[ExperimentResult]:
        """
        Run FER experiment across multiple error rates and configurations.

        Parameters
        ----------
        config : FERExperimentConfig
            Experiment configuration.
        channel_factory : callable
            Function that returns a channel with given (Pd, Pi, Ps).
        encode_fn : callable
            Encoding function(message) -> (dna, metadata).
        decode_fn : callable
            Decoding function(dna, metadata) -> decoded_message.

        Returns
        -------
        results : List[ExperimentResult]
            Per-configuration results.
        """
        results = []

        for Pd, Pi, Ps in config.Pd_Pi_Ps_ratios:
            for error_rate in config.error_rate_range:
                for inner_r in config.inner_redundancy:
                    trial_fer = []
                    trial_times = []

                    for trial in range(config.iterations):
                        message = np.random.randint(0, 2, config.file_size_bits).tolist()
                        dna, meta = encode_fn(message)

                        channel = channel_factory(Pd, Pi, Ps)
                        received, edits = channel.transmit(dna)

                        start = time.perf_counter()
                        decoded, info = decode_fn(received, meta)
                        elapsed_ms = (time.perf_counter() - start) * 1000

                        trial_times.append(elapsed_ms)

                        if decoded != message:
                            trial_fer.append(1.0)
                        else:
                            trial_fer.append(0.0)

                    fer = np.mean(trial_fer)
                    result = ExperimentResult(
                        fer=fer,
                        bit_error_rate=fer / config.file_size_bits,
                        mean_decoding_time_ms=np.mean(trial_times),
                        std_decoding_time_ms=np.std(trial_times),
                        p95_decoding_time_ms=np.percentile(trial_times, 95),
                        num_trials=config.iterations,
                        num_failures=sum(trial_fer),
                        config={
                            'Pd': Pd, 'Pi': Pi, 'Ps': Ps,
                            'error_rate': error_rate,
                            'inner_redundancy': inner_r,
                        },
                    )
                    results.append(result)
                    self.results.append(result)

        return results

    def run_scalability_experiment(
        self,
        lengths: List[int],
        fixed_error_rate: float,
        channel_factory: Callable,
        encode_fn: Callable,
        decode_fn: Callable,
        iterations: int = 100,
    ) -> ScalabilityResult:
        """
        Test decoding time and accuracy across different sequence lengths.
        """
        mean_times, std_times, p95_times, fer_list = [], [], [], []

        for length in lengths:
            times = []
            failures = 0

            for _ in range(iterations):
                message = np.random.randint(0, 2, length * 8).tolist()
                dna, meta = encode_fn(message)

                channel = channel_factory(fixed_error_rate)
                received, _ = channel.transmit(dna)

                start = time.perf_counter()
                decoded, _ = decode_fn(received, meta)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)

                if decoded != message:
                    failures += 1

            mean_times.append(np.mean(times))
            std_times.append(np.std(times))
            p95_times.append(np.percentile(times, 95))
            fer_list.append(failures / iterations)

        return ScalabilityResult(
            lengths=lengths,
            mean_times=mean_times,
            std_times=std_times,
            p95_times=p95_times,
            fer_per_length=fer_list,
            config={'error_rate': fixed_error_rate, 'iterations': iterations},
        )

    def summarize_results(self) -> dict:
        """Compute summary statistics across all results."""
        if not self.results:
            return {}

        fers = [r.fer for r in self.results]
        times = [r.mean_decoding_time_ms for r in self.results]

        return {
            'total_experiments': len(self.results),
            'mean_fer': np.mean(fers),
            'min_fer': np.min(fers),
            'max_fer': np.max(fers),
            'mean_time_ms': np.mean(times),
            'total_time_s': sum(r.mean_decoding_time_ms for r in self.results) / 1000,
        }
