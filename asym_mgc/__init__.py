"""
Asym-MGC: Asymmetric Marker-Genie Code for Nanopore DNA Storage

Implementation of the Asym-MGC system as described in IMPROVEMENT_PLAN.md v2.1.
Target: FER improvement from ~10^-3 to ~10^-6 on nanopore sequencing data.
"""

__version__ = "2.1"
__author__ = "Asym-MGC Team"

from .pipeline import DNAPipeline, StrandResult, EncoderMetadata, compute_soft_branch_metric, build_strand_copies, decode_single_strand

__all__ = [
    "DNAPipeline",
    "StrandResult",
    "EncoderMetadata",
    "compute_soft_branch_metric",
    "build_strand_copies",
    "decode_single_strand",
]
