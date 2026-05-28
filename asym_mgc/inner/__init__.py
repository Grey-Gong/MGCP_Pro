"""
Inner code: RS + CRC + FSM-constrained encoding for Asym-MGC.
Phase 1.01 from IMPROVEMENT_PLAN.md.
"""

from .encode import ConstrainedRSEncoder, create_test_message, dna_to_binary, dna_to_base_ints, binary_to_dna
from .asymmetric_window import AsymmetricWindow
from .trellis import TrellisState, HomopolymerState, ViterbiTrellis
from .fsm_joint import FSMJointDecoder, FSMViterbiState, FSMPathMetric, FSMPathMetricTopK, HPState
from .decode import AsymMGCDecoder, levenshtein_distance, detect_markers, MarkerPositions, split_at_strong_markers, fallback_for_missing_strong_marker, FallbackState
from .bcjr import FSMBCJRDecoder, posterior_to_extrinsic_llr, posterior_to_reliability

__all__ = [
    "ConstrainedRSEncoder",
    "create_test_message",
    "dna_to_binary",
    "dna_to_base_ints",
    "binary_to_dna",
    "AsymmetricWindow",
    "TrellisState",
    "HomopolymerState",
    "ViterbiTrellis",
    "FSMJointDecoder",
    "FSMViterbiState",
    "FSMPathMetric",
    "FSMPathMetricTopK",
    "HPState",
    "AsymMGCDecoder",
    "levenshtein_distance",
    "detect_markers",
    "MarkerPositions",
    "split_at_strong_markers",
    "fallback_for_missing_strong_marker",
    "FallbackState",
    "FSMBCJRDecoder",
    "posterior_to_extrinsic_llr",
    "posterior_to_reliability",
]
