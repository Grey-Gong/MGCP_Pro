"""
Constrained RS Encoder for Asym-MGC.

Encoding pipeline (revised v2.1):
  1. Pad input to multiple of l bits
  2. Group into l-bit blocks, convert to GF(2^l) symbols
  3. RS encode: add c_rs parity symbols (GF(2^l))
  4. CRC: compute per-block CRC-8 checksum (shared with decoder via crc_utils)
  5. Convert to DNA bits
  6. Homopolymer constraint (max_run, deterministic, no metadata needed)
  7. GC balance (bit-level correction, reversible)
  8. Insert strong markers only (TACGTA, no weak markers)

Reference: IMPROVEMENT_PLAN.md v2.1 + ARCHITECTURE_REVISION_v2_1.md.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from reedsolo import RSCodec

from ..utils.crc_utils import crc8_batch


# DNA binary mapping
DNA_TO_BITS = {
    'A': [0, 0],
    'C': [0, 1],
    'G': [1, 0],
    'T': [1, 1],
}
BITS_TO_DNA = {(0, 0): 'A', (0, 1): 'C', (1, 0): 'G', (1, 1): 'T'}
DNA_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

INT_TO_DNA = ['A', 'C', 'G', 'T']

# Strong marker only (no weak markers)
STRONG_MARKER = 'TACGTA'
STRONG_MARKER_LEN = 6


def binary_to_dna(bits: List[int]) -> str:
    """Convert a list of bits to a DNA sequence."""
    if len(bits) % 2 != 0:
        bits = bits + [0]
    dna = []
    for i in range(0, len(bits), 2):
        dna.append(BITS_TO_DNA[(bits[i], bits[i + 1])])
    return ''.join(dna)


def dna_to_binary(dna: str) -> List[int]:
    """Convert a DNA sequence to a list of bits."""
    bits = []
    for base in dna:
        bits.extend(DNA_TO_BITS[base])
    return bits


def dna_to_base_ints(dna: str) -> List[int]:
    """Convert a DNA string to a list of base integers (0-3)."""
    return [DNA_TO_BITS[b][0] * 2 + DNA_TO_BITS[b][1] for b in dna]


def binary_to_decimal_blocks(bits: List[int], block_length: int) -> List[int]:
    """Group bits into blocks and convert each to a decimal integer."""
    if len(bits) % block_length != 0:
        raise ValueError(
            f"Bit length {len(bits)} not divisible by {block_length}"
        )
    blocks = []
    for i in range(0, len(bits), block_length):
        block_bits = bits[i:i + block_length]
        decimal = int(''.join(str(b) for b in block_bits), 2)
        blocks.append(decimal)
    return blocks


def decimal_to_binary_blocks(decimals: List[int], block_length: int) -> List[str]:
    """Convert decimal integers back to fixed-length binary blocks."""
    return [f"{x:0{block_length}b}" for x in decimals]


# ---------------------------------------------------------------------------
# GC balance
# ---------------------------------------------------------------------------

def compute_gc_fraction(dna: str) -> float:
    """Compute GC fraction of a DNA string."""
    if not dna:
        return 0.5
    gc = dna.count('G') + dna.count('C')
    return gc / len(dna)


def _find_gc_flip_candidates(bits: List[int], gc_frac: float) -> List[int]:
    """
    Find positions where flipping a bit would improve GC balance.

    Returns indices into the bit list where flipping would:
    - Increase GC: flip a 0 in an A/T position
    - Decrease GC: flip a 1 in a G/C position
    """
    candidates = []
    pos_in_base = 0
    last_base = -1

    for idx, bit in enumerate(bits):
        if pos_in_base == 0:
            last_base = bit
            pos_in_base = 1
        else:
            base_int = last_base * 2 + bit
            is_gc = base_int in (1, 2)  # C=1, G=2
            if gc_frac < 0.40 and not is_gc:
                if bit == 0:
                    candidates.append(idx)
            elif gc_frac > 0.60 and is_gc:
                if bit == 1:
                    candidates.append(idx)
            pos_in_base = 0
            last_base = -1
    return candidates


def apply_gc_balance(bits: List[int], gc_low: float = 0.40, gc_high: float = 0.60,
                     max_flips: int = 16) -> Tuple[List[int], int]:
    """
    Correct GC fraction to [gc_low, gc_high] via bit flips.

    Uses at most max_flips flips; flips are chosen at evenly-spaced
    positions to avoid concentrating changes.

    Returns (corrected_bits, num_flips_made).
    Decoder reverses by re-running the same function (idempotent
    when already balanced).
    """
    dna = binary_to_dna(bits)
    gc_frac = compute_gc_fraction(dna)

    if gc_low <= gc_frac <= gc_high:
        return bits[:], 0

    candidates = _find_gc_flip_candidates(bits, gc_frac)
    if not candidates:
        return bits[:], 0

    flips_needed = 0
    if gc_frac < gc_low:
        flips_needed = max(1, int((gc_low - gc_frac) * len(bits) / 2))
    else:
        flips_needed = max(1, int((gc_frac - gc_high) * len(bits) / 2))

    flips_needed = min(flips_needed, max_flips, len(candidates))
    step = max(1, len(candidates) // flips_needed)
    indices_to_flip = candidates[::step][:flips_needed]

    result = bits[:]
    for idx in indices_to_flip:
        result[idx] ^= 1

    return result, len(indices_to_flip)


def apply_gc_balance_idempotent(bits: List[int], gc_low: float = 0.40,
                                gc_high: float = 0.60,
                                max_flips: int = 16) -> List[int]:
    """
    GC balance with idempotent encoding: decoder runs same function.

    After one pass, subsequent passes find no candidates (or flipping
    them again would hurt balance), so the result stabilizes.
    """
    corrected, n = apply_gc_balance(bits, gc_low, gc_high, max_flips)
    if n == 0:
        return corrected
    verified, n2 = apply_gc_balance(corrected, gc_low, gc_high, max_flips)
    return verified


# ---------------------------------------------------------------------------
# Main Encoder
# ---------------------------------------------------------------------------

class ConstrainedRSEncoder:
    """
    Complete encoding pipeline: RS + CRC + Homopolymer Constraint + GC Balance + Strong Markers.

    Encoding pipeline (v2.1):
    1. Pad input to multiple of l bits
    2. Group into l-bit blocks, convert to GF(2^l) symbols
    3. RS encode: add c_rs parity symbols (GF(2^l))
    4. CRC: compute per-block CRC-8 checksum (via shared crc_utils)
    5. Convert to DNA bits
    6. Homopolymer constraint: max_run-limited, deterministic
    7. GC balance: bit-level correction, reversible
    8. Insert strong markers only (TACGTA, no weak markers)
    """

    def __init__(
        self,
        l: int = 8,
        c_rs: int = 8,
        c_crc: int = 8,
        max_run: int = 4,
        crc_poly: int = 0x107,
        gc_low: float = 0.40,
        gc_high: float = 0.60,
    ):
        self.l = l
        self.c_rs = c_rs
        self.c_crc = c_crc
        self.max_run = max_run
        self.crc_poly = crc_poly
        self.crc_mask = (1 << c_crc) - 1
        self.gc_low = gc_low
        self.gc_high = gc_high

        self.q = 2 ** l
        self.rs_codec = RSCodec(c_rs, c_exp=l)

    def encode(self, message_bits: List[int]) -> Tuple[str, dict]:
        """
        Encode a binary message into a DNA strand.

        Parameters
        ----------
        message_bits : List[int]
            Binary message (list of 0/1).

        Returns
        -------
        dna_strand : str
            Final DNA sequence with strong markers.
        metadata : dict
            Encoding metadata for the decoder.
        """
        k_bits = len(message_bits)
        padded = self._pad_to_block(message_bits)

        K = len(padded) // self.l
        symbols = binary_to_decimal_blocks(padded, self.l)

        rs_symbols = list(self.rs_codec.encode(symbols))
        N = len(rs_symbols)

        crc_values = crc8_batch(symbols, self.l, self.crc_poly, self.crc_mask)

        dna_bits = self._symbols_to_bits(rs_symbols)

        constrained_bits = self._apply_homopolymer_constraint(dna_bits)

        balanced_bits = apply_gc_balance_idempotent(
            constrained_bits, self.gc_low, self.gc_high
        )

        dna_with_markers = self._insert_strong_markers(balanced_bits)

        metadata = {
            'k_bits': k_bits,
            'K': K,
            'N': N,
            'l': self.l,
            'c_rs': self.c_rs,
            'c_crc': self.c_crc,
            'crc_values': crc_values,
            'max_run': self.max_run,
            'gc_low': self.gc_low,
            'gc_high': self.gc_high,
            'strong_marker': STRONG_MARKER,
        }

        return dna_with_markers, metadata

    def _pad_to_block(self, bits: List[int]) -> List[int]:
        """Pad message to multiple of l bits."""
        remainder = len(bits) % self.l
        if remainder == 0:
            return bits[:]
        return bits + [0] * (self.l - remainder)

    def _symbols_to_bits(self, symbols: List[int]) -> List[int]:
        """Convert RS symbols to flat bit list (MSB-first)."""
        bits = []
        for sym in symbols:
            for i in range(self.l - 1, -1, -1):
                bits.append((sym >> i) & 1)
        return bits

    def _apply_homopolymer_constraint(self, dna_bits: List[int]) -> List[int]:
        """
        Apply homopolymer run-length constraint via deterministic substitution.

        When a run of max_run+1 identical bases would occur, substitute
        with a neighboring base. The substitution choice is deterministic
        (position-indexed), so decoder can reverse it without metadata.
        """
        result = []
        run_length = 0
        last_base = -1
        out_pos = 0

        i = 0
        while i < len(dna_bits):
            base_bits = dna_bits[i:i + 2]
            if len(base_bits) < 2:
                base_bits = base_bits + [0] * (2 - len(base_bits))
            base_int = (base_bits[0] << 1) | base_bits[1]

            if last_base < 0:
                last_base = base_int
                run_length = 1
                result.extend(base_bits)
                i += 2
                out_pos += 1
                continue

            if base_int == last_base:
                run_length += 1
            else:
                run_length = 1
                last_base = base_int

            if run_length > self.max_run:
                candidates = [b for b in range(4) if b != base_int]
                replace_idx = out_pos % len(candidates)
                replacement_int = candidates[replace_idx]
                result.append(replacement_int >> 1)
                result.append(replacement_int & 1)
                run_length = 1
                last_base = replacement_int
            else:
                result.extend(base_bits)

            i += 2
            out_pos += 1

        return result

    def _insert_strong_markers(self, dna_bits: List[int],
                                blocks_per_strong: int = 32) -> str:
        """
        Insert strong markers into the DNA sequence.

        Strong markers (TACGTA, 6bp) are inserted every
        blocks_per_strong data blocks.
        """
        dna = binary_to_dna(dna_bits)
        l_dna = self.l // 2  # DNA bases per block

        result = []
        i = 0
        block_idx = 0
        while i < len(dna):
            block_end = min(i + l_dna, len(dna))
            result.append(dna[i:block_end])
            i = block_end
            block_idx += 1

            if block_idx > 0 and block_idx % blocks_per_strong == 0:
                result.append(STRONG_MARKER)

        return ''.join(result)


def create_test_message(n_bits: int, seed: int = 42) -> List[int]:
    """Create a random test message."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n_bits).tolist()


# =============================================================================
# Integration test
# =============================================================================
if __name__ == '__main__':
    print("=== Asym-MGC Encoder Test ===")

    encoder = ConstrainedRSEncoder(l=8, c_rs=8, c_crc=8, max_run=4)

    message = create_test_message(960)
    print(f"Input message: {len(message)} bits")

    dna, meta = encoder.encode(message)
    print(f"Output DNA: {len(dna)} bases")
    print(f"Metadata keys: {list(meta.keys())}")
    print(f"K symbols: {meta['K']}, N symbols: {meta['N']}")
    print(f"CRC values (first 5): {meta['crc_values'][:5]}")
    print(f"GC fraction: {compute_gc_fraction(dna):.4f}")

    marker_count = dna.count(STRONG_MARKER)
    print(f"Strong markers inserted: {marker_count}")
