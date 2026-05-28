"""
Shared CRC-8 utilities for Asym-MGC encoding and decoding.

All encoders and decoders MUST use these functions to ensure
CRC syndrome consistency across the pipeline.

CRC-8 is computed per RS symbol (l bits), MSB-first.
Each DNA base contributes 2 bits: A=00, C=01, G=10, T=11.
Four bases form one RS symbol (l=8 bits).

Reference: CRC-8/MPEG-2 polynomial 0x07, but we use 0x107 for
the LFSR feedback formulation where the MSB of the syndrome is
fed back (equivalent when interpreted correctly).
"""

from __future__ import annotations

from typing import List


def crc8_symbol(
    syndrome: int,
    symbol: int,
    l: int,
    crc_poly: int,
    crc_mask: int,
) -> int:
    """
    Compute CRC syndrome for one RS symbol (l bits), MSB-first.

    This is the ONLY correct CRC computation method. It processes
    an entire symbol at once in MSB-first order, which is equivalent
    to standard CRC computation on a byte stream.

    The LFSR feedback formulation: at each step, the MSB of the
    syndrome register is fed back and XORed with the data bit.

    Parameters
    ----------
    syndrome : int
        Current CRC syndrome value (starts at 0).
    symbol : int
        RS symbol value (l bits, typically 8 bits = 1 byte).
    l : int
        Bits per RS symbol.
    crc_poly : int
        CRC generator polynomial (default 0x107 for CRC-8).
    crc_mask : int
        Bit mask (default 0xFF for CRC-8).

    Returns
    -------
    int
        Updated CRC syndrome.
    """
    s = syndrome
    for bit_pos in range(l):
        feedback = (s >> (l - 1)) & 1
        s = ((s << 1) & crc_mask)
        data_bit = (symbol >> (l - 1 - bit_pos)) & 1
        if feedback ^ data_bit:
            s ^= crc_poly
    return s


def crc8_batch(
    symbols: List[int],
    l: int,
    crc_poly: int,
    crc_mask: int,
) -> List[int]:
    """
    Compute CRC-8 for a batch of symbols (used by encoder).

    Each symbol gets its own CRC computed independently (per-block CRC).

    Parameters
    ----------
    symbols : List[int]
        List of RS symbol values.
    l : int
        Bits per RS symbol.
    crc_poly : int
        CRC generator polynomial.
    crc_mask : int
        Bit mask.

    Returns
    -------
    List[int]
        List of syndrome values, one per symbol.
    """
    return [crc8_symbol(0, sym, l, crc_poly, crc_mask) for sym in symbols]


def crc8_from_bases(
    base_ints: List[int],
    l: int,
    crc_poly: int,
    crc_mask: int,
) -> int:
    """
    Compute CRC syndrome from a list of DNA base integers (2 bits each).

    Concatenates bases into l bits (MSB-first bit ordering) and computes CRC.
    Used by decoders that process bases sequentially and accumulate bits
    until a full symbol is collected.

    Parameters
    ----------
    base_ints : List[int]
        List of DNA base integers (0-3), in emission order.
        Assumes exactly l/2 bases (forms one complete symbol).
    l : int
        Bits per RS symbol (must be divisible by 2).
    crc_poly : int
        CRC generator polynomial.
    crc_mask : int
        Bit mask.

    Returns
    -------
    int
        CRC syndrome of the formed symbol.
    """
    num_bases = l // 2
    if len(base_ints) != num_bases:
        raise ValueError(
            f"Expected {num_bases} bases to form {l}-bit symbol, "
            f"got {len(base_ints)}"
        )
    symbol = 0
    for b in base_ints:
        symbol = (symbol << 2) | (b & 3)
    return crc8_symbol(0, symbol, l, crc_poly, crc_mask)


def verify_all_crcs(
    base_ints_all: List[List[int]],
    expected_syndromes: List[int],
    l: int,
    crc_poly: int,
    crc_mask: int,
) -> List[bool]:
    """
    Verify CRC syndromes for all blocks.

    Parameters
    ----------
    base_ints_all : List[List[int]]
        List of base-int lists, one per block (each with l/2 bases).
    expected_syndromes : List[int]
        Expected CRC syndrome values from encoder metadata.
    l : int
        Bits per RS symbol.
    crc_poly : int
        CRC generator polynomial.
    crc_mask : int
        Bit mask.

    Returns
    -------
    List[bool]
        True if CRC checks out for each block.
    """
    results = []
    for bases, expected in zip(base_ints_all, expected_syndromes):
        computed = crc8_from_bases(bases, l, crc_poly, crc_mask)
        results.append(computed == expected)
    return results
