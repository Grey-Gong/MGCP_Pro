"""
Simplified Consensus Decoding Pipeline.

Implements the CHN-inspired consensus pipeline:
1. Collect multiple reads of the same DNA strand
2. Viterbi decode each read
3. MUSCLE multiple sequence alignment
4. Consensus formation
5. RS syndrome check

Reference: IMPROVEMENT_PLAN.md Section 3.7.8
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .encode import ConstrainedRSEncoder
from .decode import AsymMGCDecoder
from .muscle_wrapper import ConsensusPipeline, is_muscle_available
from reedsolo import RSCodec


@dataclass
class ConsensusDecodeResult:
    """Result of consensus decoding."""
    decoded_bits: List[int]
    success: bool
    num_copies: int
    consensus_quality: float  # Average consensus quality
    rs_syndrome: int
    info: dict


class SimplifiedConsensusDecoder:
    """
    Simplified consensus decoder for multi-copy DNA storage.

    This decoder accepts pre-decoded sequences (already Viterbi-decoded)
    and forms consensus from them using MUSCLE alignment.

    Parameters
    ----------
    N : int
        RS codeword length.
    l : int
        Bits per GF symbol.
    c_crc : int
        CRC bits per block.
    D_max : int
        Maximum deletion drift.
    I_max : int
        Maximum insertion drift.
    """

    DNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    INT_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def __init__(
        self,
        N: int = 120,
        l: int = 8,
        c_crc: int = 8,
        c_rs: int = 8,
        k_rs: int = 30,
        D_max: int = 10,
        I_max: int = 2,
        Pd: float = 0.05,
        Pi: float = 0.01,
        Ps: float = 0.04,
    ):
        self.N = N
        self.l = l
        self.c_crc = c_crc
        self.c_rs = c_rs
        self.k_rs = k_rs
        self.D_max = D_max
        self.I_max = I_max
        self.Pd = Pd
        self.Pi = Pi
        self.Ps = Ps

        # Inner decoder for each read
        self.inner_decoder = AsymMGCDecoder(
            N=N, l=l, c_crc=c_crc,
            D_max=D_max, I_max=I_max,
            Pd=Pd, Pi=Pi, Ps=Ps,
            K_best=100, T_threshold=15.0,
            list_k=1,  # Simplified: no list decoding
            enable_bcjr=False,
            adaptive_drift=False,
            k_rs=k_rs,  # 关键：传入正确的 k_rs
        )

        # Consensus pipeline
        self.consensus_pipeline = ConsensusPipeline()

        # RS codec for final check
        self.rs_codec = RSCodec(c_rs, c_exp=l)

    def decode_copies(
        self,
        copies: List[Tuple[str, np.ndarray]],
        raw_consensus: bool = True,
    ) -> Tuple[str, dict]:
        """
        Decode from multiple copies using consensus.

        Parameters
        ----------
        copies : list of (sequence, quality)
            Multiple copies of received DNA sequences.
        raw_consensus : bool
            If True, form consensus from raw reads (before Viterbi).
            If False, form consensus from Viterbi-decoded sequences.

        Returns
        -------
        decoded : str
            Decoded DNA sequence.
        info : dict
            Decoding info.
        """
        if len(copies) == 0:
            return '', {'success': False, 'num_copies': 0}

        # Step 1: Viterbi decode each copy (for RS syndrome check)
        decoded_copies = []
        for seq, qual in copies:
            dec, info = self.inner_decoder.decode(seq, quality=qual)
            decoded_copies.append((dec, info))

        # Step 2: Try to find a copy with zero RS syndrome
        best_copy = None
        best_syn = 999999
        for dec, info in decoded_copies:
            syn = info.get('rs_syndrome_nonzero', 999999)
            if syn < best_syn:
                best_syn = syn
                best_copy = (dec, info)

        if best_syn == 0:
            return best_copy[0], {'success': True, 'rs_syndrome': 0, **best_copy[1]}

        # Step 3: Consensus formation
        if len(copies) >= 2:
            # Form consensus from raw reads (better for indel channels)
            if raw_consensus:
                seqs = [seq for seq, _ in copies]
            else:
                seqs = [dec for dec, _ in decoded_copies]

            consensus, quality, aligned = self.consensus_pipeline.run(seqs)

            avg_quality = float(np.mean(quality)) if len(quality) > 0 else 0.0

            # Try to find perfect consensus
            try:
                consensus_bits = []
                for base in consensus:
                    if base == '-':
                        continue
                    base_int = self.DNA_TO_INT.get(base, -1)
                    if base_int >= 0:
                        consensus_bits.extend([(base_int >> 1) & 1, base_int & 1])
                decoded_rs = self.rs_codec.decode(consensus_bits)
                return consensus, {
                    'success': True,
                    'num_copies': len(copies),
                    'consensus_quality': avg_quality,
                    'rs_syndrome': 0,
                }
            except:
                pass

            # Return consensus even if RS fails
            return consensus, {
                'success': False,
                'num_copies': len(copies),
                'consensus_quality': avg_quality,
                'rs_syndrome': best_syn,
                'best_copy_syn': best_syn,
            }

        # Step 4: Return best copy
        return best_copy[0], {
            'success': best_syn == 0,
            'num_copies': len(copies),
            'rs_syndrome': best_syn,
        }


def end_to_end_test():
    """
    End-to-end test of simplified consensus pipeline.
    """
    from .channel.memory_k_nanopore import MemoryKNanoporeChannel

    print('='*60)
    print('简化共识 pipeline 端到端测试')
    print('='*60)

    # Encode
    encoder = ConstrainedRSEncoder(l=8, c_rs=8, c_crc=8, max_run=3)
    msg = [0, 1, 0, 1, 1, 0, 0, 1] * 30
    dna_original, meta = encoder.encode(msg)
    N = meta['N']

    print(f'DNA 长度: {len(dna_original)}, N={N}')
    print()

    # Test different coverage
    for num_copies in [1, 3, 5, 10]:
        print(f'{num_copies} 副本测试:')

        # Collect reads
        ch = MemoryKNanoporeChannel(Pd=0.02, Pi=0.004, Ps=0.016, seed=42)
        copies = []
        for i in range(num_copies):
            recv, qual = ch.transmit_with_quality(dna_original, base_quality_mean=15.0)
            copies.append((recv, qual))

        # Decode with consensus
        decoder = SimplifiedConsensusDecoder(N=N, l=8, c_crc=8)
        decoded, info = decoder.decode_copies(copies)

        # Check result
        ed = sum(a != b for a, b in zip(dna_original, decoded))

        print(f'  ED={ed}, success={info.get("success", False)}, '
              f'rs_syn={info.get("rs_syndrome", "N/A")}, '
              f'cons_q={info.get("consensus_quality", "N/A"):.2f if info.get("consensus_quality") else "N/A"}')
        print()

    # Test with zero-error channel
    print('零错误信道测试:')
    ch0 = MemoryKNanoporeChannel(Pd=0, Pi=0, Ps=0)
    recv0, qual0 = ch0.transmit_with_quality(dna_original, base_quality_mean=30.0)
    decoder = SimplifiedConsensusDecoder(N=N, l=8, c_crc=8)
    decoded0, info0 = decoder.decode_copies([(recv0, qual0)])
    ed0 = sum(a != b for a, b in zip(dna_original, decoded0))
    print(f'  1 副本: ED={ed0}')


if __name__ == '__main__':
    end_to_end_test()
