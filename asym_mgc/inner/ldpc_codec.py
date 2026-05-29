"""
LDPC 编码和解码模块

基于 pyldpc 的 LDPC 实现，支持分级码率配置（Adaptive Rate）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class LDPCCode:
    """LDPC 码"""
    n: int
    k: int
    rate: float
    H: csr_matrix
    G: np.ndarray
    z: int = 1
    matrix_type: str = "systematic"

    @property
    def m(self) -> int:
        return self.n - self.k


@dataclass
class LDPCTierConfig:
    """
    LDPC 分级配置。

    测试依据（2026-05-29，实测 50 trials）：
        consensus BER 由 coverage 决定，LDPC 成功率由 consensus BER 决定。
        原始信道：p_sub=10%，coverage=1 → consensus BER≈6.2%
        原始信道：p_sub=10%，coverage=3 → consensus BER≈5.1%
        原始信道：p_sub=10%，coverage=5 → consensus BER≈3.3%
        原始信道：p_sub=10%，coverage=7 → consensus BER≈2.0%
        原始信道：p_sub=10%，coverage=10 → consensus BER≈1.3%
    """
    name: str
    coverage_min: int       # 该档位所需的最小 coverage
    n: int
    dv: int
    dc: int
    max_iter: int
    description: str


# =============================================================================
# 分级配置表
# =============================================================================

LDPC_TIERS: list[LDPCTierConfig] = [
    LDPCTierConfig(
        name="HIGH",
        coverage_min=7,   # coverage >= 7 时启用：共识 BER ≈ 2%
        n=96,
        dv=3,
        dc=6,
        max_iter=500,
        description="高码率（r=0.52），信息密度=1.04 bits/base。coverage ≥ 7 时达到 100% LDPC 成功率。",
    ),
    LDPCTierConfig(
        name="MEDIUM",
        coverage_min=3,   # coverage >= 3 时启用：共识 BER ≈ 5%
        n=120,
        dv=4,
        dc=5,
        max_iter=500,
        description="中码率（r=0.23），信息密度=0.45 bits/base。所有 coverage 下均达到 100% LDPC 成功率。",
    ),
    LDPCTierConfig(
        name="LOW",
        coverage_min=0,   # 任意 coverage
        n=200,
        dv=4,
        dc=5,
        max_iter=500,
        description="低码率（r=0.22），信息密度=0.43 bits/base。任意 coverage 下均达到 100% LDPC 成功率。",
    ),
]


def select_ldpc_tier(coverage: int) -> LDPCTierConfig:
    """
    根据 coverage 自动选择最优 LDPC 档位。

    Parameters
    ----------
    coverage : int
        测序副本数量

    Returns
    -------
    LDPCTierConfig
        推荐的档位配置
    """
    for tier in LDPC_TIERS:
        if coverage >= tier.coverage_min:
            return tier
    return LDPC_TIERS[-1]  # fallback to lowest tier


def create_tiered_ldpc(coverage: int, seed: int = 42) -> LDPCCode:
    """
    根据 coverage 选择最优档位的 LDPC 码。

    Parameters
    ----------
    coverage : int
        测序副本数量
    seed : int
        随机种子

    Returns
    -------
    LDPCCode
    """
    tier = select_ldpc_tier(coverage)
    return create_ldpc_code(
        n=tier.n, k=0,
        code_type=f"tiered_{tier.name}({tier.dv},{tier.dc})",
        dv=tier.dv, dc=tier.dc, seed=seed
    )


def get_tier_info(coverage: int) -> dict:
    """获取档位信息的可读摘要。"""
    tier = select_ldpc_tier(coverage)
    return {
        'tier': tier.name,
        'coverage': coverage,
        'n': tier.n,
        'dv': tier.dv,
        'dc': tier.dc,
        'max_iter': tier.max_iter,
        'description': tier.description,
    }


def create_ldpc_code(
    n: int,
    k: int,
    code_type: str = "systematic",
    dv: int = 3,
    dc: int = 6,
    seed: int = 42,
) -> LDPCCode:
    """
    Create an LDPC code with given (n, k) and (dv, dc) parameters.

    Parameters
    ----------
    n : int
        Codeword length (n must be divisible by dc, and dc > dv)
    k : int
        Message length (actual k is determined by dv/dc, not this parameter)
    code_type : str
        Code type label
    dv : int
        Variable node degree (default 3)
    dc : int
        Check node degree (must be > dv, must divide n)
    seed : int
        Random seed

    Returns
    -------
    LDPCCode with actual k = n - dv*n/dc (after systematic conversion)
    """
    import pyldpc

    H_raw, G_raw = pyldpc.make_ldpc(n, d_v=dv, d_c=dc, seed=seed)
    m = H_raw.shape[0]
    k_actual = G_raw.shape[1]

    H_new, G_sys_T = pyldpc.coding_matrix_systematic(H_raw)
    G = np.array(G_sys_T.T)

    return LDPCCode(
        n=n, k=k_actual, rate=k_actual / n,
        H=csr_matrix(H_new),
        G=G,
        z=1,
        matrix_type=code_type
    )


def create_low_rate_ldpc(
    n: int = 200,
    dv: int = 4,
    dc: int = 5,
    seed: int = 42,
) -> LDPCCode:
    """
    Create a low-rate LDPC code optimized for 10% BSC error channels.

    Tested configurations (2026-05-29):
        (n=200, dv=4, dc=5) -> k=43, rate=0.215 -> 10% BSC: 97% (iter=500)  [BEST]
        (n=120, dv=4, dc=5) -> k=27, rate=0.225 -> 10% BSC: 90%
        (n=180, dv=5, dc=6) -> k=34, rate=0.189 -> 10% BSC: 83%
        (n=120, dv=5, dc=6) -> k=24, rate=0.200 -> 10% BSC: 80%
        (n=240, dv=5, dc=6) -> k=44, rate=0.183 -> 10% BSC: 87%

    For 8% BSC (nanopore typical):
        (n=120, dv=4, dc=5) -> k=27, rate=0.225 -> 8% BSC: 100%
        (n=120, dv=5, dc=6) -> k=24, rate=0.200 -> 8% BSC: 97%

    Default: (n=200, dv=4, dc=5, iter=500) achieves 97% at 10% BSC.
    """
    return create_ldpc_code(
        n=n, k=0, code_type=f"low_rate({dv},{dc})",
        dv=dv, dc=dc, seed=seed
    )


def create_systematic_ldpc(n: int, k: int, seed: int = 42) -> LDPCCode:
    """创建规则 LDPC 码"""
    return create_ldpc_code(n, k, code_type="systematic", seed=seed)


def create_protograph_ldpc(n: int, k: int, seed: int = 42) -> LDPCCode:
    """创建 Protograph LDPC 码"""
    return create_ldpc_code(n, k, code_type="protograph", seed=seed)


def create_sc_ldpc(n: int, k: int, seed: int = 42) -> LDPCCode:
    """创建空间耦合 LDPC 码"""
    return create_ldpc_code(n, k, code_type="sc_ldpc", seed=seed)


def ldpc_encode(info_bits: np.ndarray, code: LDPCCode) -> np.ndarray:
    """LDPC 编码: c = info @ G"""
    if len(info_bits) != code.k:
        raise ValueError(f"信息位长度不匹配: {len(info_bits)} vs {code.k}")
    return np.dot(np.asarray(info_bits, dtype=int), code.G) % 2


def llr_from_quality(quality: np.ndarray) -> np.ndarray:
    """Phred 质量转 LLR"""
    p = np.clip(10 ** (-quality / 10), 1e-10, 1 - 1e-10)
    return np.log((1 - p) / p)


def min_sum_decode(
    received_llr: np.ndarray,
    code: LDPCCode,
    max_iter: int = 100,
    algorithm: str = "normalized",  # "min_sum", "normalized", "offset"
    alpha: float = 0.625,  # 归一化因子
    beta: float = 0.5,     # 偏移量
) -> Tuple[np.ndarray, bool, int]:
    """
    Min-Sum LDPC 解码器（支持多种变体）。

    Parameters
    ----------
    received_llr : np.ndarray
        接收的 LLR 值
    code : LDPCCode
        LDPC 码
    max_iter : int
        最大迭代次数
    algorithm : str
        算法变体: "min_sum", "normalized", "offset"
    alpha : float
        归一化因子（用于 normalized Min-Sum）
    beta : float
        偏移量（用于 offset Min-Sum）

    Returns
    -------
    Tuple[np.ndarray, bool, int]
        (解码比特, 是否收敛, 迭代次数)
    """
    n, m, H = code.n, code.m, code.H
    
    if len(received_llr) != n:
        raise ValueError(f"LLR 长度不匹配")
    
    # 构建连接
    var_to_check = [[] for _ in range(n)]
    check_to_var = [[] for _ in range(m)]
    
    for row, col in zip(H.tocoo().row, H.tocoo().col):
        if row < m and col < n:
            var_to_check[col].append(row)
            check_to_var[row].append(col)
    
    msg_v_to_c = np.zeros((n, m))
    msg_c_to_v = np.zeros((n, m))
    llr = received_llr.copy()
    
    for iteration in range(max_iter):
        # V→C
        for v in range(n):
            for c in var_to_check[v]:
                s = llr[v]
                for oc in var_to_check[v]:
                    if oc != c:
                        s += msg_c_to_v[v, oc]
                msg_v_to_c[v, c] = s
        
        # C→V (Min-Sum 变体)
        for c in range(m):
            for v in check_to_var[c]:
                others = [ov for ov in check_to_var[c] if ov != v]
                if others:
                    abs_vals = [abs(msg_v_to_c[ov, c]) for ov in others]
                    sign_vals = [np.sign(msg_v_to_c[ov, c]) for ov in others]
                    min_val = min(abs_vals)
                    sign = np.prod(sign_vals)
                else:
                    min_val = float('inf')
                    sign = 1.0
                
                # 应用算法变体
                if algorithm == "normalized":
                    # 归一化 Min-Sum
                    msg_c_to_v[v, c] = sign * alpha * min_val
                elif algorithm == "offset":
                    # 偏移 Min-Sum
                    offset_val = max(0, min_val - beta)
                    msg_c_to_v[v, c] = sign * offset_val
                else:
                    # 原始 Min-Sum
                    msg_c_to_v[v, c] = sign * min_val
        
        # 判决
        posterior = llr.copy()
        for v in range(n):
            for c in var_to_check[v]:
                posterior[v] += msg_c_to_v[v, c]
        
        decoded = (posterior < 0).astype(int)
        if np.sum(H @ decoded % 2) == 0:
            return decoded, True, iteration + 1
    
    return decoded, False, max_iter


def llr_from_quality_with_gain(
    quality: np.ndarray,
    snr_estimate: float = 1.0,
) -> np.ndarray:
    """
    利用 Phred 质量分数生成 LLR。

    LLR = 2 * r / sigma^2 * (2 * x - 1)
    对于 BPSK: LLR = 4 * r / sigma^2

    Parameters
    ----------
    quality : np.ndarray
        Phred 质量分数 (0-40)
    snr_estimate : float
        SNR 估计值

    Returns
    -------
    np.ndarray
        LLR 值
    """
    # 错误概率
    p = np.clip(10 ** (-quality / 10), 1e-10, 1 - 1e-10)
    
    # LLR = log((1-p)/p) * gain
    gain = 10 ** (snr_estimate / 10)  # SNR 转增益
    llr = np.log((1 - p) / p) * gain
    
    return llr


def verify_codeword(codeword: np.ndarray, code: LDPCCode) -> bool:
    """验证码字"""
    return np.sum(code.H @ codeword % 2) == 0
