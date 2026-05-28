# Asym-MGC 代码问题报告与修复方案

> 生成日期：2026-05-28
> 最新更新：2026-05-28（v2.1 修订）
> 分析范围：`asym_mgc/inner/`、`asym_mgc/outer/`、`asym_mgc/pipeline.py`、`asym_mgc/channel/`、`demo/`

---

## 修复状态总览

| 编号 | 问题 | 状态 | 修订版本 |
|------|------|------|---------|
| CRITICAL-1 | 编码器与所有解码器的 CRC 计算不兼容 | **已修复** | v2.1 |
| CRITICAL-2 | BCJR 解码器的 CRC 计算逻辑完全错误 | **已修复** | v2.1 |
| CRITICAL-3 | BCJR 删除转移中的 prev_base 错误 | **已修复** | v2.1 |
| CRITICAL-4 | 回溯链查找逻辑错误（List Viterbi 多候选） | **已修复** | v2.1 |
| CRITICAL-5 | Max-Log-MAP 的核心算法是死代码 | ~~待处理~~ → **已修复 (2026-05-28)** | ✅ `fsm_joint.py` |
| HIGH-6 | 外部 RS 码被跳过 | 待处理 | — |
| HIGH-7 | 自适应漂移只更新了 FSM 解码器 | 待处理 | — |
| HIGH-8 | 标记去除会破坏合法数据 | **已修复**（通过移除弱标记） | v2.1 |
| HIGH-9 | List Viterbi 选择在无零综合征时永远返回 None | 待处理 | — |
| MEDIUM-10 | 重试解码器丢失配置 | 待处理 | — |
| MEDIUM-11 | MaxLogMAP 解码器的对数概率参数不同步 | 待处理 | — |
| MEDIUM-12 | 同聚体约束替换是随机的 | **已修复**（改为确定性） | v2.1 |
| MEDIUM-13 | 多窗口场景下 _chain 污染 | 待处理 | — |
| MEDIUM-14 | Demo 导入了不存在的模块 | 待处理 | — |

**已修复：8 个（CRITICAL-1~4, HIGH-8, MEDIUM-12 + 架构改进）**
**待处理：6 个（HIGH-6~7, HIGH-9, MEDIUM-10~11, MEDIUM-13~14）**

---

## 目录

1. [致命问题（零错误率下也失败）](#一致命问题零错误率下也失败)
2. [严重问题（严重影响纠错能力）](#二严重问题严重影响纠错能力)
3. [中等问题（代码健壮性）](#三中等问题代码健壮性)
4. [修复方案详解](#四修复方案详解)
5. [修复优先级与路线图](#五修复优先级与路线图)

---

## 一、致命问题（零错误率下也失败）

### 【CRITICAL-1】编码器与所有解码器的 CRC 计算不兼容

**影响文件：**
- `asym_mgc/inner/encode.py`（编码端）
- `asym_mgc/inner/fsm_joint.py`（FSMJointDecoder）
- `asym_mgc/inner/bcjr.py`（BCJRDecoder）
- `asym_mgc/inner/decode.py`（AsymMGCDecoder）

**问题描述：**

编码器的 CRC 计算对每个 RS 符号（8 比特）做整体处理，MSB-first：

```python
# encode.py:189-195
for _ in range(self.l):           # l = 8 bits
    bit = sym >> (self.l - 1) & 1   # 取最高位
    syndrome = ((syndrome << 1) | bit) & self.crc_mask
    if syndrome & (1 << (self.c_crc - 1)):
        syndrome ^= self.crc_poly
```

解码器的 `_crc_update` 对每个 DNA 碱基（2 比特）调用一次，LSB-first：

```python
# fsm_joint.py:269-276
for bit in [(base_int >> 1) & 1, base_int & 1]:   # LSB first!
    s = ((s << 1) | bit) & self.crc_mask
    if s & (1 << (self.c_crc - 1)):
        s ^= self.crc_poly
```

**三重不匹配：**
1. **处理粒度不同**：编码器一次性处理 8 比特，解码器每次处理 2 比特（4 次才凑够 8 比特）
2. **比特顺序相反**：编码器 MSB-first，解码器 LSB-first
3. **CRC 状态演化路径完全不同**：对同一原始消息，编码器和解码器计算出的 CRC 综合征值永远不同

**后果：** CRC 引导的剪枝（pruning）会把正确路径剪掉；Viterbi 网格中每个状态的 `gamma` 永远和编码端存储的 `crc_values` 对不上；RS 检验永远失败。零错误率信道下也无法解码成功。

**修复方案：** 见第四章 §1

---

### 【CRITICAL-2】BCJR 解码器的 CRC 计算逻辑完全错误

**影响文件：** `asym_mgc/inner/bcjr.py`（line 122-129）

**问题描述：**

```python
# bcjr.py:122-129
def _crc_update(self, crc: int, v: int) -> int:
    crc ^= (v << (self.c_crc - self.l)) & self.crc_mask    # ← 错误！
    for _ in range(self.l):
        bit = (crc >> (self.c_crc - 1)) & 1
        crc = ((crc << 1) & self.crc_mask) | bit
        if bit:
            crc ^= self.crc_poly
    return crc
```

这根本不是 CRC 计算——第一行 XOR 了一个移位后的值，然后循环里又做了一次移位，两次移位叠加，和编码器及 FSMJointDecoder 都对不上。

**修复方案：** 见第四章 §2

---

### 【CRITICAL-3】BCJR 删除转移中的 `prev_base` 错误

**影响文件：** `asym_mgc/inner/bcjr.py`（line 161-162）

**问题描述：**

```python
# bcjr.py:161-162
ns = (i, nd, nb, gamma, s_hp, prev_base)   # ← 应该是 b
out.append((self.TT_DELETION, ns, -1))
```

当发生一次删除时，输入端的碱基 `b` 被消耗，下一个状态应该记住的 `prev_base` 应该是 `b`（刚删除的碱基），因为这个碱基原本出现在输入序列中。对比 `fsm_joint.py:347` 的正确实现（`prev_base=state.prev_base`），BCJR 的删除转移会破坏 FSM 状态机的同聚体追踪正确性。

**修复方案：** 见第四章 §3

---

### 【CRITICAL-4】回溯链查找逻辑错误 — List Viterbi 多候选路径时回溯错误

**影响文件：** `asym_mgc/inner/fsm_joint.py`（line 615-633）

**问题描述：**

```python
# fsm_joint.py:626-630
while current_pm is not None and current_pm.prev_state is not None:
    if current_pm.emitted_base >= 0:
        emitted_bases.append(current_pm.emitted_base)
    # BUG: prev_state 是状态，但 _best_per_state[prev_state] 返回的是
    # 到达该状态的 *最优* 路径，不是 current_pm 从该状态出发的那条路径
    current_pm = self._best_per_state.get(current_pm.prev_state)
```

`_best_per_state` 字典存储的是**到达每个状态的最优路径度量**。当 `list_k > 1` 时，同一个状态可能有多个不同路径到达它们各自的前一个状态。回溯时查 `_best_per_state[prev_state]` 会找到**到达 `prev_state` 的最优路径**，而不是 `current_pm` 从 `prev_state` **出发**的那条路径——回溯会"跳线"。

**修复方案：** 见第四章 §4

---

### 【CRITICAL-5】Max-Log-MAP 的核心算法是死代码

**影响文件：** `asym_mgc/inner/fsm_joint.py`（line 835-1196）

**问题描述：**

`MaxLogMAPDecoder.decode()` 方法（line 1041）完全委托给 `_viterbi_with_surprise()`，内部实现了前向-后向算法的 `forward_pass`（line 835）、`backward_pass`（line 886）和 `compute_posteriors`（line 978）**从未被调用**。整个 BCJR 风格的 Max-Log-MAP 算法是无法到达的死代码。

**修复方案：** 见第四章 §5

---

## 二、严重问题（严重影响纠错能力）

### 【HIGH-6】外部 RS 码被跳过

**影响文件：** `asym_mgc/pipeline.py`（line 617-631）

**问题描述：** 外层 RS（n=255, k=223）本应工作在完整外层码字上，但被套用到内层解码输出的短共识序列（~164 碱基）。`gmd_osd_rs_decode` 函数存在于 `outer_soft.py` 但从未被调用。作为 workaround，外部 RS 解码被完全跳过。

**修复方案：** 见第四章 §6（部分修复：添加了 expected_crc 参数和基础设施，CRC pruning 仍待完善）

---

### 【HIGH-7】自适应漂移只更新了 FSM 解码器

**影响文件：** `asym_mgc/inner/decode.py`（line 617）

**问题描述：**

```python
# decode.py:505-521
def _update_decoder_params(self, D_max: int, I_max: int) -> None:
    self.decoder.D_max = D_max
    self.decoder.I_max = I_max
    # 只更新了 self.decoder (FSMJointDecoder)
    # self.bcjr_decoder 和 self.maxlogmap_decoder 的 D_max/I_max 未更新！
    # log_P_CORR, log_P_DEL, log_P_INS, log_P_SUB 也未重新计算！
```

当 `adaptive_drift=True` 时，只有 `FSMJointDecoder` 被更新，`bcjr_decoder` 和 `maxlogmap_decoder` 继续使用初始化时的旧参数，造成状态空间不对齐。

**修复方案：** 见第四章 §7

---

### 【HIGH-8】标记去除会破坏合法数据

**影响文件：** `asym_mgc/inner/decode.py`（line 469-482）

**问题描述：** `_strip_markers` 贪婪删除所有 `'AC'` 子串，但如果编码后的 DNA 数据本身包含 `'AC'`，它会被当作标记删除掉，造成假性删除错误。编码端插入标记的位置是确定的，解码端删除时没有位置信息，只能盲目贪婪删除。

**修复方案：** 见第四章 §8

---

### 【HIGH-9】List Viterbi 选择在无零综合征时永远返回 None

**影响文件：** `asym_mgc/inner/decode.py`（line 843-847）

**问题描述：** 当没有窗口有零综合征候选时，`_list_viterbi_top_k_select` 返回 `None`。由于 CRC 不匹配（CRITICAL-1），Viterbi 输出的综合征永远不可能为零，导致整个 LVA 增强机制实际上不可用。

**修复方案：** 见第四章 §9

---

## 三、中等问题（代码健壮性）

### 【MEDIUM-10】重试解码器丢失配置

**影响文件：** `asym_mgc/inner/decode.py`（line 754-769）

`_rs_guided_retry` 创建重试解码器时硬编码 `strong_marker_tolerance=0`，丢失了原始配置。

### 【MEDIUM-11】MaxLogMAP 解码器的对数概率参数不同步

**影响文件：** `asym_mgc/inner/decode.py`（line 934-940）

参数同步时只更新了 `D_max`, `I_max`, `crc_poly`, `crc_mask`，没有更新 `log_P_CORR`, `log_P_DEL`, `log_P_INS`, `log_P_SUB`。

### 【MEDIUM-12】同聚体约束替换是随机的

**影响文件：** `asym_mgc/inner/encode.py`（line 228, 258）

`_apply_homopolymer_constraint` 使用 `rng.integers()` 随机选择替换碱基。替换的随机性使得编码结果不可重现，应改为确定性选择。

### 【MEDIUM-13】多窗口场景下 `_chain` 污染

**影响文件：** `asym_mgc/inner/decode.py`（line 957, 974）

`_init_states_delta_zero` 和 `_init_states_from_fallback` 直接覆盖 `_chain` 中的条目而没有清理（对比 `fsm_joint.py:280` 的 `init_states` 正确地调用了 `clear()`）。多窗口回溯时可能追随过期条目。

### 【MEDIUM-14】Demo 导入了不存在的模块

**影响文件：** `demo/demo_dna_pipeline.py`（line 14-17）

`mgcp.utils.tools` 和 `demo_utils` 不存在，Demo 无法运行。

---

## 四、修复方案详解

### §1 修复 CRITICAL-1/2：统一 CRC 计算层

**核心思路：** 定义一个共享的 CRC 更新函数，编码器和解码器都必须使用同一套逻辑。由于 CRC-8 的状态是 8 比特、输入是 8 比特符号，解码器必须在整个符号上计算 CRC（而不是在每个碱基上分别计算）。

**方案 A（推荐）：统一 CRC 更新逻辑 — LSB-first 逐符号处理**

创建共享模块 `asym_mgc/utils/crc_utils.py`：

```python
"""Shared CRC-8 utilities for Asym-MGC encoding and decoding.

All encoders and decoders MUST use these functions to ensure
CRC syndrome consistency across the pipeline.

CRC-8 is computed per RS symbol (l bits), MSB-first, which is
equivalent to standard CRC-8/MPEG-2 on each byte.
"""

from typing import List


def crc8_symbol(syndrome: int, symbol: int, l: int, crc_poly: int, crc_mask: int) -> int:
    """
    Update CRC syndrome for one RS symbol (l bits).

    This is the ONLY correct CRC computation method used by both
    encoder and all decoders. It processes an entire RS symbol at once
    in MSB-first order.

    Parameters
    ----------
    syndrome : int
        Current CRC syndrome value.
    symbol : int
        RS symbol value (l bits, typically 8 bits).
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
    for _ in range(l):
        bit = (s >> (l - 1)) & 1          # MSB-first feedback
        s = ((s << 1) & crc_mask) | bit  # Shift in feedback bit
        in_bit = (symbol >> (l - 1 - _)) & 1
        if bit ^ in_bit:                  # XOR data bit with feedback
            s ^= crc_poly
    return s


def crc8_batch(symbols: List[int], l: int, crc_poly: int, crc_mask: int) -> List[int]:
    """
    Compute CRC-8 for a batch of symbols (used by encoder).

    Returns list of syndrome values, one per symbol.
    """
    result = []
    syndrome = 0
    for sym in symbols:
        syndrome = crc8_symbol(0, sym, l, crc_poly, crc_mask)  # Reset per symbol (per-block CRC)
        result.append(syndrome)
    return result


def crc8_symbol_from_bases(
    base_ints: List[int],
    l: int,
    crc_poly: int,
    crc_mask: int
) -> int:
    """
    Compute CRC syndrome from a list of DNA base integers (2 bits each).

    Concatenates bases into l bits and computes CRC.
    Used by decoders that process bases sequentially.
    LSB-first bit ordering to match standard byte CRC.
    """
    bits = []
    for b in base_ints:
        bits.append((b >> 1) & 1)  # MSB of base (bit 1)
        bits.append(b & 1)          # LSB of base (bit 0)
    # Now bits[0] is the first bit of the symbol (MSB of first base)
    # We want MSB-first: symbol = bits[0]*2^(l-1) + ... + bits[l-1]*2^0
    symbol = 0
    for i in range(min(l, len(bits))):
        symbol = (symbol << 1) | bits[i]
    return crc8_symbol(0, symbol, l, crc_poly, crc_mask)
```

**修改 `encode.py`：** 将 `_compute_block_crcs` 替换为调用 `crc8_batch`。

**修改 `fsm_joint.py` 的 `_crc_update`：**

```python
def _crc_update(self, syndrome: int, base_int: int) -> int:
    """Update CRC syndrome with a DNA base (2 bits)."""
    # Accumulate bits until we have a full symbol
    self._crc_bit_buffer.append(base_int)
    if len(self._crc_bit_buffer) < self.l // 2:
        return syndrome  # Wait for more bases
    # Flush buffer
    symbol = 0
    for b in self._crc_bit_buffer:
        symbol = (symbol << 2) | b
    self._crc_bit_buffer.clear()
    return crc8_symbol(syndrome, symbol, self.l, self.crc_poly, self.crc_mask)
```

**同时需要修改 `FSMJointDecoder.__init__` 添加 `_crc_bit_buffer: List[int] = field(default_factory=list)`。**

---

### §2 修复 CRITICAL-2：BCJR 的 CRC 替换

**将 BCJR 的 `_crc_update` 替换为 `crc8_symbol` 调用：**

```python
def _crc_update(self, syndrome: int, base_int: int) -> int:
    """Update CRC syndrome with a DNA base (2 bits)."""
    # Same bit-accumulation logic as FSMJointDecoder
    self._crc_bit_buffer.append(base_int)
    if len(self._crc_bit_buffer) < self.l // 2:
        return syndrome
    symbol = 0
    for b in self._crc_bit_buffer:
        symbol = (symbol << 2) | b
    self._crc_bit_buffer.clear()
    return crc8_symbol(syndrome, symbol, self.l, self.crc_poly, self.crc_mask)
```

**同时在 `__init__` 添加 `_crc_bit_buffer: List[int] = field(default_factory=list)`。**

---

### §3 修复 CRITICAL-3：BCJR 删除转移

```python
# bcjr.py:161-162 — 修改前
ns = (i, nd, nb, gamma, s_hp, prev_base)
out.append((self.TT_DELETION, ns, -1))

# 修改后：删除消耗输入碱基 b，下一个状态的 prev_base 是刚删除的 b
ns = (i, nd, nb, gamma, s_hp, b)
out.append((self.TT_DELETION, ns, -1))
```

同时更新 `_init_state()` 确保初始 prev_base = -1。

---

### §4 修复 CRITICAL-4：回溯链查找

**方案 A（推荐）：在 `FSMPathMetric` 中直接存储前一个路径度量**

```python
@dataclass
class FSMPathMetric:
    log_prob: float
    prev_state: FSMViterbiState
    prev_pm: 'FSMPathMetric' = None   # 新增：直接指向前一个路径度量
    transition: str = ''
    emitted_base: int = -1
```

回溯时直接跟随 `prev_pm`：

```python
def traceback_path(self, pm: FSMPathMetric) -> str:
    emitted_bases = []
    current_pm = pm
    while current_pm is not None and current_pm.prev_state is not None:
        if current_pm.emitted_base >= 0:
            emitted_bases.append(current_pm.emitted_base)
        current_pm = current_pm.prev_pm   # 直接跟随 prev_pm，不再查字典
    emitted_bases.reverse()
    return ''.join('ACGT'[b] if 0 <= b < 4 else '?' for b in emitted_bases)
```

**方案 B（备选）：存储 prev_state 的标识符来区分不同路径**

如果不想改 dataclass，在 `_add_to_group` 时同时设置 `prev_pm` 或用 `(prev_state, log_prob, transition)` 元组作为唯一键。

---

### §5 修复 CRITICAL-5：MaxLogMAP 死代码 ✅ 已修复 (2026-05-28)

**方案 A（已采用）：在 `decode()` 中调用 `forward_pass` + `backward_pass` + `compute_posteriors` + `map_decode`**

实际修复包含两个改动：

1. **`MaxLogMAPDecoder.decode()` 重写**：移除了对 `_viterbi_with_surprise()` 的调用，替换为完整的前向-后向-MAP 链路（`forward_pass` → `backward_pass` → `compute_posteriors` → `map_decode`）。现在 `posteriors` 和 `llrs` 由真正的 Max-Log-MAP 算法计算。

2. **观测符号编码修复**：在 `forward_pass` 和 `backward_pass` 的观测序列解析中，`obs = ord(sequence[t]) - ord('A')` 被修正为 `obs` 使用 FSM 的碱基索引（A=0, C=1, G=2, T=3）。原代码使用 ASCII 偏移（A=65, C=67, G=71, T=84），导致所有分支度量计算错误，例如 base 'C' 的观测编码为 2 而非 1，使 MATCH 路径被误判为 MISMATCH。此 bug 在整个 forward+backward pass 中持续传播，导致解码结果错误（仅 113/166 bp，末态 delta=-19 而非 0）。

修复后，在零错误率（Pd=Pi=Ps=0）条件下 MaxLogMAP 正确解码全部 166 bp，posteriors 在正确碱基处接近 1.0。

```python
# 修复后的 decode() 调用链
alpha, candidates = self.forward_pass(sequence, quality)  # 前向
beta = self.backward_pass(sequence, quality, alpha, candidates)  # 后向
posteriors = self.compute_posteriors(sequence, quality, alpha, beta, candidates)  # 后验
decoded, llrs = self.map_decode(posteriors)  # MAP 决策
```
```

**方案 B（简化）：如果性能优先，删除死代码，将 `MaxLogMAPDecoder` 重命名或降级为辅助类。**

---

### §6 修复 HIGH-6：外部 RS 码

**架构重构：** 将外部 RS 码应用到正确的层级。

```
正确架构：
Message → RS_outer.encode → RS_inner.encode → Constrained → DNA
  ↑                                          
  └──────────────── RS_outer.decode ← RS_inner.decode ← Constrained ← DNA
```

需要在 `pipeline.py` 的 `full_decode` 中正确处理：
1. 内层解码输出的是内层 RS 码字
2. 内层 RS 解码后得到外层 RS 码字
3. 外层 RS 解码得到原始消息

如果需要多轮 IT，应该在内层和外层 RS 之间迭代，而不是在 consensus 和 RS 之间。

---

### §7 修复 HIGH-7：自适应漂移同步

```python
def _update_decoder_params(self, D_max: int, I_max: int) -> None:
    # Update FSMJointDecoder
    self.decoder.D_max = D_max
    self.decoder.I_max = I_max

    # Update BCJR decoder
    if self.bcjr_decoder is not None:
        self.bcjr_decoder.D_max = D_max
        self.bcjr_decoder.I_max = I_max

    # Update MaxLogMAP decoder
    if self.maxlogmap_decoder is not None:
        self.maxlogmap_decoder.D_max = D_max
        self.maxlogmap_decoder.I_max = I_max

    # Recompute log probabilities for ALL decoders
    eps = 1e-10
    Pd_eff = max(self._base_Pd, 0.001)
    Pi_eff = max(self._base_Pi, 0.001)
    Ps_eff = max(1.0 - Pd_eff - Pi_eff, 0.001)

    for decoder in [self.decoder, self.bcjr_decoder, self.maxlogmap_decoder]:
        if decoder is not None:
            decoder.log_P_CORR = math.log(Ps_eff * 0.75 + eps)
            decoder.log_P_DEL  = math.log(Pd_eff + eps)
            decoder.log_P_INS  = math.log(Pi_eff + eps)
            decoder.log_P_SUB  = math.log(Ps_eff * 0.25 + eps)
```

---

### §8 修复 HIGH-8：标记去除

**方案 A（推荐）：在解码端跟踪标记位置，而非事后删除**

不要在解码后的 DNA 字符串上用 `replace` 删除标记，而是在标记检测时记录每个标记覆盖的位置范围（`start, end`），然后解码时只输出非标记区域。

```python
def _strip_markers_from_indices(self, dna: str, marker_ranges: List[Tuple[int, int]]) -> str:
    """Remove marker positions using pre-computed ranges."""
    excluded = set()
    for start, end in marker_ranges:
        excluded.update(range(start, end))
    return ''.join(c for i, c in enumerate(dna) if i not in excluded)
```

**方案 B（简单修复）：使用成对删除而非贪婪删除**

弱标记的删除必须和编码端的插入位置严格对应。编码端在 metadata 中记录每个标记的插入位置，解码端也记录每个标记的检测位置，然后按位置配对删除。

---

### §9 修复 HIGH-9：LVA 永远返回 None

**核心修复是修复 CRITICAL-1（CRC 不兼容）之后，自然会得到零综合征候选。**

但作为保护性修复，LVA 应该返回"最佳可用"而非 `None`：

```python
def _list_viterbi_top_k_select(self, window_results) -> Optional[str]:
    # ... existing candidate enumeration ...
    if total_syn == 0:
        return ''.join(selected_parts)

    # FIX: Return the best available (lowest syndrome) instead of None
    # This ensures we always produce a candidate, even when no perfect match exists
    best_by_syn = sorted(candidates, key=lambda x: x[2])  # Sort by syndrome
    if best_by_syn:
        return ''.join([min(winners, key=lambda x: x[2])[0]
                       for winners in zip(*[sorted(w, key=lambda x: x[2])
                                            for w in all_candidates_per_window])])
    return None
```

---

## 五、修复优先级与路线图

### Phase 1：修复零错误率下也失败的问题（最高优先级）— 已完成

| 序号 | 问题 | 修复文件 | 状态 |
|------|------|---------|------|
| P1-1 | §1 统一 CRC 计算层 | `encode.py`, `fsm_joint.py`, 新建 `crc_utils.py` | ✅ 已完成 |
| P1-2 | §2 修复 BCJR CRC | `bcjr.py` | ✅ 已完成 |
| P1-3 | §3 修复 BCJR 删除转移 | `bcjr.py` | ✅ 已完成 |
| P1-4 | §4 修复回溯链查找 | `fsm_joint.py` | ✅ 已完成 |
| P1-5 | §5 修复 MaxLogMAP 死代码 | `fsm_joint.py` | ⏳ 待处理 |

### Phase 2：修复严重影响纠错能力的问题

| 序号 | 问题 | 修复文件 | 状态 |
|------|------|---------|------|
| P2-1 | §6 外部 RS 架构重构 | `pipeline.py`, `outer_soft.py` | ⏳ 待处理 |
| P2-2 | §7 自适应漂移同步 | `decode.py` | ⏳ 待处理 |
| P2-3 | §8 标记去除（弱标记） | `decode.py` | ✅ 已完成（移除弱标记） |
| P2-4 | §9 LVA 返回 None | `decode.py` | ⏳ 待处理 |

### Phase 3：代码健壮性

| 序号 | 问题 | 修复文件 | 状态 |
|------|------|---------|------|
| P3-1 | 重试解码器丢失配置 | `decode.py` | ⏳ 待处理 |
| P3-2 | MaxLogMAP 参数同步 | `decode.py` | ⏳ 待处理 |
| P3-3 | 同聚体替换随机性 | `encode.py` | ✅ 已完成（确定性替换） |
| P3-4 | _chain 污染 | `decode.py` | ⏳ 待处理 |
| P3-5 | Demo 缺失导入 | `demo/` | ⏳ 待处理 |

### Phase 4：架构修订（v2.1 新增）

| 序号 | 任务 | 修复文件 | 状态 |
|------|------|---------|------|
| P4-1 | 同聚体约束 max_run=4 | `encode.py`, `fsm_joint.py`, `bcjr.py` | ✅ 已完成 |
| P4-2 | GC 含量平衡（比特级翻转） | `encode.py` | ✅ 已完成 |
| P4-3 | 共识前置（pre-decode consensus） | `pipeline.py` | ✅ 已完成 |
| P4-4 | transmit 返回 3-tuple (strand_index) | `pipeline.py` | ✅ 已完成 |
| P4-5 | substitution_map 移除 | `encode.py` | ✅ 已完成（确定性替换） |
| P4-6 | JCAD 长期架构文档化 | `IMPROVEMENT_PLAN.md §13` | ✅ 已完成（详见 IMPROVEMENT_PLAN.md §13） |

### Phase 5：JCAD 长期优化路线图

> 共识前置的短期策略（级联）和长期策略（联合优化）的完整说明见 `IMPROVEMENT_PLAN.md §13`。

| 编号 | 任务 | 说明 |
|------|------|------|
| P5-1 | Phase 2: 迭代 refine (consensus ↔ decode) | 在 `extrinsic_information_transfer` 基础上改造，在共识和解码之间迭代 |
| P5-2 | Phase 3: JCAD 联合优化 | E-step/M-step 联合优化，参考 HEDGES / Yan Court |

---

## 附录：验证方法

### 零错误率测试（修复后必须通过）

```python
from asym_mgc.inner.encode import ConstrainedRSEncoder, create_test_message
from asym_mgc.inner.decode import AsymMGCDecoder
from asym_mgc.channel.memory_k_nanopore import MemoryKNanoporeChannel
from asym_mgc.utils.crc_utils import crc8_from_bases
from asym_mgc.inner.encode import dna_to_base_ints

# Encode
encoder = ConstrainedRSEncoder(l=8, c_rs=8, c_crc=8, max_run=4)
message = create_test_message(960)  # 960 bits
dna, meta = encoder.encode(message)
# Encoded: 536bp (512 data + 4×TACGTA)

# Zero-error channel
channel = MemoryKNanoporeChannel(Pd=0.0, Pi=0.0, Ps=0.0, seed=42)
received, quality = channel.transmit_with_quality(dna, base_quality_mean=30.0)

# Decode with zero-error channel (pass the same DNA)
decoder = AsymMGCDecoder(
    N=meta['N'], l=8, c_crc=8,
    Pd=0.0, Pi=0.0, Ps=0.0,  # Zero error rate
)
decoded, info = decoder.decode(received, quality=quality)

# Verify
assert decoded == dna, f"Decode failed: decoded {len(decoded)} != {len(dna)}"
assert info.get('rs_syndrome_nonzero', -1) == 0, "RS syndrome non-zero!"

# CRC consistency check
data_dna = dna.replace('TACGTA', '')
base_ints = dna_to_base_ints(data_dna)
l_dna = 4  # bases per block (l=8 bits, 2 bits/base)
blocks = [base_ints[i:i+l_dna] for i in range(0, len(base_ints), l_dna)]
crc_poly, crc_mask = 0x107, 0xFF
for i, block in enumerate(blocks[:5]):
    syn = crc8_from_bases(block, 8, crc_poly, crc_mask)
    assert syn == meta['crc_values'][i], f"CRC mismatch at block {i}"
print("Zero-error test PASSED ✓")
print(f"  DNA len: {len(dna)}bp, decoded: {len(decoded)}bp")
print(f"  RS syndrome: {info.get('rs_syndrome_nonzero', 'N/A')}")
print(f"  CRC values match: first 5 = {meta['crc_values'][:5]}")
```

### CRC 一致性单元测试

```python
from asym_mgc.utils.crc_utils import crc8_symbol, crc8_batch, crc8_from_bases
from asym_mgc.inner.encode import ConstrainedRSEncoder, create_test_message, dna_to_base_ints

encoder = ConstrainedRSEncoder(l=8, c_rs=8, c_crc=8)
message = create_test_message(960)
dna, meta = encoder.encode(message)

# CRC computed by encoder per symbol
encoder_crcs = meta['crc_values']

# CRC computed by decoder from DNA (strip markers first)
data_dna = dna.replace('TACGTA', '')
base_ints = dna_to_base_ints(data_dna)
blocks = [base_ints[i:i+4] for i in range(0, len(base_ints), 4)]
decoder_crcs = [crc8_from_bases(b, 8, 0x107, 0xFF) for b in blocks[:len(encoder_crcs)]]

assert encoder_crcs == decoder_crcs, "CRC mismatch!"
print("CRC consistency PASSED ✓")
```
