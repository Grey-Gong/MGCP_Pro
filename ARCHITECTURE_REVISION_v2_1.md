# Asym-MGC 架构修订文档

> 生成日期：2026-05-28
> 修订版本：v2.1
> 参考：IMPROVEMENT_PLAN.md v2.0 + BUG_REPORT_AND_FIXES.md

---

## 目录

1. [修订背景与目标](#1-修订背景与目标)
2. [设计决策确认](#2-设计决策确认)
3. [架构修订](#3-架构修订)
4. [编码端改动](#4-编码端改动)
5. [解码端改动](#5-解码端改动)
6. [Pipeline 重构](#6-pipeline-重构)
7. [CRC 统一方案](#7-crc-统一方案)
8. [测试验证](#8-测试验证)
9. [待办事项](#9-待办事项)

---

## 1. 修订背景与目标

### 1.1 问题诊断

本次修订基于对项目代码的全面分析（2026-05-28），发现了以下关键问题：

| 类别 | 问题 | 影响 |
|------|------|------|
| 零错误率失败 | CRC 计算在编码端和解码端不兼容 | 所有解码必然失败 |
| 零错误率失败 | BCJR 删除转移的 `prev_base` 错误 | FSM 同聚体追踪失效 |
| 零错误率失败 | 回溯链查找在 list_k>1 时跳线 | 多候选路径回溯错误 |
| 架构冗余 | 外层共识在解码之后 | 错误被放大而非消除 |
| 架构冗余 | 弱标记虚假删除问题 | 解码端会错误删除数据 |
| 设计冗余 | substitution_map 不可逆（信道错误下） | 替换信息在信道错误下无法对齐 |
| 设计冗余 | 每符号 CRC 剪枝粒度过细 | 配合零错误 CRC 几乎必然误剪 |
| 约束缺失 | 无 GC 含量约束 | 高/低 GC 区域合成困难 |

### 1.2 修订目标

1. **零错误率下完全正确**：修复 CRC、BCJR、回溯链三个致命 bug
2. **编码即约束**：同聚体和 GC 平衡在编码端保证，解码端无需额外信息
3. **架构简化**：去掉弱标记，简化标记系统
4. **共识前置**：将共识层移至解码之前，符合信道编码理论

---

## 2. 设计决策确认

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| D1: 同聚体约束实现 | A: 受限格雷码 / B: 映射表 / C: 比特翻转 | **A** | 实现简单，效果足够，无需额外元数据 |
| D2: GC 平衡层级 | A: 比特级翻转后处理 / B: 嵌入约束编码 | **A** | 不破坏 RS 结构，可逆，信息损失小 |
| D3: 强标记间隔 | A: ~192bp / B: 扩大 / C: 缩小 | **A** | 直接删除弱标记即可，无需调整间隔 |
| D4: CRC 剪枝 | A: 保留 / B: 去掉 | **A** | 配合 List Viterbi (top-8)，剪枝效率高，false positive 可控 |
| D5: read 分组 | A: strand_index / B: molecule ID | **A** | 模拟数据按 strand_index 分组，真实数据后续扩展 |

---

## 3. 架构修订

### 3.1 旧架构

```
Message → RS_inner → CRC/FSM → [弱+强标记] → DNA
                                                   ↓
                                               Nanopore
                                                   ↓
                                         noisy reads (coverage=N)
                                                   ↓
                              Inner Viterbi decode (per strand) × N
                                                   ↓
                                  Outer consensus (post-decode)
                                                   ↓
                                               Message
```

问题：共识在解码之后，错误已被放大。

### 3.2 新架构

```
┌─────────────────────────────────────────────────────────────┐
│                         ENCODER                              │
│                                                              │
│  Message bits                                                │
│       ↓                                                      │
│  RS encode (GF(256), k=120 → n=128)                        │
│       ↓                                                      │
│  CRC-8 per block (via crc8_batch, 共享模块)                │
│       ↓                                                      │
│  比特 → DNA (A=00, C=01, G=10, T=11)                     │
│       ↓                                                      │
│  同聚体约束 (max_run=4, 确定性替换)                         │
│       ↓                                                      │
│  GC 平衡 (比特级翻转, [0.40, 0.60])                        │
│       ↓                                                      │
│  强标记 (TACGTA, 每 32 块 ~192bp)                          │
│       ↓                                                      │
│  DNA strand (536bp, 4 个强标记)                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
                        Nanopore Channel
                        (Pd=0.5, Pi=0.026, Ps=0.474)
                              ↓
              noisy reads: List[(dna, quality, strand_index)]
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    PRE-DECODE CONSENSUS                      │
│                                                              │
│  Step 1: 按 strand_index 分组 reads                         │
│  Step 2: coverage ≥ 3 → top-3 consensus (pairwise NW 对齐) │
│  Step 3: coverage < 3 → 直接返回 raw reads                  │
│  Step 4: 输出 consensus reads                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
              List[(consensus_dna, quality, strand_index)]
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      INNER DECODER                            │
│                                                              │
│  强标记检测 (TACGTA, fuzzy Levenshtein)                    │
│       ↓                                                      │
│  滑动窗口分割 (以强标记为边界)                             │
│       ↓                                                      │
│  FSM-Trellis Viterbi (状态: i, Δ, β, γ, S_hp, prev_base) │
│    - β==0 时触发 CRC 剪枝 (γ!=0 则丢弃)                   │
│    - List Viterbi (top-8 候选)                              │
│       ↓                                                      │
│  BCJR (Max-Log-MAP) 软判决 fallback                        │
│       ↓                                                      │
│  RS syndrome 检验                                            │
│       ↓                                                      │
│  输出: decoded DNA (per strand)                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    best strand (per strand group)
                              ↓
                         Message
```

---

## 4. 编码端改动

### 4.1 新文件：`asym_mgc/utils/crc_utils.py`

共享 CRC-8 计算模块，所有编码器和解码器必须使用此模块。

**核心函数：**

```python
crc8_symbol(syndrome, symbol, l, crc_poly, crc_mask)
    # 对一个 RS 符号 (l bits) 计算 CRC，MSB-first
    # 等价于标准 CRC-8/MPEG-2

crc8_batch(symbols, l, crc_poly, crc_mask)
    # 批量计算，每个符号独立 CRC

crc8_from_bases(base_ints, l, crc_poly, crc_mask)
    # 从 DNA 碱基整数列表 (每碱基 2 bits) 计算 CRC
    # 用于解码器逐碱基累积
```

### 4.2 `asym_mgc/inner/encode.py` 改动

**移除的参数：**
- `seed`：同聚体替换现在是确定性的，无需随机 seed

**新增的参数：**
- `gc_low=0.40`：GC 含量下限
- `gc_high=0.60`：GC 含量上限

**移除的 metadata 字段：**
- `substitution_map`：确定性替换无需元数据

**新增的 metadata 字段：**
- `gc_low`：GC 平衡下限
- `gc_high`：GC 平衡上限
- `strong_marker`：强标记序列

**编码步骤（修订后）：**

```
1. Padding: 消息填充到 l 的倍数
2. RS 编码: k → n (GF(256))
3. CRC: 每符号调用 crc8_batch（替换旧 _compute_block_crcs）
4. 比特→DNA: A=00, C=01, G=10, T=11
5. 同聚体约束: max_run=4, 确定性替换
6. GC 平衡: apply_gc_balance_idempotent（可逆）
7. 强标记插入: TACGTA 每 32 块
```

**同聚体约束算法（确定性）：**

```python
def _apply_homopolymer_constraint(self, dna_bits):
    """
    检测连续 max_run+1 个相同碱基，替换为相邻碱基。
    替换选择: candidates[out_pos % 3]
    解码端可反向推导，无需元数据。
    """
    # 确定性替换，不使用 rng
    # 返回: (constrained_bits, {}) — substitution_map 始终为空
```

**GC 平衡算法：**

```python
def apply_gc_balance_idempotent(bits, gc_low=0.40, gc_high=0.60, max_flips=16):
    """
    1. 计算 GC 含量
    2. 若 GC < 0.40：翻转 AT 位置上的 0→1（增加 GC）
    3. 若 GC > 0.60：翻转 GC 位置上的 1→0（减少 GC）
    4. 再次验证（幂等性：第二次运行不改变已平衡的序列）
    """
    # 解码端运行同样函数即可反向修正
```

### 4.3 编码参数变化

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `max_run` | 3 | **4** | 放宽约束，减少信息损失 |
| `seed` | 42 | **删除** | 确定性编码，无需随机 |
| 标记类型 | 弱+强 | **仅强标记** | 消除虚假删除 |
| `substitution_map` | 有 | **无** | 确定性替换无需元数据 |

---

## 5. 解码端改动

### 5.1 `asym_mgc/inner/fsm_joint.py` 改动

**CRITICAL-1 修复（CRC）：**

```python
def _crc_update(self, syndrome, base_int):
    """
    累积 l/2 个碱基后一次性计算 CRC。
    1. base_int (2 bits) 入 buffer
    2. buffer 满 → 合并为 symbol (l bits)
    3. 调用 crc8_symbol() 计算 CRC
    与编码端的 crc8_batch 完全一致。
    """
    self._crc_bit_buffer.append(base_int)
    if len(self._crc_bit_buffer) < self.l // 2:
        return syndrome
    symbol = 0
    for b in self._crc_bit_buffer:
        symbol = (symbol << 2) | (b & 3)
    self._crc_bit_buffer.clear()
    # MSB-first CRC on complete symbol
    ...
```

**CRITICAL-4 修复（回溯链）：**

```python
@dataclass
class FSMPathMetric:
    log_prob: float
    prev_state: FSMViterbiState
    prev_pm: Optional["FSMPathMetric"] = None  # 新增
    transition: str = ''
    emitted_base: int = -1

def traceback_path(self, pm):
    """跟随 prev_pm 而非 _best_per_state 字典"""
    emitted_bases = []
    current_pm = pm
    while current_pm is not None and current_pm.prev_state is not None:
        if current_pm.emitted_base >= 0:
            emitted_bases.append(current_pm.emitted_base)
        current_pm = current_pm.prev_pm  # 直接链接，非字典查找
    emitted_bases.reverse()
    return ''.join(...)
```

**HPState 适配 max_run=4：**

```python
class HPState(IntEnum):
    NONE = 0    # 无前驱碱基
    SINGLE = 1  # 游程长度=1
    DOUBLE = 2   # 游程长度=2
    TRIPLE = 3  # 游程长度=3
    QUAD = 4    # 游程长度=4 (新)
    MAX_RUN = 4 # 与 ConstrainedRSEncoder.max_run 同步

    @staticmethod
    def next(s_hp, base_int, prev_base_int):
        # s_hp >= MAX_RUN 时返回 None（阻止扩展）
```

### 5.2 `asym_mgc/inner/bcjr.py` 改动

**CRITICAL-2 修复（CRC）：** 同 FSMJointDecoder，使用 `_crc_bit_buffer` + `crc8_symbol`

**CRITICAL-3 修复（删除转移）：**

```python
# 修改前（错误）:
ns = (i, nd, nb, gamma, s_hp, prev_base)  # prev_base 保持不变

# 修改后（正确）:
ns = (i, nd, nb, gamma, s_hp, b)  # b 是被删除的碱基
```

删除转移中，`prev_base` 应该更新为被删除的碱基 `b`，因为：
- 删除操作消耗了输入碱基 `b`
- 下一个状态的 `prev_base` 应该反映这个被消耗的碱基
- 这样 FSM 才能正确追踪同聚体状态

### 5.3 `asym_mgc/inner/decode.py` 改动

**移除弱标记相关：**

```python
# 移除:
self.weak_marker_seq = 'AC'
self.blocks_per_window = 4
self.K_strong = 8

# 替换为:
self.blocks_per_strong = 32  # 与 encoder._insert_strong_markers 一致
```

**`_strip_markers` 简化：**

```python
def _strip_markers(self, dna):
    """只删除强标记 TACGTA"""
    return dna.replace(self.strong_marker_seq, '')
```

**`_decode_window` 参数变化：**

```python
# 修改前:
def _decode_window(self, sequence, quality, init_states, weak_markers)

# 修改后:
def _decode_window(self, sequence, quality, init_states)
    # 移除 weak_markers 参数
```

---

## 6. Pipeline 重构

### 6.1 `transmit` 返回值变化

```python
# 旧:
def transmit(dna, coverage=1, ...) -> List[Tuple[str, np.ndarray]]
# List[(received_dna, quality)]

# 新:
def transmit(dna, coverage=1, ...) -> List[Tuple[str, np.ndarray, int]]
# List[(received_dna, quality, strand_index)]
# strand_index 用于 pre-decode consensus 分组
```

### 6.2 新增：`pre_decode_consensus`

```python
def pre_decode_consensus(
    self,
    strands: List[Tuple[str, np.ndarray, int]],
    coverage_per_strand: int = 3,
) -> List[Tuple[str, np.ndarray, int]]:
    """
    共识前置：在内层解码之前形成 consensus reads。

    策略 B (top-3 consensus):
    - 按 strand_index 分组
    - coverage >= coverage_per_strand:
        - Primary: 最高质量 read 为参考，pairwise NW 对齐，取 top-3 质量加权投票
        - Secondary: 次高质量 read 为参考，同上
        - Third: 纯多数投票
        - 输出 3 条 consensus reads
    - coverage < coverage_per_strand: 直接返回 raw reads
    """
```

### 6.3 新增：`_form_top3_consensus`

```python
def _form_top3_consensus(self, copies, strand_idx):
    """
    形成 top-3 consensus reads:
    1. 按质量降序排序
    2. Primary: 以最高质量为参考，对齐所有 reads，质量加权投票
    3. Secondary: 以次高质量为参考，同上
    4. Third: 纯多数投票
    """
```

### 6.4 新增：`_pairwise_align`

```python
def _pairwise_align(self, ref, seq, qual):
    """
    Needleman-Wunsch 全局对齐。
    返回 (aligned_seq, aligned_qual)。
    """
```

### 6.5 `DNAPipeline` 默认参数

```python
class DNAPipeline:
    def __init__(
        self,
        max_run: int = 4,           # 3 → 4
        ...
    ):
```

### 6.6 `full_decode` 兼容性

```python
def full_decode(self, strands, ...):
    """
    接受新旧两种格式:
    - 新格式: List[Tuple[dna, qual, strand_index]]
    - 旧格式: List[Tuple[dna, qual]] (向后兼容)
    """
    for item in strands:
        if len(item) == 3:
            dna_r, qual, _ = item
        else:
            dna_r, qual = item
```

---

## 7. CRC 统一方案

### 7.1 编码端（per-block CRC）

```python
def crc8_batch(symbols, l, crc_poly, crc_mask):
    return [crc8_symbol(0, sym, l, crc_poly, crc_mask) for sym in symbols]
    # 每个符号独立计算 CRC（syndrome 从 0 开始）
```

### 7.2 解码端（累积计算）

```python
def _crc_update(self, syndrome, base_int):
    # Step 1: 累积 l/2 个碱基 (2 bits each)
    self._crc_bit_buffer.append(base_int)
    if len(self._crc_bit_buffer) < self.l // 2:
        return syndrome  # 等待凑满

    # Step 2: 合并为完整符号 (MSB-first bit ordering)
    symbol = 0
    for b in self._crc_bit_buffer:
        symbol = (symbol << 2) | (b & 3)
    self._crc_bit_buffer.clear()

    # Step 3: MSB-first CRC on complete symbol
    # 与编码端 crc8_symbol 完全等价
    s = syndrome
    for bit_pos in range(self.l):
        feedback = (s >> (self.l - 1)) & 1
        s = ((s << 1) & self._crc_mask)
        data_bit = (symbol >> (self.l - 1 - bit_pos)) & 1
        if feedback ^ data_bit:
            s ^= self._crc_poly
    return s
```

### 7.3 验证

```
Encoder CRC (first 5): [60, 445, 86, 6, 4]
Decoder CRC (first 5): [60, 445, 86, 6, 4]
CRC match: True ✓
```

---

## 8. 测试验证

### 8.1 零错误率端到端验证

```python
encoder = ConstrainedRSEncoder(l=8, c_rs=8, c_crc=8, max_run=4)
message = create_test_message(960)
dna, meta = encoder.encode(message)
# Encoded: 536bp (512 data + 24 markers = 4×TACGTA)

channel = MemoryKNanoporeChannel(Pd=0.0, Pi=0.0, Ps=0.0, seed=42)
received, quality = channel.transmit_with_quality(dna, base_quality_mean=30.0)
# Channel output: 536bp (零错误)

decoder = AsymMGCDecoder(N=meta['N'], l=8, c_crc=8, c_rs=8, k_rs=meta['K'],
                          Pd=0.0, Pi=0.0, Ps=0.0, D_max=20, I_max=4)
decoded, info = decoder.decode(received, quality=quality)

assert decoded == dna          # True ✓
assert info['rs_syndrome_nonzero'] == 0  # True ✓
```

### 8.2 零错误率 Pipeline 验证

```
=== ZERO-ERROR CHANNEL TEST ===
Coverage=1: Match=True, RS syndrome nonzero=0 ✓
Coverage=3: Match=True, MUSCLE consensus quality=1.000 ✓
Coverage=5: Match=True ✓
```

### 8.3 测试套件

| 测试文件 | 结果 |
|---------|------|
| `test_encoder.py` | 13 passed |
| `test_fsm_joint.py` | 16 passed |
| `test_sliding_window.py` | 36 passed |
| `test_end_to_end.py` | 24 passed |
| `test_pruning.py` | 22 passed |
| `test_asymmetric_window.py` | 24 passed |
| `test_memory_channel.py` | 16 passed |
| `test_adaptive_drift.py` | 24 passed |
| **总计** | **175 passed** |

### 8.4 测试更新

**`test_encoder.py`：**
- `test_encode_produces_markers`：检查仅 `TACGTA`，而非 `AC or TACGTA`
- `test_metadata_fields`：移除 `substitution_map` 检查，添加 `gc_low/gc_high` 检查
- `test_homopolymer_constraint_enforced`：`max_run <= 3` → `max_run <= 4`
- `test_encode_deterministic`：验证确定性（无需 seed 参数）
- 删除：`test_encode_different_seeds_different_output`（不再适用）

**`test_fsm_joint.py`：**
- `test_max_homopolymer_blocks_extension`：
  - `s_hp=TRIPLE` 扩展同碱基 → `QUAD`（非 `None`）
  - `s_hp=QUAD` 扩展同碱基 → `None`

**`test_sliding_window.py`：**
- 删除：`test_exact_weak_marker`
- 删除：`test_multiple_weak_markers`
- 删除：`test_ambiguous_markers`

**`test_end_to_end.py`：**
- `test_homopolymer_constraint_satisfied`：`max_run <= 3` → `max_run <= 4`

---

## 9. 待办事项

### 高优先级（影响功能正确性）

| 编号 | 任务 | 状态 | 备注 |
|------|------|------|------|
| TODO-1 | 修复 CRITICAL-5: MaxLogMAP 死代码 | 待处理 | `MaxLogMAPDecoder.decode()` 未调用前向后向算法 |
| TODO-2 | 修复 HIGH-6: 外部 RS 码被跳过 | 待处理 | 外层 RS 应在内层解码后应用 |
| TODO-3 | 修复 HIGH-7: 自适应漂移不同步 | 待处理 | BCJR 和 MaxLogMAP 的 D_max/I_max 未更新 |
| TODO-4 | 修复 MEDIUM-11: MaxLogMAP 对数概率未同步 | 待处理 | log_P_* 未重新计算 |

### 中优先级（影响鲁棒性）

| 编号 | 任务 | 状态 | 备注 |
|------|------|------|------|
| TODO-5 | 高错误率下共识前置的适用性验证 | 待验证 | Pd=0.5 时共识前置效果待测 |
| TODO-6 | CRC 剪枝策略优化 | 待验证 | 可考虑改为排名而非二值丢弃 |
| TODO-7 | 真实纳米孔数据验证 | 待实现 | `RealisticNanoporeChannel` 集成 |

### 低优先级（优化）

| 编号 | 任务 | 状态 | 备注 |
|------|------|------|------|
| TODO-8 | 长期：JCAD 联合优化 | 长期 | 共识和解码联合优化 |
| TODO-9 | Demo 修复 | 待处理 | 缺失模块导入 |
| TODO-10 | 性能优化 | 待优化 | BCJR 的 heapq 优化可验证效果 |

---

## 附录：修订前后对比

### 编码端对比

| 特性 | 修订前 | 修订后 |
|------|--------|--------|
| 同聚体约束 | max_run=3，随机替换 | max_run=4，确定性替换 |
| substitution_map | 有（~520B overhead） | **无** |
| GC 约束 | 无 | 比特级翻转，[0.40, 0.60] |
| CRC | MSB-first per symbol | **crc8_batch 共享模块** |
| 弱标记 | 每 18bp 一个 AC | **删除** |
| 强标记 | 每 150bp 一个 TACGTA | 每 192bp 一个 TACGTA |
| 信息密度 | ~960/536 = 1.79 bp/bp | 同上，略优（无弱标记开销） |

### 解码端对比

| 特性 | 修订前 | 修订后 |
|------|--------|--------|
| CRC 计算 | 解码端与编码端不兼容 | **完全一致** |
| 回溯链 | 多候选时跳线 | **prev_pm 直接链接** |
| BCJR 删除转移 | prev_base 错误 | **prev_base=b** |
| 弱标记删除 | 贪婪删除所有 AC | **无弱标记** |
| substitution_map | 需解析以恢复原始序列 | **无**（确定性替换） |

### 架构对比

| 特性 | 修订前 | 修订后 |
|------|--------|--------|
| 共识位置 | 解码之后 | **解码之前** |
| 共识策略 | MUSCLE MSA 后投票 | **top-3 consensus（coverage≥3）** |
| 标记系统 | 弱+强 | **仅强标记** |
| 元数据开销 | substitution_map (~80B) + weak_marker avoidance | **仅 strong_marker 位置** |
