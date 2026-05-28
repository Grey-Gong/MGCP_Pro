# MGCP_Pro 改进计划 v2.1：Asym-MGC — 基于非对称网格与有限状态机的纳米孔 DNA 存储纠错码

> **目标：** 适应纳米孔测序平台的错误特性（长读长，indels 高，且 deletion-domination），实现 FER 从 ~10⁻³ 到 ~10⁻⁶ 的量级提升，算法复杂度从组合级降至多项式级。面向 IEEE TCOM 或同等水平期刊。

> **版本说明：**
> - **v2.1（2026-05-28）**：修复了 4 个 CRITICAL bug（CRC 不兼容、BCJR 删除转移、回溯链跳线），完成架构修订（编码即约束、去掉弱标记、共识前置）。详见 `ARCHITECTURE_REVISION_v2_1.md`。
> - **v2.0**：整合 Asym-MGC 理论框架（分层 Marker、滑窗 Viterbi、非对称漂移窗口）与 ATC-MGC 实现方案（CRC 引导剪枝、软判决外码）。

> **v2.1 架构概要：**
> - 编码：`RS → CRC8 → 同聚体(≤4, 确定性) → GC平衡 → 强标记(TACGTA)`
> - 信道：Nanopore (Pd=0.5, Pi=0.026, Ps=0.474)
> - 解码：`Pre-decode consensus → FSM-Trellis Viterbi → BCJR fallback → RS syndrome`
> - 共识前置：coverage ≥ 3 时先共识再解码，符合信道编码理论
> - GC 约束：无（比特级平衡后处理）
> - substitution_map：无（确定性替换）


---

## 1. 研究背景与问题陈述

### 1.1 DNA 存储与纳米孔测序

DNA 存储因其极高的数据密度（理论可达 215 EB/g）和超长保存时间被视为下一代冷数据存储的终极形态。纳米孔测序（Nanopore Sequencing）凭借便携性、低成本和长读长（10kb+）优势，是 DNA 数据读取的理想平台。

然而，纳米孔信道存在三个致命的错误特性，**现有 MGC+（Marker-Genie Code）专为理想随机信道设计，无法应对：**

| 错误特性 | 描述 | 对现有 MGC+ 的影响 |
|---------|------|------------------|
| **极高的 Indel 错误率** | 总错误率 5-15%，indels 占主导 | 穷举 erasure pattern 组合爆炸 |
| **Deletion-domination** | 删除概率远大于插入（Pd ≈ 0.5, Pi ≈ 0.03） | 对称窗口浪费状态空间 |
| **Homopolymer 敏感** | ~47-50% 错误发生在同聚体区域 | 编码端无预防，完全依赖纠错 |

### 1.2 现有 MGCP 的核心缺陷

```
| 缺陷 | 根因 | 严重程度 |
|------|------|---------|
| 解码器穷举所有 erasure pattern | preCompute_Patterns 枚举所有整数拆分 | 致命 |
| 无软信息利用 | 硬判决 Hamming 距离验证 | 严重 |
| 外码 majority vote 丢弃可靠性 | consensus_by_strand_id 无权重 | 严重 |
| 信道模型纯 i.i.d. | DNA_iid_channel 无记忆 | 中 |
| 缺乏与 SOTA 的公平对比 | 无统一基准框架 | 中 |
| 无信息论分析 | 缺乏理论贡献感 | 中 |
| 无 homopolymer 约束 | 编码端无预防 | 严重 |
```

### 1.3 解决思路

- **从被动纠错 → 联合预防 + 主动纠错：** 通过同聚体约束从源头消除 homopolymer 错误
- **从穷举搜索 → 结构化动态规划：** 通过非对称 Viterbi 网格将组合复杂度变为多项式
- **从硬判决 → 软判决：** 通过 LLR 加权的分支度量充分利用 basecaller 的置信度信息
- **从固定窗口 → 滑动窗口：** 通过分层 Marker 将长序列内存从 O(N) 降至 O(1)

---

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                      Encoding Pipeline                            │
│                                                                  │
│  Message → RS_encode → CRC_encode → Constrained_Coding → DNA    │
│                                          ↓                       │
│                                   ~~Hierarchical Markers~~        │
│                                   ~~(weak + strong)~~            │
│                                   → Strong Markers (TACGTA)      │
│                                          ↓                       │
│                                   DNA Strands                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Nanopore Sequencing
                    (Indels, Deletion-domination,
                     Homopolymer errors)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Decoding Pipeline                           │
│                                                                  │
│  1. Strong Marker Detection (TACGTA only, no weak markers)       │
│     → window boundary + drift state reset                        │
│                                          ↓                       │
│  2. Asym-MGC Trellis Decoder                                    │
│     State: (i, Δ, β, γ, S_hp)                                   │
│     Branch metric: LLR + drift prior + FSM constraint            │
│     Pruning: CRC early-termination + metric threshold + Top-K     │
│                                          ↓                       │
│  3. Consensus-first (pre-decode) → Inner decode → RS check      │
│     ~~Outer Soft-Decision Layer~~                                │
│     ~~Reliability-weighted consensus~~                           │
│     ~~DNA-specific error prediction → erasure conversion~~        │
│     ~~GMD/OSD RS decoding + Extrinsic Information Transfer~~     │
│                                          ↓                       │
│  Recovered Message                                              │
└─────────────────────────────────────────────────────────────────┘
```

> **v2.1 重大变化：** 标记系统从两级（弱+强）简化为仅强标记；共识从解码之后移至解码之前；FSM 同聚体约束从 ≤3 放宽至 ≤4。

**与 v1.0 的关键区别：** v1.0 的 ATC-MGC 和 Constrained Coding 是两个独立改进；v2.0 将 FSM 约束显式嵌入 Viterbi 状态空间（张量积 `(Δ, S_hp)`），使同聚体约束成为解码器的一等公民，而非外挂的 post-filter。

---

## 3. 核心创新一：Asym-MGC 内码 — 非对称网格与联合状态空间

### 3.1 非对称漂移窗口设计（Asymmetric Drift Window）

#### 3.1.1 动机

原版 MGC+ 假设 insertion 和 deletion 概率对称，将漂移状态空间设为 `Δ ∈ [-W, W]`。这在对称信道下合理，但在纳米孔的 deletion-domination 信道下：

```
典型 nanopore 参数:  Pd ≈ 0.50,  Pi ≈ 0.03,  Ps ≈ 0.47
Δ = net_deletions - net_insertions
E[Δ] = -(Pd - Pi) ≈ -0.47   (整体呈缩短趋势)
Var[Δ] ∝ N · Pd · (1 - Pd)
```

对称窗口 `[-W, W]` 对正负漂移等宽处理，导致**状态空间浪费**——大量状态永远不会被正确路径访问。

#### 3.1.2 非对称窗口定义

```
D_max = ceil(δ_d * N * 1.5)   # deletion 预算（负漂移）
I_max = ceil(δ_i * N * 0.5)   # insertion 预算（正漂移）

Δ ∈ [−D_max, +I_max]
|Δ| = D_max + I_max + 1

对比对称窗口:
  - 若 D_max = 3 * I_max（对应 Pd ≈ 3 * Pi）
  - 对称窗口 |Δ_sym| = 2W + 1
  - 若取 W ≈ D_max（安全选择），则 |Δ_sym| ≈ 2 · D_max + 1
  - 非对称窗口 |Δ_asym| = D_max + I_max + 1 ≈ D_max + D_max/3 + 1
  - 状态数减少: (|Δ_asym|) / (|Δ_sym|) ≈ (1 + 1/3) / 2 = 67%
  - 即状态数降低约 33%
```

#### 3.1.3 理论保证：非对称窗口的正确路径捕获率

**Theorem 1 (Asymmetric Window Completeness):** 设信道为 i.i.d. deletion/insertion 信道，deletion 概率为 Pd，insertion 概率为 Pi，满足 Pd > Pi。令漂移随机变量 `Δ_t = sum_{k=1}^t (D_k - I_k)`，其中 D_k ~ Bernoulli(Pd)，I_k ~ Bernoulli(Pi)。则在任意时刻 t，正确路径的漂移状态 `Δ_t` 满足 `−D_max ≤ Δ_t ≤ +I_max` 的概率至少为：

```
P(−D_max ≤ Δ_t ≤ +I_max) ≥ 1 − 2 · exp(−2 · (D_max − E[Δ_t])² / t)  [Hoeffding]
```

**Proof Sketch:** 设 X_k = D_k − I_k，则 E[X_k] = Pd − Pi = μ > 0，Var[X_k] ≤ 1/4。对 sum_{k=1}^t X_k 应用 Hoeffding 不等式：

```
P(Δ_t ≤ −D_max) = P(−Δ_t ≥ D_max) ≤ exp(−2 · D_max² / t)
```

类似地，对负方向的 Chernoff bound 可得 `P(Δ_t ≥ +I_max)` 的上界。合并两边得到定理。

**Corollary:** 取 `D_max = ceil((Pd − Pi) · t + 3√{t · log(1/δ)})`，则正确路径被非对称窗口捕获的概率 ≥ 1 − δ。

**实践意义：** 在 typical nanopore 参数下（Pd = 0.5, Pi = 0.03），取 δ = 10⁻⁶，对 N = 120 的序列，D_max ≈ 58，I_max ≈ 3，|Δ| ≈ 62。相比对称窗口 W = 58（|Δ_sym| = 117），**状态数减少 47%**，且正确路径被捕获的概率仍 ≥ 1 − 10⁻⁶。

### 3.2 联合状态空间：FSM-Trellis 张量积

#### 3.2.1 状态定义

将同聚体约束状态 `S_hp` 显式纳入 Viterbi 状态空间，与漂移状态 `Δ` 做张量积：

```
状态:  s = (i, Δ, β, γ, S_hp)

  i     ∈ [0, N]                          已处理的输入符号数
  Δ     ∈ [−D_max, +I_max]                累计 net deletion 计数
  β     ∈ [0, l−1]                        当前 block 内符号偏移
  γ     ∈ [0, 2^{c_crc}−1]                当前 block 的 CRC syndrome
  S_hp  ∈ {0, 1, 2, 3}                    末尾同聚体 run length
```

**总状态数：**

```
|S| = (N + 1) · (D_max + I_max + 1) · l · 2^{c_crc} · 4
```

以典型参数 `N=120, D_max=18, I_max=3, l=8, c_crc=8` 为例：

```
|S| ≈ 121 · 22 · 8 · 256 · 4 ≈ 21.8M 状态
```

虽然绝对数量不小，但关键性质：**这是一个固定上界，与 erasure pattern 数量无关，且活跃状态数由剪枝策略控制**。

#### 3.2.2 FSM 约束状态机

同聚体约束状态机（4 状态）：

```
S_hp = 当前末尾同聚体 run length

转移规则:
  若 x_next == prev_base:
    S_hp' = S_hp + 1   (当 S_hp < 3)
    拒绝         (当 S_hp ≥ 3)
  若 x_next != prev_base:
    S_hp' = 1          (新碱基，重置 run length)

转移指示函数:
  I(S_hp → S_hp') = 1  若转移合法
                     = 0  若转移违反同聚体约束（概率 −∞，硬剪枝）
```

**homopolymer 长度阈值 sweep：** 通过实验确定最优阈值：

| 阈值 | 状态数 | 信息损失 | 错误预防率 |
|------|--------|---------|----------|
| ≤ 2 | 3 | ~1% | ~30% |
| ≤ 3 | 4 | ~0.5% | ~47-50% |
| ≤ 4 | 5 | ~0.1% | ~55% |

默认值取 **≤ 3**，在状态数（4）和错误预防率（47-50%）之间取得最佳平衡。若实验发现解码失败主要由超长 homopolymer 引发，可提升至 ≤ 4。

#### 3.2.3 联合分支度量

这是 Asym-MGC 的核心公式——将 Viterbi 漂移状态、同聚体 FSM 约束和软信息融合为统一的分支度量：

```
γ_t((Δ_{t-1}, S_{t-1}) → (Δ_t, S_t))
    = log P(y_t | x_t)                          ← 软信息（LLR from basecaller）
    + log P(Δ_t | Δ_{t-1})                      ← 漂移先验（asymmetric）
    + log I(S_{t-1} → S_t)                      ← FSM 约束（硬剪枝）
    + log P(x_t | β_{t-1})                      ← RS block parity prior
```

**各分项详解：**

1. **软信息项** `log P(y_t | x_t)`：从 nanopore basecaller（ Dorado）获取 per-base Phred score Q，转换为 LLR：
   ```
   LLR(b) = Q · log(10)  若 b == y_t（接收与假设一致）
            = −Q · log(10)  若 b ≠ y_t（替换错误）
   ```
   对于 insertion/deletion，单独建模（见 3.4 节）。

2. **漂移先验项** `log P(Δ_t | Δ_{t-1})`：刻画 net deletion 的累积：
   ```
   P(Δ' | Δ) = P_DELETION  若 Δ' = Δ − 1
              = P_INSERTION  若 Δ' = Δ + 1
              = P_NO_ERROR  若 Δ' = Δ
              = 0            否则（不可能转移）
   ```
   其中 `P_DELETION = Pd · (1 − Pi − Ps)`，`P_INSERTION = Pi · (1 − Pd − Ps)`，`P_NO_ERROR = (1 − Pd − Pi − Ps)`。

3. **FSM 约束项** `log I(S_{t-1} → S_t)`：如果转移不满足同聚体约束，返回 `−∞`（硬剪枝）。

### 3.3 分层 Marker 与滑动窗口解码

#### 3.3.1 分层 Marker 设计

为解决超长序列（10kb+）中累积漂移无限放大的问题，引入两级 marker：

| 类型 | 序列 | 间隔 | 作用 |
|------|------|------|------|
| **弱 Marker** | `'AC'` | 每 4 blocks | 局部同步：允许 Viterbi 在局部窗口内修正小范围漂移 |
| **强 Marker** | `'TACGTA'` | 每 K 个弱 marker | 全局截断：将长序列分成独立的短窗口，重置漂移状态 |

**Marker 序列选择原则：**
- 与 DNA 编码字保持最小 Hamming 距离（避免被误识别为数据）
- 自身满足同聚体约束（run-length ≤ 3）
- 强 Marker 长度 6 bp，在随机序列中出现的期望概率为 `4^{-6} ≈ 1.5 × 10^{-4}`，足够稀疏

**Marker 插入算法：**

```
for block_idx in range(num_blocks):
    insert_block(block_idx)

    if block_idx % marker_period_weak == marker_period_weak - 1:
        insert_marker(weak_marker)  # 'AC'

    if block_idx % (marker_period_weak * K_strong) == 0:
        insert_marker(strong_marker)  # 'TACGTA'
        flush_and_reset_drift_state()  # 重置 Δ = 0
```

#### 3.3.2 滑动窗口 Viterbi 算法

强 Marker 作为窗口边界，将长序列的 Viterbi 解码从 O(N) 内存降至 **O(1) 内存**（仅需存储当前窗口状态）：

```python
def sliding_window_viterbi(Y, strong_markers, window_params):
    """
    核心思想: 以强 Marker 分割序列，对每个窗口独立运行 Viterbi，
    通过强 Marker 处的强制 Δ=0 状态保证窗口间一致性。
    """
    windows = split_at_strong_markers(Y, strong_markers)

    decoded_strong = []
    for win_idx, (Y_win, start_pos) in enumerate(windows):
        # 初始化：强 Marker 后的第一个状态强制 Δ=0
        init_states = build_init_states(Delta=0, S_hp=0, i=0)

        # 运行 Viterbi（局部窗口，内存 = O(窗口大小 × |Δ| × |S|)）
        # 窗口大小典型值: 50-200 blocks（~200-800 bp），内存可控
        viterbi_result = local_viterbi_decode(
            Y_win,
            init_states,
            max_delta=D_max,
            max_insertion=I_max,
            c_crc=8,
            K_max=200,        # 每 i-group 最多保留 200 个活跃状态
            T_threshold=15.0  # path metric 剪枝阈值
        )

        # 提取当前窗口的解码结果
        decoded_win, final_states = viterbi_result

        # 若非最后一个窗口：强 Marker 强制 Δ=0 验证
        if win_idx < len(windows) - 1:
            assert_strong_marker_detected(decoded_win, strong_marker)

        decoded_strong.append(decoded_win)

        # 为下一个窗口传递信息（软信息传递，非硬重置）
        next_init = propagate_soft_info(final_states, decoded_win)

    # 拼接所有窗口解码结果
    return concatenate(decoded_strong)
```

**内存复杂度对比：**

| 方案 | 内存复杂度 | N=120 | N=10,000 |
|------|-----------|-------|---------|
| MGCP 穷举 | `O(N · |P2|)` | ~MB | TB 级（不可行） |
| ATC-MGC DP | `O(N · |Δ| · |S|)` | ~10 MB | ~800 MB |
| **Asym-MGC 滑窗** | `O(W · |Δ| · |S|)` | — | **~10 MB（常数）** |

其中 W 为窗口大小（典型 200-800 bp），|Δ| ≈ 22，|S| = 4（固定小状态机）。

#### 3.3.3 强 Marker 丢失的 Fallback 机制

强 Marker 本身可能被 deletion 或 severe substitution 破坏。需要 explicit fallback：

```python
def detect_strong_marker(Y_segment, tolerance=2):
    """
    容忍最多 tolerance 个 substitution/indel 的 marker 检测。
    tolerance=0: 精确匹配（严格模式）
    tolerance=1: 允许 1 个编辑距离
    tolerance=2: 允许 2 个编辑距离（推荐）
    """
    # Levenshtein distance 检测，阈值 tolerance
    for pos in range(len(Y_segment) - len(strong_marker) + 1):
        window = Y_segment[pos:pos + len(strong_marker)]
        dist = levenshtein_distance(window, strong_marker)
        if dist <= tolerance:
            return pos, window  # Marker detected

    # Marker 丢失：使用前一个窗口的 final Δ 作为当前窗口的 init Δ
    return None  # Marker NOT detected → fallback needed
```

**Fallback 策略：** 若强 Marker 检测失败，使用前一个窗口的 final Δ 值作为当前窗口的初始化偏移，并在解码结果中标记"高不确定性"，供外码层处理。

### 3.4 软判决分支度量：LLR 的完整形式

#### 3.4.1 对 Match 转移的 LLR

给定接收符号 `y_t` 和假设输入 `x_t`：

```
LLR_match = log [P(y_t | x_t, MATCH) / P(y_t | x_t, DELETION)]
          = log [P_correct / P_deletion]
```

其中 `P_correct = 1 − Pd − Pi − Ps`，`P_deletion = Pd · (1 − Pi − Ps)`。

当 basecaller 提供 per-base quality Q 时：
```
P_error = 10^{−Q/10}
P_correct ≈ 1 − P_error
LLR_match ≈ Q · log(10)   (当 P_error 很小)
```

#### 3.4.2 对 Insertion 转移的 LLR

insertion 的特殊性：接收符号 `y_t` 不对应任何输入符号，此时：

```
LLR_insertion = log [P(y_t | INSERTION) / P(y_t | MATCH)]
              = log [P_insertion / P_correct]
```

若 basecaller 报告低 confidence（Q 很小）且 `y_t` 与相邻接收符号高度相似，则 LLR_insertion 增大（更可能是插入错误）。

#### 3.4.3 同聚体感知的 LLR 调整

homopolymer 边界处的 deletion 概率更高，需要动态调整 LLR：

```python
def homopolymer_aware_LLR(y_t, x_t, transition, hp_state, basecaller_llr):
    """
    根据同聚体上下文调整分支 LLR。

    若当前 x_t 与前一个符号相同（处于 homopolymer 中），
    则 deletion 的后验概率上调，insertion 的后验概率下调。
    """
    if hp_state > 0 and x_t == prev_base:
        # 处于 homopolymer run 中
        deletion_boost = math.log(homopolymer_penalty)  # penalty > 1
        return {
            'MATCH':     basecaller_llr - deletion_boost,
            'DELETION':  basecaller_llr + deletion_boost,
            'INSERTION': basecaller_llr  # insertion 不受 homopolymer 影响
        }
    else:
        return {
            'MATCH':     basecaller_llr,
            'DELETION':  0,
            'INSERTION': 0
        }
```

### 3.5 剪枝策略

继承 v1.0 ATC-MGC 的三种剪枝策略，并在联合状态空间下重新表述：

#### 策略 A：CRC Early Termination

在每个 block 边界（`β = 0`）检查 CRC syndrome `γ`：

```
if β == 0 and γ != 0:
    prune(state)  # 该 block 必有错误传播，提前丢弃
```

#### 策略 B：FSM Constraint（自动执行）

联合状态空间的 FSM 转移矩阵自动拒绝不合法转移——这是硬剪枝，无 false positive 风险。

#### 策略 C：Path Metric Threshold

```python
# 每个 Viterbi 步骤后
best_pm = max(PM[s] for s in active_states)

for s in active_states:
    if PM[s] < best_pm - T_threshold:
        prune(s)  # T_threshold 由 Chernoff bound 确定
```

#### 策略 D：Adaptive Top-K（按漂移状态分组）

```python
# 每个 (i, Δ) 组合最多保留 K_best 个状态
for (i, delta), states in group_by((i, delta)):
    if len(states) > K_best:
        states = top_K_by_PM(states, K=K_best)
```

典型值：`T_threshold = 15.0`（对应 FER ~10⁻⁹ 的置信度），`K_best = 200`。

### 3.6 复杂度分析

| 维度 | MGCP (v1.0 前) | ATC-MGC (v1.0) | Asym-MGC (v2.0) |
|------|---------------|----------------|----------------|
| **搜索策略** | 穷举 pattern | DP + CRC剪枝 | DP + FSM约束 + 滑窗 |
| **时间复杂度** | `O(lim·|P2|·K²)` | `O(M·K_active·l)` | `O(M·K_active·l)` |
| **空间复杂度** | `O(N·|P2|)` | `O(N·|Δ|·4)` | `O(W·|Δ|·4)` ⭐ |
| **可扩展性** | 随 N 指数 | 随 N 线性 | **随 N 常数**（窗口） |
| **长读长支持** | ❌ (~200 bp) | △ (~2kb) | ✅ (10kb+) |
| **软信息** | 无 | 部分 | **完整 LLR** |
| **FSM 约束** | 无 | 外挂 post-filter | **联合状态空间** |
| **Marker 丢失处理** | 无 | 无 | **Fallback 机制** |
| **理论保证** | 无 | CRC completeness | **漂移概率 bound** |

---

## 4. 核心创新二：软判决外码（Extrinsic Information Transfer）

### 4.1 系统定位

外码层解决的是** oligo 层面的 sequence dropout**（某些 strand 完全丢失）和**内码输出的残余错误**。这与内码的 symbol-level 纠错形成互补：

```
内码输出: 每个 strand 的解码结果（可能有残留错误）
         ↓
外码输入: N_strands 份可能有错的 strand copies
         ↓
外码任务: (1) consensus formation (reliability-weighted)
          (2) strand dropout 补偿 (RS erasure decoding)
          (3) 残余错误纠正 (GMD/OSD)
          (4) 软信息传递回内码 (Extrinsic IT)
```

### 4.2 Reliability-Weighted Consensus

```python
def soft_consensus(copies_with_quality, length):
    """
    copies_with_quality: list of (sequence: str, phred_array: ndarray) tuples
    每份 copy 携带 per-position Phred score，直接映射为权重。
    """
    consensus = []
    pos_confidence = []

    for pos in range(length):
        score = {b: 0.0 for b in 'ATCG'}

        for seq, qual in copies_with_quality:
            base = seq[pos] if pos < len(seq) else '-'
            q = qual[pos] if pos < len(qual) else 0
            weight = 10 ** (q / 10)  # Phred weight（指数映射，差异更显著）

            if base != '-':
                score[base] += weight

        best_base = max(score, key=score.get)
        consensus.append(best_base)
        pos_confidence.append(score[best_base])

    return consensus, pos_confidence
```

### 4.3 DNA-Specific Error Prediction → Erasure Conversion

参考 Derrick (Nature Science Review, 2024)，将 systematic error 位置强制转换为 erasure，让 RS 解码器利用 erasure capability（每个 erasure 消耗一半的纠错能力）：

```python
def dna_error_predictor(sequence, context_window=5):
    """
    基于序列上下文预测易出错的位置，返回 per-position error probability。
    参考: real nanopore error profile statistics。
    """
    error_probs = []

    for i, base in enumerate(sequence):
        ctx = sequence[max(0, i - context_window):i + context_window]
        hp_before = count_run_length_backward(sequence, i)
        hp_after = count_run_length_forward(sequence, i)

        # Homopolymer boundary: deletion 概率升高
        if hp_before + hp_after >= 3:
            # 当前在 homopolymer run 内或边界处
            p_hp_boundary = 0.15 if hp_before > 0 else 0.05
        else:
            p_hp_boundary = 0.0

        # Error-prone motifs (GA, CU 等)
        p_motif = 0.08 if ctx.upper() in ['GAGA', 'CUCU', 'AGAG', 'CTCT'] else 0.0

        # GC content bias
        gc_ratio = (ctx.count('G') + ctx.count('C')) / max(len(ctx), 1)
        p_gc_bias = 0.03 if (gc_ratio > 0.7 or gc_ratio < 0.3) else 0.0

        error_probs.append(p_hp_boundary + p_motif + p_gc_bias)

    return error_probs
```

### 4.4 GMD + OSD 软判决 RS 解码

```python
def gmd_osd_rs_decode(received, confidence, error_probs, rs_n, rs_k, field):
    """
    1. 综合 Phred weight + DNA error prediction 生成综合可靠性向量
    2. 最低 20% 位置标记为 erasure
    3. 尝试 RS erasure decoding
    4. 若失败，尝试 OSD (order=3)
    """
    # 综合可靠性: Phred weight × (1 - error_prob)
    reliability = [w * (1 - ep) for w, ep in zip(confidence, error_probs)]

    threshold = np.percentile(reliability, 20)  # 最低 20% 为 erasure
    erasure_positions = [i for i, r in enumerate(reliability) if r < threshold]

    decoded = rs_erasure_decode(received, erasure_positions,
                                n=rs_n, k=rs_k, field=field)
    if decoded is not None:
        return decoded, 'erasure_success'

    # OSD: 按可靠性排序，尝试翻转最不可靠的位置
    for order in range(1, 4):
        decoded = osd_decode(received, reliability, order=order,
                            n=rs_n, k=rs_k, field=field)
        if decoded is not None:
            return decoded, f'osd_order_{order}'

    return None, 'failed'
```

### 4.5 Extrinsic Information Transfer（外码 → 内码）

这是 v2.0 相对于 v1.0 的关键增强：**软信息不仅在内码层流动，还在外码层和内码层之间双向流动**：

```
Basecaller LLR
     ↓
┌─────────────────────────────────────────────┐
│  Inner: Asym-MGC Trellis Decoder             │
│  输出: per-strand reliability + decoded bits  │
└────────────────────┬────────────────────────┘
                     ↓
        Reliability-Weighted Consensus
                     ↓
        DNA Error Prediction → Erasure Map
                     ↓
        ┌──────────────────────────────────────────┐
        │  Outer: GMD/OSD RS Decoding              │
        │  输出: per-position extrinsic LLR         │
        └────────────────────┬─────────────────────┘
                             ↓
              若外码成功纠正: 生成 extrinsic LLR
              LLR_ext(b) = log P(b | 解码结果) − log P(b | 内码输出)
                             ↓
              反馈回内码: 更新 per-strand 权重
              内码重新解码（若时间允许）→ 迭代 refinement
```

```python
def extrinsic_information_transfer(inner_results, outer_result, max_iters=3):
    """
    迭代 extrinsic IT。
    每次迭代: 内码 → 外码 → 内码。
    """
    current_results = inner_results

    for iteration in range(max_iters):
        # Consensus formation with current weights
        consensus, weights = soft_consensus(current_results)

        # GMD/OSD outer decoding
        decoded, status = gmd_osd_rs_decode(consensus, weights,
                                             error_probs=None,
                                             rs_n=outer_n, rs_k=outer_k,
                                             field=gf)

        if status.startswith('success'):
            # 生成 extrinsic LLR 反馈
            extrinsic_llr = compute_extrinsic(decoded, consensus, weights)

            # 更新内码权重（用于下次迭代或最终输出）
            for i, result in enumerate(current_results):
                result.confidence = extrinsic_llr[i]

            if iteration > 0:
                print(f"Iteration {iteration}: extrinsic IT improved result")
            return decoded

    return outer_result  # fallback to best outer decode
```

---

## 5. 改进三：Memory-k Nanopore 信道模型

### 5.1 信道模型

```python
class MemoryKNanoporeChannel:
    def __init__(self, k=3, Pd=0.5, Pi=0.026, Ps=0.474):
        self.k = k
        self.Pd = Pd
        self.Pi = Pi
        self.Ps = Ps
        self.homopolymer_penalty = 2.0   # homopolymer 边界处 Pd × 2
        self.gc_bias = 0.15              # 高/低 GC 区域 Pd × 1.15

    def error_profile(self, context):
        """给定前 k 个输入符号，返回 (Pd, Pi, Ps)。"""
        ctx = context[-self.k:] if len(context) >= self.k else context

        # Homopolymer penalty
        if len(ctx) >= 2 and ctx[-1] == ctx[-2]:
            Pd = self.Pd * self.homopolymer_penalty
        else:
            Pd = self.Pd

        # GC bias
        gc_ratio = (ctx.count('G') + ctx.count('C')) / max(len(ctx), 1)
        if gc_ratio > 0.7:
            Pd *= (1 + self.gc_bias)
        elif gc_ratio < 0.3:
            Pd *= (1 - self.gc_bias * 0.5)

        return {'Pd': min(Pd, 0.95), 'Pi': self.Pi, 'Ps': self.Ps}
```

### 5.2 BCJR AIR 计算

```python
def compute_BCJR_AIR(channel, sequence_length=10000):
    """
    BCJR-once: 在发送端使用完美 CSI，计算可达信息速率。
    参考: Hamoum et al., ISTC 2021; Maarouf et al., 2023.
    """
    n_states = 4 ** channel.k
    alpha = np.full(n_states, -np.inf)
    beta = np.full(n_states, -np.inf)
    alpha[0] = 0.0  # 初始状态

    # Forward pass
    for pos in range(sequence_length):
        alpha_new = np.full(n_states, -np.inf)
        for s in range(n_states):
            for prev_s in range(n_states):
                if T[s, prev_s] > -np.inf:  # valid transition
                    alpha_new[s] = np.logaddexp(
                        alpha_new[s],
                        alpha[prev_s] + T[s, prev_s]
                    )
        alpha = alpha_new

    # 计算 mutual information: I(X;Y) ≈ H(X) - H(X|Y)
    # 使用 alpha 的 stationary distribution
    H_alpha = -np.sum(alpha * np.exp(alpha)) / sequence_length
    AIR = 1.0 - H_alpha  # H(X) = 2 bits/symbol for DNA
    return AIR
```

---

## 6. 改进四：性能基准对比

### 6.1 对比方案

| 对比对象 | 类型 | 代码可得性 |
|---------|------|----------|
| **HEDGES** | Hash-based, sync-aware | 开源 |
| **DNA Fountain** | Fountain code | 开源 |
| **DNA-Aeon** | Concatenated + RS | 需联系作者 |
| **DNA StairLoop** | Staircase interleaver | 开源 |
| **MGCP (现有)** | Guess-and-check | 已复现 |
| **ATC-MGC (v1.0)** | Trellis DP + CRC pruning | 本计划 |

### 6.2 对比指标与实验参数

```python
EXPERIMENTS = {
    'fer_vs_error': {
        'file_size': 5000,
        'max_length': 120,
        'inner_redundancy': [4, 6, 8],
        'outer_redundancy': [100, 200, 500],
        'error_rate_range': [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
        'Pd_Pi_Ps_ratios': [
            (0.447, 0.026, 0.527),  # 真实 nanopore
            (0.6, 0.1, 0.3),        # deletion-dominant
            (0.33, 0.33, 0.34),     # i.i.d.
        ],
        'iterations': 1000,
        'coverage': [5, 10, 30],
    },

    'scalability': {
        'lengths': [100, 500, 1000, 5000, 10000],
        'fixed_error_rate': 0.10,
        'measure': ['mean', 'std', 'p95'],
    },

    'homopolymer_ablation': {
        'homopolymer_max': [2, 3, 4, float('inf')],  # 约束阈值 sweep
        'measure': ['FER', 'encoding_rate'],
    },

    'real_data': {
        'dataset': 'NIST DNA Storage Benchmark',
        'source': 'https://www.nist.gov/programs-projects/dna-data-storage-benchmark',
        'metrics': ['FER', 'bit_error_rate', 'decoding_time'],
    }
}
```

---

## 7. 改进五：信息论分析

### 7.1 DT Bound

```python
def DT_bound(n, k, delta, epsilon=1e-6):
    """
    Dependency Testing bound for DNA storage channel.
    Maor et al., "Concatenated Codes for the DNA Storage Channel", TIT 2023.
    """
    R = k / n
    H2 = -delta * np.log2(delta) - (1 - delta) * np.log2(1 - delta)
    I_lower = 1 - H2 - delta * np.log2(3)  # |X| = 4 for DNA
    D_KL = R * np.log2(R / I_lower) + (1 - R) * np.log2((1 - R) / (1 - I_lower))
    P_e_lower = np.exp(-n * D_KL * np.log(2))
    return P_e_lower
```

### 7.2 Half-Singleton Bound

```
R ≤ (1 − δ) / 2 + O(1 / log q)
Cheng et al., "Constructions of Linear Codes for Insertions and Deletions", ICALP 2023.
```

---

## 8. 实现路线图

```
Phase 1: 基础实现 (Month 1-2)
┌─────────────────────────────────────────────────────────────┐
│ 11.01 基础编码器：RS + CRC + 二进制→DNA 转换                │
│ 11.02 非对称漂移窗口 Viterbi（无剪枝、无 FSM）               │
│         验证：输出 == MGCP 穷举输出（正确性 baseline）        │
│ 11.03 同聚体 FSM 状态机：encode + decode                    │
│         测试：约束覆盖率、信息损失率                         │
│ 11.04 FSM-Trellis 张量积联合状态空间                        │
│ 11.05 单元测试：单错、多错、burst、homopolymer             │
└─────────────────────────────────────────────────────────────┘
                              ↓
Phase 2: 剪枝 + Marker (Month 3)
┌─────────────────────────────────────────────────────────────┐
│ 11.06 CRC early termination + metric threshold pruning      │
│ 11.07 活跃状态数分布测量 vs. MGCP pattern 数               │
│ 11.08 分层 Marker 插入（弱 + 强）                          │
│ 11.09 滑动窗口 Viterbi + strong marker fallback             │
│ 11.10 内存使用 profiling：短序列 vs. 长序列                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
Phase 3: 软判决 + 外码 (Month 4)
┌─────────────────────────────────────────────────────────────┐
│ 11.11 Basecaller LLR 提取与整合                            │
│ 11.12 Reliability-weighted consensus                       │
│ 11.13 DNA error prediction → erasure conversion             │
│ 11.14 GMD + OSD outer RS decoding                          │
│ 11.15 Extrinsic Information Transfer（内-外迭代）          │
│ 11.16 端到端集成：FER vs. error rate (MGCP vs. Asym-MGC)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
Phase 4: 性能评估 (Month 5)
┌─────────────────────────────────────────────────────────────┐
│ 11.17 Memory-k 信道模型 + BCJR AIR                         │
│ 11.18 Homopolymer 约束阈值 sweep (≤2, ≤3, ≤4)             │
│ 11.19 基准对比: HEDGES, DNA Fountain, DNA StairLoop        │
│ 11.20 真实 nanopore 数据测试 (NIST benchmark)               │
│ 11.21 DT bound + half-Singleton bound 对比图               │
│ 11.22 理论贡献整理: 漂移概率 bound + FSM 约束分析          │
└─────────────────────────────────────────────────────────────┘
                              ↓
Phase 5: 论文写作 (Month 6)
┌─────────────────────────────────────────────────────────────┐
│ 11.23 初稿: 系统模型 + Asym-MGC 算法                       │
│ 11.24 实验: FER 曲线 + 基准对比 + 真实数据                 │
│ 11.25 理论: 漂移 bound + 复杂度分析 + FSM 约束性质        │
│ 11.26 导师修改 → 投稿 (TCOM)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 预期成果

### 9.1 性能目标

| 指标 | MGCP | ATC-MGC (v1.0) | Asym-MGC (v2.0) |
|------|------|----------------|----------------|
| FER (10% error, 5KB) | ~10⁻³ | ~10⁻⁴ | **< 10⁻⁶** |
| 解码复杂度 | 组合级 | 多项式 O(N) | **多项式 O(1) 内存** |
| 最大序列长度 | ~200 bp | ~2 kb | **10 kb+** |
| 软信息 | 无 | 部分 | **完整 LLR + extrinsic IT** |
| FSM 约束 | 无 | 后置过滤 | **联合状态空间** |
| Homopolymer 错误预防 | 无 | 无 | **~47-50%** |
| DT bound gap | 未测量 | < 15% | **< 10%** |

### 9.2 论文贡献声明（Target: IEEE TCOM）

1. **算法贡献 I（Asymmetric Trellis）：** 提出了针对纳米孔 deletion-domination 特性的非对称漂移窗口设计，证明了在删除主导信道下正确路径被非对称窗口捕获的概率下界（漂移 Hoeffding bound），并将 Viterbi 解码的状态空间减少约 33-47%。

2. **算法贡献 II（FSM-Trellis Joint Decoding）：** 将同聚体约束（homopolymer run-length ≤ 3）显式建模为有限状态机，与 Viterbi 漂移状态做张量积构建联合状态空间 `(Δ, S_hp)`。在解码过程中通过 FSM 转移矩阵实现硬剪枝，将约 47-50% 的同聚体错误从被动纠错升级为主动预防。

3. **算法贡献 III（Hierarchical Sliding-Window）：** 设计了分层 Marker（弱 Marker 局部同步 + 强 Marker 全局截断）和滑动窗口 Viterbi 解码算法，将长序列解码的内存复杂度从 O(N) 降至 O(1)，使 Asym-MGC 可 scale 到 10kb+ 的纳米孔长读长。

4. **系统贡献（Extrinsic IT）：** 设计了内码-外码双向软信息传递架构：外码层综合 Phred 权重、DNA error prediction 生成 erasure map，GMD/OSD 解码产生 extrinsic LLR 反馈回内码，实现迭代 refinement，将端到端 FER 压至 10⁻⁶ 量级。

5. **实验贡献：** 在模拟 memory-k nanopore 信道和 NIST 真实数据集上验证了 Asym-MGC 的性能，相较于 HEDGES、DNA Fountain、DNA StairLoop 等 SOTA 系统在 FER vs. redundancy trade-off 和 scalability 上达到更优或 competitive 的性能。

---

## 10. 文件结构规划

```
asym_mgc/
├── inner/
│   ├── __init__.py
│   ├── encode.py               # RS + CRC + FSM constrained encoding
│   ├── markers.py              # Hierarchical marker insertion
│   ├── decode.py               # Asym-MGC main decoder
│   ├── trellis.py              # Viterbi DP, state machine, transitions
│   ├── asymmetric_window.py    # Asymmetric drift window definition
│   ├── fsm_joint.py            # FSM-Trellis tensor product
│   ├── pruning.py             # CRC + metric + Top-K pruning
│   ├── sliding_window.py       # Sliding-window Viterbi
│   ├── soft_branch_metric.py   # LLR computation, homopolymer-aware
│   └── marker_fallback.py      # Strong marker detection + fallback
│
├── outer/
│   ├── __init__.py
│   ├── soft_consensus.py       # Reliability-weighted consensus
│   ├── gmd_osd.py             # GMD + OSD soft-decision RS
│   ├── error_predictor.py     # DNA-specific error prediction
│   └── extrinsic_it.py        # Extrinsic Information Transfer
│
├── channel/
│   ├── __init__.py
│   ├── memory_k_nanopore.py   # Memory-k nanopore channel
│   ├── bcjr_air.py             # BCJR AIR computation
│   └── fit_params.py           # Fit from real nanopore data
│
├── benchmarks/
│   ├── experiment_framework.py # Unified experiment runner
│   ├── compare_hedges.py
│   ├── compare_fountain.py
│   ├── compare_stairloop.py
│   ├── dt_bound.py
│   └── real_data_eval.py
│
├── analysis/
│   ├── complexity.py
│   ├── drift_bound.py          # Theorem 1 proof implementation
│   └── plotting.py
│
└── tests/
    ├── test_asymmetric_window.py
    ├── test_fsm_joint.py
    ├── test_sliding_window.py
    ├── test_marker_fallback.py
    ├── test_soft_outer.py
    └── test_memory_channel.py
```

---

## 11. 关键创新汇总对照

| 创新点 | 来源 | 论文价值 | 实施难度 |
|--------|------|---------|---------|
| 非对称漂移窗口 + Hoeffding bound | 用户提案 + v1.0 | 高（理论） | 低 |
| FSM-Trellis 张量积联合状态空间 | 用户提案 + v1.0 | 高（算法+理论） | 中 |
| 分层 Marker + 滑窗 Viterbi | 用户提案（原创性强） | **极高** | 中高 |
| Extrinsic Information Transfer | v1.0 + 新增 | 高（系统） | 中 |
| Memory-k 信道 + BCJR AIR | v1.0 | 中（配套） | 低 |
| DNA error prediction → erasure | v1.0 (Derrick) | 中（配套） | 低 |
| DT bound + SOTA 基准对比 | v1.0 | 中（审稿必需） | 低 |

---

## 12. v2.1 变更日志（2026-05-28）

### 12.1 Bug 修复

| Bug ID | 描述 | 修复文件 |
|--------|------|---------|
| CRITICAL-1 | CRC 计算在编码端（MSB/符号级）和解码端（LSB/碱基级）完全不兼容，导致零错误率解码失败 | `utils/crc_utils.py`（新建）, `encode.py`, `fsm_joint.py`, `bcjr.py` |
| CRITICAL-2 | BCJR 的 `_crc_update` 逻辑完全错误，第一行 XOR 一个移位后的值然后循环里又移位，两次叠加 | `bcjr.py` |
| CRITICAL-3 | BCJR 删除转移中 `prev_base` 未更新为被删除的碱基 `b`，破坏同聚体追踪 | `bcjr.py` |
| CRITICAL-4 | 回溯链查找在 `list_k>1` 时查 `_best_per_state` 返回到达该状态的最优路径，而非当前路径，导致跳线 | `fsm_joint.py` |

### 12.2 架构修订

| 修订 | 描述 |
|------|------|
| 同聚体约束 max_run=4 | 原 max_run=3，改为 4 以减少替换频率 |
| 确定性替换 | 移除 `substitution_map` 和随机 `rng.integers()`，改用 `out_pos % len(candidates)` 确定性选择 |
| GC 含量平衡 | 新增 `apply_gc_balance_idempotent`，比特级翻转修正 GC% ∈ [0.40, 0.60]，可逆 |
| 移除弱标记 | 删除所有 `AC` 弱标记，简化标记系统，消除虚假删除问题 |
| 共识前置 | 新增 `pre_decode_consensus` 和 `_form_top3_consensus`，将共识层从解码之后移至解码之前 |
| transmit 返回 3-tuple | `transmit` 返回 `(dna, quality, strand_index)`，用于共识前置分组 |
| encoder 移除 seed 参数 | 同聚体约束确定性，无需随机 seed |

### 12.3 关键参数变化

| 参数 | v2.0 | v2.1 |
|------|------|------|
| `max_run` | 3 | **4** |
| `seed` | 42 | **删除** |
| `substitution_map` | 有 | **无** |
| 弱标记 | 有 | **无** |
| 强标记间隔 | ~150bp | ~192bp |
| 共识位置 | 解码后 | **解码前** |
| CRC 计算 | 不兼容 | **统一（crc8_batch）** |

### 12.4 新增文件

- `asym_mgc/utils/__init__.py`
- `asym_mgc/utils/crc_utils.py`：共享 CRC-8 计算模块

### 12.5 待处理（v2.1 之后）

| 编号 | 任务 | 说明 |
|------|------|------|
| TODO-1 | 修复 CRITICAL-5: MaxLogMAP 死代码 | `MaxLogMAPDecoder.decode()` 未调用前向后向算法 |
| TODO-2 | 修复 HIGH-6: 外部 RS 码被跳过 | 外层 RS 应在内层解码后应用 |
| TODO-3 | 修复 HIGH-7: 自适应漂移不同步 | BCJR 和 MaxLogMAP 的 D_max/I_max 未更新 |
| TODO-4 | 高错误率下共识前置的适用性验证 | Pd=0.5 时共识前置效果待测 |
| TODO-5 | 长期：JCAD 联合优化 | 共识和解码联合优化（长期目标，详见 IMPROVEMENT_PLAN.md §13） |

详细变更见 `ARCHITECTURE_REVISION_v2_1.md`。

---

## 13. 长期方向：JCAD（Joint Consensus and Decoding）

> 本节描述共识前置架构的长期演进方向，属于当前版本的规划目标，而非已实现功能。

### 13.1 短期策略的局限

当前 v2.1 采用级联架构：

```
raw reads → [pre-decode consensus] → Viterbi + BCJR → RS check → message
                   ↑ 观测改善                   ↑ 硬决策 + 错误检测
```

级联架构的局限：
1. **consensus 阶段的错误会传播到 decode 阶段**：如果 consensus 形成了错误的 indel 对齐，解码器只能被动接受
2. **decode 无法反馈给 consensus**：Viterbi 发现 consensus 质量低时，无法修正 consensus
3. **两阶段独立优化**：consensus 和 decode 分别优化，无法利用解码器的 posterior 信息来改善 consensus

### 13.2 JCAD 的核心思想

JCAD 将 consensus 和 decoding 在同一个概率模型中联合优化：

```
最大化后验：P(X | Y_1, ..., Y_K) ∝ P(Y_1 | X) · P(Y_2 | X) · ... · P(Y_K | X)

其中：
  X  = consensus 序列（隐变量）
  Y_k = 第 k 条 read 的观测序列
```

目标：找到一个 X 使得所有 read 的似然乘积最大。这个 X 同时就是 consensus 和最终的解码结果。

### 13.3 关键参考文献

| 论文 | 方法 | 适用性 |
|------|------|--------|
| HEDGES (Schwartz et al.) | hash-based, sync-aware decoding | 开源，适合作为 baseline |
| Yan Court et al. | bit-flipping MCMC for joint consensus | 高 coverage 时效果优 |
| DNA Fountain (Grass et al.) | 喷泉码思路 | 冗余度高，适合长期存储 |
| DNA StairLoop (Chen et al.) | 嵌套循环码 | 硬件友好 |

### 13.4 实现路径

```
Phase 1 (当前，级联): consensus → decode
Phase 2 (短期): decode → feedback to consensus (迭代 refine)
Phase 3 (长期): JCAD — E-step 和 M-step 联合优化
```

**Phase 2 迭代 refine 示例：**

```
1. 初始 consensus C_0（当前 pre-decode consensus）
2. 内层解码 → 得到 posterior probabilities P_k(X_t | Y_k)
3. E-step: 计算每个 read 对每个 consensus 位置的后验
4. M-step: 用后验加权的 consensus 替换 C_0
5. 重复 2-4 直到收敛或达到最大迭代
```

这本质上是一个 **turbo-like iteration**（类似 extrinsic IT），已经在 `outer_soft.py` 的 `extrinsic_information_transfer` 中有雏形。区别在于迭代应该在内层（Viterbi/BCJR）和 consensus 之间进行，而非在 consensus 和外层 RS 之间。

### 13.5 与现有代码的衔接

当前已存在的相关模块：

- `outer_soft.py::extrinsic_information_transfer`：外层的 extrinsic IT 框架
- `inner/posterior_guided_align.py`：BCJR posterior 引导的对齐
- `inner/bcjr.py::FSMBCJRDecoder`：提供 per-position posterior

Phase 2 的实现可以在现有 `extrinsic_information_transfer` 基础上改造：将其从"内层→外层"迭代扩展为"内层↔consensus"迭代。核心改动是 `consensus.py` 需要能够接受 posterior-weighted reads 并重新对齐。

---

## 14. v2.0 Legacy（已废弃内容）

> 以下内容已在 v2.1 中被废弃，保留供历史参考。

### 14.1 两级标记系统（Hierarchical Markers）

v2.0 原采用两级标记：

| ~~类型~~ | ~~序列~~ | ~~间隔~~ | ~~作用~~ |
| ~~------~~ | ~~------~~ | ~~------~~ | ~~------~~ |
| ~~弱 Marker~~ | ~~'AC'~~ | ~~每 4 blocks~~ | ~~局部同步：允许 Viterbi 在局部窗口内修正小范围漂移~~ |
| **强 Marker** | `'TACGTA'` | 每 32 blocks | 全局截断：将长序列分成独立窗口，重置漂移状态 |

v2.1 移除了弱标记，理由：
1. `AC` 在随机 DNA 中出现概率较高（`4^{-2} = 6.25%`），虚假删除问题严重
2. 弱标记提供的"局部同步"收益不如预期
3. 仅靠强标记已足够控制漂移

### 14.2 非确定性同聚体替换

v2.0 的 `_apply_homopolymer_constraint` 使用 `rng.integers()` 随机选择替换碱基，并通过 `substitution_map` 记录位置。v2.1 改用 `out_pos % len(candidates)` 确定性选择，无需 `substitution_map`。

### 14.3 解码后共识（Post-Decode Consensus）

v2.0 的解码 pipeline 为：

```
raw reads → [Viterbi decode × N] → [consensus after decode] → message
```

v2.1 改为共识前置（Pre-Decode Consensus）：

```
raw reads → [pre-decode consensus] → [Viterbi decode] → message
```

理由见 §3.3 和 §13。

### 14.4 CRC LSB/碱基层计算

v2.0 解码器对每个碱基（2 bits）独立计算 CRC，与编码端按 RS 符号（8 bits）整体计算不兼容。v2.1 统一使用 `crc8_batch`（MSB-first per symbol）。

---

## 15. 下一步计划

### 当前完成度

| 模块 | 状态 | 说明 |
|------|------|------|
| CRC 统一 | ✅ 已完成 | crc8_batch, 零错误率验证通过 |
| 同聚体约束(≤4) | ✅ 已完成 | 确定性替换, 无 substitution_map |
| GC 平衡 | ✅ 已完成 | apply_gc_balance_idempotent |
| 移除弱标记 | ✅ 已完成 | 仅强标记, _strip_markers 简化 |
| 共识前置 | ✅ 已完成 | pre-decode consensus, strand_index 分组 |
| FSM Viterbi (HPState≤4) | ✅ 已完成 | QUAD=4, prev_pm 回溯链修复 |
| BCJR 删除转移 | ✅ 已完成 | prev_base=b |
| 测试套件 | ✅ 已完成 | 175 tests passing |

### 高优先级待办

| 编号 | 任务 | 状态 | 说明 |
|------|------|------|------|
| **TODO-H1** | 零错误率 FER benchmark | ⏳ 待验证 | 在 Pd=0, Pi=0, Ps=0 下验证 FER=0 |
| **TODO-H2** | CRC 剪枝策略重新启用 | ⚠️ 发现根本问题 | 解码器无expected CRC值，无法blind prune。需等HIGH-6修复后传递expected CRC |
| **TODO-H3** | 高错误率下的共识前置效果验证 | ⏳ 待验证 | Pd=0.5 时 pre-decode consensus 是否有效 |
| **TODO-H4** | CRITICAL-5: MaxLogMAP 死代码 | ✅ 已修复 | `decode()` 现在调用 forward+backward+posteriors；修复了观测符号 ASCII 编码 bug |
| **TODO-H5** | HIGH-6: 外部 RS 码被跳过 | ⏳ 待处理 | 外层 RS 应在内层解码后应用 |

### 中优先级待办

| 编号 | 任务 | 状态 |
|------|------|------|
| **TODO-M1** | HIGH-7: 自适应漂移不同步 | ⏳ 待处理 |
| **TODO-M2** | HIGH-9: LVA 返回 None 保护 | ⏳ 待处理 |
| **TODO-M3** | MEDIUM-10: 重试解码器丢失配置 | ⏳ 待处理 |
| **TODO-M4** | MEDIUM-11: MaxLogMAP 对数概率未同步 | ⏳ 待处理 |
| **TODO-M5** | MEDIUM-13: _chain 污染 | ⏳ 待处理 |
| **TODO-M6** | MEDIUM-14: Demo 缺失导入 | ⏳ 待处理 |

### 长期路线图

| 阶段 | 任务 | 说明 |
|------|------|------|
| **Phase 2** | 迭代 refine（consensus ↔ decode） | 在 Viterbi/BCJR 和 consensus 之间迭代 |
| **Phase 3** | JCAD 联合优化 | E-step/M-step 联合优化，参考 HEDGES/Yan Court |
| **Phase 4** | 真实纳米孔数据验证 | Squigulator + Dorado 端到端验证 |

---

*文档版本: v2.1 | 2026-05-28*
