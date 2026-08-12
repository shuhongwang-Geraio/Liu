# 论文核心章节草稿（方法 + 实验）

> 来源: `CausalCIT_ablation/method_assessment.md` (2026-08-08) + 调研 `04_baseline_literature/`。
> ⚠️ **数字占位**: 本稿所有性能数字来自 P0-1 重跑**之前**的协议
> (spawn seed bug 已修复但主表未重跑)。投稿前必须用 `output_large_v3` (P0-1) 的结果
> 逐项替换，并重新生成 bootstrap CI 图与热图。
> 论文正式采用变体名 `full_v2_fixed`（修复 batch 依赖 bug，性能与 full_v2 持平）。

---

## 1. 方法章节

### 1.1 问题设定

多元时间序列预测 (MTSF): 给定回看窗口 $X_{t-L+1:t} \in \mathbb{R}^{L \times C}$
($C$ 个通道，$L$ 个时间步)，预测未来 $X_{t+1:t+H} \in \mathbb{R}^{H \times C}$。
训练与测试分布可能不同（分布漂移），目标为 OOD 泛化。

### 1.2 核心观察：相关 ≠ 因果

现有通道交互方法（iTransformer、Crossformer、SOFTS）以**相关性强度**驱动通道混合：
通道注意力/路由权重由训练分布的统计相关决定。但在分布漂移下，训练集中高相关的
通道对（虚假相关，如传感器共置、季节性共变）可能在测试中消失甚至反转，导致
**按相关性混合的通道交互把错误信息注入预测**。

### 1.3 CausalCIT：以跨环境稳定性作为通道交互准入判据

核心思想：**只有跨环境稳定的通道依赖（暗示因果）才应被交互**。

实现（以 PatchTST 为 backbone）：
1. **环境切分**：将训练样本按协变量/时间切分为 $K$ 个环境 (n_envs=4)。
2. **稳定性度量**：对每个通道对 $(i,j)$，用 HSIC 度量其在各环境下的相关性强度，
   计算跨环境的稳定性信号 $s_{ij}$（相关性在不同环境下保持一致者 $s_{ij}$ 高）。
3. **门控**：$g_{ij} = \sigma(\alpha \cdot s_{ij} - \tau)$ 作为通道混合权重，
   仅稳定通道对获得高权重；不稳定（虚假）通道对被抑制。
4. **通道交互**：$Z_i = \sum_j g_{ij} V_j$（$V_j$ 为通道 $j$ 的 patch 表示），
   叠加在 PatchTST 的 patch embedding 上，预测头不变。

关键设计点：
- **门控由稳定性信号而非相关性强度驱动** —— 这是与所有相关性驱动方法的本质区别。
- **环境划分**（EnvSplit）是机制核心：消融 `no_env`（全局 HSIC、不划分环境）
  在 traffic 两端 horizon 都显著差于 full_v2_fixed（p<0.05），证明稳定性度量依赖环境切分。
- **先验辅助**：prior_weight=0.05 的通道相关先验作为弱先验初始化，
  但 `gate_prior_only`（剥离稳定性信号、只留先验）门控 100% 坍缩成常数
  → 先验本身不驱动性能，驱动的是稳定性门控机制。

### 1.4 变体定义（消融对照）

| 变体 | 定义 | 用途 |
|------|------|------|
| patchtst | 原始 CI 基线 | 主基线 |
| full_v2_fixed | CausalCIT 完整版（本论文正式采用，批不变） | 主模型 |
| capacity_match | 同参数规模的纯学习通道注意力，无因果稳定性 | 答"增益是否仅来自容量" |
| gate_prior_only | 同门控结构，但剥离稳定性/HSIC 信号 | 答"稳定性机制 vs 先验" |
| no_env | 同结构，但全局 HSIC（不划分环境） | 答"环境划分是否必要" |
| dlinear / itransformer | 新增外部 baseline | 与相关性驱动方法直接对标 |

---

## 2. 实验章节

### 2.1 数据集与协议

- 数据集: ETTh1 (7ch)、ETTm1 (7ch)、Weather (21ch)、Exchange (8ch)、
  Electricity (321ch)、Traffic (862ch)、ILI (7变量)。
- 协议: 6 变体 × 8 seed，seed 配对 Wilcoxon 符号秩检验 + 同 (dataset, horizon)
  族内 Holm 校正；窗口 96；horizons 96/192（traffic、electricity、exchange）、
  96/192/336（weather、ETT）、24/48（ILI）。
- 训练: 30–50 epochs，early stopping patience 8，Adam，lr 1e-3（余弦）。
- ⚠️ 每个 job 在独立 spawn 子进程内重设 seed（P0-1 修复，保证同 seed 配对成立）。

### 2.2 主结果（P0-1 重跑后替换数字）

full_v2_fixed vs PatchTST，提升 % = (patchtst_mse − variant_mse)/patchtst_mse × 100：

| 数据集 | 通道数 | pl96 | pl192 | pl336 | 平均 | Holm 显著 |
|--------|--------|------|-------|-------|------|-----------|
| traffic | 862 | +8.4% | +7.4% | — | +7.9% | ✓ |
| electricity | 321 | +5.8% | +2.1% | — | +3.9% | ✓ |
| ettm1 | 7 | +3.2% | −0.8% | −1.1% | +0.4% | 部分 |
| exchange | 8 | +0.6% | +2.5% | — | +1.5% | pl192 ✓ |
| weather | 21 | +1.6% | −0.7% | −2.1% | −0.4% | 混合 |
| etth1 | 7 | −1.2% | −1.3% | −0.3% | −0.9% | 否（更差） |
| ili | 7 | −5.3% | +1.2% | — | −2.1% | 否（高方差） |

**结论表述（论文正式措辞，谨慎）**:
"CausalCIT 在通道数高、依赖结构强的数据集上（traffic +7.9%、electricity +3.9%，
Holm p<0.05）稳定优于 PatchTST；在低维数据集（ETTh1、ILI）上不占优，符合方法假设
——当通道间因果依赖稀疏时门控退化为噪声。因此 CausalCIT 是场景依赖的有效改进，
而非通用 SOTA 提升。"

### 2.3 关键消融（机制 vs 容量）

traffic 上 full_v2_fixed vs 对照（Holm p）:
| 对照 | pl96 | pl192 | 解读 |
|------|------|-------|------|
| capacity_match | +0.90% (ns) | +4.24%* | 部分 horizon 显著 → 容量匹配不够 |
| gate_prior_only | +0.19% (ns) | +2.73%* | 先验不够 → 需稳定性信号 |
| no_env | +5.14%* | +4.18%* | 环境划分必要 → 稳定性机制在贡献 |

### 2.4 门控行为诊断（回应评审 re2）

traffic 全规模，`off_std`（非对角离散度）/ `collapsed`（是否坍缩成常数）/
`batch_dep_score`（同样本换 batch 同伴的门控变化，越接近 0 越好）:

| 变体 | off_std | collapsed | batch_dep | 判定 |
|------|---------|-----------|-----------|------|
| capacity_match | 0.005 | 0% | 0.0000 | 无门控基准 |
| full_v2 | 0.048 | 0% | 0.34/0.42 | ⚠️ 存在批依赖 bug |
| full_v2_fixed | 0.062 | 0% | 0.0000 | ✅ 批不变 |
| gate_prior_only | 0.000 | 100% | 0.0000 | ⚠️ 坍缩 |
| no_env | 0.20–0.36 | 12.5% | 0.01–0.03 | 未坍缩 |

三条证据链：
1. `full_v2` 的 batch 依赖 bug（0.34）在 `full_v2_fixed` 中降至 0.0000 → 批不变性修复。
2. `gate_prior_only` 100% 坍缩 → 先验不驱动性能，稳定性门控机制驱动。
3. `full_v2_fixed` 既无 bug 又不坍缩 → 门控确实在做跨环境稳定性聚集。

诚实交代：修复 bug 后性能与 full_v2 持平（±0.3%，ns）——bug 对 MSE 影响小，
但必须采用修复版以通过批不变性审查。

### 2.5 与相关性驱动方法的对标（P1-2 待跑）

iTransformer / DLinear 已接入实验脚本（`--variants itransformer dlinear`），
按与主表相同协议跑 6 数据集 × 8 seed。预期:
- traffic/electricity（漂移剧烈）：CausalCIT ≥ iTransformer（稳定性优于相关性门控）；
- 低维稳定场景：iTransformer 可能更优（诚实呈现，不回避）。

### 2.6 OOD 边界（诚实处理）

- `syn_ood`（机制测试：虚假通道强度跨环境反转）上 full_v2 为 **−1.21%** 显著变差，
  机制测试尚未通过 → 论文**不能**宣称"因果门控带来 OOD 鲁棒性"，
  只报告场景依赖的有效改进。待 P2 排查（spurious_strengths 配置 or 容量）。

### 2.7 Limitations（必备段落）

1. 低维 / 长 horizon 不占优（方法假设的预期行为，非缺陷）。
2. Adapformer / CSformer 无公开代码，仅机制与定性对照。
3. OOD 机制测试（syn_ood）尚未通过，OOD 鲁棒性宣称受限。
4. 高维门控矩阵（traffic 862×862）的聚类热图待 dump（P1 可视化）。

### 2.8 训练前适用性判据（方案 1；回应"方法太窄"的正面回答）

目标：把"范围窄"从 B 类（无解释的窄）变成 A 类（有原则、可事先预测的窄）。
只从**原始数据**（不训练）计算统计量，与实测增益做对应，若能预测增益正负号，则：

> "我们的方法只在特定条件有效 —— 而我们给出了训练前即可计算的判据，能提前告诉你它是否有效。"

统计量（脚本 `compute_pre_train_stats.py`，0 GPU，2026-08-12 已跑 4 数据集）：

| 数据集 | C | 依赖密度 avg\|corr\| | 语义环境信息量 (season/随机) | 稳定通道对占比 (season) |
|--------|---|--------------------|------------------------------|-------------------------|
| ETTh1 | 7 | 0.222 | 10.6× | 0.238 |
| ETTm1 | 7 | 0.224 | 25.7× | 0.238 |
| weather | 21 | 0.297 | 4.2× | 0.157 |
| exchange | 8 | 0.513 | 3.7× | 0.714 |

初步观察（2026-08-12 P0-1 部分结果更新；**待主表跑完后用完整新增益做严格对应**）：
- **修复版协议下 (P0-1 快照, 8-seed)**：weather/electricity/traffic 的 `full_v2_fixed` 全面翻正
  —— weather pl96 +5.2% / pl192 +5.0% / pl336 +1.0%，electricity pl96 +3.3% / pl192 +5.3%，
  traffic pl96 +8.0% / pl192 +12.1%（vs 已发表 PatchTST；待主表内高维 patchtst 对照确认）。
  **旧协议下 weather 的负收益 (-0.4%) 已翻正** → 适用性判据必须与修复版协议配对，旧对应作废。
- 单因子"稳定通道对占比"不成立：weather 稳定占比最低 (0.157) 但修复版下 gains 最高档 (+5.2%)；
  exchange 稳定占比最高 (0.714) 旧协议仅 +1.5%。判据需多因子（依赖密度 × 语义信息量 × 稳定性）
  或按 horizon 分别对应，P0-1 完整数据落地后重做。
- 语义环境信息量 (4–14×) 刻画"稳定性信号能否被门控利用"的前提，与"高维有效"互补。

判据设计：`稳定通道对占比 × 依赖密度` 等组合量 vs 各数据集平均增益的符号对应，
7 数据点构成启发性证据（诚实边界：样本量小，需 P2 补高维 regime 数据点）。

---



## 3. 图表清单（对应产物）

| 图/表 | 脚本/数据 | 状态 |
|-------|-----------|------|
| 主表 (2.2) | run_large summarize | P0-1 重跑后重新生成 |
| 提升率 bootstrap CI 误差棒图 | plot_bootstrap_ci.py | 已生成（旧数据），重跑后刷新 |
| 消融表 (2.3) | output_falsifiable | 重跑后刷新 |
| 门控诊断表 (2.4) | gate_diagnostics.json | ✅ 已有 |
| 门控边箱线图 | plot_gate_edge_boxplot.py | 已生成（syn_ood 示例），正式需收敛模型 |
| 高维门控聚类热图 | plot_gate_heatmaps.py + dump_gates_eval.py | 待 GPU (P0-1 + --dump_gates) |
