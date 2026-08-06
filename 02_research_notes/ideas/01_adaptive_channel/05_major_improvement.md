# CausalCIT 重大改进方案：三根因诊断与修复路线

> 性质标记：**重大改进（MAJOR IMPROVEMENT）** —— 本文件不是新增对照变体，而是修复已确认的实现/设计缺陷，使方法首次按设计意图运行。
> 归档日期：2026-08-06
> 来源：针对 `electricity/weather 效果差、归因不清` 问题的代码级诊断（审稿链第三轮 `reviewer_critique_re2.md` 之后的延续动作）。
> 前置文档：`01_proposal.md`（方法提案）、`../reviewer_critique_re2.md`（三审）、`../../03_experiments/CausalCIT/CausalCIT_ablation/诊断报告_3seed消融结果分析.md`（消融诊断）。
> 关键结论：**三个根因同时解释了「为什么归因说不清」和「为什么偏偏 electricity/weather 差」；且效果排序的反序可归因于 `d_model`，而非通道数——"维度单调律"很可能是混淆变量。**

---

## 一、背景

审稿链三轮后，核心争论点收敛为：full_v2 相对 `capacity_match`/`gate_prior_only` 的增益证据薄弱、方向不稳定，且 electricity/weather 上为负。本文件从代码出发定位到三个可复现的根因，并给出（a）低成本归因诊断方案 和（b）对应三项修复，均无需新增变体。

## 二、三个根因（含代码定位）

### 根因 1：RFF 核带宽硬编码 σ=1，从未适配 d_model

- 位置：`CausalCIT_demo/models/causal_channel.py` — `RFFKernel.__init__` 默认 `sigma=1.0`；`CausalStabilityGate` 构造 `RFFKernel(d_model, rff_dim)` 未传 sigma（永远用默认 1.0）。
- 机理：`proj = x @ W` 的尺度 ≈ `||x||` ≈ √d_model（LayerNorm 后）。d_model=64 时 proj 标准差 ≈ 8，`cos(proj)` 剧烈震荡 → RFF 特征退化为伪随机向量 → **HSIC 估计为噪声**。核方法带宽不做 median heuristic 基本必废。
- 旁证（效果排序 = d_model 反序）：

| 数据集 | d_model | n_vars | full_v2 平均提升 |
|---|---|---|---|
| traffic | 16 | 862 | +7.88% |
| electricity | 32 | 321 | +3.33% |
| ettm1 | 32 | 7 | +1.00% |
| etth1 | 32 | 7 | −0.65% |
| weather | 64 | 21 | −0.58% |

- 推论：配置里通道越多 `d_model` 越小（为省显存，`run_large.py:105-109`），"高维有效"很可能是"`d_model` 小 → 核尚未失效"的**混淆变量**。这也自洽解释了审稿人抓住的 exchange 打脸（exchange 为 8 通道但 d_model=32，按 d_model 解释完全一致）。

### 根因 2：未归一化 HSIC 淹没了稳定性信号，门控退化为相关性强度门控

- 位置：`causal_channel.py:177` — `stability = hsic_mean / (1.0 + cv + self.stability_bias.abs())`。
- 机理：分母 `(1+cv+|bias|)` 实际只在约 [1,3] 内变化（最多 3 倍），分子 `hsic_mean` 为**未归一化** HSIC，跨通道对可差 1–2 个数量级。相乘后门控 ≈ `hsic_mean` 的单调函数。
- 后果：方法实质退化为 Adapformer 式"相关性强度门控"——正是 full_v2 打不过 `capacity_match` 的直接原因；且未归一化 HSIC 被**方差大的通道**主导而非依赖强的通道对，weather（21 通道语义高度异质）受害最深。

### 根因 3：「环境」= 窗口内均分，非真实机制变化

- 位置：`causal_channel.py:97-102`（`env_size = patch_num // n_envs`）+ `run_large.py:143-145`（`n_envs=4`）。
- 参数推演：seq_len=96（pl≤192）+ patch_len=16/stride=8/padding=end → patch_num=12，n_envs=4 → **每环境仅 3 个 patch ≈ 24 时间步**。
- 物理时间对照：

| 数据集 | 采样粒度 | 96步窗口 | 单"环境" | 环境间有真实机制差异吗 |
|---|---|---|---|---|
| weather | 10 分钟 | 16 小时 | 4 小时 | ❌ 几乎为零，纯日内平滑 |
| ettm1 | 15 分钟 | 24 小时 | 6 小时 | ❌ 弱 |
| electricity | 1 小时 | 4 天 | 1 天 | △ 仅偶尔跨周末 |
| traffic | 1 小时 | 4 天 | 1 天 | ✅ 862 路段传感器，工作日/周末通道关系剧变 |

- 推论：只有 traffic 的环境划分**偶然**捕捉到真实机制变化；其余数据集的"跨环境稳定性"测的是近同分布段间的抖动 = 噪声。weather 同时踩中根因 1（d_model 最大）与根因 3（时间跨度最短），负收益可预期。

### 顺带发现：被低估的正面信号（full_v2_fixed）

`output_falsifiable/large_scale_report.md` 中 `full_v2_fixed`（唯一区别 `running_stats=True`）全面优于 `full_v2`：
- pl96：+9.36% vs +8.42%，std **0.0060 vs 0.0117（方差减半）**
- pl192：+8.35% vs +7.38%
- pl192 上 full_v2 vs capacity_match **+4.24%，Holm p=0.0234，显著**

→ `full_v2_fixed` 应升级为主模型，而非仅当"bug 修复对照组"。

## 三、归因诊断方案（按成本排序，前两步几乎不耗算力）

### 第 0 步：静态诊断（零训练成本，随机初始化 + 单 batch 前向）
打印四个量：
1. `proj = x @ W` 的 std（验证根因 1：若 ≫1 则核已失效）
2. `hsic_mean` 动态范围（max/min、分位数）
3. `stability` 中 `log(hsic_mean)` 与 `log(1/(1+cv))` 的方差贡献占比（验证根因 2：前者 >90% 则稳定性信号是装饰品）
4. `cv` 在真实数据上的分布（环境划分是否有信息）

在 weather(d64) / electricity(d32) / traffic(d16) 上各跑一次。若 (1)(3) 排序与效果排序一致 → 根因 1、2 定性成立。

### 第 1 步：决定性实验——解耦 d_model 与 n_vars
- 固定 n_vars=21（weather），扫 d_model ∈ {16, 32, 64}
- 固定 d_model=32，比 etth1(7ch) / weather(21ch) / electricity(321ch)

三种结局均有论文价值：
- 效果随 d_model 变小而单调变好 → "维度律"为伪相关，真实机制是核带宽（诚实且有价值的发现）
- 效果只随 n_vars 变化 → 维度律站住，可正面写
- 两者均不显著 → 明确止损信号，降级 claim

## 四、改进措施（修缺陷，非加变体）

| 改法 | 对应根因 | 做法 | 预期受益 | 成本 |
|---|---|---|---|---|
| A. median heuristic 自适应带宽 | 1 | RFF sigma 改为按 batch 内成对距离中位数在线估计（或至少 σ=√d_model），并相应归一 W | weather(d64) 最大，electricity 次之 | 一行级 |
| B. 归一化 HSIC（改 CKA） | 2 | `hsic_ij / sqrt(hsic_ii·hsic_jj)` 压到 [0,1]，再乘 `1/(1+cv)` —— 让稳定性因子真正掌话语权 | 通道异质的 weather 最大；同时成为"不是相关性门控"的直接证据 | 十行内 |
| C. 语义化环境 | 3 | 环境改用时间戳外生变量（月份/星期/工作日-周末/时段），跨 batch 用 EMA 按环境累积 HSIC（复用 full_v2_fixed 的 running EMA 机制） | electricity（季节+周末漂移）、weather（季节漂移）最大；顺带回应审稿人 §1.5「OOD 名不副实」 | 最大 |
| D. full_v2_fixed 升为主模型 | — | `running_stats=True` 设为默认 | 已有数据显示提升且方差减半 | 改默认值 |

注：C 是理论上最正统的一项——方法立论是"环境变化导致相关性失效"，环境理应由真实外生漂移定义，而非窗口内切 4 段；且按季节/节假日切分才是真正对标 FOIL/COGS 的分布漂移设定。

**执行顺序**：A → B → D → C。A+B 仅在 weather/electricity 上、用现有 8-seed 协议验证。

## 五、诚实的止损点

先做第 0 步 + A/B 在 weather/electricity 上的小规模验证。若 **A+B 修复后 weather/electricity 仍为负收益** → "跨环境稳定性"信号在这两类数据上本质不存在（非实现不到位），应：
- 将方法定位收敛为"高维、强机制切换场景（traffic 类）下的稳定性正则化通道注意力"；
- 标题去掉 causal，改 stability-regularized channel attention；
- 以诚实、范围清晰的增量贡献投稿（二线会议/专题期刊梯队）。

## 六、关联归档

- 本方案与审稿链的关系：属于 `reviewer_critique_re2.md` 第六节第 4 条"修复 entropy_weight、batch 依赖门控等代码/设计缺陷"的延续与扩展（新定位 3 处实现缺陷）。
- 判定准则：本文件四项改动性质为**修复**而非新增变体，与审稿人"停止新增变体"的指令不冲突；归档时需在报告/回应中明确此区分。
