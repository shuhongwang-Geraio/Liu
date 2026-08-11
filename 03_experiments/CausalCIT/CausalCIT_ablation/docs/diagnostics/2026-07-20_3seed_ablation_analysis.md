# CausalCIT 消融实验诊断报告（3-seed 结果分析）

> 数据来源：`output_seed42/ablation_report.md`、`output_seed123/ablation_report.md`、
> `output_seed2024/ablation_report.md`（对应 seed=42/123/2024，其余超参完全一致，见
> `run_3seeds.sh`）。
> 报告撰写时间：2026-07-20。
> 状态：**未解决 / 待进一步诊断**（历史状态，见下方更新）。本报告不下最终结论，目的是把
> 已发现的两个异常现象、定量证据、可疑代码位置和后续排查步骤记录清楚，方便后续继续排查时
> 不用从头翻数据。

---

## 更新（2026-08-04）：本报告的异常已通过 full_v2 修复并复验

> 补充说明，不改动上面的历史分析内容。

- **2026-07-22**：基于本报告第3节的假设，定位到比"先验权重"更根本的两个缺陷——(1) 通道注意力
  在池化表示上做、`expand_as` 广播回所有patch，抹平了滞后因果所需的时间信息；(2) 稳定性分数
  `1/(1+cv)` 只看跨环境变异系数、忽略依赖强度本身，导致独立通道也能拿到高门控。实施
  `temporal_mix`（时间分辨率保留的通道交互）+ `stability_v2`（依赖强度×跨环境一致性）+
  `per_channel_alpha`（逐通道融合系数、优雅回退）三项修复，得到 `full_v2`变体。详见
  `experiments/2026-07-22_multiseed/sota_v2/SOTA_report.md`：门控非对角线std从
  0.0007→0.2476，正确识别合成数据因果簇并压制独立噪声通道；weather pred_len=96 上 3 seed
  稳健 +2.83%~+5.35%。即本报告"门控未分化"“Full≈w/o EnvSplit"两个异常均已定位并修复，
  **假设A（先验主导）和假设C（门控饱和）成立，假设B（环境划分退化）部分成立**。
- **2026-08-04（评审 re2 修复后）**：`full_v2` 的 `stability_v2` 门控被指出测试时依赖 batch
  组成（用当前batch统计量而非全局统计量），已改为 `running_stats`（EMA全局统计量）修复，
  修复版记为 `full_v2_fixed`。GPU 上 traffic 数据集 8-seed 验证显示 `full_v2` vs
  `full_v2_fixed` 无显著差异（pred_len=96: p=1.0；pred_len=192: p=0.055 边缘不显著），说明
  修 batch 依赖 bug 没有把效果修没。同时在本机CPU 上用 `run_diag.py --data syn` 重新跑了一次
  单seed诊断复验，`full_v2` 门控分化能力依然完好、甚至更强（非对角线 std=0.4322，优于
  SOTA_report记录的0.2476），因果簇 Base/C1/C2/S1 互相高门控(≈0.99)、独立噪声 I1/I2 被压制到
  0.005~0.06，**说明门控识别因果结构的能力对本次bug修复是稳健的**。
- **仍未做（需GPU）**：`next_steps.md` 里的"正式多seed管线"——把 `full_v2` 接入正式的多seed
  显著性检验管线——目前用 `run_large.py`（而非 `run_ablation.py`）实现，已在 traffic 上跑出
  8-seed Holm校正结果（见 `output_falsifiable/large_scale_report.md`），但 weather/ETTh1 等
  其他数据集尚未用同样严格的多seed+Holm校正协议重跑，只有 SOTA_report 里 3-seed 的初步结果。

---

## 0. 结论摘要（TL;DR）

1. **噪声量级 ≈ 组件间差异量级**：3个seed跑出来的 vs-PatchTST 提升率，同一变体跨seed的
   标准差普遍在 ±1.5~2.5 个百分点，而4个变体之间的均值差异也在同一量级。**目前的数据不足以
   证明 Full CausalCIT 比任何一个简化变体（w/o HSIC / w/o EnvSplit / w/o Gate）更好**，
   包括不能证明它比 PatchTST 基线更好（合成数据上 Full 3-seed均值是 **-0.55%**，即比基线差）。
2. **更严重的异常**：`Full CausalCIT` 和 `w/o EnvSplit` 这两个理论上应该给出不同门控矩阵的
   变体，在全部 9 组测试点（3 seed × 3 个实验设置）上的 MSE **几乎逐点重合**，平均相对差异
   仅 **0.04%**，比它们各自与 `w/o HSIC`（均值 0.93%）、`w/o Gate`（均值 1.56%）的差异小一个
   数量级以上，且远小于同一变体跨seed的自然噪声（1~5%）。这不太可能是随机噪声，更像是两条
   代码路径在训练收敛后学到了**功能上等价**的门控（例如都收敛到了"几乎全通"的 gate≈1），
   即"环境划分"这个机制目前对结果没有产生可观测的独立贡献。
3. 本报告第 3 节给出了具体的可疑代码位置和假设，第 4 节给出了不需要重新训练、只需要在训练时
   顺手保存的诊断手段，供下一轮实验直接落地验证。

---

## 1. 原始数据汇总（3 seed × 3 实验设置，vs PatchTST 提升率）

> 提升率 = (PatchTST_MSE − 变体_MSE) / PatchTST_MSE × 100%，正数=比基线好，负数=比基线差。
> 数值直接取自 3 份 `ablation_report.md`，未做任何加工。

### 1.1 合成数据 (Synthetic, d_model=64, 50 epochs)

| 变体 | seed42 | seed123 | seed2024 | **均值** | **标准差(样本)** |
|---|---|---|---|---|---|
| w/o Gate | -0.55% | -0.88% | -1.40% | **-0.94%** | ±0.43pp |
| w/o EnvSplit | -2.28% | +0.36% | +0.20% | **-0.58%** | ±1.47pp |
| w/o HSIC | -1.14% | -0.70% | -0.12% | **-0.65%** | ±0.51pp |
| **Full CausalCIT** | -2.32% | +0.48% | +0.20% | **-0.55%** | ±1.53pp |

### 1.2 ETTh1, pred_len = 96

| 变体 | seed42 | seed123 | seed2024 | **均值** | **标准差(样本)** |
|---|---|---|---|---|---|
| w/o Gate | -2.11% | +1.02% | -1.56% | **-0.88%** | ±1.71pp |
| w/o EnvSplit | -0.97% | +0.22% | -2.43% | **-1.06%** | ±1.33pp |
| w/o HSIC | -1.44% | +0.60% | -0.97% | **-0.60%** | ±1.11pp |
| **Full CausalCIT** | -0.99% | +0.17% | -2.43% | **-1.08%** | ±1.35pp |

### 1.3 ETTh1, pred_len = 336

| 变体 | seed42 | seed123 | seed2024 | **均值** | **标准差(样本)** |
|---|---|---|---|---|---|
| w/o Gate | -0.89% | +1.75% | +3.33% | **+1.40%** | ±2.12pp |
| w/o EnvSplit | +2.35% | -1.29% | +3.39% | **+1.48%** | ±2.46pp |
| w/o HSIC | +5.18% | -1.77% | +3.26% | **+2.22%** | ±3.62pp |
| **Full CausalCIT** | +2.45% | -1.29% | +3.36% | **+1.51%** | ±2.46pp |

**观察**：4个变体在三个设置里的均值都挤在一个很窄的区间内，且区间宽度和跨seed标准差
基本同量级 → **单个变体的跨seed噪声，大到足以吞掉"变体之间"的真实差异**。目前无法从这组数据
判断任何两个变体谁更优。

---

## 2. 核心异常：`Full CausalCIT` 与 `w/o EnvSplit` 高度收敛

### 2.1 逐点 MSE 对比

| 实验设置 | seed | Full MSE | w/o EnvSplit MSE | 绝对差 | 相对差 |
|---|---|---|---|---|---|
| Synthetic | 42 | 0.496238 | 0.496067 | 0.000171 | 0.0345% |
| Synthetic | 123 | 0.483718 | 0.484320 | 0.000602 | 0.1244% |
| Synthetic | 2024 | 0.486201 | 0.486182 | 0.000019 | 0.0039% |
| pred_len=96 | 42 | 0.378064 | 0.378014 | 0.000050 | 0.0132% |
| pred_len=96 | 123 | 0.382809 | 0.382619 | 0.000190 | 0.0497% |
| pred_len=96 | 2024 | 0.383371 | 0.383369 | 0.000002 | 0.0005% |
| pred_len=336 | 42 | 0.468565 | 0.469039 | 0.000474 | 0.1011% |
| pred_len=336 | 123 | 0.484923 | 0.484884 | 0.000039 | 0.0080% |
| pred_len=336 | 2024 | 0.468979 | 0.468812 | 0.000167 | 0.0356% |

**9组测试点，相对差全部落在 0.0005% ~ 0.1244% 区间，平均 0.041%。**

### 2.2 和其他变体对比的差异量级（作为对照基线）

同样对全部9组测试点计算 `Full` 与其他变体的平均相对MSE差异：

| 对比对象 | 平均相对差异 | 相当于 EnvSplit 差异的倍数 |
|---|---|---|
| Full vs **w/o EnvSplit** | **0.041%** | 1× (基准) |
| Full vs w/o HSIC | 0.925% | ≈ 22× |
| Full vs w/o Gate | 1.556% | ≈ 38× |

同一个变体（如 w/o Gate 自身）跨3个seed的自然噪声平均在 **1~2%** 量级（见第1节标准差），
而 `Full` 与 `w/o EnvSplit` 的系统性差异只有 **0.04%**，比训练随机噪声还小一个数量级。
**如果两者的门控机制真的独立起作用，不应该在9组独立训练（不同seed、不同数据、不同pred_len）
里都稳定地比噪声还接近。** 这更像是两条代码路径最终收敛到了行为上等价的门控输出。

---

## 3. 可疑代码位置与假设

对比 `models/causal_channel.py` 里的 `CausalStabilityGate`（Full 用，环境划分版）
和 `models_ablation.py` 里的 `GlobalHSICGate`（w/o EnvSplit 用，全局版），公式结构几乎一致：

```python
# CausalStabilityGate.forward (causal_channel.py:101-122)
stability = self.compute_stability_score(x)         # 环境划分版稳定性分数, ∈ (0,1]
prior = torch.sigmoid(self.channel_prior)
stability = stability * 0.7 + prior.unsqueeze(0) * 0.3   # <- 先验项占30%
logit = self.gate_mlp(stability.unsqueeze(-1)).squeeze(-1)
gate = torch.sigmoid(logit / temp)

# GlobalHSICGate.forward (models_ablation.py:130-156)
hsic_norm = hsic_global / (hsic_global.max() + 1e-8)      # 全局HSIC版
score = hsic_norm * 0.7 + prior.unsqueeze(0) * 0.3        # <- 同样30%先验
gate = self.gate_mlp(score.unsqueeze(-1)).squeeze(-1)
```

**假设 A：`channel_prior`（可学习的、与输入无关的先验矩阵）在训练中占据主导，
把"环境划分算出的稳定性分数"和"全局HSIC分数"这两路本该不同的信号都压制/淹没了。**
两个模块结构几乎一样（都是 `Linear(1,16)+GELU+Linear(16,1)` 的小MLP + 30%先验混合），
如果输入的稳定性分数本身对最终gate的影响很小，两条分支自然会训出接近的门控矩阵和接近的下游MSE。

**假设 B：环境划分的 `n_envs=4` 在当前 `patch_num` 下切分出的每个环境样本量太小，
导致"跨环境标准差"这个信号本身就很弱/很嘈杂，等效于退化成了全局统计。**
以当前配置核算：`seq_len=96, patch_len=16, stride=8, padding_patch='end'` →
`patch_num = (96-16)/8 + 1 + 1 = 12`；`env_size = patch_num // n_envs = 12 // 4 = 3`。
每个"环境"只有 **3 个 patch**，用3个点算的HSIC方差本身噪声就很大，`compute_stability_score`
里的 `cv = hsic_std / hsic_mean` 在小样本下可能表现不稳定，训练后期梯度可能主要来自
`stability_bias` 和 `channel_prior` 这两个可学习参数，而不是真正的跨环境统计量。

**假设 C：门控整体收敛到饱和区（gate≈1 或 gate≈常数）。**
`CausalChannelAttention` 里门控是加性 log-bias（`attn + log(gate)`），如果两个变体的gate
都收敛到接近1（即约等于不加门控/全连接注意力），那么无论底层稳定性分数怎么算，下游行为都会
趋同——这也能同时解释为什么 `Full` 有时也和 `w/o Gate`（完全无门控）很接近
（如 pred_len=336, seed2024: Full=0.468979 vs w/o Gate=0.469125，相对差仅0.031%）。

> 以上三个假设目前都只是基于代码结构的推理，**没有直接证据**（训练时没有保存/打印门控矩阵的
> 具体数值分布），需要第4节的诊断手段验证。

---

## 4. 建议的后续诊断步骤（不需要重新完整跑一轮消融）

### 步骤1：直接保存并对比门控矩阵（最低成本，优先做）

`AblationBackbone`/`CausalCIT_backbone` 已经有 `get_gate_matrix()` 接口
（见 `models_ablation.py:296-297`、`causalcit.py:105-106`），`run_ablation.py` 的
`run_synthetic_ablation` 里也已经在测试集上采样了 `gate_matrices[variant_key]`
（见 `run_ablation.py:147-159`），只是目前只用来画图，**没有落盘保存数值**。

建议：在 `_plot_synthetic_ablation` 之前，把 `gate_matrices` 里 `full` 和 `no_env` 两个
key 的均值矩阵（`gate_matrices[vk].mean(axis=0)`，形状 `[n_vars, n_vars]`）分别打印出来，
或者 `np.save` 到 `output_dir` 下，直接肉眼对比数值：
- 如果两个矩阵几乎逐元素相同（尤其都接近全1或都接近某个共同的先验模式）→ 支持假设A/C。
- 如果两个矩阵差异明显、但下游MSE却接近 → 说明门控矩阵的差异对最终输出的影响本来就很小，
  需要往回查 `alpha`（`fusion_alpha`，融合系数，当前默认0.3）是否把通道交互整体的贡献
  稀释掉了——如果 `alpha` 学到的值很小，那不管门控多不同，最终输出都会趋近于"不做通道交互"。

### 步骤2：单独打印 `channel_prior` 参数训练前后的变化和量级

对比 `full` 和 `no_env` 两个变体训练完之后 `stability_gate.channel_prior` 的
`sigmoid()` 值，如果两者非常接近（或者都趋于某个常数），说明先验项确实主导了门控，
验证假设A。同时打印 `stability_bias`（Full）和检查 `temperature_param` 最终收敛值——
如果温度收敛到很大（`clamp(max=10.0)` 触顶），`sigmoid(logit/temp)` 会被压扁到接近0.5，
反而更可能被 `gate*(1-eye)+eye` 之外的常数项主导。

### 步骤3：加大环境划分粒度做对照实验，验证假设B

当前 `n_envs=4` 在 `patch_num=12` 下每环境仅3个patch。可以做一个对照：
把 `seq_len` 加大（比如336或512，让 `patch_num` 变大），或者把 `n_envs` 调小（比如2），
重跑一次小规模实验（不需要3个seed，1个seed即可，只是诊断用），如果 `Full` 与 `w/o EnvSplit`
的差异在更大 `patch_num` 下明显拉开，说明当前 `patch_num=12` 确实是"环境划分信号太弱"的
元凶；如果差距依然很小，则更支持假设A（先验主导）而非样本量问题。

### 步骤4：统计显著性——多seed是否需要扩到5~10个

第1节的标准差量级（1~2.5pp）意味着若要在均值上分辨出2pp左右的组件效果，
按标准误差 ≈ std/√n 估算，3个seed的标准误差约为 std/√3 ≈ 0.6~1.4pp，仍然和效应量同阶。
如果步骤1-3排查后确认代码逻辑本身没问题（只是效应量小+方差大），需要把seed数量扩到
**5~10个** 才可能让均值差异的置信区间不再和0重叠，届时再决定是否需要配对t检验或
Wilcoxon符号秩检验（因为是同一批数据、不同模型的配对比较，配对检验比独立样本检验更有效）。

---

## 5. 附录：3份原始报告关键数字全表（合成数据部分）

| 变体 | seed | MSE | MAE |
|---|---|---|---|
| PatchTST | 42 | 0.484986 | 0.527619 |
| PatchTST | 123 | 0.486046 | 0.529632 |
| PatchTST | 2024 | 0.487155 | 0.528755 |
| w/o Gate | 42 | 0.487643 | 0.527911 |
| w/o Gate | 123 | 0.490300 | 0.531767 |
| w/o Gate | 2024 | 0.493962 | 0.533332 |
| w/o EnvSplit | 42 | 0.496067 | 0.532370 |
| w/o EnvSplit | 123 | 0.484320 | 0.527984 |
| w/o EnvSplit | 2024 | 0.486182 | 0.527477 |
| w/o HSIC | 42 | 0.490490 | 0.529310 |
| w/o HSIC | 123 | 0.489426 | 0.530204 |
| w/o HSIC | 2024 | 0.487717 | 0.529404 |
| Full CausalCIT | 42 | 0.496238 | 0.532446 |
| Full CausalCIT | 123 | 0.483718 | 0.527542 |
| Full CausalCIT | 2024 | 0.486201 | 0.527487 |

完整的 ETTh1（pred96/336）原始 MSE/MAE 见各自的
`output_seed42/ablation_report.md`、`output_seed123/ablation_report.md`、
`output_seed2024/ablation_report.md`，本报告第1节已经把 vs-PatchTST 百分比整理成表，
不再重复摘录原始 MSE/MAE。
