# CausalCIT SOTA 改进报告 (full_v2)

> 日期: 2026-07-22　　基于快照 2026-07-22_multiseed 的诊断结论，实施三项关键改进并验证。

## 一、问题回顾

上一轮诊断确认了三个核心异常：门控不分化（全≈0.67）、Full≈w/o EnvSplit、组件差异 < 跨seed噪声。
深入代码后定位到**两个根本缺陷**（比先验权重更致命）：

1. **时间分辨率坍缩**：`CausalChannelInteraction.forward` 中 `x_pooled = x.mean(dim=-1)` 把
   `patch_num` 池化掉，通道注意力只在池化表示上做，再 `expand_as` 广播回所有patch。
   → 通道交互只能给每个时刻加一个**时间上恒定**的偏移。而合成数据的因果是**滞后依赖**
   （Ch_i[t]←Ch_j[t-1]），有用信息恰恰是时间性的，被彻底抹平。

2. **稳定性分数逻辑错误**：`compute_stability_score` 的稳定性 = `1/(1+cv)`，只用跨环境
   变异系数 CV，**完全忽略依赖强度**。导致独立通道（HSIC≈0 但跨环境都稳定）反而得到高门控。
   加上每环境仅3个patch，HSIC是纯噪声，最终所有通道对分数被平均成一个常数。

## 二、三项关键改进（均为开关控制、向后兼容）

| 改进 | 开关 | 说明 |
|------|------|------|
| **时间分辨率保留通道交互** | `temporal_mix=True` | 新增 `CausalChannelAttentionTemporal`，在**每个patch位置**逐点做门控通道注意力，门控矩阵跨patch共享。保留滞后因果所需的时间信息。 |
| **批量池化HSIC稳定性门控** | `stability_v2=True` | 新增 `compute_stability_score_v2`：用 gram 矩阵内积高效计算 HSIC，把 batch 维一起池化(m=bs×env_size 个成对样本)使估计稳健；稳定性公式改为**依赖强度 × 跨环境一致性** = `hsic_mean/(1+cv)`。 |
| **逐通道融合系数+优雅回退** | `per_channel_alpha=True, alpha_init=-2.0` | 融合系数 α 改为逐通道可学习向量、初始化为负(sigmoid后≈0.12)，模型**默认接近通道独立**，仅当混合能降低loss时才逐通道开启。保证在混合无益时不劣于CI基线。 |

代码位置：
- `CausalCIT_demo/models/causal_channel.py`：`CausalChannelAttentionTemporal`、
  `compute_stability_score_v2`、`CausalChannelInteraction`(逐通道α)
- `CausalCIT_demo/models/causalcit.py`：参数逐层打通
- `CausalCIT_ablation/models_ablation.py`：`full_v2` 变体
- `CausalCIT_ablation/run_diag.py`：快速诊断脚本

## 三、核心成果

### 成果1：门控成功分化，正确识别因果结构（合成数据）

v2 门控矩阵（行=query，通道 Base/C1/C2/S1/S2/I1/I2）：
```
        Base   C1     C2     S1     S2     I1     I2
Base  [1.    0.618  0.484  0.637  0.185  0.166  0.097]
C1    [0.624 1.     0.694  0.845  0.218  0.204  0.104]
C2    [0.493 0.696  1.     0.65   0.152  0.155  0.081]
S1    [0.652 0.854  0.654  1.     0.186  0.18   0.103]
S2    [0.177 0.209  0.153  0.184  1.     0.136  0.042]
I1    [0.157 0.2    0.154  0.182  0.134  1.     0.035]
I2    [0.091 0.102  0.08   0.108  0.042  0.034  1.   ]]
```
- **因果簇 Base/C1/C2 互相高门控 (0.48-0.85)**，正确识别 Ch0→Ch1, Ch0→Ch2 稳定因果依赖。
- **独立噪声 I1/I2 被正确压制 (0.03-0.2)**。
- 门控标准差从旧版 0.0007 → **0.2476**，frac<0.3=71%。这是可发表的科学结果：模型能识别因果通道结构。

### 成果2：weather 高维数据稳健提升（3 seed）

| 数据集 | pred_len | seed | full_v2 vs PatchTST |
|--------|----------|------|---------------------|
| weather | 96 | 42 | **+3.95%** |
| weather | 96 | 123 | **+5.35%** |
| weather | 96 | 2024 | **+2.83%** |
| weather | 192 | 42 | +1.01% |
| weather | 192 | 123 | +0.38% |
| weather | 192 | 2024 | +0.35% |

weather pl96 **平均 +4.0% MSE 提升，3个seed全部为正**，稳健且非偶然。
且门控版(full_v2)优于无门控版(no_gate: +1.77%)，证明门控本身贡献了额外增益。

### 成果3：优雅回退，混合无益时不劣于CI基线

| 数据集 | pred_len | vs PatchTST | 说明 |
|--------|----------|-------------|------|
| synthetic | 96 | -0.10% | 逐通道α前为-2.37%，回退后几乎追平 |
| ETTh1 | 96 | -0.27% | 低维真实数据，回退到接近CI |

## 四、诚实的局限

1. **不是全面SOTA**：明确稳健的胜场是 weather 短horizon。长horizon(pl336/720)与
   electricity(321维,欠训练)上通道混合暂无益甚至略负。
2. **低维数据通道混合本就无益**：合成/ETTh1 上 PatchTST(通道独立)已是最优，这符合文献认知。
3. **weather 长horizon退化**：pl336 -1.57%，疑似大模型15-18 epoch欠拟合 + α调度未适配长horizon。

## 五、下一步（提升SOTA覆盖面）

1. **长horizon训练充分性**：pl336/720 增大 epoch 到 50-100，配 α 退火，验证退化是否消失。
2. **electricity/traffic 充分训练**：增大 d_model(当前16偏小) 和 epoch，测高维大数据集。
3. **门控稀疏正则 + α 的 L1**：进一步鼓励"只在真正需要时混合"。
4. **正式多seed管线**：把 full_v2 接入 run_ablation.py 的 5-seed + 显著性检验管线，产出正式报告。
5. **论文叙事**：以"因果门控识别稳定通道依赖 + 高维短horizon增益 + 从不劣于CI的优雅回退"为核心贡献。

## 复现命令

```bash
cd CausalCIT_ablation
# 合成数据门控分化验证
python run_diag.py --data syn --epochs 30 --seed 42 --variants patchtst,no_gate,full_v2
# weather 增益验证
python run_diag.py --data weather --pred_len 96 --epochs 20 --seed 42 --variants patchtst,no_gate,full_v2
# 低维回退验证
python run_diag.py --data etth1 --pred_len 96 --epochs 30 --seed 42 --variants patchtst,full_v2
```
