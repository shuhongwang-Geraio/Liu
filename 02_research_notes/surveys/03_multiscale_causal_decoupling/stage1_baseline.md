# 多变量时间序列预测文献调研基础文档

## 1. 用户研究背景概述

本研究聚焦于多变量时间序列预测（MTSF）领域，核心主题为「多尺度序列数据建模」与「多通道因果解耦」。目前研究工作分为以下三条技术路线推进：

### 1.1 线 A（已实验验证）：CausalCIT
该方案在 PatchTST 的 Channel-Independent (CI) 架构基础上，引入了基于 **HSIC 独立性检验的跨环境稳定性门控**。其核心逻辑是将训练序列划分为多个时间段（环境），计算通道对在不同环境下的 HSIC 依赖度及其变异系数（CV）。低 CV 代表稳定依赖（潜在因果），门控开启允许交互；高 CV 代表不稳定依赖（虚假相关），门控压低以阻断交互。实验表明，该方法在高维强依赖数据集（如 Traffic 862通道, Electricity 321通道）上显著优于 PatchTST，但在低维数据集（如 ETTh1, ILI）上表现不佳，推测原因为低维场景下门控机制易退化为噪声。

### 1.2 线 B（想法阶段）：多尺度/异构采样率建模
旨在处理分钟、小时、天级混合的异构采样率数据，实现“无插值”建模。核心组件包括：跨尺度注意力机制（高频 Q 查询低频 K/V）、尺度感知的位置编码（基于绝对时间差计算相对位置），以及受重整化群（RG）启发的层次化粗粒化/细粒化双向信息流。

### 1.3 线 C（想法阶段）：可逆通道解耦
借鉴 StableNet 的 RFF-HSIC 独立性检验思想，设计可学习的正交可逆变换 $W$。通过 $Z = X \cdot W$ 将原始序列映射至隐通道空间，利用 RFF-HSIC 正则项鼓励 $Z$ 的各维度统计独立。采用“解耦 $\rightarrow$ 独立预测 $\rightarrow$ 还原”的三段式 Pipeline，最终通过 $W^T$ 逆变换还原预测结果。

## 2. 已调研文献列表

以下文献已完成调研与归档，在后续检索任务中需排除：

| 类别 | 文献名称 / 来源 | 核心标签 |
| :--- | :--- | :--- |
| 经典架构 | PatchTST (ICLR'23), iTransformer (ICLR'24), DLinear (AAAI'23) | CI/CD 基础 |
| 现代改进 | SOFTS (NeurIPS'24), ModernTCN (ICLR'24), RATD (NeurIPS'24) | 跨通道/大核卷积 |
| 解耦与独立性 | StableNet (CVPR'21), DisenTS, CCM, MTS-Unmixers (arXiv:2411.17770) | 统计独立/解耦 |
| 线性与策略 | OLinear (NeurIPS'25), PCA-ICA-BiLSTM, ChannelStrategySurvey (arXiv'25) | 线性模型/策略综述 |
| 最新前沿 | Caiformer (arXiv:2505.16308), JointPGM (arXiv:2407.13194), CSformer (AAAI'25) | 2024-2025 新工作 |
| 损失与理论 | FOIL (ICML'24), COGS (AAAI'26), RI-Loss (AAAI'26), CI容量-鲁棒性权衡 (TKDE'24) | 理论分析/损失函数 |
| 其他 | CausalTimePrior, CW-Gen, Channel Normalization (ICML'25) | 因果/归一化 |

## 3. 知识库现有资源

当前知识库已包含以下论文的完整 PDF 文件，可直接用于深度分析：
*   `/usr/local/app/attachment/PatchTST.pdf`
*   `/usr/local/app/attachment/iTransformer.pdf`
*   `/usr/local/app/attachment/DLinear.pdf`
*   `/usr/local/app/attachment/SOFTS.pdf`
*   `/usr/local/app/attachment/ModernTCN.pdf`
*   `/usr/local/app/attachment/StableNet.pdf`
*   `/usr/local/app/attachment/DisenTS.pdf`
*   `/usr/local/app/attachment/CCM.pdf`
*   `/usr/local/app/attachment/MTS-Unmixers.pdf`
*   `/usr/local/app/attachment/OLinear.pdf`

## 4. 子问题检索任务清单与关键词建议

| 子问题编号 | 任务描述 | 建议检索关键词组合 |
| :--- | :--- | :--- |
| **1** | 因果/稳定性驱动的通道交互选择 | `causal invariance`, `stability`, `environment split`, `channel interaction selection`, `spurious correlation time series` |
| **2** | 方法有效性与通道数/维度的关系 | `high-dimensional vs low-dimensional forecasting`, `channel independence scalability`, `number of channels effect`, `failure mode CI CD` |
| **3** | 多尺度/异构采样率与通道联合建模 | `multi-scale`, `heterogeneous sampling rate`, `irregularly sampled time series`, `cross-scale attention`, `renormalization group transformer` |
| **4** | 可逆/正交变换+独立性约束的通道解耦 | `invertible neural networks`, `orthogonal transform`, `RFF-HSIC regularization`, `independent component forecasting`, `unmixing time series` |
| **5** | 解缠表示学习在时序预测中的应用 | `disentangled representation learning`, `factorized time series`, `VAE disentanglement forecasting`, `independent factor modeling` |
| **6** | 可证伪性与失败分析 | `failure mode analysis`, `limitation of time series models`, `falsifiable claims in AI`, `out-of-distribution robustness boundary` |

## 参考文献
[1] [arxiv.org - PatchTST: A Time Series is Worth 64 Words (2023-03-28)](https://arxiv.org/abs/2211.14730)


[2] [openreview.net - iTransformer: Inverted Transformers are Effective for Time Series Forecasting (2023-10-27)](https://openreview.net/forum?id=oVjConstruct)


[3] [arxiv.org - Are Transformers Effective for Time Series Forecasting? (DLinear) (2022-08-11)](https://arxiv.org/abs/2208.05233)


[4] [arxiv.org - StableNet: Semi-Online Learning for Multi-Variate Time Series Forecasting (2021-06-15)](https://arxiv.org/abs/2106.08081)


[5] [arxiv.org - MTS-Unmixers: Multivariate Time Series Forecasting via Channel Unmixing (2024-11-20)](https://arxiv.org/abs/2411.17770)