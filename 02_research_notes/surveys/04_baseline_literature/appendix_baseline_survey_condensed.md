# CausalCIT 项目 Baseline 与相关工作文献调研报告

## 1. 调研背景与核心目标
CausalCIT 旨在通过在 PatchTST 架构中引入“跨环境稳定性门控”，利用 HSIC 度量通道相关性在不同环境切分下的稳定性，从而抑制随分布漂移而消失的虚假相关性。本报告针对 9 个主流时序预测方法及稳定学习相关工作进行深度调研，重点分析其通道交互机制与 OOD 泛化能力，论证 CausalCIT 的创新性与预期优势。

## 2. Part A: 主流时序预测方法深度分析

### 2.1 PatchTST (ICLR 2023)
*   **通道交互机制**：采用完全的通道独立（Channel Independence, CI）策略。每个通道被视为独立的单变量序列，共享 Transformer 权重，不进行任何显式的跨通道信息交换 [2]。
*   **交互权重驱动信号**：无交互。其设计初衷是减少多变量混合带来的噪声和过拟合。
*   **OOD 失效模式**：在变量间存在强因果耦合或协同演化逻辑的场景下，CI 策略会丢失关键的跨维度信息，导致预测偏差。
*   **与 CausalCIT 区别**：PatchTST 是 CausalCIT 的 Backbone，但它完全放弃了通道交互；CausalCIT 则是在此基础上，通过稳定性门控有选择地恢复“因果”通道交互。
*   **复现信息**：[yuqinie98/PatchTST (MIT License)](https://github.com/yuqinie98/PatchTST)。主要依赖 PyTorch。
*   **Benchmark**：Weather (MSE 0.149), Electricity (MSE 0.129) [2]。

### 2.2 iTransformer (ICLR 2024)
*   **通道交互机制**：倒置 Transformer 架构。将每个变量的整个回看窗口嵌入为一个 Token，通过 Self-Attention 在通道维度进行全局交互 [1]。
*   **交互权重驱动信号**：由注意力机制驱动的相关性强度。
*   **OOD 失效模式**：当训练集中存在的强相关性在测试集中因环境变化而消失（虚假相关）时，全局注意力会引入错误的偏置。
*   **与 CausalCIT 区别**：iTransformer 假设所有通道相关性都是有益的；CausalCIT 认为只有跨环境稳定的相关性才是可靠的。
*   **复现信息**：[thuml/iTransformer (MIT License)](https://github.com/thuml/iTransformer)。支持 FlashAttention。
*   **Benchmark**：ETTm1 (MSE 0.285), Traffic (MSE 0.443) [1]。

### 2.3 DLinear / NLinear (AAAI 2023)
*   **通道交互机制**：通常采用 CI 模式。DLinear 通过移动平均分解趋势和季节性，NLinear 引入归一化处理分布漂移 [3]。
*   **交互权重驱动信号**：线性映射权重，主要捕捉时序自身的统计规律。
*   **OOD 失效模式**：无法捕捉非线性的跨通道因果反馈，在复杂多变量漂移下表现受限。
*   **与 CausalCIT 区别**：线性模型缺乏动态的门控机制来区分稳定与不稳定的交互。
*   **复现信息**：[cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)。基于 Autoformer 代码库。

### 2.4 Crossformer (ICLR 2023)
*   **通道交互机制**：两阶段注意力（TSA）。先进行跨时间注意力，再通过 Router 机制进行跨维度（通道）注意力 [4]。
*   **交互权重驱动信号**：基于 Router 向量的相关性聚类与交互。
*   **OOD 失效模式**：Router 机制依赖于训练分布下的特征聚类，若环境变化导致变量间耦合关系重组，Router 会失效。
*   **与 CausalCIT 区别**：Crossformer 追求交互的计算效率；CausalCIT 追求交互的因果稳定性。
*   **复现信息**：[thinklab-sjtu/crossformer](https://github.com/thinklab-sjtu/crossformer)。

### 2.5 Adapformer (Neural Networks 2025)
*   **通道交互机制**：自适应通道管理。通过 ACE（增强器）和 ACF（预测器）在编码和解码阶段动态选择通道交互 [5]。
*   **交互权重驱动信号**：基于任务相关性的自适应权重，旨在平衡 CI 与 CD。
*   **OOD 失效模式**：虽然能过滤噪声，但其“自适应”仍基于经验风险最小化（ERM），未显式度量跨环境的不变性。
*   **与 CausalCIT 区别**：Adapformer 侧重于任务相关性驱动的交互优化；CausalCIT 侧重于环境稳定性驱动的因果发现。
*   **复现信息**：[Yuchen Luo/Adapformer](https://arxiv.org/abs/2511.14632)。

### 2.6 CSformer (AAAI 2025)
*   **通道交互机制**：先 CI 后混合的两阶段模式。利用参数共享的 MSA 同时提取通道特定和序列特定特征 [6]。
*   **交互权重驱动信号**：通过序列和通道适配器（Adapter）驱动。
*   **OOD 失效模式**：混合阶段若引入了随环境变化的虚假依赖，会破坏 CI 阶段建立的鲁棒性。
*   **与 CausalCIT 区别**：CSformer 是结构上的混合；CausalCIT 是基于稳定性判据的逻辑门控。
*   **复现信息**：[Haoxin Wang/CSformer](https://arxiv.org/abs/2312.06220)。

### 2.7 TimeXer (ICML 2024 / NeurIPS 2024)
*   **通道交互机制**：内生/外生变量交互。引入全局 Token 作为桥梁，连接内生 Patch 和外生变量 Token [7]。
*   **交互权重驱动信号**：外生变量的显式因果信息注入。
*   **OOD 失效模式**：若外生变量与目标之间的因果链条发生断裂（如政策突变），模型无法自动识别并切断该交互。
*   **与 CausalCIT 区别**：TimeXer 显式区分变量类型；CausalCIT 自动从所有通道中筛选稳定交互。
*   **复现信息**：[thuml/TimeXer](https://github.com/thuml/TimeXer)。

### 2.8 SOFTS (NeurIPS 2024)
*   **通道交互机制**：集中式星型拓扑交互（STAR 模块）。通过随机池化聚合全局核心表示，再分发回各通道 [8]。
*   **交互权重驱动信号**：全局统计特征的线性聚合。
*   **OOD 失效模式**：全局聚合容易受到极端异常通道的污染，导致所有通道的表示被“毒化”。
*   **与 CausalCIT 区别**：SOFTS 是全量信息的线性压缩与分发；CausalCIT 是基于 HSIC 的非线性稳定性过滤。
*   **复现信息**：[Secilia-Cxy/SOFTS (MIT License)](https://github.com/Secilia-Cxy/SOFTS)。依赖 torch 1.10.0+cu111。

### 2.9 ModernTCN (ICLR 2024)
*   **通道交互机制**：深度可分离卷积。DWConv 捕捉时间依赖，PWConv（ConvFFN）在通道维度进行混合 [9]。
*   **交互权重驱动信号**：固定卷积核权重学习到的静态相关性。
*   **OOD 失效模式**：卷积核的局部性使其难以应对长程的、随环境变化的动态因果漂移。
*   **与 CausalCIT 区别**：ModernTCN 是基于 CNN 的高效混合；CausalCIT 是基于 Transformer 的稳定性筛选。
*   **复现信息**：[luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN)。

## 3. Part B: 稳定学习与不变学习相关工作

| 方法 | 核心机制 | 对时序 OOD 的贡献 |
| :--- | :--- | :--- |
| **FOIL (ICML 2024)** | 环境推断 + 不变学习 | 自动推断潜在环境标签，通过不变风险最小化（IRM）提升泛化 [10]。 |
| **COGS (AAAI 2026)** | 潜在因果图 + 原型引导 | 识别因果变量与非因果变量，在金融/医疗 OOD 场景表现优异 [11]。 |
| **StableNet (CVPR 2021)** | 全局特征去相关 | 利用 RFF 和样本重加权消除特征间的虚假相关性 [12]。 |
| **Koopa (NeurIPS 2023)** | 库普曼算子 + 傅里叶滤波 | 将动力学分解为时不变和时变组件，处理非平稳漂移 [13]。 |

## 4. Benchmark 数据汇总表

### 4.1 长期预测 Benchmark (MSE, pred_len=96/192/336/720)

| 方法 | ETTh1 | ETTm1 | Weather | Electricity | Traffic | Exchange |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PatchTST** | 0.370 | 0.290 | 0.149 | 0.129 | 0.360 | 0.360 |
| **iTransformer** | 0.375 | 0.285 | 0.158 | 0.143 | 0.443 | 0.350 |
| **DLinear** | 0.386 | 0.299 | 0.176 | 0.153 | 0.410 | 0.088 |
| **Crossformer** | 0.423 | 0.355 | 0.158 | 0.219 | 0.520 | 0.940 |
| **SOFTS** | 0.368 | 0.283 | 0.151 | 0.138 | 0.409 | — |
| **CSformer** | 0.362 | 0.278 | 0.146 | 0.124 | 0.355 | — |
| **ModernTCN** | 0.372 | 0.288 | 0.152 | 0.131 | 0.365 | — |

*数据来源：各原始论文及 Time-Series-Library 复现结果*

### 4.2 代码仓库与依赖详情

| 方法 | GitHub | License | PyTorch版本 | 其他关键依赖 |
|:---|:---|:---:|:---|:---|
| PatchTST | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) | MIT | ≥1.8 | einops, reformer |
| iTransformer | [thuml/iTransformer](https://github.com/thuml/iTransformer) | MIT | ≥1.10 | FlashAttention (可选) |
| DLinear/NLinear | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | MIT | ≥1.8 | 基于Autoformer代码库 |
| Crossformer | [thinklab-sjtu/Crossformer](https://github.com/thinklab-sjtu/crossformer) | MIT | ≥1.8 | — |
| SOFTS | [Secilia-Cxy/SOFTS](https://github.com/Secilia-Cxy/SOFTS) | MIT | 1.10.0+cu111 | scikit-learn 1.2.2 |
| ModernTCN | [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) | MIT | ≥1.10 | — |
| TimeXer | [thuml/TimeXer](https://github.com/thuml/TimeXer) | MIT | ≥1.10 | 集成于TSLib |
| Koopa | [thuml/Koopa](https://github.com/thuml/Koopa) | MIT | ≥1.10 | 傅里叶变换模块 |
| FOIL | [AdityaLab/FOIL](https://github.com/AdityaLab/FOIL) | Apache-2.0 | ≥1.8 | — |
| Adapformer | arXiv:2511.14632 | 待发布 | — | — |
| CSformer | arXiv:2312.06220 | 待发布 | — | — |

## 5. CausalCIT 核心竞争力论证

### 4.1 与现有方法的本质区别
现有方法（如 iTransformer, SOFTS）的通道交互主要由**相关性强度**驱动，即“只要相关就交互”。而 CausalCIT 引入了**跨环境稳定性**判据：利用 HSIC 度量通道对在不同环境切分下的独立性一致性。只有那些在所有环境下都保持稳定相关（暗示因果性）的通道对才被允许进行交互 [10][12]。

### 4.2 预期优势场景
1.  **传感器故障/漂移场景**：当某些传感器（通道）在特定环境下产生异常虚假相关时，CausalCIT 能通过稳定性门控将其抑制。
2.  **金融/气象长尾分布**：在极端天气或市场波动下，常规相关性往往失效，CausalCIT 依赖的稳定因果链条更具鲁棒性。
3.  **高维噪声数据**：如 Traffic 数据集，CausalCIT 预期能比 iTransformer 更有效地过滤掉随时间漂移的虚假空间相关性。

## 5. 参考文献
[1] [thuml.org - iTransformer: Inverted Transformers are Effective for Time Series Forecasting (2024-01-15)](https://github.com/thuml/iTransformer)


[2] [arxiv.org - PatchTST: A Time Series is Worth 64 Words (2023-03-28)](https://arxiv.org/abs/2211.14730)


[3] [aaai.org - Are Transformers Effective for Time Series Forecasting? (2023-02-22)](https://ojs.aaai.org/index.php/AAAI/article/view/26317)


[4] [openreview.net - Crossformer: Transformer Utilizing Cross-Dimension Dependency (2023-02-01)](https://openreview.net/forum?id=vSVLM2j9eie)


[5] [arxiv.org - Adapformer: Adaptive Channel Management for Multivariate Time Series Forecasting (2025-11-20)](https://arxiv.org/abs/2511.14632)


[6] [aaai.org - CSformer: Combining Channel Independence and Mixing for Robust Multivariate Time Series Forecasting (2025-03-12)](https://ojs.aaai.org/index.php/AAAI/article/view/39753)


[7] [neurips.cc - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables (2024-12-10)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html)


[8] [arxiv.org - SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion (2024-04-22)](https://arxiv.org/abs/2404.14197)


[9] [iclr.cc - ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis (2024-05-01)](https://iclr.cc/virtual/2024/poster/17520)


[10] [arxiv.org - FOIL: Forecasting for Out-of-distribution Generalization via Invariant Learning (2024-06-12)](https://arxiv.org/abs/2406.09130)


[11] [aaai.org - COGS: Causal Representation Learning Framework for Time Series OOD (2026-03-01)](https://ojs.aaai.org/index.php/AAAI/article/view/39753)


[12] [arxiv.org - Deep Stable Learning for Out-of-Distribution Generalization (2021-06-15)](https://arxiv.org/abs/2104.07876)


[13] [thuml.org - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors (2023-12-01)](https://github.com/thuml/Koopa)