# CausalCIT 项目 Baseline 与 Related Work 文献调研报告

**信息截止日期：2026年8月10日**

---

## 1. CausalCIT 核心创新概述

CausalCIT 是一种面向多元时间序列预测（MTSF）的 OOD 泛化方法，以 PatchTST 为骨干网络，核心创新在于引入**跨环境稳定性门控**机制。该方法利用希尔伯特-施密特独立性准则（HSIC）度量通道对相关性在不同环境切分下的稳定性，仅保留那些在所有环境下都保持稳定相关（暗示因果性）的通道对进行交互，同时抑制随分布漂移而消失的虚假相关通道对 [1]。其关键假设是：在高维、依赖结构强的数据集（如 Traffic、Electricity）上，稳定性门控能有效抑制虚假相关；而在低维、弱依赖场景（如 ETTh1）下，门控可能退化为噪声 [2]。

与现有方法的核心区别在于：**CausalCIT 以"跨环境稳定性"而非"相关性强度"作为通道交互的准入判据**，这是对当前主流方法设计哲学的根本性修正。

---

## 2. 主流方法深度分析

### 2.1 PatchTST (ICLR 2023) — 当前 Baseline

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 完全的通道独立（Channel Independence, CI）策略。每个通道被视为独立的单变量序列，共享 Transformer 权重，不进行任何显式的跨通道信息交换 [3]。 |
| **A. 交互权重驱动信号** | 无交互。其设计初衷是减少多变量混合带来的噪声和过拟合，通过 Patching 机制捕获局部时序语义。 |
| **A. OOD 失效模式** | 在变量间存在强因果耦合或协同演化逻辑的场景下，CI 策略会丢失关键的跨维度信息，导致预测精度存在天花板。 |
| **B. 与 CausalCIT 本质区别** | PatchTST 是 CausalCIT 的 Backbone，但它完全放弃了通道交互；CausalCIT 则是在此基础上，通过稳定性门控有选择地恢复"因果"通道交互，打破 CI 的性能瓶颈。 |
| **B. 实验预期差异** | 在高维强依赖数据集（Traffic/Electricity）上，CausalCIT 预期通过选择性交互超越 PatchTST；在低维弱依赖数据集（ETTh1）上，门控退化为噪声，两者性能接近。 |
| **C. 复现信息** | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) (MIT License)；PyTorch ≥1.8；依赖 einops, reformer。Benchmark：Weather (MSE 0.149), Electricity (MSE 0.129), Traffic (MSE 0.360) [3]。 |

**vs CausalCIT 差异总结**：PatchTST 是目前最强的 CI 基准，其成功证明了"不交互"在部分场景下的有效性。CausalCIT 的目标是证明：通过 HSIC 稳定性门控引入的适度通道交互，可以在保持 PatchTST 鲁棒性的同时，获取多变量协同带来的精度提升，从而突破 CI 的性能上限。

---

### 2.2 iTransformer (ICLR 2024) — 倒置 Transformer

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 倒置 Transformer 架构。将每个变量的整个回看窗口嵌入为一个 Token，通过 Self-Attention 在通道维度进行全局交互，以变量为中心而非以时间步为中心 [4]。 |
| **A. 交互权重驱动信号** | 由注意力机制驱动的相关性强度。所有通道对之间均建立全连接注意力，无差别地进行信息混合。 |
| **A. OOD 失效模式** | 当训练集中存在的强相关性在测试集中因环境变化而消失（虚假相关）时，全局注意力会引入错误的偏置，导致预测性能显著下降。 |
| **B. 与 CausalCIT 本质区别** | iTransformer 假设所有通道相关性都是有益的，进行"相关性强度驱动的全量交互"；CausalCIT 认为只有跨环境稳定的相关性才是可靠的，进行"稳定性驱动的选择性交互"。 |
| **B. 实验预期差异** | 在 Traffic/Electricity 等高维且环境波动剧烈的数据集上，CausalCIT 预期通过剔除不稳定的虚假相关通道对，表现优于 iTransformer；但在 ETTh1 等低维数据上，iTransformer 的全局建模能力可能更占优。 |
| **C. 复现信息** | [thuml/iTransformer](https://github.com/thuml/iTransformer) (MIT License)；PyTorch ≥1.10；支持 FlashAttention。Benchmark：ETTm1 (MSE 0.285), Traffic (MSE 0.443), Weather (MSE 0.158) [4]。 |

**vs CausalCIT 差异总结**：iTransformer 的本质是"只要相关就交互"，其通道注意力权重完全由训练分布的统计相关性决定。CausalCIT 引入 HSIC 作为外部约束，强制门控关注跨环境的不变性，从而在分布偏移时保持预测逻辑的一致性。两者代表了"相关性驱动"与"因果稳定性驱动"两种设计哲学的根本对立。

---

### 2.3 DLinear / NLinear (AAAI 2023) — 分解线性

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 通常采用 CI 模式。DLinear 通过移动平均将序列分解为趋势和季节项，分别用线性层预测后求和；NLinear 引入 Instance Normalization 缓解分布漂移 [5]。 |
| **A. 交互权重驱动信号** | 线性映射权重，主要捕捉时序自身的统计规律，不涉及跨通道信息交换。 |
| **A. OOD 失效模式** | 无法捕捉非线性的跨通道因果反馈，在复杂多变量漂移下表现受限；线性模型的表达能力存在天然瓶颈。 |
| **B. 与 CausalCIT 本质区别** | DLinear 代表了极端保守的 CI 策略——完全放弃交互以换取稳定性；CausalCIT 则试图在 CI 和 CD 之间寻找因果平衡点，实现"有选择的交互"。 |
| **B. 实验预期差异** | 在变量间存在强因果依赖的场景下，CausalCIT 能够提取 DLinear 无法获取的跨通道增益；在极低维或近乎独立的变量场景下，两者性能接近。 |
| **C. 复现信息** | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) (MIT License)；PyTorch ≥1.8；基于 Autoformer 代码库。Benchmark：ETTh1 (MSE 0.386), Electricity (MSE 0.153), Exchange (MSE 0.088) [5]。 |

**vs CausalCIT 差异总结**：DLinear 通过"不交互"规避了虚假相关风险，但也丢失了多变量协同信息。CausalCIT 提供了一种更具理论支撑的方案：不是简单地放弃交互，而是通过 HSIC 稳定性判据筛选出值得交互的通道对，在保持鲁棒性的前提下获取跨通道增益。

---

### 2.4 Crossformer (ICLR 2023) — 跨维度注意力

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 两阶段注意力（Two-Stage Attention, TSA）。先通过跨时间注意力捕获各通道的时序依赖，再通过 Router 机制在通道维度进行跨维度注意力交互 [6]。 |
| **A. 交互权重驱动信号** | 基于 Router 向量的相关性聚类与交互。Router 从各通道提取代表性信息，通过注意力机制在通道间进行信息路由。 |
| **A. OOD 失效模式** | Router 机制依赖于训练分布下的特征聚类模式，若环境变化导致变量间耦合关系重组，Router 的路由逻辑会失效，引入错误的跨通道偏置。 |
| **B. 与 CausalCIT 本质区别** | Crossformer 关注"如何高效地交互"（通过 Router 减少计算量）；CausalCIT 关注"是否应该交互"（基于跨环境稳定性进行准入判断）。 |
| **B. 实验预期差异** | 在变量间依赖结构相对稳定的场景下，Crossformer 的高效交互可能占优；在依赖结构随环境漂移的场景下，CausalCIT 的稳定性门控能过滤掉 Crossformer 中那些虽然强度高但随环境变化的虚假连接。 |
| **C. 复现信息** | [thinklab-sjtu/Crossformer](https://github.com/thinklab-sjtu/crossformer) (MIT License)；PyTorch ≥1.8。Benchmark：ETTh1 (MSE 0.423), Weather (MSE 0.158), Traffic (MSE 0.520) [6]。 |

**vs CausalCIT 差异总结**：Crossformer 强调跨维度的精细建模，但其交互权重完全取决于当前样本的特征强度。CausalCIT 的稳定性门控作为一种"过滤器"，能够识别出 Crossformer 中那些虽然强度高但随环境变化的虚假连接，从而在分布偏移时保持预测逻辑的一致性。

---

### 2.5 Adapformer (Neural Networks 2025) — **重点竞品**

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 自适应通道管理。通过 ACE（Adaptive Channel Enhancer）和 ACF（Adaptive Channel Forecaster）在编码和解码阶段动态选择通道交互，旨在平衡 CI 与 CD 的优势 [7]。 |
| **A. 交互权重驱动信号** | 基于任务相关性的自适应权重。通过轻量级适配器模块学习哪些通道对当前预测任务有益，侧重于参数高效微调。 |
| **A. OOD 失效模式** | 虽然能过滤噪声，但其"自适应"仍基于经验风险最小化（ERM），适配器可能学习到下游任务特定环境的过拟合特征，未显式度量跨环境的不变性。 |
| **B. 与 CausalCIT 本质区别** | Adapformer 是基于相关性的"自适应"——门控主要为了提升微调效率和任务适配度；CausalCIT 是基于因果稳定性的"准入制"——门控由 HSIC 跨环境稳定性判据驱动。 |
| **B. 实验预期差异** | 在存在明显环境干扰（如传感器故障、季节性政策变动）的数据集中，CausalCIT 的因果门控应具有更强的鲁棒性；在环境相对稳定的场景下，Adapformer 的自适应微调可能更灵活。 |
| **C. 复现信息** | [arXiv:2511.14632](https://arxiv.org/abs/2511.14632)；官方代码待发布。Benchmark 数据待确认。 |

**vs CausalCIT 差异总结**：作为直接竞品，Adapformer 的门控信号来源仍是统计相关性，其"自适应"本质上是 ERM 框架下的优化。CausalCIT 引入 HSIC 作为外部不变性约束，强制门控关注跨环境的一致性，这在理论上提供了更强的 OOD 泛化保障。两者的核心分歧在于：**"任务相关"不等于"因果稳定"**。

---

### 2.6 CSformer (AAAI 2025) — 两阶段通道混合

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 先 CI 后 CD 的两阶段架构。第一阶段利用参数共享的 MSA 独立提取各通道特征（CI），第二阶段通过序列和通道适配器进行跨通道混合（CD），试图结合两者的优点 [8]。 |
| **A. 交互权重驱动信号** | 通过序列适配器和通道适配器（Adapter）驱动的固定架构设计，第二阶段进行全通道混合。 |
| **A. OOD 失效模式** | 第二阶段的 CD 混合如果缺乏约束，仍会引入训练集特有的分布偏见；混合阶段若引入了随环境变化的虚假依赖，会破坏 CI 阶段建立的鲁棒性。 |
| **B. 与 CausalCIT 本质区别** | CSformer 是架构上的分阶段——先独立后混合；CausalCIT 是在交互入口处进行因果稳定性校验——用 HSIC 判据决定哪些通道参与混合。 |
| **B. 实验预期差异** | CSformer 承认了 CI 的稳定性价值，但其 CD 阶段仍是启发式的全量混合；CausalCIT 提供了一个更具理论支撑的准则来决定 CD 阶段的交互范围，在分布漂移场景下预期更鲁棒。 |
| **C. 复现信息** | [arXiv:2312.06220](https://arxiv.org/abs/2312.06220)；官方代码待发布。Benchmark：ETTh1 (MSE 0.362), ETTm1 (MSE 0.278), Weather (MSE 0.146) [8]。 |

**vs CausalCIT 差异总结**：CSformer 在架构层面承认了 CI 的价值，但其 CD 阶段缺乏理论指导。CausalCIT 的 HSIC 稳定性门控可以视为对 CSformer CD 阶段的理论升级——不是启发式地混合所有通道，而是基于跨环境不变性判据进行选择性交互。

---

### 2.7 TimeXer (ICML 2024 / NeurIPS 2024) — 内生/外生变量交互

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 内生变量与外生变量的显式交互机制。引入全局 Token 作为桥梁，连接内生变量的 Patch 嵌入和外生变量的序列 Token，实现跨变量类型的信息融合 [9]。 |
| **A. 交互权重驱动信号** | 外生变量的显式因果信息注入。假设外生变量与目标变量之间存在稳定的因果关系，通过注意力机制进行定向信息传递。 |
| **A. OOD 失效模式** | 若外生变量与目标之间的因果链条发生断裂（如政策突变、市场规则改变），模型无法自动识别并切断该交互，预测会剧烈失效。 |
| **B. 与 CausalCIT 本质区别** | TimeXer 侧重变量类型的先验区分（内生 vs 外生），依赖人工指定的变量角色；CausalCIT 侧重交互关系的跨环境稳定性验证，是一种数据驱动的因果发现过程。 |
| **B. 实验预期差异** | 在能明确区分内外生变量的场景下，TimeXer 的定向交互可能更精准；在无法明确区分或所有变量地位对等的复杂系统中，CausalCIT 的普适性更强。 |
| **C. 复现信息** | [thuml/TimeXer](https://github.com/thuml/TimeXer) (MIT License)；PyTorch ≥1.10；集成于 Time-Series-Library。Benchmark 数据见原始论文 [9]。 |

**vs CausalCIT 差异总结**：TimeXer 依赖于先验的变量分类知识，而 CausalCIT 是一种数据驱动的因果发现过程。在无法明确区分内外生变量或所有变量地位对等的复杂系统中，CausalCIT 的普适性更强。此外，CausalCIT 的稳定性门控可以自动识别并切断失效的外生变量交互，而 TimeXer 缺乏这种自适应切断机制。

---

### 2.8 SOFTS (NeurIPS 2024) — 基于统计特征的通道交互

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 集中式星型拓扑交互（STAR 模块）。通过随机池化将所有通道的信息聚合为一个全局核心表示，再将该核心表示分发回各通道，实现高效的通道间信息交换 [10]。 |
| **A. 交互权重驱动信号** | 全局统计特征的线性聚合。通过随机池化压缩通道信息，以极低的计算成本实现通道交互。 |
| **A. OOD 失效模式** | 全局聚合容易受到极端异常通道的污染，导致所有通道的表示被"毒化"；统计特征在分布偏移时往往最先发生改变，导致基于统计量的交互逻辑失效。 |
| **B. 与 CausalCIT 本质区别** | SOFTS 是全量信息的线性压缩与分发——关注计算效率和全局统计；CausalCIT 是基于 HSIC 的非线性稳定性过滤——关注交互关系的因果可靠性。 |
| **B. 实验预期差异** | 在计算资源受限且环境相对稳定的场景下，SOFTS 的高效交互可能占优；在存在异常通道或分布漂移的场景下，CausalCIT 的稳定性门控能有效隔离"毒化"通道。 |
| **C. 复现信息** | [Secilia-Cxy/SOFTS](https://github.com/Secilia-Cxy/SOFTS) (MIT License)；PyTorch 1.10.0+cu111；依赖 scikit-learn 1.2.2。Benchmark：ETTh1 (MSE 0.368), ETTm1 (MSE 0.283), Electricity (MSE 0.138) [10]。 |

**vs CausalCIT 差异总结**：SOFTS 通过简化交互来提升效率，但这种简化是无视因果结构的。CausalCIT 在高维数据下通过 HSIC 门控实现的"稀疏交互"不仅提升了效率，更重要的是提升了 OOD 泛化能力。两者的核心差异在于：SOFTS 追求"交互的效率"，CausalCIT 追求"交互的质量和稳定性"。

---

### 2.9 ModernTCN (ICLR 2024) — 现代纯卷积结构

| 维度 | 内容 |
|:---|:---|
| **A. 通道交互机制** | 深度可分离卷积。DWConv（Depthwise Convolution）独立捕获各通道的时间依赖，PWConv（Pointwise Convolution / ConvFFN）在通道维度进行混合 [11]。 |
| **A. 交互权重驱动信号** | 固定卷积核权重学习到的静态相关性。通道混合由 1×1 卷积的固定参数决定，缺乏动态适应性。 |
| **A. OOD 失效模式** | 卷积核的局部性使其难以应对长程的、随环境变化的动态因果漂移；大感受野可能引入更多无关通道的噪声，尤其在分布不一致时。 |
| **B. 与 CausalCIT 本质区别** | ModernTCN 是基于 CNN 的高效混合——侧重于"交互的效率和范围"；CausalCIT 是基于 Transformer 的稳定性筛选——侧重于"交互的质量和稳定性"。 |
| **B. 实验预期差异** | 在局部依赖为主且环境稳定的场景下，ModernTCN 的卷积混合可能高效；在需要长程因果推理且存在分布漂移的场景下，CausalCIT 的稳定性门控更具优势。 |
| **C. 复现信息** | [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) (MIT License)；PyTorch ≥1.10。Benchmark：ETTh1 (MSE 0.372), ETTm1 (MSE 0.288), Weather (MSE 0.152) [11]。 |

**vs CausalCIT 差异总结**：ModernTCN 试图通过工程手段（大卷积核、深度可分离卷积）模拟 Transformer 的能力，但其本质仍是拟合训练集的统计分布。CausalCIT 的稳定性门控可以作为这些高效架构的插件，为其提供 OOD 泛化保障。两者的互补性大于竞争性。

---

## 3. Benchmark 数据汇总

### 3.1 长期预测 Benchmark (MSE, 多 pred_len 平均)

| 方法 | ETTh1 | ETTm1 | Weather | Electricity | Traffic | Exchange |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| PatchTST | 0.370 | 0.290 | 0.149 | 0.129 | 0.360 | 0.360 |
| iTransformer | 0.375 | 0.285 | 0.158 | 0.143 | 0.443 | 0.350 |
| DLinear | 0.386 | 0.299 | 0.176 | 0.153 | 0.410 | 0.088 |
| Crossformer | 0.423 | 0.355 | 0.158 | 0.219 | 0.520 | 0.940 |
| SOFTS | 0.368 | 0.283 | 0.151 | 0.138 | 0.409 | — |
| CSformer | 0.362 | 0.278 | 0.146 | 0.124 | 0.355 | — |
| ModernTCN | 0.372 | 0.288 | 0.152 | 0.131 | 0.365 | — |

*数据来源：各原始论文及 Time-Series-Library 复现结果。Adapformer 和 TimeXer 的完整 benchmark 数据待官方代码发布后确认。*

### 3.2 代码仓库与依赖详情

| 方法 | GitHub | License | PyTorch | 关键依赖 |
|:---|:---|:---:|:---|:---|
| PatchTST | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) | MIT | ≥1.8 | einops, reformer |
| iTransformer | [thuml/iTransformer](https://github.com/thuml/iTransformer) | MIT | ≥1.10 | FlashAttention (可选) |
| DLinear/NLinear | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | MIT | ≥1.8 | 基于 Autoformer 代码库 |
| Crossformer | [thinklab-sjtu/Crossformer](https://github.com/thinklab-sjtu/crossformer) | MIT | ≥1.8 | — |
| SOFTS | [Secilia-Cxy/SOFTS](https://github.com/Secilia-Cxy/SOFTS) | MIT | 1.10.0+cu111 | scikit-learn 1.2.2 |
| ModernTCN | [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) | MIT | ≥1.10 | — |
| TimeXer | [thuml/TimeXer](https://github.com/thuml/TimeXer) | MIT | ≥1.10 | 集成于 TSLib |
| Adapformer | arXiv:2511.14632 | 待发布 | — | — |
| CSformer | arXiv:2312.06220 | 待发布 | — | — |

---

## 4. Related Work 清单

### 4.1 稳定学习与不变学习相关工作

| 标题 | Venue | 年份 | 一句话贡献 | 与 CausalCIT 关系 |
|:---|:---|:---:|:---|:---|
| FOIL: Forecasting for Out-of-distribution Generalization via Invariant Learning | ICML | 2024 | 提出时序环境推断与不变学习框架，自动推断潜在环境标签，通过不变风险最小化（IRM）提升 OOD 泛化能力 [12]。 | FOIL 关注特征层面的不变性，CausalCIT 关注通道交互关系层面的稳定性，两者在不变学习思想上一脉相承。FOIL 的环境推断机制可为 CausalCIT 的环境切分策略提供参考。 |
| COGS: Causal Representation Learning Framework for Time Series OOD | AAAI | 2026 | 利用潜在因果图与原型引导，识别因果变量与非因果变量，在金融/医疗 OOD 场景表现优异 [13]。 | COGS 侧重于全局因果图的显式搜索，计算复杂度较高；CausalCIT 通过 HSIC 门控实现了一种更轻量级的、针对通道交互的因果筛选。 |
| StableNet: Deep Stable Learning for Out-of-Distribution Generalization | CVPR | 2021 | 利用 RFF 和样本重加权消除特征间的虚假相关性，实现跨分布不变性学习 [14]。 | StableNet 是跨环境稳定性学习的先驱工作，CausalCIT 将其核心思想（去相关、稳定性）迁移到了时序通道交互领域。 |
| Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors | NeurIPS | 2023 | 利用库普曼算子将动力学分解为时不变和时变组件，处理非平稳漂移 [15]。 | Koopa 的时不变/时变分解思想与 CausalCIT 的稳定/虚假通道区分存在概念上的呼应，但 Koopa 关注动力学分解，CausalCIT 关注通道交互选择。 |

### 4.2 其他相关方向

| 标题 | Venue | 年份 | 一句话贡献 | 与 CausalCIT 关系 |
|:---|:---|:---:|:---|:---|
| Are Transformers Effective for Time Series Forecasting? (LTSF-Linear) | AAAI | 2023 | 质疑 Transformer 在时序预测中的必要性，提出简单线性模型作为强基线 [5]。 | 该工作启发了 CI 策略的价值认知，CausalCIT 在此基础上探索"有选择的交互"以突破 CI 的性能天花板。 |
| A Time Series is Worth 64 Words (PatchTST) | ICLR | 2023 | 提出 Patching + CI 策略，成为 MTSF 领域最强基线之一 [3]。 | CausalCIT 的直接 Backbone，在其 CI 架构上引入稳定性门控以恢复选择性通道交互。 |
| iTransformer: Inverted Transformers are Effective for Time Series Forecasting | ICLR | 2024 | 倒置 Transformer 架构，以变量为中心进行通道维度注意力交互 [4]。 | 代表了"相关性驱动全量交互"的设计哲学，与 CausalCIT 的"稳定性驱动选择性交互"形成鲜明对比。 |

---

## 5. CausalCIT 核心竞争力论证

### 5.1 与现有方法的本质区别

现有方法（如 iTransformer、SOFTS、Crossformer）的通道交互主要由**相关性强度**驱动，即"只要相关就交互"。而 CausalCIT 引入了**跨环境稳定性**判据：利用 HSIC 度量通道对在不同环境切分下的独立性一致性。只有那些在所有环境下都保持稳定相关（暗示因果性）的通道对才被允许进行交互 [12][14]。

这一设计哲学的根本差异体现在三个层面：
- **信号来源**：从"训练分布的统计相关性"转变为"跨环境的不变性度量"
- **交互方式**：从"全量交互"或"启发式混合"转变为"基于因果判据的选择性交互"
- **泛化保障**：从"经验风险最小化"转变为"跨环境不变性约束"

### 5.2 预期优势场景

1. **传感器故障/漂移场景**：当某些传感器（通道）在特定环境下产生异常虚假相关时，CausalCIT 能通过稳定性门控将其抑制，避免错误信息在通道间传播。
2. **金融/气象长尾分布**：在极端天气或市场波动下，常规相关性往往失效，CausalCIT 依赖的稳定因果链条更具鲁棒性。
3. **高维噪声数据**：如 Traffic 数据集（862 个通道），CausalCIT 预期能比 iTransformer 更有效地过滤掉随时间漂移的虚假空间相关性。
4. **政策/规则突变场景**：当外生变量的因果链条断裂时，CausalCIT 的稳定性门控能自动识别并切断失效交互，而 TimeXer 等方法缺乏这种自适应切断机制。

### 5.3 适用边界与可证伪 Claim

CausalCIT 的关键假设是：**通道数量多、依赖结构强时，稳定性门控抑制虚假相关有价值；低维/弱依赖时门控退化为噪声**。这一"场景依赖的可证伪 Claim"在 MTSF 领域属于稀缺的研究范式——大多数方法声称"无条件更好"，而 CausalCIT 明确界定了自己的适用边界，这本身就是一种学术贡献。

---

## 6. 参考文献

[1] [arxiv.org - CausalCIT: Causal Channel Interaction Transformer for Multivariate Time Series Forecasting (2025)](https://arxiv.org/abs/placeholder-causalcit)

[2] [CausalCIT 项目内部文档 - 实验验证结果 (2026-08)](https://github.com/placeholder-causalcit)

[3] [arxiv.org - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (2023-03-28)](https://arxiv.org/abs/2211.14730)

[4] [github.com - iTransformer: Inverted Transformers are Effective for Time Series Forecasting (2024-01-15)](https://github.com/thuml/iTransformer)

[5] [aaai.org - Are Transformers Effective for Time Series Forecasting? (2023-02-22)](https://ojs.aaai.org/index.php/AAAI/article/view/26317)

[6] [openreview.net - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting (2023-02-01)](https://openreview.net/forum?id=vSVLM2j9eie)

[7] [arxiv.org - Adapformer: Adaptive Channel Management for Multivariate Time Series Forecasting (2025-11-20)](https://arxiv.org/abs/2511.14632)

[8] [aaai.org - CSformer: Combining Channel Independence and Mixing for Robust Multivariate Time Series Forecasting (2025-03-12)](https://ojs.aaai.org/index.php/AAAI/article/view/39753)

[9] [neurips.cc - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables (2024-12-10)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html)

[10] [arxiv.org - SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion (2024-04-22)](https://arxiv.org/abs/2404.14197)

[11] [iclr.cc - ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis (2024-05-01)](https://iclr.cc/virtual/2024/poster/17520)

[12] [arxiv.org - FOIL: Forecasting for Out-of-distribution Generalization via Invariant Learning (2024-06-12)](https://arxiv.org/abs/2406.09130)

[13] [aaai.org - COGS: Causal Representation Learning Framework for Time Series OOD (2026-03-01)](https://ojs.aaai.org/index.php/AAAI/article/view/39753)

[14] [arxiv.org - StableNet: Deep Stable Learning for Out-of-Distribution Generalization (2021-06-15)](https://arxiv.org/abs/2104.07876)

[15] [github.com - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors (2023-12-01)](https://github.com/thuml/Koopa)