# 多变量时序预测创新研究方向深度调研报告

## 1. 自适应通道交互机制 (Adaptive CI/CD Strategy)

### 1.1 背景介绍
在多变量时序预测（MTSF）中，如何处理变量间的依赖关系是核心挑战。传统的通道独立（Channel-Independent, CI）策略将每个变量视为独立序列，虽能缓解分布偏移但忽略了跨变量相关性；而通道相关（Channel-Dependent, CD）策略虽能建模交互，却易引入噪声导致过拟合 [16]。自适应通道交互机制旨在通过可学习的架构动态平衡两者。

### 1.2 最新相关论文 (2023-2025)
* [1] ICLR 2024. iTransformer: Inverted Transformers are Effective for Time Series Forecasting.
* [2] IEEE ICASSP 2024. CGN: A simple yet effective multi-channel gated network.
* [3] KDD 2023. TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting.
* [4] NeurIPS 2023. CrossGNN: Confronting Noisy Multivariate Time Series via Cross Interaction Refinement.
* [5] IEEE TKDE 2024. The capacity and robustness trade-off: Revisiting the channel independent strategy.
* [6] arXiv 2024. CMamba: Channel correlation enhanced state space models.

### 1.3 技术路径分析
* **维度反转与全局交互**：iTransformer 通过将维度反转，使 Transformer 的注意力机制作用于变量维度而非时间维度，从而在 CI 骨干上实现 CD 效果 [1]。
* **门控动态选择**：CGN 设计了通道门控单元（Channel Gate），通过公式 $Y = G \cdot f_{CD}(X) + (1-G) \cdot f_{CI}(X)$ 实现策略的软切换，其中 $G$ 是基于输入特征学习的权重 [4]。
* **图结构自适应学习**：利用动态图卷积网络，根据时序片段的相似度实时演化邻接矩阵 $A_t$，实现非平稳的变量关系建模 [11]。

### 1.4 潜在挑战与实验建议
**挑战**：高维变量下的计算复杂度呈平方增长；动态权重在噪声数据下的不稳定性。
**实验方案**：
* **数据集**：Traffic, Weather, Solar-Energy。
* **评估指标**：MSE, MAE。
* **基线模型**：PatchTST, DLinear, iTransformer。

---

## 2. 因果时序预测 (Causal Time Series Forecasting)

### 2.1 背景介绍
相关性不等于因果性。在非平稳环境下，变量间的统计相关性会随时间改变，而因果结构通常保持稳定。因果时序预测通过发现变量间的因果图（DAG），利用结构不变性提升预测的鲁棒性 [3]。

### 2.2 最新相关论文 (2023-2025)
* [7] ICLR 2024. CausalTime: Realistically generated time-series for benchmarking.
* [8] IEEE 2023. CausalFormer: Causal discovery-based transformer.
* [9] Nature Reviews 2023. Causal inference for time series.
* [10] arXiv 2023. CUTS: Neural causal discovery from irregular time-series data.
* [11] ACM Computing Surveys 2024. Causal discovery from temporal data: An overview.
* [12] IEEE Access 2025. Causal-Aware Multimodal Transformer for Supply Chain Demand.

### 2.3 技术路径分析
* **因果掩码注意力**：CausalFormer 利用 Granger 因果检验预先构建因果矩阵 $M$，并将其作为掩码应用于 Transformer 的注意力层：$Attn(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d}} \odot M)V$ [2]。
* **不变表示学习**：通过干预增强（Interventional Augmentation）学习环境不变特征，确保预测模型在分布偏移下依然有效 [1]。

### 2.4 潜在挑战与实验建议
**挑战**：观测数据中存在未观测混杂因素；因果发现算法在大规模变量下的搜索空间爆炸。
**实验方案**：
* **数据集**：CausalTime 合成集, PhysioNet 医疗数据。
* **评估指标**：SHD (Structural Hamming Distance), RMSE。
* **基线模型**：TFT, DeepAR, CausalFormer。

---

## 3. 高效扩散模型用于时序预测 (Efficient Diffusion)

### 3.1 背景介绍
扩散模型（Diffusion Models）通过逆转噪声注入过程生成概率分布，解决了时序预测中的不确定性建模问题。然而，其推理速度慢是主要瓶颈。

### 3.2 最新相关论文 (2023-2025)
* [13] ICLR 2024. mr-Diff: Multi-resolution Diffusion Model for Time Series Forecasting.
* [14] arXiv 2024. TSFlow: Flow Matching with Gaussian Process Priors.
* [15] ICLR 2025. Consistency models made easy.
* [16] AAAI 2025. ARMD: Auto-Regressive Moving Diffusion.
* [17] arXiv 2025. Non-stationary diffusion for probabilistic time series forecasting.
* [18] KDD 2025. Stochastic Diffusion: A Diffusion Based Model for Time Series.

### 3.3 技术路径分析
* **流匹配（Flow Matching）**：TSFlow 放弃了复杂的 SDE 框架，采用确定性的概率流路径，通过 $x_t = (1-t)x_0 + tx_1$ 实现更直的采样轨迹，大幅减少采样步数 [12]。
* **多分辨率去噪**：mr-Diff 采用季节性-趋势分解，先在低分辨率下生成趋势，再逐步细化高频细节，实现了“由易到难”的生成过程 [17]。

### 3.4 潜在挑战与实验建议
**挑战**：一步生成（One-step）下的精度损失；长序列生成的累积误差。
**实验方案**：
* **数据集**：Electricity, Exchange-Rate。
* **评估指标**：CRPS (Continuous Ranked Probability Score), 推理延迟。
* **基线模型**：TimeGrad, CSDI, TSFlow。

---

## 4. 分布外泛化与非平稳性建模 (OOD & Non-stationarity)

### 4.1 背景介绍
现实世界中的时序数据具有强烈的非平稳性，表现为均值和方差的动态漂移。传统的离线训练模型难以应对测试阶段出现的“新常态”。

### 4.2 最新相关论文 (2023-2025)
* [19] AAAI 2025. Battling the non-stationarity via test-time adaptation (TAFAS).
* [20] NeurIPS 2024. Test-time adaptation in non-stationary environments via adaptive alignment.
* [21] IEEE 2024. TimeDRL: Disentangled representation learning for multivariate time-series.
* [22] AAAI 2023. Dish-TS: A general paradigm for alleviating distribution shift.
* [23] arXiv 2026. Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts.
* [24] arXiv 2025. Accurate parameter-efficient test-time adaptation for time series.

### 4.3 技术路径分析
* **测试时适应（TTA）**：TAFAS 在推理阶段利用当前观测到的窗口数据，通过最小化自监督损失（如掩码重建）在线更新模型参数，实现即时适应 [15]。
* **可逆实例归一化（RevIN）**：通过在输入端减去均值并除以标准差，在输出端还原，消除实例间的分布差异 [22]。

### 4.4 潜在挑战与实验建议
**挑战**：在线更新导致的参数崩塌；缺乏真实标签时的无监督适应稳定性。
**实验方案**：
* **数据集**：ETTh1, ETTm2 (含明显概念漂移)。
* **评估指标**：Online MSE, 适应速度。
* **基线模型**：Non-stationary Transformer, RevIN+PatchTST。

---

## 5. 状态空间模型与Transformer混合架构 (SSM-Transformer Hybrid)

### 5.1 背景介绍
Transformer 的二次复杂度限制了其在超长序列（LTSF）中的应用。Mamba 等选择性状态空间模型（SSM）提供了线性复杂度，但对复杂语义的建模能力略逊。混合架构旨在结合两者的长处。

### 5.2 最新相关论文 (2023-2025)
* [25] ECAI 2024. TimeMachine: A Time Series is Worth 4 Mambas.
* [26] ICLR 2025. Linear Oscillatory State-Space Models (LinOSS).
* [27] AI21 Labs 2024. Jamba 1.5 Technical Report.
* [28] arXiv 2024. Mambaformer in Time Series.
* [29] arXiv 2025. SST: Multi-Scale Hybrid Mamba-Transformer Experts.
* [30] IEEE 2025. TSM-ATTN: A Hybrid Time Series Forecasting Model.

### 5.3 技术路径分析
* **多尺度Mamba集成**：TimeMachine 利用四个 Mamba 块分别处理不同尺度的上下文，通过线性扫描替代注意力机制，将显存占用降低了 80% 以上 [16]。
* **交替堆叠架构**：Jamba 风格的混合层在每 7 层 Mamba 后插入 1 层 Transformer，利用 Transformer 修正 SSM 的长期记忆衰减 [18]。

### 5.4 潜在挑战与实验建议
**挑战**：SSM 在多变量通道混合时的信息瓶颈；混合架构的超参数搜索空间巨大。
**实验方案**：
* **数据集**：Traffic (长序列), PEMS04。
* **评估指标**：Memory Usage, MSE, Inference Throughput。
* **基线模型**：Mamba, PatchTST, TimeMachine。

---

## 参考文献

[1] ICLR, 2024-05. Causaltime: Realistically generated time-series for benchmarking of causal discovery. https://proceedings.iclr.cc/paper_files/paper/2024/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html

[2] IEEE, 2023-12. CausalFormer: Causal discovery-based transformer for multivariate time series forecasting. https://ieeexplore.ieee.org/abstract/document/10373365/

[3] Nature Reviews Earth & Environment, 2023-04. Causal inference for time series. https://www.nature.com/articles/s43017-023-00431-y

[4] IEEE ICASSP, 2024-03. Cgn: A simple yet effective multi-channel gated network for long-term time series forecasting. https://ieeexplore.ieee.org/abstract/document/10448209/

[5] arXiv, 2024-03. TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting. https://arxiv.org/abs/2403.09898

[6] ICLR, 2025-01. Linear Oscillatory State-Space Models (LinOSS). https://openreview.net/forum?id=Ai8Hw3AXqks

[7] AI21 Labs, 2024-08. Jamba 1.5 Technical Report. https://arxiv.org/abs/2408.12570

[8] NeurIPS, 2023-12. Encoding time-series explanations through self-supervised model behavior consistency. https://proceedings.neurips.cc/paper_files/paper/2023/hash/65ea878cb90b440e8b4cd34fe0959914-Abstract-Conference.html

[9] AAAI, 2025-02. Battling the non-stationarity in time series forecasting via test-time adaptation (TAFAS). https://ojs.aaai.org/index.php/AAAI/article/view/33965

[10] arXiv, 2026-02. Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets. https://arxiv.org/abs/2602.00073

[11] Expert Systems with Applications, 2024-01. Dynamic multi-fusion spatio-temporal graph neural network for multivariate time series forecasting. https://www.sciencedirect.com/science/article/pii/S0957417423032311

[12] arXiv, 2024-10. Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting (TSFlow). https://arxiv.org/abs/2410.03024

[13] ICLR, 2025-01. Consistency models made easy. https://proceedings.iclr.cc/paper_files/paper/2025/hash/bb166dd4de5dba363bf1023eb956a826-Abstract-Conference.html

[14] IEEE, 2024-06. Timedrl: Disentangled representation learning for multivariate time-series. https://ieeexplore.ieee.org/abstract/document/10597874/

[15] AAAI, 2025-02. Battling the non-stationarity in time series forecasting via test-time adaptation. https://ojs.aaai.org/index.php/AAAI/article/view/33965

[16] ECAI, 2024-10. TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting. https://arxiv.org/abs/2403.09898

[17] ICLR, 2024-05. mr-Diff: Multi-resolution Diffusion Model for Time Series Forecasting. https://iclr.cc/virtual/2024/poster/18144

[18] AI21 Labs, 2024-08. Jamba: A Hybrid Transformer-Mamba Language Model. https://www.ai21.com/jamba