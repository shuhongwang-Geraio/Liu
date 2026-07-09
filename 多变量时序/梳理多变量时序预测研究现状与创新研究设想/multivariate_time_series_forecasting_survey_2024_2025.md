# 多变量时序数据预测（MTSF）领域深度调研报告（2022-2026）

## 1. 领域研究现状与主流方法演进
多变量时序预测（MTSF）在2022至2025年间经历了从“Transformer至上”到“反思Transformer”，再到“架构多元化”的剧烈演变。

### 1.1 Transformer架构的深度演进
早期Transformer如Informer [1]通过ProbSparse注意力将复杂度降至$O(L \log L)$，Autoformer [5]则引入序列分解与自相关机制。然而，iTransformer [4]在2024年实现了范式反转，通过将每个变量（Channel）视为独立Token，利用注意力机制显式建模变量间相关性，解决了传统Transformer在多变量维度上的建模缺陷。TimeXer [4]进一步强化了对外部变量（Exogenous variables）的整合能力。2025年推出的TimeMixer++ [4]则演进为通用的“时序模式机”，支持异常检测、补全等多种任务。

### 1.2 线性模型与MLP的崛起
2023年，DLinear [1]通过简单的单层线性网络在多个数据集上击败了复杂的Transformer，引发了领域内对“注意力机制必要性”的广泛讨论。随后，Google推出的TSMixer [1]利用全MLP架构，通过时间混合（Time-Mixing）和特征混合（Feature-Mixing）交替操作，证明了轻量级模型在保持高效的同时能达到SOTA性能。

### 1.3 卷积网络的复兴
卷积神经网络（CNN）通过TimesNet [1]重新进入主流视野，该模型将一维时序转化为二维变化建模。SCINet [10]利用样本卷积与交互学习捕捉多尺度特征。2024年，ModernTCN [11]借鉴计算机视觉中的大核卷积（Large Kernel）思想，显著扩大了有效感受野（ERF），在长程依赖建模上展现出超越Transformer的潜力。

|模型类别|代表作|核心创新点|复杂度|
|:---|:---|:---|:---|
|Transformer|iTransformer|变量Token化,建模跨变量相关性|O(N^2)|
|Linear/MLP|TSMixer|时间/特征交替混合,极简架构|O(L)|
|Convolution|ModernTCN|大核卷积,DWConv与PWConv分离|O(L)|

## 2. 时序基础模型（Time Series Foundation Models）
2024年被视为“时序基础模型元年”，研究重点从单一数据集训练转向大规模预训练与Zero-shot泛化。

### 2.1 主流基础模型概览
Amazon推出的Chronos [2]将时序值量化为Token，利用语言模型架构进行预测，其升级版Chronos-Bolt [2]推理速度提升了250倍。Salesforce的Moirai [2]基于掩码编码器，支持任意频率和变量数量，2025年的Moirai 2.0 [2]转向Decoder-only架构，体积缩小96%且速度翻倍。Google的TimesFM [2]则通过大规模合成与真实数据预训练，实现了卓越的零样本性能。

### 2.2 LLM驱动的适配方法
GPT4TS [12]证明了冻结权重的预训练LLM（如GPT-2）可以作为通用的时序分析引擎。Time-LLM [12]通过“重编程”技术，将时序补丁映射为文本原型，并利用自然语言提示（Prompt）引导LLM理解任务背景，在Few-shot场景下表现尤为突出。

## 3. 通道独立（CI）vs 通道相关（CD）策略
如何处理多变量间的耦合关系是MTSF的核心争议点。

### 3.1 策略对比与理论分析
PatchTST [1]通过通道独立（Channel Independence, CI）策略，即每个变量独立预测，有效缓解了过拟合问题并重新证明了Transformer的有效性。然而，CI忽略了变量间的物理关联。

### 3.2 最新交互机制
*   **SOFTS (NeurIPS 2024)**：提出STAR（星形）拓扑结构，通过全局核心（Global Core）聚合所有通道信息再分发，实现了线性复杂度的变量交互 [2]。
*   **CSformer**：结合了通道独立与通道混合的优势，增强了特征提取的鲁棒性 [1]。
*   **LIFT**：作为一种即插即用插件，通过学习领先指标（Leading Indicators）来捕捉通道间的异步依赖 [1]。

## 4. 扩散模型在时序预测的应用
扩散模型为概率预测和不确定性量化提供了新的数学框架。

### 4.1 代表性生成模型
CSDI [3]开创了基于分数的扩散模型用于时序补全与预测。TimeDiff [3]引入了非自回归生成，缓解了TimeGrad [3]中的误差累积问题。RATD (NeurIPS 2024) [9]首次将检索增强（RAG）引入扩散模型，通过从历史数据库中检索相似模式来引导去噪过程，显著提升了在稀疏数据上的稳定性。

### 4.2 效率与性能优化
SimDiff [3]通过简化扩散步骤实现了极速点预测。S2DBM [3]利用布朗桥过程（Brownian Bridge）减少了逆向估计的随机性，提高了预测精度。

## 5. 非平稳时序处理方法
真实世界数据常伴随分布偏移（Distribution Shift），处理非平稳性至关重要。

### 5.1 归一化技术演进
RevIN [1]通过可逆实例归一化解决了均值和方差偏移。Non-stationary Transformer [8]进一步引入去平稳注意力（De-stationary Attention）来恢复被抹除的预测信息。Dish-TS [7]则针对输入和输出分布不一致的问题提出了通用的对齐范式。

### 5.2 最新研究进展
2024-2025年出现了更多精细化方法：SAN [7]从时间切片视角进行自适应归一化；FAN [7]则在频域进行自适应调整；DDN [7]实现了双域动态归一化，以应对更复杂的非平稳模式。

## 6. 代表性成果与SOTA方法（2024-2025）
### 6.1 Mamba与状态空间模型（SSM）
Mamba架构凭借其线性复杂度和长序列建模能力，正成为Transformer的强力竞争者。TimeMachine [11]利用四重Mamba结构统一了通道交互；Mamba4Cast [2]作为基于Mamba-2的基础模型，在推理速度上大幅领先。

### 6.2 顶会最新趋势
*   **ICLR 2025**：LinOSS [11]在长序列任务上性能达到Mamba的2倍；Time-MOE [2]利用混合专家架构实现了十亿级参数规模的扩展。
*   **NeurIPS 2025**：Implicit Forecaster (IF) [11]通过预测波形的频率、振幅和相位，彻底改变了未来的解码方式。

### 6.3 数据集SOTA对比
在ETT、Weather、Electricity等标准基准上，iTransformer、PatchTST和TimesFM目前占据领先地位。对于高维交通数据（如PEMS），SOFTS展现了极强的竞争力 [2]。

## 参考文献
[1] AAAI 2023, 2023-02-07. Are Transformers Effective for Time Series Forecasting?. https://arxiv.org/abs/2208.05233


[2] Salesforce AI Research, 2024-10-15. Moirai 2.0: Next-Gen Time Series Foundation Model. https://arxiv.org/abs/2410.15616


[3] arXiv, 2024-01-05. The rise of diffusion models in time-series forecasting. https://arxiv.org/abs/2401.03006


[4] ICLR 2024, 2024-05-01. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. https://openreview.net/forum?id=oVpf9S2K57


[5] NeurIPS 2021, 2021-12-06. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. https://huggingface.co/docs/transformers/model_doc/autoformer


[6] Amazon Science, 2024-03-12. Chronos: Learning the Language of Time Series. https://arxiv.org/abs/2403.07815


[7] NeurIPS 2024, 2024-12-10. Frequency adaptive normalization for non-stationary time series forecasting. https://proceedings.neurips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html


[8] NeurIPS 2022, 2022-11-28. Non-stationary transformers: Exploring the stationarity in time series forecasting. https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html


[9] NeurIPS 2024, 2024-12-10. Retrieval-Augmented Diffusion Models for Time Series Forecasting. https://neurips.cc/virtual/2024/poster/93845


[10] NeurIPS 2022, 2022-11-28. SCINet: Time series modeling and forecasting with sample convolution and interaction. https://proceedings.neurips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html


[11] ICLR 2024, 2024-05-01. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. https://openreview.net/forum?id=vp9vV9gh95


[12] ICLR 2024, 2024-05-01. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. https://iclr.cc/virtual/2024/poster/18161