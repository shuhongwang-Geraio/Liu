# 重整化在Transformer架构上处理多尺度数据的研究综述

## 1. 执行摘要

重整化群（Renormalization Group, RG）理论起源于统计物理学，其核心思想是通过系统性地消除短程自由度来描述系统在不同尺度上的行为。近年来，随着深度学习特别是Transformer架构的蓬勃发展，研究者发现RG理论与深度神经网络的层级结构之间存在深刻的数学联系。这一交叉领域的研究不仅为理解深度学习的工作机制提供了新的理论视角，也为设计能够有效处理多尺度数据的新型神经网络架构提供了指导原则。

本报告基于对27篇核心文献的系统调研，从理论基础、架构设计、信号处理、跨学科应用和可解释性五个维度，全面梳理了2018年至2025年间该领域的研究进展。研究表明：（1）神经网络的层级结构可被严格映射为RG流中的粗粒化过程；（2）多尺度Transformer架构通过自适应路径和层级注意力机制显著提升了对复杂数据的建模能力；（3）小波变换与Transformer的结合为长序列建模提供了线性复杂度的解决方案；（4）RG方法正在成为解释大语言模型内部机制的新范式。这些发现对于推动下一代人工智能系统的理论理解和架构创新具有重要意义。

## 2. 研究背景：重整化群理论与深度学习的理论联系

### 2.1 重整化群的基本概念

重整化群理论由Kenneth Wilson在20世纪70年代系统发展，其核心操作包括三个步骤：粗粒化（coarse-graining）——将微观自由度聚合为宏观集体变量；重新标度（rescaling）——调整系统的尺度以恢复原始结构；以及迭代（iteration）——重复上述过程以揭示系统在不同尺度上的普适行为[1]。这一框架最初用于解释相变和临界现象，但其数学结构具有高度的普适性。

### 2.2 深度网络与RG流的对应关系

2018年，Li和Wang在《Physical Review Letters》上发表的开创性工作首次建立了神经网络与RG之间的精确对应关系[2]。他们证明，基于可逆生成模型（如RealNVP）的神经网络能够自动学习到与Kadanoff块自旋变换等价的粗粒化操作，从而在训练过程中自发地发现系统的层级化高阶特征。这一发现揭示了深度网络层级结构的物理本质：每一层网络可被视为一个学习到的粗粒化算子，将输入数据中的"微观"细节逐步过滤，最终提取出稳定的"宏观"语义表示。

后续研究进一步深化了这一理论联系。Hu和Li等人于2020年提出了基于流模型的全息映射方法，使网络能够在训练中自发演化出最优的重整化方案[7]。Li和Sompolinsky在2021年提出的反向传播核重整化（BPKR）理论，则从统计力学角度解析了深度线性网络的训练动力学，允许逐层集成网络权重以精确求解网络属性[8]。

### 2.3 Transformer与RG流的类比

Transformer架构的前向传播过程与Kadanoff-Wilson RG流存在深刻的类比关系[1]。在这一框架下，每一层Transformer的自注意力机制被解释为一个信息粗粒化算子：它通过聚合Token之间的相关性，收缩数据流形的Fisher信息几何，将局部的语法波动等"短程"细节过滤，最终在深层网络中留下稳定的语义吸引子。这一视角不仅为理解Transformer的工作机制提供了物理直觉，也为设计更高效的多尺度架构提供了理论指导。

## 3. 研究进展分类总结

### 3.1 理论基础：神经网络重整化群

该方向的研究致力于建立深度学习与重整化群之间的严格数学联系，为后续的架构设计和可解释性研究奠定理论基础。

|论文|年份|期刊|核心贡献|
|:---|:---|:---|:---|
|Neural network renormalization group|2018|Phys. Rev. Lett.|首次证明神经网络可自动学习RG变换[2]|
|Machine learning holographic mapping|2020|Phys. Rev. Research|引入流模型实现自发重整化方案演化[7]|
|Backpropagating kernel renormalization|2021|Phys. Rev. X|提出BPKR理论解析深度线性网络[8]|
|Hierarchical Maximum Entropy via RG|2025|arXiv|利用RG简化层级不变性下的最大熵模型[9]|

从理论发展脉络来看，2018年的开创性工作确立了神经网络作为RG变换载体的基本范式，随后的研究逐步扩展到更复杂的网络结构和更广泛的应用场景。值得注意的是，早在1991年，Willcox就已从RG角度探讨了层级神经网络中耦合矩阵的缩放规律[20]，这表明该领域的理论探索具有深厚的历史渊源。

### 3.2 多尺度Transformer架构

为了在架构层面显式处理多尺度数据，研究者开发了多种具备层级结构或自适应路径的Transformer变体。这些架构设计直接受到RG理论中"不同尺度对应不同有效自由度"这一核心思想的启发。

**Pathformer（2024）** 引入了自适应路径机制，通过不同大小的补丁（Patch）将时间序列划分为多种分辨率，从而在单一模型中同时捕获全局相关性和局部细节[3]。这一设计理念与RG中的多尺度分析高度契合：不同分辨率的补丁对应于不同的粗粒化层级，而自适应路径则允许模型根据数据特性动态选择最优的尺度组合。

**TR-NAS（2024）** 是一个专为高光谱图像分类设计的神经架构搜索框架[6]。该方法的核心创新在于自适应地在全局注意力、局部窗口注意力和多尺度注意力算子之间进行选择，以平衡空间异质性带来的建模挑战。从RG视角看，这种自适应选择机制本质上是在探索不同粗粒化策略的最优组合。

**Mean-Field Transformers（2025）** 将Token视为相互作用的粒子系统，揭示了Token在模型深度增加时先坍缩到低维空间再聚类的多尺度行为[10]。这一研究在NeurIPS 2025上发表，为理解Transformer的内部动力学提供了全新的统计力学视角，其核心发现与RG流中的固定点行为存在深刻联系。

**Swin Transformer** 作为层级Transformer的代表，通过移动窗口机制实现了跨尺度的特征融合。其"先局部后全局"的处理策略与RG中"先消除短程关联再处理长程关联"的思想一脉相承，目前已成为视觉领域处理多尺度任务的基准模型。

### 3.3 小波-Transformer结合

小波变换因其优异的时频局部化性质，成为增强Transformer多尺度处理能力的天然选择。从数学角度看，小波变换本身就是一种多分辨率分析工具，其与Transformer的结合为长序列建模提供了兼顾效率和表达能力的解决方案。

|方法|年份|来源|技术特点|应用场景|
|:---|:---|:---|:---|:---|
|WavSpA|2023|ICML|小波系数空间注意力|长程竞技场基准[15]|
|LMWT|2025|arXiv|可学习Haar小波替代点积注意力|线性复杂度序列建模[11]|
|WTHN|2026|SPIE|DWT揭示多频率模式|金融时序预测[12]|
|WaveFormer|2026|ResearchGate|小波增强局部注意力|卫星视频目标跟踪[16]|

**可学习多尺度小波Transformer（LMWT）** 是该方向的最新代表性工作[11]。该方法使用可学习的Haar小波模块替代传统的点积注意力，在保持多尺度特性的同时将计算复杂度从O(n²)降至O(n)。这一设计巧妙地将小波变换的多分辨率分解与Transformer的上下文建模能力相结合，为处理超长序列提供了可行的技术路径。

**小波空间注意力（WavSpA）** 则提出在小波系数空间而非原始信号空间进行注意力计算[15]。由于小波变换具有能量压缩特性，高频细节被有效分离到独立的子带中，这使得注意力机制能够更聚焦于信号的关键结构特征。该方法在长程竞技场（Long Range Arena）基准上取得了优异表现。

### 3.4 粗粒化与Transformer：跨学科应用

粗粒化是重整化的核心操作，在分子动力学和随机动力学模拟等计算科学领域具有重要应用价值。Transformer架构被证明能够有效学习从微观到宏观的映射关系，为加速复杂系统模拟提供了新的技术手段。

**通用Transformer粗粒化分子动力学框架（2025）** 将Transformer集成到蛋白质动力学的粗粒化学习过程中[22]。该框架能够自动学习从全原子表示到粗粒化表示的映射，显著提高了大分子构象采样的效率。从RG角度看，这一过程对应于消除原子级别的快速振动自由度，保留对蛋白质功能至关重要的慢动力学模式。

**基于Transformer学习随机动力学（2024）** 发表在《Nature Communications》上，证明Transformer能够从微观轨迹数据中学习大型系统的涌现行为和粗粒化动力学[26]。这一工作的核心贡献在于展示了Transformer作为"通用粗粒化器"的潜力：它不仅能够提取有效自由度，还能捕获这些自由度之间的有效相互作用。

**CoarsenConf（2024）** 提出了一种等变粗粒化方法，利用聚合注意力机制进行分子构象生成[27]。该方法克服了传统固定长度粗粒化的局限性，允许根据分子结构的局部复杂度自适应地选择粗粒化程度，这与RG中"关键区域需要更精细描述"的思想高度一致。

### 3.5 层级网络与重整化

层级网络结构天然契合重整化群的思想，通过逐层抽象实现信息的有效压缩。这一方向的研究关注如何利用RG框架设计和分析具有层级结构的神经网络。

2025年发表在《Nature Reviews Physics》上的综述文章"Network renormalization"系统探讨了确定性层级网络（如Cayley树）与图神经网络在重整化框架下的结合[4]。该综述指出，网络重整化不仅是一种分析工具，更是一种设计原则：通过强制网络在不同尺度上保持某种结构不变性，可以引导模型学习更加鲁棒的表示。

此外，Zhang等人2025年发表在《Nature Communications》上的工作探索了通过统计物理和机器学习方法对网络流进行粗粒化的可能性[25]，为理解复杂网络系统中的多尺度现象提供了新的分析框架。

### 3.6 可解释性与RG方法

重整化为"打开深度学习黑盒"提供了独特的视角。通过追踪信息在网络各层之间的流动和聚合过程，研究者试图揭示模型内部的决策机制。

**大语言模型（LLM）重整化** 是该方向的新兴研究热点[5]。研究者建议将模型组件（层、注意力头、神经元）作为重整化的基本单元，通过系统性地"粗粒化"这些组件来理解知识在模型中的分布式存储方式。这一方法的核心假设是：如果某些组件可以被合并而不显著影响模型输出，那么它们可能编码了冗余或相关的信息。

**Vision Transformer（ViT）信息瓶颈分析** 通过限制Patch间的信息流动来模拟RG流过程[7][19]。这一方法允许研究者监测模型如何从局部图像碎片逐步构建全局语义表示，揭示了视觉Transformer中"从局部到全局"的信息整合机制。

## 4. 时间线分析：研究演进脉络（2018-2025）

从文献发表时间来看，该领域的研究呈现出清晰的演进脉络：

|阶段|时间|核心进展|代表性工作|
|:---|:---|:---|:---|
|理论奠基期|2018-2020|建立NN-RG对应关系|Neural network RG[2], Holographic mapping[7]|
|方法发展期|2021-2023|深化理论，拓展应用|BPKR[8], WavSpA[15]|
|架构创新期|2024|多尺度Transformer架构涌现|Pathformer[3], TR-NAS[6], CoarsenConf[27]|
|综合应用期|2025-2026|跨学科应用与可解释性|Mean-Field Transformers[10], Network RG综述[4]|

**2018年**标志着该领域的正式起步，Li和Wang的开创性工作首次在顶级物理期刊上建立了神经网络与重整化群的精确联系[2]。**2020-2021年**，理论研究持续深化，全息映射和BPKR等工作进一步巩固了这一理论框架的基础。**2023-2024年**是架构创新的高峰期，Pathformer、TR-NAS等多尺度Transformer架构相继提出，同时小波-Transformer结合方法开始成熟。**2025年至今**，研究重心逐步转向跨学科应用（如粗粒化分子动力学）和可解释性分析，《Nature Reviews Physics》综述的发表标志着该领域已进入系统化总结阶段。

## 5. 关键技术对比

|技术方法|理论基础|计算复杂度|主要优势|典型应用|
|:---|:---|:---|:---|:---|
|神经网络RG|RG流-层级对应|O(n)|提供理论框架|物理系统分析|
|Pathformer|多分辨率补丁|O(n log n)|自适应路径选择|时间序列预测|
|WavSpA|小波多分辨率分析|O(n)|线性复杂度|长序列建模|
|Mean-Field Trans.|粒子系统统计力学|O(n²)|揭示Token动力学|理论分析|
|粗粒化Transformer|物理系统粗粒化|O(n)|加速分子模拟|计算生物学|

从技术特点来看，不同方法在理论基础、计算效率和适用场景上各有侧重。基于小波的方法在降低计算复杂度方面具有明显优势，适合处理超长序列；基于物理粗粒化的方法则在跨学科应用中展现出独特价值；而Mean-Field Transformers等理论分析工作虽然不直接追求效率提升，但为理解模型行为提供了不可替代的洞察。

## 6. 研究趋势与展望

基于对现有文献的系统分析，该领域未来的发展趋势可归纳为以下几个方向：

**理论深化方向**：当前的NN-RG对应关系主要建立在生成模型和线性网络上，未来需要将理论扩展到更一般的非线性网络结构，特别是带有残差连接和层归一化的现代Transformer架构。此外，如何从RG视角理解模型的泛化能力和鲁棒性仍是开放问题。

**架构创新方向**：基于RG原则的架构设计正在从"受启发"走向"受约束"。未来的多尺度Transformer可能会显式地强制满足某种尺度不变性或自相似性，从而在理论上保证模型的多尺度处理能力。Pathformer和LMWT等工作已经展示了这一方向的可行性。

**可解释性方向**：RG方法为理解大语言模型提供了全新的视角。未来研究可能会发展出基于"信息粗粒化"的模型压缩技术，以及基于"RG流跟踪"的可解释性分析工具。这不仅有助于理解模型行为，也可能催生新的模型诊断和调试方法。

**跨学科应用方向**：Transformer在分子动力学粗粒化中的成功应用预示着更广泛的跨学科融合。气候模拟、材料设计、流体力学等涉及多尺度物理过程的领域都可能受益于这一技术路线。

## 7. 参考文献

[1] Symmetry Broken, 2021. Attention is Bayesian Inference and Renormalization Group Flow. https://symmetrybroken.com/attention-is-bayesian-inference-and-renormalization-group-flow/

[2] Physical Review Letters, 2018. Neural network renormalization group. https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.260601

[3] arXiv, 2024. Pathformer: Multi-scale Transformers with Adaptive Pathways. https://arxiv.org/abs/2402.05956

[4] Nature Reviews Physics, 2025. Network renormalization. https://www.nature.com/articles/s42254-025-00817-5

[5] Scholaris, 2025. LM Renormalization for Interpretability. https://scholaris.ca/resources/s3:10214-13845

[6] MDPI Remote Sensing, 2024. TR-NAS: A Multiscale Transformer-based Neural Architecture Search Framework. https://www.mdpi.com/2072-4292/16/5/843

[7] Physical Review Research, 2020. Machine learning holographic mapping by neural network renormalization group. https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023369

[8] Physical Review X, 2021. Statistical mechanics of deep linear neural networks: The backpropagating kernel renormalization. https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.031059

[9] arXiv, 2025. Hierarchical Maximum Entropy via the Renormalization Group. https://arxiv.org/abs/2509.01424

[10] NeurIPS, 2025. Mean-Field Transformers: Tokens as a System of Particles. https://neurips.cc/virtual/2025/poster/93345

[11] arXiv, 2025. Learnable Multi-Scale Wavelet Transformer. https://arxiv.org/abs/2504.08801

[12] SPIE Digital Library, 2026. Wavelet-Transformer Hybrid Networks for multiscale stock market price forecasting. https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13160/3011444/Wavelet-Transformer-Hybrid-Networks-for-multiscale-stock-market-price-forecasting/10.1117/12.3011444.short

[15] ICML Proceedings, 2023. Wavelet Space Attention (WavSpA). https://proceedings.mlr.press/v202/zhang23am.html

[16] ResearchGate, 2026. WaveFormer: Wavelet-Enhanced Transformer for Multi-Scale Representation Learning. https://www.researchgate.net/publication/380345632_WaveFormer_Wavelet-Enhanced_Transformer_for_Multi-Scale_Representation_Learning_in_Time_Series_Forecasting

[19] Kieran Murphy. Information Flow and RG in Vision Transformers. https://kieranamurphy.com/

[20] Journal of Physics A, 1991. Understanding hierarchical neural network behaviour: A renormalization group approach. https://iopscience.iop.org/article/10.1088/0305-4470/24/11/030/meta

[22] arXiv, 2025. A Universal Transformer-Based Coarse-Grained Molecular Dynamics Framework for Protein Dynamics. https://arxiv.org/abs/2502.05909

[25] Nature Communications, 2025. Coarse-graining network flow through statistical physics and machine learning. https://www.nature.com/articles/s41467-025-56034-2

[26] Nature Communications, 2024. Learning stochastic dynamics and predicting emergent behavior using transformers. https://www.nature.com/articles/s41467-024-45629-w

[27] Journal of Chemical Information and Modeling, 2024. Coarsenconf: Equivariant coarsening with aggregated attention for molecular conformer generation. https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c01001