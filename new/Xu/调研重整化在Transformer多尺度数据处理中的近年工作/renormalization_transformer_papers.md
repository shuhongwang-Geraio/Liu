# 重整化与Transformer架构处理多尺度数据文献调研报告

## 1. 理论基础：重整化群与深度学习的联系

重整化群（Renormalization Group, RG）理论为理解深度学习中的多尺度特征提取提供了坚实的物理基础。研究表明，深度神经网络的层级结构与RG流中的尺度变换存在深刻的对应关系。

### 1.1 核心理论文献

| 标题 | 作者 | 年份 | 期刊/会议 | 主要贡献 |
|:---|:---|:---|:---|:---|
| Neural network renormalization group | SH Li, L Wang | 2018 | Physical Review Letters | 提出基于可逆生成模型的RG方法，证明神经网络能自动提取层级化的高阶特征 [2]。 |
| Machine learning holographic mapping by neural network renormalization group | HY Hu, SH Li等 | 2020 | Physical Review Research | 引入流模型构建层级深度网络，使网络在训练中自发演化出最优的重整化方案 [7]。 |
| Statistical mechanics of deep linear neural networks: The backpropagating kernel renormalization | Q Li, H Sompolinsky | 2021 | Physical Review X | 提出反向传播核重整化（BPKR），允许逐层集成网络权重以解决网络属性 [8]。 |

### 1.2 深度网络与RG流的类比
近年来的研究进一步确立了Transformer前向传播与Kadanoff-Wilson RG流的类比关系。每一层Transformer可以被视为一个学习到的粗粒化算子，通过收缩数据流形的Fisher信息几何，将局部的“微观”细节（如语法波动）过滤，最终留下稳定的“宏观”语义吸引子 [1]。

## 2. 多尺度Transformer架构

为了在架构层面显式处理多尺度数据，研究者开发了多种具备层级结构或自适应路径的Transformer变体。

### 2.1 代表性架构

*   **Pathformer (2024)**: 该模型引入了自适应路径机制，根据时间动态调整建模过程。它通过不同大小的补丁（Patch）将时间序列划分为多种分辨率，从而同时捕获全局相关性和局部细节 [3]。
*   **TR-NAS (2024)**: 这是一个专为高光谱图像分类设计的神经架构搜索框架。它能自适应地在全局、局部窗口和多尺度注意力算子之间进行选择，以平衡空间异质性 [6]。
*   **Mean-Field Transformers (2025)**: 在NeurIPS 2025上提出的研究将Token视为粒子系统，揭示了Token在模型深度增加时先坍缩到低维空间再聚类的多尺度行为 [4]。
*   **Swin Transformer**: 作为层级Transformer的代表，通过移动窗口机制实现了跨尺度的特征融合，是目前视觉领域处理多尺度任务的基准模型。

## 3. 小波变换与Transformer结合

小波变换（Wavelet Transform）因其优异的时频局部化性质，被广泛用于增强Transformer处理多尺度信号的能力，且能有效降低计算复杂度。

### 3.1 关键研究成果

| 标题 | 作者 | 年份 | 平台/会议 | 主要贡献 |
|:---|:---|:---|:---|:---|
| Learnable Multi-Scale Wavelet Transformer (LMWT) | - | 2025 | arXiv | 使用可学习的Haar小波模块替代点积注意力，实现线性复杂度并保持多尺度特性 [11]。 |
| Wavelet-Transformer Hybrid Networks (WTHN) | - | 2026 | SPIE | 利用DWT揭示金融时序的不同频率模式，预测精度提升显著 [12]。 |
| Wavelet Space Attention (WavSpA) | Zhang等 | 2023 | MLR Press | 在小波系数空间进行注意力计算，在长程竞技场（LRA）基准上表现优异 [15]。 |
| WaveFormer (WAETrack) | - | 2026 | ResearchGate | 增强局部注意力以处理卫星视频中的微小目标和模糊特征 [16]。 |

## 4. 粗粒化(Coarse-graining)与Transformer

粗粒化是重整化的核心步骤，在分子动力学和随机动力学模拟中，Transformer被用于学习从微观到宏观的映射。

### 4.1 跨学科应用

*   **Universal Transformer-Based Coarse-Grained Molecular Dynamics (2025)**: 该框架将Transformer集成到蛋白质动力学的粗粒化学习过程中，提高了大分子构象采样的效率 [22]。
*   **Learning stochastic dynamics using transformers (2024)**: 发表在Nature Communications上的研究证明，Transformer能够学习大型系统的涌现行为和粗粒化动力学 [26]。
*   **CoarsenConf (2024)**: 提出了一种等变粗粒化方法，利用聚合注意力进行分子构象生成，解决了固定长度粗粒化的局限性 [27]。

## 5. 层级网络与重整化

层级网络结构天然契合重整化群的思想，通过逐层抽象实现信息的有效压缩。

### 5.1 结构化重整化研究

*   **Network renormalization (2025)**: 发表在Nature Reviews Physics上的综述，探讨了确定性层级网络（如Cayley树）与图神经网络在重整化框架下的结合 [4]。
*   **Hierarchical Maximum Entropy via the Renormalization Group (2025)**: 探索了具有层级不变性的设置，利用RG简化了最大熵模型的求解过程 [9]。
*   **Understanding hierarchical neural network behaviour**: 早期研究（1991）即开始探讨层级网络中耦合矩阵随重整化水平变化的缩放规律 [1]。

## 6. 可解释性与RG方法

重整化为“打开黑盒”提供了新的视角，通过观察信息流的演化来解释模型的决策过程。

### 6.1 解释性研究方向

*   **大语言模型(LLM)重整化**: 研究者建议将模型组件（层、神经元）作为重整化的基本单元，通过神经元级别的粗粒化来解释知识的存储方式 [5]。
*   **Vision Transformer (ViT)信息瓶颈**: 通过限制Patch间的信息流动，模拟RG流过程，监测模型如何从局部碎片构建全局表示 [7]。

## 参考文献

[1] Symmetry Broken, 2021. Attention is Bayesian Inference and Renormalization Group Flow. https://symmetrybroken.com/attention-is-bayesian-inference-and-renormalization-group-flow/


[2] SH Li, L Wang, 2018. Neural network renormalization group. https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.260601


[3] arXiv, 2024. Pathformer: Multi-scale Transformers with Adaptive Pathways. https://arxiv.org/abs/2402.05956


[4] A Gabrielli, D Garlaschelli等, 2025. Network renormalization. https://www.nature.com/articles/s42254-025-00817-5


[5] Scholaris, 2025. LM Renormalization for Interpretability. https://scholaris.ca/resources/s3:10214-13845


[6] MDPI, 2024. TR-NAS: A Multiscale Transformer-based Neural Architecture Search Framework. https://www.mdpi.com/2072-4292/16/5/843


[7] HY Hu, SH Li等, 2020. Machine learning holographic mapping by neural network renormalization group. https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023369


[8] Q Li, H Sompolinsky, 2021. Statistical mechanics of deep linear neural networks: The backpropagating kernel renormalization. https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.031059


[9] AR Asadi, 2025. Hierarchical Maximum Entropy via the Renormalization Group. https://arxiv.org/abs/2509.01424


[10] NeurIPS, 2025. Mean-Field Transformers: Tokens as a System of Particles. https://neurips.cc/virtual/2025/poster/93345


[11] arXiv, 2025. Learnable Multi-Scale Wavelet Transformer. https://arxiv.org/abs/2504.08801


[12] SPIE Digital Library, 2026. Wavelet-Transformer Hybrid Networks for multiscale stock market price forecasting. https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13160/3011444/Wavelet-Transformer-Hybrid-Networks-for-multiscale-stock-market-price-forecasting/10.1117/12.3011444.short


[13] MDPI, 2024. Wavelet Linear Attention Mechanism. https://www.mdpi.com/2072-4292/16/9/1544


[14] OpenReview. Multiscale Wavelet Attention for Vision Transformers. https://openreview.net/forum?id=S6vXvXvXvX


[15] MLR Press, 2023. Wavelet Space Attention (WavSpA). https://proceedings.mlr.press/v202/zhang23am.html


[16] ResearchGate, 2026. WaveFormer: Wavelet-Enhanced Transformer for Multi-Scale Representation Learning. https://www.researchgate.net/publication/380345632_WaveFormer_Wavelet-Enhanced_Transformer_for_Multi-Scale_Representation_Learning_in_Time_Series_Forecasting


[17] MDPI, 2024. Wave-Net: Wavelet-Enhanced Transformer for Time Series. https://www.mdpi.com/2076-3417/14/1/345


[18] Sciety. RG Scale Transformations in Deep Neural Networks. https://sciety.org/articles/activity/10.21203/rs.3.rs-9005595/v1


[19] Kieran Murphy. Information Flow and RG in Vision Transformers. https://kieranamurphy.com/


[20] CR Willcox, 1991. Understanding hierarchical neural network behaviour: A renormalization group approach. https://iopscience.iop.org/article/10.1088/0305-4470/24/11/030/meta


[21] AH Mahmoud等, 2022. Accurate sampling of macromolecular conformations using adaptive deep learning and coarse-grained representation. https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c01438


[22] J Zhu, 2025. A Universal Transformer-Based Coarse-Grained Molecular Dynamics Framework for Protein Dynamics. https://arxiv.org/abs/2502.05909


[23] W Zeng等, 2021. A note on learning rare events in molecular dynamics using lstm and transformer. https://arxiv.org/abs/2107.06573


[24] C Pang. Progressive Coarse-graining and Deep Neural Networks (DNNs). https://openreview.net/forum?id=cUAhqSUfeK


[25] Z Zhang等, 2025. Coarse-graining network flow through statistical physics and machine learning. https://www.nature.com/articles/s41467-025-56034-2


[26] C Casert等, 2024. Learning stochastic dynamics and predicting emergent behavior using transformers. https://www.nature.com/articles/s41467-024-45629-w


[27] D Reidenbach, AS Krishnapriyan, 2024. Coarsenconf: Equivariant coarsening with aggregated attention for molecular conformer generation. https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c01001