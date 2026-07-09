# 多变量时序预测领域创新研究设想报告

## 1. 关键技术瓶颈综合分析

多变量时序预测（Multivariate Time Series Forecasting, MTSF）领域在2021至2025年间经历了从"Transformer至上"到"架构多元化与基础模型涌现"的剧烈演变。通过对7篇核心论文（DLinear、PatchTST、SOFTS、ModernTCN、RATD、Stable Learning等）的系统分析，以及对领域最新进展的扩展检索，本报告识别出六大关键技术瓶颈，这些瓶颈构成了当前研究的核心挑战，也为创新方向提供了明确的切入点。

### 1.1 自适应CI/CD策略缺失

通道独立（Channel-Independence, CI）与通道相关（Channel-Dependence, CD）策略的选择是当前领域最具争议性的问题之一。CI策略（如DLinear、PatchTST）在ETT、ECL、Weather等多数基准上表现更优，具有更强的分布漂移鲁棒性和抗过拟合能力[1]。然而，从物理意义上看，完全忽略变量间信息在多传感器同步采集、供应链协同预测等场景下显然不合理。

当前方法的根本问题在于策略选择的静态性：要么完全采用CI（如DLinear、PatchTST），要么采用固定的CD结构（如Crossformer、iTransformer），要么采用静态折中方案（如SOFTS的星形拓扑）。缺乏能够根据数据特性、变量相关强度、预测任务需求动态调整变量交互程度的自适应机制。Channel Clustering（NeurIPS 2024）提出根据通道相似性进行聚类，LIFT学习领先指标捕捉异步依赖[15]，但这些方法仍缺乏端到端的自适应学习能力。

|策略类型|代表方法|优势|劣势|适用场景|
|---|---|---|---|---|
|纯CI|DLinear,PatchTST|抗过拟合、分布漂移鲁棒|丢失变量间信息|变量相关性弱|
|纯CD|Crossformer,iTransformer|充分利用变量关系|O(C²)复杂度、易过拟合|小规模通道|
|星形折中|SOFTS|线性复杂度、鲁棒性好|存在信息瓶颈|大规模通道|

### 1.2 深层非平稳性处理不足

时序数据的分布随时间变化是普遍现象，但当前主流方法仅处理了浅层的均值-方差漂移。RevIN通过可逆实例归一化消除实例间的分布差异[22]，已成为多数时序预测模型的标准组件。然而，深层非平稳性（如趋势突变、周期性变化、概念漂移）尚未得到充分解决。

Non-stationary Transformer引入去平稳注意力（De-stationary Attention）机制[8]，其洞见是标准归一化虽然解决了输入层的非平稳问题，但也抹除了对预测有用的统计信息。2024-2025年出现了SAN（时间切片自适应归一化）、FAN（频域自适应归一化）、DDN（双域动态归一化）等精细化方法[7]，但仍是预设的处理模式，难以适应复杂多变的非平稳场景。Stable Learning提供了从因果推断角度处理非平稳性的理论框架，但其依赖多环境数据，在单一环境下如何区分稳定/不稳定特征仍是开放问题。

### 1.3 超长序列建模效率瓶颈

虽然PatchTST通过Patch设计将输入Token数量从L减少到约L/S，使注意力机制的计算复杂度呈平方级降低[1]，ModernTCN通过51×51甚至71×71的超大卷积核显著扩大有效感受野[11]，但在数万个时间步的超长序列场景下仍面临计算和内存挑战。Mamba/SSM架构（如TimeMachine、LinOSS）提供了线性复杂度O(L)的解决方案，ICLR 2025发表的LinOSS在长序列任务上实现了Mamba两倍的性能[6]，但其在多变量场景下的有效性尚需进一步验证。

### 1.4 扩散模型推理效率低下

RATD等扩散模型方法在概率预测和不确定性估计方面表现突出，在罕见病预测任务中MSE较iTransformer降低51%，较CSDI降低59%[9]。然而，多步迭代采样导致推理时间显著高于确定性预测方法，限制了实时应用。SimDiff通过简化扩散步骤实现极速点预测，S2DBM利用布朗桥过程减少逆向估计的随机性[3]，但与确定性模型相比仍有数量级的效率差距。

### 1.5 变量间因果关系建模

现有方法主要建模变量间的统计相关性而非因果关系。在存在虚假相关的场景下（如由共同混杂因素导致的伪相关），基于统计相关的预测可能失效。Stable Learning提出的特征去相关思想为识别因果关系提供了启发，通过样本重加权使模型更关注稳定特征，从而在分布偏移下保持预测鲁棒性。然而，该方法依赖多环境数据，在单一环境下难以区分稳定特征与不稳定特征。将因果发现算法（如PC、FCI、Granger Causality）与时序预测系统性结合是重要的开放问题。

### 1.6 计算效率与精度权衡

从O(T²N²)的全注意力到O(TN)的线性方法，效率提升显著，但精度上限仍有争议。如何在保证预测精度的前提下实现实时预测和边缘部署是持续挑战。

|方法类型|计算复杂度|精度水平|推理延迟|适用场景|
|---|---|---|---|---|
|线性模型|O(TN)|中等|极低|边缘部署|
|MLP/TCN|O(TN)|高|低|通用场景|
|Transformer|O((T/S)²N)|高|中|云端推理|
|扩散模型|O(T·步数)|最高|高|离线分析|
|SSM/Mamba|O(TN)|高|低|超长序列|

## 2. 创新研究设想

基于上述技术瓶颈分析和最新文献调研，本报告提出五个具有创新潜力的研究方向，每个方向均包含详细的研究动机、技术路径、相关文献支持和实验验证思路。

### 2.1 自适应通道交互机制（Adaptive CI/CD Strategy）

#### 2.1.1 研究动机与目标

当前CI与CD策略的二元对立限制了模型在不同场景下的适应能力。理想的解决方案应能根据数据特性和任务需求自动调整变量交互程度，实现"该独立时独立，该交互时交互"的智能决策。核心研究目标是设计一种端到端可学习的自适应通道交互机制，能够在单一模型框架内动态平衡CI与CD策略的优势。

#### 2.1.2 技术路径

**路径一：门控动态选择机制**。借鉴CGN的通道门控单元设计[4]，通过可学习的门控权重实现策略的软切换。核心公式为：

$$Y = G \cdot f_{CD}(X) + (1-G) \cdot f_{CI}(X)$$

其中$G \in [0,1]^{C \times T}$是基于输入特征学习的权重矩阵，$f_{CD}$和$f_{CI}$分别表示通道相关和通道独立的处理分支。门控权重可通过以下方式计算：

$$G = \sigma(W_g \cdot [\text{ChannelSim}(X); \text{TemporalVar}(X)])$$

其中$\text{ChannelSim}$计算通道间相似度矩阵，$\text{TemporalVar}$计算时间维度的变异系数。

**路径二：维度反转与全局交互**。参考iTransformer的设计思想[4]，将Transformer的注意力机制作用于变量维度而非时间维度，从而在CI骨干上实现CD效果。技术要点包括：将每个变量整体视为一个Token，利用变量间注意力显式建模跨变量相关性，同时保持时间维度的独立处理。

**路径三：图结构自适应学习**。利用动态图卷积网络，根据时序片段的相似度实时演化邻接矩阵$A_t$[11]。具体步骤包括：（1）计算滑动窗口内各变量的嵌入表示；（2）通过可学习的相似度函数构建动态邻接矩阵；（3）利用图卷积聚合邻域变量信息；（4）结合CI分支的输出进行融合预测。

#### 2.1.3 相关文献支持

|论文|会议/年份|核心贡献|与本方向关联|
|---|---|---|---|
|iTransformer|ICLR 2024|变量Token化|维度反转思想|
|CGN|ICASSP 2024|通道门控单元|门控机制设计|
|CrossGNN|NeurIPS 2023|跨通道图交互|图结构学习|
|CMamba|arXiv 2024|通道相关SSM|SSM路线参考|
|Channel Clustering|NeurIPS 2024|通道聚类策略|聚类思想借鉴|

#### 2.1.4 实验验证思路

**数据集选择**：Traffic（862变量，高维场景）、Weather（21变量，中等规模）、Solar-Energy（137变量，强相关场景）、ETTh1（7变量，弱相关场景）。通过覆盖不同变量规模和相关性强度的数据集，验证自适应机制的通用性。

**评估指标**：MSE、MAE作为主要预测性能指标；门控权重分布分析验证自适应行为；消融实验量化各组件贡献。

**基线模型**：PatchTST（CI代表）、iTransformer（CD代表）、SOFTS（折中代表）、DLinear（简单基线）。

**关键实验设计**：（1）在不同变量相关性强度的数据集上对比自适应机制与固定策略的性能差异；（2）可视化门控权重随输入变化的动态行为；（3）在分布漂移场景下测试自适应机制的鲁棒性。

### 2.2 因果时序预测（Causal Time Series Forecasting）

#### 2.2.1 研究动机与目标

相关性不等于因果性。在非平稳环境下，变量间的统计相关性会随时间改变，而因果结构通常保持稳定[3]。传统时序预测方法依赖数据中的统计规律，当测试分布与训练分布不一致时容易失效。因果时序预测通过发现变量间的因果图（DAG），利用结构不变性提升预测的鲁棒性。核心研究目标是将因果发现与时序预测深度融合，构建在分布偏移下仍能保持稳定预测性能的模型。

#### 2.2.2 技术路径

**路径一：因果掩码注意力**。借鉴CausalFormer的设计[2]，利用Granger因果检验或PC算法预先构建因果矩阵$M$，并将其作为掩码应用于Transformer的注意力层：

$$\text{Attn}(Q,K,V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} \odot M\right)V$$

其中$M_{ij} = 1$表示变量$j$对变量$i$存在因果影响，$M_{ij} = 0$表示无因果关系。这种设计强制模型仅关注因果相关的变量，忽略虚假相关。

**路径二：不变表示学习**。通过干预增强（Interventional Augmentation）学习环境不变特征。具体方法包括：（1）对训练数据进行时间切片，将不同时段视为不同"环境"；（2）学习在所有环境中都稳定的特征表示；（3）基于稳定特征进行预测，确保模型在分布偏移下依然有效。

**路径三：端到端因果发现与预测联合学习**。将因果图学习作为可微分模块嵌入预测网络：（1）使用神经网络参数化因果图的邻接矩阵$A$；（2）添加无环约束（DAG Constraint）确保图结构有效；（3）联合优化预测损失和因果发现损失。

#### 2.2.3 相关文献支持

|论文|会议/年份|核心贡献|与本方向关联|
|---|---|---|---|
|CausalTime|ICLR 2024|因果时序基准|评估标准|
|CausalFormer|IEEE 2023|因果掩码注意力|技术路径参考|
|CUTS|arXiv 2023|不规则时序因果发现|因果发现算法|
|Causal Inference for TS|Nature Reviews 2023|因果推断综述|理论基础|
|StableNet|CVPR 2021|稳定学习框架|不变表示思想|

#### 2.2.4 实验验证思路

**数据集选择**：CausalTime合成数据集（具有已知因果图的ground truth）、PhysioNet医疗数据（存在明确因果关系的生理信号）、Exchange金融数据（存在复杂因果结构）。

**评估指标**：RMSE、MAE评估预测性能；SHD（Structural Hamming Distance）评估因果图发现准确性；在人工注入分布偏移后的性能下降幅度评估鲁棒性。

**基线模型**：TFT、DeepAR、CausalFormer、iTransformer、PatchTST。

**关键实验设计**：（1）在CausalTime数据集上验证因果图发现的准确性；（2）在人工构造的分布偏移场景下对比因果模型与统计模型的鲁棒性差异；（3）在真实数据集上分析学习到的因果结构是否符合领域知识。

### 2.3 高效扩散模型用于时序预测（Efficient Diffusion for Time Series）

#### 2.3.1 研究动机与目标

扩散模型通过逆转噪声注入过程生成概率分布，能够输出完整的预测分布而非单一点预测，对于不确定性量化和风险敏感型应用至关重要[3]。然而，标准扩散模型需要数十到上百步迭代采样，推理时间比确定性模型高1-2个数量级。核心研究目标是在保持扩散模型概率建模优势的同时，将推理效率提升至实用水平，实现"既要概率预测，又要实时响应"的目标。

#### 2.3.2 技术路径

**路径一：流匹配（Flow Matching）**。TSFlow放弃了复杂的SDE框架，采用确定性的概率流路径[12]：

$$x_t = (1-t)x_0 + tx_1$$

通过更直的采样轨迹，大幅减少采样步数（从100步降至5-10步），同时保持生成质量。技术要点包括：（1）使用最优传输理论设计更高效的传输路径；（2）结合高斯过程先验提供时序特有的归纳偏置。

**路径二：多分辨率去噪**。mr-Diff采用季节性-趋势分解，实现"由易到难"的生成过程[17]：（1）首先在低分辨率下生成趋势成分（简单任务）；（2）逐步细化高频季节性细节（困难任务）；（3）各分辨率层级共享扩散框架但使用独立参数。这种设计减少了高维空间的扩散步数需求。

**路径三：一致性模型（Consistency Models）**。借鉴Consistency Models Made Easy的思想[13]，训练模型直接从任意噪声水平映射到数据分布，实现一步或少步生成。核心技术包括：（1）一致性训练目标确保轨迹上任意点都能正确映射到终点；（2）渐进式蒸馏从多步教师模型提炼一步学生模型。

**路径四：知识蒸馏加速**。将训练好的多步扩散模型作为教师，蒸馏出快速的学生模型：（1）学生模型学习教师的输出分布而非逐步去噪过程；（2）可结合确定性预测器提供粗预测，扩散模型仅负责残差的概率建模。

#### 2.3.3 相关文献支持

|论文|会议/年份|核心贡献|加速效果|
|---|---|---|---|
|TSFlow|arXiv 2024|流匹配+高斯过程先验|采样步数减少90%|
|mr-Diff|ICLR 2024|多分辨率去噪|推理速度提升5倍|
|Consistency Models|ICLR 2025|一步生成|推理速度提升200倍|
|ARMD|AAAI 2025|自回归移动扩散|效率精度平衡|
|RATD|NeurIPS 2024|检索增强扩散|概率预测基线|

#### 2.3.4 实验验证思路

**数据集选择**：Electricity（需要不确定性估计的电力预测）、Exchange-Rate（金融场景的尾部风险建模）、Weather（需要概率预测的气象预报）。

**评估指标**：CRPS（Continuous Ranked Probability Score）评估概率预测质量；推理延迟（ms/样本）评估效率；MSE/MAE评估点预测性能。

**基线模型**：TimeGrad、CSDI、RATD（扩散模型基线）；PatchTST、iTransformer（确定性模型对照）。

**关键实验设计**：（1）CRPS vs 推理延迟的帕累托曲线分析，寻找最优效率-精度权衡点；（2）不同采样步数下的性能衰减曲线；（3）在需要不确定性估计的实际应用场景（如电力调度）中的端到端效果验证。

### 2.4 分布外泛化与非平稳性建模（OOD Generalization & Non-stationarity）

#### 2.4.1 研究动机与目标

现实世界的时序数据具有强烈的非平稳性，表现为均值、方差、趋势、周期的动态漂移。传统的离线训练模型难以应对测试阶段出现的"新常态"。RevIN等方法仅处理了浅层的均值-方差漂移，深层非平稳性（如概念漂移、突变点）尚未得到充分解决。核心研究目标是设计能够在线检测和适应分布变化的机制，实现"训练一次，持续适应"的在线学习能力。

#### 2.4.2 技术路径

**路径一：测试时适应（Test-Time Adaptation, TTA）**。TAFAS在推理阶段利用当前观测到的窗口数据，通过最小化自监督损失在线更新模型参数[9]：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_{SSL}(x_t; \theta_t)$$

其中$\mathcal{L}_{SSL}$可以是掩码重建损失、对比学习损失等不依赖标签的自监督目标。关键技术挑战包括：（1）选择合适的自监督任务避免灾难性遗忘；（2）控制更新幅度防止参数崩塌；（3）仅更新部分参数（如归一化层）提高效率。

**路径二：自监督分布不变表示学习**。通过预训练学习跨时段、跨分布的不变表示：（1）将时间序列划分为多个时段，视为不同"域"；（2）利用域自适应技术学习域不变特征；（3）下游预测基于不变特征进行，天然具有分布漂移鲁棒性。

**路径三：动态归一化机制**。扩展RevIN为多域动态归一化：（1）时域归一化处理均值-方差漂移；（2）频域归一化处理周期性变化（如FAN[7]）；（3）自适应选择归一化策略，针对不同非平稳模式采用不同处理。

**路径四：因果视角的稳定预测**。将Stable Learning的理论框架系统性引入时序预测：（1）识别"稳定特征"（与预测目标存在因果关系）和"不稳定特征"（仅存在统计相关）；（2）通过样本重加权使模型更关注稳定特征；（3）在分布偏移下保持预测鲁棒性。

#### 2.4.3 相关文献支持

|论文|会议/年份|核心贡献|与本方向关联|
|---|---|---|---|
|TAFAS|AAAI 2025|测试时适应框架|TTA技术路径|
|TimeDRL|IEEE 2024|解耦表示学习|不变表示思想|
|Dish-TS|AAAI 2023|分布偏移范式|分布对齐方法|
|FAN|NeurIPS 2024|频域自适应归一化|多域归一化|
|StableNet|CVPR 2021|稳定学习框架|因果视角|

#### 2.4.4 实验验证思路

**数据集选择**：ETTh1/ETTm2（存在明显概念漂移的能源数据）、金融数据（存在regime shift的市场数据）、合成数据（可控的分布偏移场景）。

**评估指标**：Online MSE（逐窗口更新的在线预测误差）；适应速度（达到稳定性能所需样本数）；分布偏移前后的性能下降幅度。

**基线模型**：Non-stationary Transformer、RevIN+PatchTST、Dish-TS、标准PatchTST（无适应）。

**关键实验设计**：（1）在人工构造的分布偏移场景（如突变均值、改变周期）下测试适应能力；（2）在真实数据的不同时段上测试泛化性能；（3）分析TTA更新对模型知识保留的影响。

### 2.5 SSM-Transformer混合架构（SSM-Transformer Hybrid）

#### 2.5.1 研究动机与目标

Transformer的二次复杂度O(L²)限制了其在超长序列中的应用，而Mamba等选择性状态空间模型（SSM）虽提供了线性复杂度O(L)，但对复杂语义的建模能力略逊[5]。混合架构旨在结合两者的长处：SSM提供高效的长程依赖建模，Transformer提供精细的语义理解能力。核心研究目标是设计计算效率与建模能力兼优的混合架构，在超长序列场景下实现SOTA性能。

#### 2.5.2 技术路径

**路径一：多尺度Mamba集成**。参考TimeMachine的设计[5]，利用四个Mamba块分别处理不同尺度的上下文：

$$H = \text{Concat}[\text{Mamba}_1(X_{s_1}), \text{Mamba}_2(X_{s_2}), \text{Mamba}_3(X_{s_3}), \text{Mamba}_4(X_{s_4})]$$

其中$X_{s_i}$表示第$i$个尺度的输入（通过不同步长的Patch划分获得）。这种设计将显存占用降低80%以上，同时捕捉多尺度时间模式。

**路径二：交替堆叠架构**。借鉴Jamba的设计思想[7]，在每7层Mamba后插入1层Transformer：

$$
\begin{aligned}
H^{(l)} &= \text{Mamba}(H^{(l-1)}), \quad l \mod 8 \neq 0 \\
H^{(l)} &= \text{Transformer}(H^{(l-1)}), \quad l \mod 8 = 0
\end{aligned}
$$

Transformer层用于修正SSM的长期记忆衰减，提供全局信息交互的"检查点"。

**路径三：LinOSS振荡状态空间**。采用ICLR 2025提出的线性振荡状态空间模型[6]，在长序列任务上实现Mamba两倍的性能。关键技术包括：（1）引入振荡项建模周期性模式；（2）保持线性复杂度的同时提升表达能力。

**路径四：时间-变量双轴混合**。设计分别处理时间维度和变量维度的混合模块：（1）时间轴使用SSM建模长程依赖（线性复杂度）；（2）变量轴使用Transformer建模跨变量关系（O(N²)但N通常较小）；（3）交替堆叠两种模块实现全面的时序-变量建模。

#### 2.5.3 相关文献支持

|论文|会议/年份|核心贡献|效率提升|
|---|---|---|---|
|TimeMachine|ECAI 2024|四重Mamba集成|显存降低80%|
|LinOSS|ICLR 2025|线性振荡SSM|性能达Mamba 2倍|
|Jamba|AI21 2024|Transformer-Mamba混合|上下文长度256K|
|SST|arXiv 2025|多尺度混合专家|动态计算分配|
|MambaTS|arXiv 2024|时序Mamba适配|线性复杂度|

#### 2.5.4 实验验证思路

**数据集选择**：Traffic（超长序列，862变量）、PEMS04（交通预测，复杂时空依赖）、ETTh1/ETTm1（标准基准对照）。

**评估指标**：Memory Usage（GPU显存占用）、MSE/MAE（预测性能）、Inference Throughput（吞吐量，样本/秒）、Training Time（训练时间）。

**基线模型**：Mamba、PatchTST、TimeMachine、iTransformer、ModernTCN。

**关键实验设计**：（1）随序列长度增长的显存占用和推理时间曲线对比；（2）不同混合比例（Mamba:Transformer层数比）的消融实验；（3）在超长序列（>10000步）场景下的性能对比。

## 3. 创新方向优先级与可行性分析

### 3.1 评估维度定义

本节从创新性、可行性、影响力三个维度对五个创新方向进行综合评估。

**创新性**：评估研究方向的新颖程度，是否填补了现有研究空白，是否提出了新的技术范式。

**可行性**：评估技术实现的难度，是否有成熟的理论基础和工具支持，预期研究周期长短。

**影响力**：评估研究成果的潜在应用价值，是否能解决实际痛点，是否能推动领域发展。

### 3.2 综合评估矩阵

|研究方向|创新性|可行性|影响力|综合评分|推荐优先级|
|---|---|---|---|---|---|
|自适应CI/CD机制|★★★★☆|★★★★★|★★★★★|4.7|**1**|
|因果时序预测|★★★★★|★★★☆☆|★★★★☆|4.0|3|
|高效扩散模型|★★★★☆|★★★★☆|★★★★☆|4.0|3|
|OOD泛化与非平稳性|★★★★☆|★★★★☆|★★★★★|4.3|**2**|
|SSM-Transformer混合|★★★☆☆|★★★★★|★★★★☆|4.0|3|

### 3.3 优先级分析与推荐

**第一优先级：自适应通道交互机制**

推荐理由：（1）CI vs CD是当前领域最核心的未解争议，解决这一问题具有里程碑意义；（2）技术路径清晰，门控机制、维度反转、图结构学习均有成熟工具支持；（3）可在现有模型基础上进行模块化改进，验证周期短；（4）对各类数据集和应用场景具有普适价值。

**第二优先级：OOD泛化与非平稳性建模**

推荐理由：（1）非平稳性是时序数据的本质特征，现有方法处理不足；（2）TTA技术在其他领域已有成熟实践，迁移至时序领域可行性高；（3）直接关联实际部署场景的鲁棒性需求，工业界关注度高；（4）可与其他创新方向（如因果学习）形成协同。

**第三优先级（并列）：因果时序预测、高效扩散模型、SSM-Transformer混合**

这三个方向各有侧重，可根据研究团队的技术积累和资源情况选择性开展。因果时序预测理论深度高但验证难度大；高效扩散模型对实时应用价值大但技术门槛较高；SSM-Transformer混合架构创新性相对较低但工程可行性最强。

### 3.4 研究路线图建议

```
阶段一（0-6个月）：自适应CI/CD机制
├── 月1-2：门控机制原型开发与基准测试
├── 月3-4：图结构自适应学习模块实现
├── 月5-6：综合实验与论文撰写

阶段二（6-12个月）：OOD泛化与非平稳性
├── 月7-8：TTA框架设计与自监督任务选择
├── 月9-10：多域动态归一化机制开发
├── 月11-12：在线学习系统集成与评估

阶段三（12-18个月）：选择性深入
├── 路线A：因果时序预测（理论导向）
├── 路线B：高效扩散模型（应用导向）
└── 路线C：SSM-Transformer混合（效率导向）
```

## 4. 参考文献

[1] AAAI, 2023-02-07. Are Transformers Effective for Time Series Forecasting?. https://arxiv.org/abs/2208.05233

[2] IEEE, 2023-12. CausalFormer: Causal discovery-based transformer for multivariate time series forecasting. https://ieeexplore.ieee.org/abstract/document/10373365/

[3] Nature Reviews Earth & Environment, 2023-04. Causal inference for time series. https://www.nature.com/articles/s43017-023-00431-y

[4] ICLR, 2024-05-01. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. https://openreview.net/forum?id=oVpf9S2K57

[5] arXiv, 2024-03. TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting. https://arxiv.org/abs/2403.09898

[6] ICLR, 2025-01. Linear Oscillatory State-Space Models (LinOSS). https://openreview.net/forum?id=Ai8Hw3AXqks

[7] AI21 Labs, 2024-08. Jamba 1.5 Technical Report. https://arxiv.org/abs/2408.12570

[8] NeurIPS, 2022-11-28. Non-stationary transformers: Exploring the stationarity in time series forecasting. https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html

[9] AAAI, 2025-02. Battling the non-stationarity in time series forecasting via test-time adaptation (TAFAS). https://ojs.aaai.org/index.php/AAAI/article/view/33965

[10] NeurIPS, 2024-12-10. Retrieval-Augmented Diffusion Models for Time Series Forecasting. https://neurips.cc/virtual/2024/poster/93845

[11] Expert Systems with Applications, 2024-01. Dynamic multi-fusion spatio-temporal graph neural network for multivariate time series forecasting. https://www.sciencedirect.com/science/article/pii/S0957417423032311

[12] arXiv, 2024-10. Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting (TSFlow). https://arxiv.org/abs/2410.03024

[13] ICLR, 2025-01. Consistency models made easy. https://proceedings.iclr.cc/paper_files/paper/2025/hash/bb166dd4de5dba363bf1023eb956a826-Abstract-Conference.html

[14] IEEE, 2024-06. Timedrl: Disentangled representation learning for multivariate time-series. https://ieeexplore.ieee.org/abstract/document/10597874/

[15] arXiv, 2024-01. Rethinking channel dependence for multivariate time series forecasting: Learning from leading indicators. https://arxiv.org/abs/2401.17548

[16] IEEE ICASSP, 2024-03. CGN: A simple yet effective multi-channel gated network for long-term time series forecasting. https://ieeexplore.ieee.org/abstract/document/10448209/

[17] ICLR, 2024-05. mr-Diff: Multi-resolution Diffusion Model for Time Series Forecasting. https://iclr.cc/virtual/2024/poster/18144

[18] ICLR, 2024-05-01. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. https://openreview.net/forum?id=vp9vV9gh95

[19] NeurIPS, 2024-12-10. Frequency adaptive normalization for non-stationary time series forecasting. https://proceedings.neurips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html

[20] ICLR, 2024-05. CausalTime: Realistically generated time-series for benchmarking of causal discovery. https://proceedings.iclr.cc/paper_files/paper/2024/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html

[21] NeurIPS, 2024. From similarity to superiority: Channel clustering for time series forecasting. https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb9b18ccb76a1156af5779ffdca1d91f-Abstract-Conference.html

[22] AAAI, 2023. Dish-TS: A general paradigm for alleviating distribution shift in time series forecasting. https://ojs.aaai.org/index.php/AAAI/article/view/25914

[23] Salesforce AI Research, 2024-10-15. Moirai 2.0: Next-Gen Time Series Foundation Model. https://arxiv.org/abs/2410.15616

[24] Amazon Science, 2024-03-12. Chronos: Learning the Language of Time Series. https://arxiv.org/abs/2403.07815

[25] NeurIPS, 2024. SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion. https://neurips.cc/virtual/2024/poster/93845

[26] arXiv, 2024-01-05. The rise of diffusion models in time-series forecasting. https://arxiv.org/abs/2401.03006

[27] IEEE TKDE, 2024. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. https://ieeexplore.ieee.org/abstract/document/10529618/

[28] NeurIPS, 2021-12-06. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. https://huggingface.co/docs/transformers/model_doc/autoformer

[29] ICLR, 2024-05-01. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. https://iclr.cc/virtual/2024/poster/18161

[30] NeurIPS, 2022-11-28. SCINet: Time series modeling and forecasting with sample convolution and interaction. https://proceedings.neurips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html