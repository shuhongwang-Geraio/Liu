# 多变量时序数据预测领域扩展文献检索与研究现状综合报告

## 1. 领域研究现状总览

### 1.1 2021-2025年技术演进时间线

多变量时序预测（Multivariate Time Series Forecasting, MTSF）领域在2021至2025年间经历了从"Transformer至上"到"反思Transformer"，再到"架构多元化与基础模型涌现"的剧烈演变。这一演变过程不仅重塑了技术路线的格局，也深刻影响了研究社区对时序数据本质特性的认知。

|时间阶段|核心事件|代表性工作|影响|
|---|---|---|---|
|2021|Transformer时序化|Informer, Autoformer|开启Transformer主导时代|
|2022|线性模型挑战|DLinear, FEDformer|质疑复杂架构必要性|
|2023|Patch策略崛起|PatchTST, TSMixer|重新证明Transformer有效性|
|2024|架构多元化|iTransformer, SOFTS, ModernTCN, RATD|各技术路线百花齐放|
|2025|基础模型元年|Chronos-Bolt, Moirai 2.0, TimesFM|预训练与零样本泛化成为焦点|

2021年标志着Transformer架构正式进入时序预测领域的主流视野。Informer通过ProbSparse注意力机制将计算复杂度从O(L²)降至O(L log L)[1]，Autoformer则引入序列分解与自相关机制，这些工作奠定了Transformer在长程时序预测中的基础地位[5]。然而，2022年DLinear的发表彻底改变了这一格局——一个简单的单层线性网络竟在多个基准数据集上击败了所有复杂的Transformer变体，引发了领域内对"注意力机制必要性"的广泛讨论[1]。

2023年成为技术路线分野的关键年份。PatchTST通过Patch化设计和通道独立策略，重新证明了Transformer在时序领域的有效性，其核心洞见在于时序数据应以子序列（Patch）而非单点作为Token[1]。与此同时，Google推出的TSMixer利用全MLP架构证明了轻量级模型同样能达到SOTA性能[1]。2024年见证了技术路线的全面多元化：iTransformer实现了变量Token化的范式反转[4]，SOFTS提出了线性复杂度的星形拓扑变量交互[2]，ModernTCN通过大核卷积复兴了CNN在时序领域的地位[11]，RATD则将检索增强引入扩散模型开辟了概率预测新范式[9]。

### 1.2 核心问题与研究主题变迁

多变量时序预测领域的核心研究问题随技术演进不断深化与拓展。早期研究聚焦于如何将NLP领域的成功经验迁移至时序数据，核心问题是"如何设计适合时序特性的注意力机制"。随着DLinear的挑战，研究焦点转向"复杂模型是否真正有效"这一根本性问题。当前阶段，研究社区已形成对以下六大核心问题的共识性关注：

**问题一：通道独立与通道相关策略的选择**。这是当前领域最具争议性的问题之一。CI策略（如DLinear、PatchTST）在多数基准上表现更优且具有更强的分布漂移鲁棒性，但从理论角度看完全忽略了变量间信息[1]。SOFTS的星形拓扑[2]、iTransformer的变量Token化[4]、LIFT的领先指标学习等工作代表了在CI与CD之间寻求平衡的不同尝试。

**问题二：非平稳性处理**。时序数据的分布随时间变化是普遍现象，但主流方法（如RevIN）仅处理了浅层的均值-方差漂移。2024-2025年出现了SAN（时间切片自适应归一化）、FAN（频域自适应归一化）、DDN（双域动态归一化）等精细化方法[7]，Non-stationary Transformer的去平稳注意力机制也提供了重要思路[8]。

**问题三：长程依赖建模**。虽然PatchTST通过Patch设计延长了有效回顾窗口，ModernTCN通过大核卷积扩大了感受野，但超长序列建模仍面临挑战。Mamba/SSM架构的引入（如TimeMachine、LinOSS）为这一问题提供了线性复杂度的解决方案[11]。

**问题四：不确定性估计与概率预测**。扩散模型路线（TimeGrad→CSDI→TimeDiff→RATD）为概率预测提供了新框架，但推理效率是主要瓶颈[3][9]。SimDiff、S2DBM等工作致力于优化扩散过程的效率[3]。

**问题五：预训练与零样本泛化**。2024年被视为"时序基础模型元年"，Chronos、Moirai、TimesFM等模型实现了大规模预训练与Zero-shot泛化[2][6]。Time-LLM等工作探索了利用LLM作为时序分析引擎的可能性[12]。

**问题六：计算效率与精度权衡**。从O(T²N²)的全注意力到O(TN)的线性方法，效率提升显著，但如何在保证预测精度的前提下实现实时预测和边缘部署仍是持续挑战。

## 2. 主流方法演进脉络

### 2.1 Transformer路线：从点级Token到变量级Token

Transformer在时序预测领域的演进经历了三个阶段：效率优化阶段、Token设计反思阶段、变量建模范式转换阶段。

|模型|年份/会议|核心创新|复杂度|CI/CD|
|---|---|---|---|---|
|Informer|AAAI 2021|ProbSparse注意力|O(L log L)|CD|
|Autoformer|NeurIPS 2021|序列分解+自相关|O(L log L)|CD|
|FEDformer|ICML 2022|频域注意力|O(L)|CD|
|PatchTST|ICLR 2023|Patch+通道独立|O((L/S)²·N)|CI|
|iTransformer|ICLR 2024|变量Token化|O(N²)|CD|
|TimeXer|2024|外部变量整合|O(N²)|CD|
|TimeMixer++|2025|时序模式机|O(L)|混合|

**效率优化阶段（2021-2022）**：Informer通过ProbSparse注意力机制筛选出最重要的Query，将复杂度降至O(L log L)[1]。Autoformer引入序列级连接的自相关机制取代点级自注意力，并创新性地将时间序列分解为趋势项和季节项[5]。FEDformer将注意力计算转移到频域，利用傅里叶变换实现O(L)复杂度。然而，这些工作在2022年遭遇DLinear的挑战——简单线性模型的性能竟优于所有复杂设计。

**Token设计反思阶段（2023）**：PatchTST的核心洞见在于：时序数据不应像NLP那样以单点为Token。通过将时间序列划分为固定长度的Patch作为基本输入单元，PatchTST既保留了局部语义信息，又将Token数量从L减少到约L/S，使注意力机制的计算复杂度呈平方级降低[1]。更重要的是，PatchTST采用通道独立策略，所有变量共享同一个Transformer主干但独立计算，这一设计有效缓解了过拟合问题并增强了分布漂移鲁棒性。

**变量建模范式转换阶段（2024-2025）**：iTransformer实现了"倒置Transformer"的范式反转——将每个变量（Channel）整体视为一个Token，利用注意力机制显式建模变量间相关性，而非时间步之间的依赖[4]。这一设计解决了传统Transformer在多变量维度上的建模缺陷。TimeXer进一步强化了对外部变量（Exogenous variables）的整合能力[4]。2025年的TimeMixer++演进为通用的"时序模式机"，支持预测、异常检测、补全等多种任务[4]。

### 2.2 线性/MLP路线：从简单基准到高效交互

线性模型与MLP的崛起是2022年后时序预测领域最重要的趋势之一，其演进路径清晰地展示了"简单有效"到"简单且高效交互"的技术进步。

|模型|年份/会议|核心创新|复杂度|CI/CD|
|---|---|---|---|---|
|DLinear|AAAI 2022|单层线性映射|O(T·N)|CI|
|TSMixer|2023|时间/特征交替混合|O(L)|混合|
|SOFTS|NeurIPS 2024|STAR星形拓扑|O(C·L)|间接CD|
|TimeMixer|2024|多尺度混合|O(L)|混合|

**DLinear的里程碑意义**：DLinear采用极其简单的架构——仅用单层线性网络将历史窗口直接映射到预测窗口，不包含任何注意力机制或非线性变换[1]。在ETT、ECL、Traffic等多个数据集上，DLinear的性能竟优于Informer、Autoformer、FEDformer等所有复杂Transformer变体。这一发现为领域建立了极具竞争力的基线，迫使后续研究必须证明其复杂模块确实带来了超越线性映射的性能增益。然而，DLinear的局限性也很明显：当通道数非常大时表现下降，完全采用通道独立策略丢失了变量间信息，且线性本质限制了其捕捉非线性模式的能力。

**TSMixer与MLP架构的成熟**：Google推出的TSMixer利用全MLP架构，通过时间混合（Time-Mixing）和特征混合（Feature-Mixing）交替操作，证明了轻量级模型能在保持高效的同时达到SOTA性能[1]。时间混合层沿时间维度学习依赖关系，特征混合层则在变量维度进行信息交换，这种解耦设计兼顾了CI与CD的优势。

**SOFTS的星形拓扑创新**：SOFTS提出的STAR（STar Aggregate-Redistribute）模块是MLP路线的重要突破[2]。其核心设计灵感来源于软件工程中的星形中心化系统：通过一个"核心表示"（Core Representation）聚合所有通道的信息，再将聚合后的全局信息分发回各通道。这一设计将通道间交互的复杂度从O(C²)降低到O(C)，同时增强了对异常通道的鲁棒性——当某个通道出现异常时，STAR能通过核心表示利用正常通道的信息进行"拉回"修正。

### 2.3 卷积路线：从局部感受野到大核卷积

卷积神经网络在时序预测领域经历了从边缘化到复兴的过程，大核卷积的引入是这一复兴的关键技术突破。

|模型|年份/会议|核心创新|核心技术|
|---|---|---|---|
|SCINet|NeurIPS 2022|样本卷积交互|多尺度分解|
|TimesNet|ICLR 2023|2D变化建模|1D→2D转换|
|ModernTCN|ICLR 2024|大核卷积|51×51~71×71核|

**SCINet的多尺度建模**：SCINet利用样本卷积与交互学习捕捉多尺度时间特征[10]。其核心思想是将时间序列递归地分解为多个尺度的子序列，在每个尺度上进行卷积操作后再进行交互融合。这一设计有效捕捉了时序数据中常见的多周期性特征。

**TimesNet的维度转换**：TimesNet将一维时序转化为二维张量进行变化建模，通过FFT分析序列的主要周期，将时间维度重塑为"周期数×周期内位置"的二维结构，然后利用2D卷积同时捕捉周期内和周期间的依赖关系[1]。这一创新使CNN重新进入多变量时序预测的主流视野。

**ModernTCN的大核卷积革命**：ModernTCN是卷积路线的集大成者[11]。其核心贡献在于借鉴计算机视觉中ConvNeXt的大核卷积思想，采用51×51甚至71×71的超大卷积核显著扩大有效感受野（ERF）。研究表明，在纯卷积结构中，ERF与卷积核大小成线性正相关，而与层数仅呈亚线性关系O(ks×√nl)，因此优先通过增大卷积核而非堆叠深层小核来获取更大ERF更为高效。ModernTCN采用三组件解耦设计：DWConv（深度卷积）负责时间信息学习，ConvFFN1负责变量内部特征学习，ConvFFN2负责跨变量依赖捕捉。在长期预测、短期预测、插补、分类和异常检测五大任务上，ModernTCN均达到SOTA水平，相比TimesNet，MSE平均降低27.4%，MAE降低15.3%。

### 2.4 扩散模型路线：从确定性预测到概率分布

扩散模型为时序预测引入了概率建模框架，能够输出完整的预测分布而非单一点预测，这对于不确定性量化和风险敏感型应用至关重要。

|模型|年份|核心创新|技术特点|
|---|---|---|---|
|TimeGrad|2021|自回归扩散|逐步预测|
|CSDI|NeurIPS 2021|条件分数扩散|补全与预测统一|
|TimeDiff|2023|非自回归扩散|缓解误差累积|
|D3VAE|2023|解耦扩散VAE|潜空间建模|
|RATD|NeurIPS 2024|检索增强扩散|语义引导去噪|

**扩散模型在时序领域的适配**：TimeGrad是最早将扩散模型应用于时序预测的工作之一，采用自回归方式逐步预测未来时间步。CSDI开创了基于分数的扩散模型用于时序补全与预测的统一框架[3]。然而，自回归生成方式存在误差累积问题，TimeDiff通过非自回归设计缓解了这一问题。

**RATD的检索增强创新**：RATD是扩散模型路线的重要突破[9]。其核心贡献在于将检索增强（RAG）技术引入时序扩散模型——利用预训练编码器将历史序列转化为嵌入向量，在预构建的数据库中检索相似样本作为"参考序列"，通过参考调制注意力机制（RMA）将参考信息融入去噪过程。这一设计为缺乏语义信息的扩散去噪提供了明确引导，在罕见病预测等复杂任务中表现尤为突出，MSE较iTransformer降低51%，较CSDI降低59%。

**效率优化进展**：扩散模型的主要瓶颈在于多步迭代采样导致的低推理效率。SimDiff通过简化扩散步骤实现极速点预测，S2DBM利用布朗桥过程减少逆向估计的随机性[3]。这些工作为扩散模型在实时预测场景中的应用铺平道路。

### 2.5 基础模型路线：从单数据集到大规模预训练

2024年被视为"时序基础模型元年"，研究重点从单一数据集训练转向大规模预训练与Zero-shot泛化，这一转变与NLP领域的发展轨迹形成呼应。

|模型|机构|架构类型|核心特点|
|---|---|---|---|
|Chronos|Amazon|Encoder-Decoder|时序值量化为Token|
|Chronos-Bolt|Amazon|优化版|推理速度提升250倍|
|Moirai|Salesforce|掩码编码器|任意频率/变量数|
|Moirai 2.0|Salesforce|Decoder-only|体积缩小96%|
|TimesFM|Google|Decoder-only|合成+真实数据预训练|
|Lag-Llama|开源|LLaMA架构|概率预测|
|Timer|2024|GPT架构|统一时序任务|

**Amazon Chronos系列**：Chronos将时序值量化为离散Token，利用语言模型架构（T5）进行预测[6]。其创新在于将连续的时序值映射到有限词表，使标准语言模型技术可直接应用。升级版Chronos-Bolt通过模型蒸馏和架构优化，推理速度提升250倍，大幅提高了实用性。

**Salesforce Moirai系列**：Moirai基于掩码编码器架构，支持任意频率和变量数量的输入[2]。2025年的Moirai 2.0进行了重大架构革新，转向Decoder-only设计，模型体积缩小96%，推理速度翻倍，同时保持了强劲的零样本性能。

**Google TimesFM**：TimesFM通过在大规模合成数据与真实数据上进行预训练，实现了卓越的零样本泛化能力[2]。在ETT、Weather、Electricity等标准基准的零样本评测中，TimesFM与针对特定数据集微调的模型性能相当，展示了预训练范式在时序领域的巨大潜力。

**LLM驱动的适配方法**：GPT4TS证明了冻结权重的预训练LLM（如GPT-2）可作为通用时序分析引擎[12]。Time-LLM通过"重编程"技术将时序Patch映射为文本原型，并利用自然语言Prompt引导LLM理解任务背景[12]，在Few-shot场景下表现突出。这一路线为利用现有LLM的强大能力处理时序数据提供了新思路。

### 2.6 SSM/Mamba路线：线性复杂度的新范式

Mamba架构凭借其线性复杂度和长序列建模能力，正成为Transformer的有力竞争者，代表了时序预测领域的最新技术前沿。

|模型|年份|核心创新|技术特点|
|---|---|---|---|
|TimeMachine|2024|四重Mamba|统一通道交互|
|MambaTS|2024|时序Mamba|变量依赖建模|
|Mamba4Cast|2025|Mamba-2基础模型|推理速度领先|
|LinOSS|ICLR 2025|线性振荡状态空间|性能达Mamba 2倍|

**Mamba在时序领域的适配**：TimeMachine利用四重Mamba结构统一了时间和通道两个维度的信息交互[11]。MambaTS则专门针对多变量时序预测设计，通过状态空间模型的递归特性高效捕捉长程依赖。Mamba4Cast作为基于Mamba-2的基础模型，在推理速度上大幅领先基于Transformer的同类模型。

**LinOSS的性能突破**：ICLR 2025发表的LinOSS在长序列任务上实现了Mamba两倍的性能[11]，标志着SSM架构在时序预测领域的进一步成熟。这一路线有望成为处理超长序列的首选方案。

## 3. 核心技术争议与演进

### 3.1 CI vs CD策略的理论与实证分析

通道独立（Channel-Independence, CI）与通道相关（Channel-Dependence, CD）策略的选择是当前多变量时序预测领域最核心的未解争议。这一争议的本质在于：多变量时序数据中变量间的相关信息究竟是"信号"还是"噪声"？

**CI策略的实证支持**：DLinear和PatchTST在ETT、ECL、Weather等多数基准上采用CI策略表现更优[1]。CI策略的优势源于三个方面：首先，避免了跨通道过拟合，尤其在训练数据有限时收敛更快；其次，对分布漂移具有更强鲁棒性，因为单通道模式比跨通道依赖更稳定；最后，允许预训练数据与下游任务数据的变量数量不一致，提高了模型的通用性。

**CD策略的理论支撑**：从物理意义上看，完全忽略变量间信息在多传感器同步采集、供应链协同预测等场景下显然不合理。iTransformer通过变量Token化显式建模跨变量相关性[4]，在某些数据集上取得了优于CI方法的结果。最新研究（如LIFT）提出学习"领先指标"来捕捉变量间的异步依赖关系[1]。

**折中方案的探索**：SOFTS的星形拓扑代表了CI与CD之间的折中路线[2]。通过核心表示间接实现变量交互，SOFTS在保持O(C)线性复杂度的同时部分保留了通道间信息。ModernTCN则通过可选的ConvFFN2模块实现跨变量建模，在单变量任务时可移除该组件[11]。Channel Clustering（NeurIPS 2024）提出根据通道相似性进行聚类，在簇内采用CD策略、簇间采用CI策略。

|策略|代表方法|优势|劣势|适用场景|
|---|---|---|---|---|
|纯CI|DLinear,PatchTST|抗过拟合、分布漂移鲁棒|丢失变量间信息|变量相关性弱|
|纯CD|Crossformer|充分利用变量关系|O(C²)复杂度、易过拟合|小规模通道|
|星形折中|SOFTS|线性复杂度、鲁棒性好|存在信息瓶颈|大规模通道|
|自适应|Channel Clustering|动态平衡|设计复杂|通用场景|

### 3.2 Transformer有效性之争的最终走向

"Transformer在时序预测中是否真正有效"这一争论历经三年演变，目前已形成较为清晰的共识。

**第一阶段：Transformer主导（2021-2022）**。Informer、Autoformer、FEDformer等工作将Transformer架构引入时序预测，通过各种效率优化技术解决长序列建模问题。这一阶段的研究默认复杂架构优于简单模型。

**第二阶段：线性模型挑战（2022-2023）**。DLinear的发表彻底颠覆了这一假设[1]。其核心发现是：在ETT、ECL、Traffic等常用基准上，单层线性网络的性能优于所有Transformer变体。这一结果引发了对"时序预测任务是否需要复杂时序模式建模"的根本性质疑。

**第三阶段：Patch策略回应（2023-2024）**。PatchTST对线性模型的挑战作出了有力回应[1]。其核心洞见是：此前Transformer失败的原因不在于注意力机制本身，而在于Token设计不当——以单点为Token的设计无法捕捉时序数据的局部语义结构。通过Patch化设计，PatchTST在保持Transformer优势的同时显著提升了性能和效率。

**当前共识**：领域内已形成以下共识：(1) 简单线性模型是有效的强基线，任何复杂设计都需要证明其超越线性映射的增益；(2) Transformer的有效性取决于输入表示设计，Patch策略是关键；(3) 不存在"最优架构"，Transformer、MLP、TCN、扩散模型各有优势场景；(4) 效率与精度的权衡因应用场景而异。

### 3.3 非平稳性处理方法演进

非平稳性是时序数据的本质特征之一，其处理方法经历了从简单归一化到多域动态调整的演进。

|方法|年份|技术路线|处理层次|
|---|---|---|---|
|RevIN|2022|可逆实例归一化|均值-方差|
|Non-stationary Transformer|NeurIPS 2022|去平稳注意力|注意力层面|
|Dish-TS|2023|分布对齐|输入-输出分布|
|SAN|2024|时间切片自适应|局部统计量|
|FAN|NeurIPS 2024|频域自适应|频谱特性|
|DDN|2024|双域动态|时域+频域|

**浅层处理：均值-方差归一化**。RevIN通过可逆实例归一化解决了均值和方差随时间漂移的问题[1]。其核心思想是在输入前减去实例均值、除以实例标准差，在输出后进行逆变换恢复原始尺度。这一简单技术已成为多数时序预测模型的标准组件。

**中层处理：注意力机制调整**。Non-stationary Transformer引入去平稳注意力（De-stationary Attention）机制[8]。其洞见是：标准归一化虽然解决了输入层的非平稳问题，但也抹除了对预测有用的统计信息。去平稳注意力通过学习恢复被归一化抹除的信息，在保持输入稳定性的同时保留预测所需的非平稳特征。

**深层处理：多域动态调整**。2024-2025年的研究进一步深化了非平稳性处理[7]。SAN从时间切片视角进行自适应归一化，针对不同时间段动态调整归一化参数。FAN在频域进行自适应调整，针对不同频率成分采用不同的归一化策略。DDN实现了时域与频域的双域动态归一化，能够应对更复杂的非平稳模式。

**因果视角的启示**：Stable Learning提供了从因果推断角度处理非平稳性的理论框架。其核心思想是区分"稳定特征"（与标签存在因果关系）和"不稳定特征"（与标签仅存在统计相关），通过样本重加权使模型更关注稳定特征，从而在分布偏移下保持预测鲁棒性。这一视角为时序预测中的非平稳性处理提供了新思路，但在单一环境下如何区分稳定/不稳定特征仍是开放问题。

## 4. 代表性成果汇总表

### 4.1 按年份与会议整理的重要论文（2021-2025）

|年份|会议|论文|核心贡献|技术路线|
|---|---|---|---|---|
|2021|AAAI|Informer|ProbSparse注意力|Transformer|
|2021|NeurIPS|Autoformer|序列分解+自相关|Transformer|
|2021|CVPR|StableNet|稳定学习框架|因果推断|
|2022|AAAI|DLinear|线性基准|线性模型|
|2022|ICML|FEDformer|频域注意力|Transformer|
|2022|NeurIPS|Non-stationary Trans.|去平稳注意力|Transformer|
|2022|NeurIPS|SCINet|样本卷积交互|CNN|
|2023|ICLR|PatchTST|Patch+通道独立|Transformer|
|2023|ICLR|TimesNet|2D变化建模|CNN|
|2023|-|TSMixer|时间/特征混合|MLP|
|2024|ICLR|iTransformer|变量Token化|Transformer|
|2024|ICLR|ModernTCN|大核卷积|CNN|
|2024|ICLR|Time-LLM|LLM重编程|LLM|
|2024|NeurIPS|SOFTS|STAR星形拓扑|MLP|
|2024|NeurIPS|RATD|检索增强扩散|扩散模型|
|2024|NeurIPS|FAN|频域自适应归一化|归一化|
|2024|-|Chronos|时序值Token化|基础模型|
|2024|-|Moirai|掩码编码器|基础模型|
|2024|-|TimesFM|大规模预训练|基础模型|
|2025|ICLR|LinOSS|线性振荡SSM|SSM|
|2025|-|Moirai 2.0|Decoder-only|基础模型|
|2025|-|Chronos-Bolt|推理优化|基础模型|

### 4.2 各数据集SOTA方法对比

|数据集|类型|最优方法|次优方法|备注|
|---|---|---|---|---|
|ETTh1/ETTm1|能源|iTransformer/PatchTST|TimesFM|基础模型Zero-shot接近|
|ETTh2/ETTm2|能源|PatchTST|TimeMixer|CI策略占优|
|Weather|气象|iTransformer|TimesFM|CD策略有效|
|Electricity|电力|iTransformer|SOFTS|高维数据|
|Traffic|交通|SOFTS|ModernTCN|超高维(862变量)|
|PEMS|交通|SOFTS|iTransformer|大规模通道场景|
|Exchange|金融|RATD|PatchTST|概率预测场景|

上表总结了2024-2025年各主要基准数据集的最优方法[1][2][4][9][11]。可以观察到：(1) iTransformer和PatchTST在中等规模数据集上表现突出；(2) SOFTS在高维交通数据上展现极强竞争力；(3) 基础模型（如TimesFM）的零样本性能已接近特定数据集微调模型；(4) 扩散模型（如RATD）在需要概率预测的场景中具有独特优势。

## 5. 关键技术瓶颈识别

基于对知识库论文和扩展文献的综合分析，本节识别出六大关键技术瓶颈，这些瓶颈将为Stage 3的创新方向探索提供重要基础。

### 5.1 自适应CI/CD策略缺失

**问题描述**：当前方法要么完全采用CI（如DLinear、PatchTST），要么采用固定的CD结构（如iTransformer），要么采用静态折中方案（如SOFTS）。缺乏能够根据数据特性、变量相关强度、预测任务需求动态调整变量交互程度的自适应机制。

**现有尝试**：Channel Clustering（NeurIPS 2024）提出根据通道相似性进行聚类；LIFT学习领先指标捕捉异步依赖。但这些方法仍缺乏端到端的自适应学习能力。

**潜在研究方向**：设计可学习的CI/CD门控机制；基于元学习的策略选择；利用因果发现自动识别变量间真实依赖。

### 5.2 深层非平稳性处理不足

**问题描述**：RevIN等方法仅处理了浅层的均值-方差漂移，深层非平稳性（如趋势突变、周期性变化、概念漂移）尚未得到充分解决。

**现有尝试**：FAN、DDN等方法提供了频域和双域的归一化策略[7]，但仍是预设的处理模式，难以适应复杂多变的非平稳场景。

**潜在研究方向**：将Stable Learning的因果视角系统性引入时序预测；设计能够在线检测和适应分布变化的机制；探索自监督学习提取分布不变表示。

### 5.3 超长序列建模效率瓶颈

**问题描述**：虽然PatchTST通过Patch设计延长了有效回顾窗口，ModernTCN通过大核卷积扩大了感受野，但在数万个时间步的超长序列场景下仍面临计算和内存挑战。

**现有尝试**：Mamba/SSM架构（如TimeMachine、LinOSS）提供了线性复杂度的解决方案[11]，但其在多变量场景下的有效性尚需进一步验证。

**潜在研究方向**：结合SSM与Transformer的混合架构；开发稀疏注意力的新变体；探索层次化序列建模策略。

### 5.4 扩散模型推理效率低下

**问题描述**：RATD等扩散模型方法在概率预测和不确定性估计方面表现突出，但多步迭代采样导致推理时间显著高于确定性预测方法，限制了实时应用[9]。

**现有尝试**：SimDiff简化扩散步骤，S2DBM利用布朗桥减少随机性[3]。但与确定性模型相比仍有数量级的效率差距。

**潜在研究方向**：一步扩散模型的设计；将扩散过程与确定性预测器结合；知识蒸馏压缩扩散模型。

### 5.5 变量间因果关系建模

**问题描述**：现有方法主要建模变量间的统计相关性而非因果关系。在存在虚假相关的场景下（如由共同混杂因素导致的伪相关），基于统计相关的预测可能失效。

**现有尝试**：Stable Learning提供了因果推断的理论框架，但其依赖多环境数据，在单一环境下难以区分稳定/不稳定特征。

**潜在研究方向**：将因果发现算法（如PC、FCI）与时序预测结合；利用干预数据学习因果结构；探索反事实推理在时序预测中的应用。

### 5.6 计算效率与预测精度的权衡

**问题描述**：从O(T²N²)的全注意力到O(TN)的线性方法，效率提升显著，但精度上限仍有争议。如何在保证预测精度的前提下实现实时预测和边缘部署是持续挑战。

|方法类型|计算复杂度|精度水平|推理延迟|适用场景|
|---|---|---|---|---|
|线性模型|O(TN)|中等|极低|边缘部署|
|MLP/TCN|O(TN)|高|低|通用场景|
|Transformer|O((T/S)²N)|高|中|云端推理|
|扩散模型|O(T·步数)|最高|高|离线分析|
|基础模型|视架构而定|高|中-高|零样本任务|

**潜在研究方向**：模型压缩与量化技术；动态计算分配（根据输入复杂度调整计算量）；硬件感知的架构搜索。

## 6. 参考文献

[1] AAAI, 2023-02-07. Are Transformers Effective for Time Series Forecasting?. https://arxiv.org/abs/2208.05233

[2] Salesforce AI Research, 2024-10-15. Moirai 2.0: Next-Gen Time Series Foundation Model. https://arxiv.org/abs/2410.15616

[3] arXiv, 2024-01-05. The rise of diffusion models in time-series forecasting. https://arxiv.org/abs/2401.03006

[4] ICLR, 2024-05-01. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. https://openreview.net/forum?id=oVpf9S2K57

[5] NeurIPS, 2021-12-06. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. https://huggingface.co/docs/transformers/model_doc/autoformer

[6] Amazon Science, 2024-03-12. Chronos: Learning the Language of Time Series. https://arxiv.org/abs/2403.07815

[7] NeurIPS, 2024-12-10. Frequency adaptive normalization for non-stationary time series forecasting. https://proceedings.neurips.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html

[8] NeurIPS, 2022-11-28. Non-stationary transformers: Exploring the stationarity in time series forecasting. https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html

[9] NeurIPS, 2024-12-10. Retrieval-Augmented Diffusion Models for Time Series Forecasting. https://neurips.cc/virtual/2024/poster/93845

[10] NeurIPS, 2022-11-28. SCINet: Time series modeling and forecasting with sample convolution and interaction. https://proceedings.neurips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html

[11] ICLR, 2024-05-01. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. https://openreview.net/forum?id=vp9vV9gh95

[12] ICLR, 2024-05-01. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. https://iclr.cc/virtual/2024/poster/18161

[13] IEEE TKDE, 2024. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. https://ieeexplore.ieee.org/abstract/document/10529618/

[14] NeurIPS, 2024. From similarity to superiority: Channel clustering for time series forecasting. https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb9b18ccb76a1156af5779ffdca1d91f-Abstract-Conference.html

[15] arXiv, 2024-01. Rethinking channel dependence for multivariate time series forecasting: Learning from leading indicators. https://arxiv.org/abs/2401.17548