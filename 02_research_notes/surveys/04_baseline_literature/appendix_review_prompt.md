# 文献调研任务 (转发给调研 agent 用)

> 请把本文件内容整体转发给调研 agent。这是 CausalCIT 项目的 baseline / related work 调研，
> 目标是支撑论文里"我们与竞品的差异"论证 (审稿人 re2 第 3 条点名要求)。

## 项目背景 (一句话)

CausalCIT 是多元时间序列预测方法: 在 PatchTST backbone 上加入"跨环境稳定性门控"
(用 HSIC 度量通道对相关性在不同环境切分下的稳定性, 以稳定/因果的通道对进行交互,
抑制随分布漂移而消失的虚假相关通道对), 目标是 OOD 泛化, 而非仅相关性强度驱动的通道混合。

## 需要调研的方法清单

1. iTransformer (ICLR 2024) — 倒置 Transformer, 通道维度注意力
2. DLinear / NLinear (AAAI 2023, LTSF-Linear) — 分解线性
3. Crossformer (ICLR 2023) — 跨维度注意力
4. Adapformer (Neural Networks 2025) — 基于相关性的自适应通道选择 (重点, 提案直接竞品)
5. CSformer (AAAI 2025) — 先 CI 后 CD 的两阶段通道混合
6. TimeXer (ICML 2024) — 内生/外生变量交互
7. SOFTS (2024) — 基于统计特征的通道交互
8. ModernTCN / MCformer (如适用) — 其它通道交互类方法
9. PatchTST (ICLR 2023) — 当前唯一 baseline, 需确认我们已正确复现

## 每个方法请回答以下问题

**A. 机制层面**
1. 通道交互 (channel interaction) 机制是什么? (相关性强度 / 注意力 / 固定混合 / 分解 / 不交互 CI)
2. 交互权重由什么信号驱动? 是否显式建模"跨环境稳定性"或区分"因果 vs 虚假"通道依赖?
3. 该方法在 OOD / 分布漂移场景下有什么失效模式 (理论上或实验上)?

**B. 与我们方法的差异论证** (最重要)
4. 一句话总结: 与 CausalCIT "用跨环境稳定性而非相关性强度决定通道交互" 的本质区别是什么?
5. 这个差异在实验上如何体现? (即: 在什么类型的数据集/条件下, 我们预期比它强; 为什么)
   - 提示: CausalCIT 的假设是"通道多、依赖结构强时, 稳定性门控抑制虚假相关有价值;
     低维/弱依赖时退化为噪声"。可据此对照各方法的适用边界。

**C. 复现信息**
6. 官方开源代码地址 (GitHub) + license + 主要依赖 (PyTorch 版本等)
7. 是否已有公开 benchmark 数字 (ETTh1/ETTm1/Weather/Electricity/Traffic/ILI/Exchange,
   各 pred_len 的 MSE/MAE)? 给出出处。

**D. Related work 补充** (可选但重要)
8. 与"稳定学习/不变学习 + 时序通道交互"更直接相关的工作有哪些? 特别关注:
   FOIL (ICML 2024, 时序环境推断+不变学习)、COGS (因果表示学习时序 OOD)、StableNet、
   以及任何把"跨环境一致性/稳定性"用于时序特征或通道选择的近作 (2023-2026)。

## 输出格式

对每个方法: 一张 markdown 表 (上述 A/B/C 各字段) + 一段 150 字以内的"vs CausalCIT 差异"总结。
最后附一份 related work 清单 (标题/venue/年份/一句话贡献/与我们的关系)。

## 注意

- 优先引用 arXiv 摘要与官方 repo README, 标注信息截止日期。
- 若某方法无开源代码, 明确说明"无官方代码", 并给可用的第三方实现。
- 不需要跑实验, 纯文献调研。
