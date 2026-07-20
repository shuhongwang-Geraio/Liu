诊断报告已写入 `03_experiments/CausalCIT/CausalCIT_ablation/诊断报告_3seed消融结果分析.md`，内容包括：

1. **结论摘要**：噪声量级≈组件差异量级，目前数据不足以证明任何组件有效；`Full` 与 `w/o EnvSplit` 高度异常趋同。
2. **原始数据全表**：3个实验设置×3个seed的vs-PatchTST百分比+标准差。
3. **核心异常定量证据**：9组测试点逐点对比 `Full` vs `w/o EnvSplit`，平均相对差仅0.041%，比与其他变体的差异小一个数量级以上。
4. **可疑代码定位**：具体指出 `CausalStabilityGate`（`causal_channel.py:101-122`）和 `GlobalHSICGate`（`models_ablation.py:130-156`）两处代码，给出3个假设（先验项主导 / 环境切分粒度太小导致 `patch_num=12` 下每环境仅3个patch / 门控整体饱和）。
5. **4步可执行诊断方案**：优先做"保存门控矩阵数值直接对比"（成本最低），再到打印 `channel_prior`/`temperature`，再到调整 `n_envs` 做对照，最后是seed数量的统计功效估算。

这份报告不含最终结论，主要是把排查线索和可复现的定量依据固定下来，方便你或后续任何人接着往下查，不用重新翻3份原始 report.md 拼数据。