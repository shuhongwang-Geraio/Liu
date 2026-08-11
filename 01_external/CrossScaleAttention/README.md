# Cross-Scale Attention (SPL 2024)

- 标题: Cross-Scale Attention for Long-Term Time Series Forecasting
- 链接: https://ieeexplore.ieee.org/abstract/document/10623694/
- PDF: paper/CrossScaleAttention_SPL2024.pdf
- 一句话: 多尺度 patching（整序列为 1 patch，逐级二分），单层注意力建模跨尺度 patch 关系，token 数压至 ~7，12x 快于 PatchTST。
- 相关性: 线 B **直接覆盖"跨尺度注意力"概念**。
  - 差异: 均匀下采样(二进制分割)，**非异构采样率**；效率导向而非物理/尺度对齐导向。
- 影响: 线 B 若只报"跨尺度注意力"会重叠；卖点须收敛到"异构采样率+无插值+尺度感知 RoPE"。
- 详细分析: surveys/03_multiscale_causal_decoupling/paper_analysis_deep.md §7
