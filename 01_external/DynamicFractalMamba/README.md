# Dynamic Fractal Mamba (ICML 2026)

- 标题: Dynamic Fractal Mamba: A Neural Renormalization Group Flow for Scale-Invariant Sequence Modeling
- 链接: https://openreview.net/forum?id=L8a9GRfoly
- PDF: paper/DynamicFractalMamba_ICML2026.pdf | 官方代码: github.com/yzlab1/Dynamic-Fractal-Mamba
- 一句话: 物理启发 RG 流：可变步长(Δt=Softplus·2^k)模拟时间膨胀 + 可学习粗粒化(参数跨尺度共享) + RG 门控融合 + 信息守恒/不动点损失。
- 相关性: 线 B（多尺度 RG）**高度覆盖**。RG 概念、递归粗粒化、物理一致性损失均被覆盖。
  - 差异: 基于 Mamba 而非 Transformer（无跨尺度注意力）；面向等间隔采样，**未处理异构采样率**；有官方代码。
- 影响: 线 B 差异化必须收敛到"异构采样率/多速率无插值"。
- 详细分析: surveys/03_multiscale_causal_decoupling/paper_analysis_deep.md §2
