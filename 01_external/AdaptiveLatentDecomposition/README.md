# Adaptive Latent Decomposition (TKDD 2026)

- 标题: Adaptive Latent Decomposition for Domain Generalization in Time Series Forecasting
- 链接: https://dl.acm.org/doi/abs/10.1145/3819822
- PDF: paper/AdaptiveLatentDecomposition_TKDD2026.pdf
- 一句话: 分解式 VAE 将输入解为"域共享+域特定"潜在成分，测试时自适应推断适配目标域。
- 相关性: 线 A（通道交互）检索报告列为 Top-1 最近似，深读修正为"部分覆盖"。
  - 相似: 关注环境/域间依赖差异、做解耦。
  - 差异: 面向**域泛化**（测试时自适应），无跨环境 HSIC 稳定性检验、无门控机制，解的是表示而非通道交互。
- 详细分析: 02_research_notes/surveys/03_multiscale_causal_decoupling/paper_analysis_deep.md §1
