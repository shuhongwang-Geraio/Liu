# Unveiling Limitations of Transformer Models in TSF (PAI 2026)

- 标题: Unveiling the Limitations of Transformer Models in Time Series Forecasting
- 链接: https://link.springer.com/article/10.1007/s13748-026-00450-y
- PDF: paper/UnveilingLimitations_PAI2026.pdf
- 一句话: 批评 Transformer 在 LTSF 上的"边际 MSE 提升"缺统计检验、缺训练稳定性分析；4 Transformer vs LTSF-Linear 在 9 数据集上的多初始化/split 方差+统计检验，结论 Transformer 更差且方差更大。
- 相关性: 方法论层面支撑我们的可证伪 claim。
  - 支撑"统计检验必要性"（我们已用 8-seed Wilcoxon+Holm）。
  - 提醒: 低维负结果须如实报告并解释，契合"场景依赖有效改进"的定位。
- 详细分析: surveys/03_multiscale_causal_decoupling/paper_analysis_deep.md §4
