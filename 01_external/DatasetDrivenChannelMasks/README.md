# Dataset-Driven Channel Masks / PCD (ICASSP 2026)

- 标题: Dataset-Driven Channel Masks in Transformers for Multivariate Time Series
- 链接: https://ieeexplore.ieee.org/abstract/document/11464024/
- PDF: paper/DatasetDrivenChannelMasks_ICASSP2026.pdf | 官方代码: github.com/YonseiML/pcd
- 一句话: Partial Channel Dependence (PCD)：channel mask M=σ(α·R̄+β)（数据集级相关矩阵+可学习域参数）与注意力矩阵逐元素相乘，统一 CI/CD/PCD 框架；提出 CD ratio。
- 相关性: 线 A **最关键竞争者 + 最强佐证**。
  - 相似: 用数据集特性（非架构）决定通道交互；可插拔；作用于注意力。
  - 差异: 静态相关矩阵（|R|），**无跨环境稳定性、无 HSIC、无因果概念**。
  - ⭐维度效应实锤: PEMS(高维) +12.7%~40.2%，ETTh1/2(低维) 仅 0.3%~2.8%——与我们"高维有效/低维失效"几乎一致。证明"数据集特性决定通道交互价值"是普遍规律。
- 影响: claim 需显式对比 PCD，卖点="跨环境 HSIC 稳定性门控"（PCD 没有），并把其维度效应作为外部佐证。
- 详细分析: surveys/03_multiscale_causal_decoupling/paper_analysis_deep.md §3
