# Learning Pattern-Specific Experts / TFPS (NeurIPS 2025)

- 标题: Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift
- 链接: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8491a7fcc218946b471b600a915c8b02-Abstract-Conference.html
- PDF: paper/PatternSpecificExperts_NeurIPS2025.pdf | 官方代码: github.com/syrGitHub/TFPS
- 一句话: TFPS：双域编码器(时+频) + 子空间聚类识别 patch 模式 + MoPE(模式专家混合)路由，面向 patch 级分布偏移(概念漂移)。
- 相关性: 线 A 部分覆盖（"自适应选择"思想相似）。
  - 差异: 路由的是**时间模式**(patch 聚类)，我们门控的是**通道对**；其偏移度量用 Wasserstein，我们用 HSIC CV；无独立性/因果概念。
- 详细分析: surveys/03_multiscale_causal_decoupling/paper_analysis_deep.md §6
