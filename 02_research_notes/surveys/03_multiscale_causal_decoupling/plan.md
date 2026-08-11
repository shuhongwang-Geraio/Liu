Plan: 文献调研：多尺度序列建模 + 多通道因果解耦
=============================

Progress: 3/3 stages completed (100%)
Status: 3 completed, 0 in progress, 0 blocked, 0 not started

Stages:

1. [✓] 查看用户上传的知识库附件，了解已有调研基础（01_external/ 和 02_research_notes/ 下的归档文献），明确检索边界，避免重复检索已读文献
2. [✓] 针对6个子问题并行进行系统性学术文献检索：(1)因果/稳定性驱动的通道交互选择；(2)方法有效性与通道数/维度关系；(3)多尺度/异构采样率与通道联合建模；(4)可逆/正交变换+独立性约束的通道解耦；(5)解缠表示学习在时序预测中的应用；(6)方法适用边界与失败分析
3. [✓] 整合所有检索结果，按子问题分节撰写调研报告：包含结论性判断（已被覆盖/部分覆盖/空白）、证据列表、Top3最近似工作差距分析、三条技术线创新点评估及差异化定位建议，确保所有论文引用包含可验证链接

## 归档记录 (2026-08-10)

调研产物按 research-org 规范归档：

- `report_final.md` — 最终综合报告（6 子问题 + 三条线创新点评估 + claim 定位建议）
- `stage1_baseline.md` — 阶段 1 基线（已有 22 篇文献 + 子问题关键词清单）
- `stage2_channel_interaction.md` — 阶段 2 通道交互与解耦调研中间稿
- `paper_analysis_deep.md` — 7 篇重点论文深度阅读分析（2026-08-10 补）
- 对应论文 PDF/代码归档于 `01_external/`（20 篇下载了 PDF、9 个 GitHub 仓库 clone、其余付费墙论文建链接索引）

## 归档更新 (2026-08-10，深读轮)

用户手动下载的 7 篇重点论文已全部阅读、分析并归档：
- 7 个 PDF 已入库：`AdaptiveLatentDecomposition`、`DynamicFractalMamba`、`DatasetDrivenChannelMasks`、`UnveilingLimitations`、`UnderstandingMoirai`、`PatternSpecificExperts`、`CrossScaleAttention`
- 补 clone 3 个官方代码：DF-Mamba (yzlab1)、PCD (YonseiML)、TFPS (syrGitHub)
- 每个目录 README 已更新为"已下载 + 核心分析"
- 关键结论见 `paper_analysis_deep.md`：PCD 论文的维度效应（PEMS +12.7~40.2% vs ETTh +0.3~2.8%）与本项目"高维有效/低维失效"观察几乎一致，是 claim 定位的最强外部佐证；DF-Mamba 与 Cross-Scale Attention 双重覆盖线 B，线 B 需转向异构采样率。
