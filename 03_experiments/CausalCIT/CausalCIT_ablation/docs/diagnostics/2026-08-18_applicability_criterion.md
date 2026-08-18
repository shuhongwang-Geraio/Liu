# 方案 1: 训练前适用性判据 — 统计量 vs P0-1 增益对应 (2026-08-18)

> 数据: `compute_pre_train_stats.py` (4 数据集, 本机) + P0-1 主表
> `output_large_v3/results_shard*.csv` (8-seed 配对)。
> 脚本: `correspond_analysis.py` (对应分析) / `_fixAB_boundary.py` (修复版边界)。

## 1. 对应表 (full_v2_fixed vs patchtst, 8-seed 配对增益)

| dataset | pl | gain% | 依赖密度 | 语义信息量(season) | 稳定占比(season) |
|---------|----|-------|---------|-------------------|-----------------|
| traffic | 96 | **+8.71** | (待补) | (待补) | (待补) |
| traffic | 192 | **+6.88** | | | |
| electricity | 96 | **+7.19** | (待补) | (待补) | (待补) |
| electricity | 192 | **+3.89** | | | |
| ili | 48 | +6.40 | (待补) | (待补) | (待补) |
| weather | 96 | +3.01 | 0.297 | 4.2× | 0.157 |
| exchange | 192 | +1.58 | 0.513 | 3.7× | 0.714 |
| ettm1 | 96 | +1.10 | 0.224 | 25.7× | 0.238 |
| exchange | 96 | +0.31 | 0.513 | 3.7× | 0.714 |
| weather | 192 | +0.60 | 0.297 | 4.2× | 0.157 |
| etth1 | 192/336 | ≈0 | 0.222 | 10.6× | 0.238 |
| ettm1 | 192/336 | -1.4~-2.2 | 0.224 | 25.7× | 0.238 |
| weather | 336 | -1.35 | 0.297 | 4.2× | 0.157 |
| etth1 | 96 | -1.80 | 0.222 | 10.6× | 0.238 |
| ili | 24 | **-11.51** | (待补) | (待补) | (待补) |

## 2. 单因子评估 (4 数据集 × 11 horizon 组)

| 统计量 | 符号一致率 | 判断 |
|--------|-----------|------|
| **依赖密度 avg\|corr\|** | **9/11** | 最有希望: 密度高→正增益 (exchange 0.513→全正; etth1 0.222→负/平) |
| 稳定通道占比 (season) | 8/11 | 次之, 与密度高度共线 |
| 语义信息量 (season) | 5/11 | **无预测力** (ettm1 25.7× 却负增益) → 该统计量与"修复版能否利用"无关 |

关键反例: ettm1 语义信息量最高 (25.7×) 但增益负 → 语义环境有信息 ≠ 门控能用上
(与"容量 vs 选择"的脱节一致)。weather 稳定占比最低 (0.157) 但 pl96 +3.01% → 单因子不成立。

## 3. 独立维度: horizon 效应

同一数据集内, 增益随 horizon 单调下降: weather +3.01→+0.60→-1.35,
ettm1 +1.10→-1.39→-2.18, etth1 -1.80→-0.07→-0.06 (pl96 特弱), traffic +8.71→+6.88。
→ 适用性判据需**按 horizon 分层**: 短 horizon (≤96) 是门控有效区间。

## 4. 修复版 (修 A+B) 的适用边界 (full_v2 vs full_v2_fixed, 8-seed 配对)

| 数据集 | 修复版相对旧版 | 判定 |
|--------|--------------|------|
| weather pl96 / pl192 | **-3.25% / -1.50% (8/8, 7/8 seed 一致)** | 修复版大幅有效 |
| electricity pl96 / pl192 | **-2.01% / -3.04% (8/8, 8/8)** | 修复版全一致有效 |
| traffic pl96 / pl192 | **+1.18% / +1.06% (0/8)** | **修复版失效 (862 通道)** |
| etth1/ettm1/exchange/ili | ±0.7% 内无方向 | 修复版无影响 (低维) |

**边界结论**: median 带宽 + CKA 归一化在中等通道数 (21~321) 有效, 在 862 通道超高维失效
(0/8 seed 一致变差) —— median heuristic 带宽在超高维失真。这解释了为何 traffic 上
full_v2_fixed 仍 +8.71% (主要来自架构容量) 但略逊旧版。

## 5. 结论与论文表述

1. **判据候选**: "依赖密度 × 短 horizon" 组合可粗分有效区间 (traffic/elec/ili48/weather96 正,
   etth1 负); 需要补 traffic/electricity/ili 统计量后验证 (若 traffic 密度高且正增益 → 判据成立)。
2. **诚实边界**: 判据是"统计趋势"非严格判别 (4 数据集样本小, 11 组里 9/11 一致);
   论文表述为 "a pre-train heuristic with 9/11 sign agreement (n=4 datasets)"。
3. **修复版边界** 本身是可写进论文的贡献: "带宽启发式的适用区间"。

## 6. 待办 (服务器)

```bash
# 补 traffic / electricity / ili 统计量 (数据在服务器, 近 0 GPU)
python compute_pre_train_stats.py --data <服务器数据目录>/traffic.csv --name traffic --out _stats_traffic.json
python compute_pre_train_stats.py --data <服务器数据目录>/electricity.csv --name electricity --out _stats_electricity.json
python compute_pre_train_stats.py --data <服务器数据目录>/ILI.csv --name ILI --out _stats_ILI.json
# 然后本机重跑 correspond_analysis.py 做 7 数据集对应
```
