# CausalCIT 多seed聚合与配对显著性检验

> seeds: [42, 123, 2024, 7, 99]
> 检验方法: scipy.stats.ttest_rel (配对t检验) / wilcoxon (配对符号秩检验)
> 解读: 提升率% = (base_mse - variant_mse) / base_mse * 100；p<0.05 表示显著优于基线

## 合成数据 (配对检验: 各变体 vs PatchTST)

| 变体 | mean MSE | std MSE | 提升% mean | 提升% std | t-test p | Wilcoxon p |
|------|---------|---------|-----------|-----------|----------|------------|
| w/o Gate (full attention) | 0.490616 | 0.002239 | -1.06 | 0.49 | 0.0120 | 0.0625 |
| w/o EnvSplit (global HSIC) | 0.487662 | 0.005507 | -0.45 | 1.08 | 0.4505 | 0.8125 |
| w/o HSIC (Pearson corr) | 0.489963 | 0.003192 | -0.93 | 0.85 | 0.0961 | 0.1875 |
| Full CausalCIT (Ours) | 0.487730 | 0.005503 | -0.46 | 1.08 | 0.4385 | 0.8125 |
| Full CausalCIT (fix prior) | 0.487812 | 0.005419 | -0.48 | 1.06 | 0.4164 | 0.8125 |

## ETTh1 pred_len=96 (配对检验 vs PatchTST)

| 变体 | mean MSE | std MSE | 提升% mean | 提升% std | t-test p | Wilcoxon p |
|------|---------|---------|-----------|-----------|----------|------------|
| w/o Gate (full attention) | 0.380131 | 0.001115 | -0.95 | 1.06 | 0.1518 | 0.1875 |
| w/o EnvSplit (global HSIC) | 0.380365 | 0.002235 | -1.01 | 0.86 | 0.0786 | 0.1250 |
| w/o HSIC (Pearson corr) | 0.380409 | 0.001409 | -1.02 | 0.86 | 0.0783 | 0.1250 |
| Full CausalCIT (Ours) | 0.380396 | 0.002244 | -1.02 | 0.84 | 0.0718 | 0.1250 |
| Full CausalCIT (fix prior) | 0.380626 | 0.002023 | -1.08 | 0.85 | 0.0638 | 0.1250 |

## ETTh1 pred_len=336 (配对检验 vs PatchTST)

| 变体 | mean MSE | std MSE | 提升% mean | 提升% std | t-test p | Wilcoxon p |
|------|---------|---------|-----------|-----------|----------|------------|
| w/o Gate (full attention) | 0.476884 | 0.006153 | +1.09 | 1.49 | 0.2142 | 0.3125 |
| w/o EnvSplit (global HSIC) | 0.478954 | 0.008200 | +0.66 | 1.98 | 0.5347 | 0.6250 |
| w/o HSIC (Pearson corr) | 0.472365 | 0.010413 | +2.03 | 2.22 | 0.1388 | 0.1250 |
| Full CausalCIT (Ours) | 0.478892 | 0.008274 | +0.68 | 1.99 | 0.5291 | 0.6250 |
| Full CausalCIT (fix prior) | 0.478883 | 0.008283 | +0.68 | 1.99 | 0.5283 | 0.6250 |

## 结论

- 若某变体提升% mean>0 但 p 值不显著，说明效应被训练噪声淹没（诊断报告异常1）。
- 若 Full 与 w/o EnvSplit 在多个 pred_len 上均不显著且提升%接近，印证两条路径学到等价门控（诊断报告异常2）。
- full_fix 与 full 的差异反映先验权重(0.3→0.1)的影响（诊断报告假设A）。
