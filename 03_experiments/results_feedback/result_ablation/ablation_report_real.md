# CausalCIT 真实数据消融报告

> 运行时间: 2026-06-04 21:00:03
> 设备: cuda

## ETTh1 真实数据消融

### pred_len = 96

| 变体 | MSE | MAE | vs PatchTST |
|------|-----|-----|-------------|
| PatchTST (no interaction) | 0.377050 | 0.398783 | +0.00% |
| w/o Gate (full attention) | 0.378378 | 0.396959 | -0.35% |
| w/o EnvSplit (global HSIC) | 0.381314 | 0.397239 | -1.13% |
| w/o HSIC (Pearson corr) | 0.380729 | 0.397439 | -0.98% |
| Full CausalCIT (Ours) | 0.379863 | 0.396639 | -0.75% |

### pred_len = 336

| 变体 | MSE | MAE | vs PatchTST |
|------|-----|-----|-------------|
| PatchTST (no interaction) | 0.477108 | 0.448774 | +0.00% |
| w/o Gate (full attention) | 0.465554 | 0.450602 | +2.42% |
| w/o EnvSplit (global HSIC) | 0.486039 | 0.458458 | -1.87% |
| w/o HSIC (Pearson corr) | 0.484203 | 0.452524 | -1.49% |
| Full CausalCIT (Ours) | 0.486507 | 0.452068 | -1.97% |

