# CausalCIT 大规模实验报告 (full_v2 vs baselines)

> 生成时间: 2026-08-11 23:55:10
> 数据集: ['electricity', 'traffic', 'weather']
> 变体: ['full_v2_fixed']
> 每个 (数据集, horizon) 下跨 seed 报告 mean±std MSE/MAE，以及 full_v2 相对 PatchTST 的提升%
> 显著性: seed 配对 Wilcoxon 符号秩检验 (双侧), 同组内 (同数据集×horizon) 跨变体 Holm 校正;
> n<5 对 seed 时不报 p 值 (功效不足). 已弃用方向不可辨的 t-test 报法 (P0-4).

## 数据集: electricity

### pred_len = 96

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| full_v2_fixed | 0.159459 | 0.002162 | 0.252891 | - | 8 | - | - |  |

### pred_len = 192

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| full_v2_fixed | 0.170626 | 0.002051 | 0.262349 | - | 8 | - | - |  |

## 数据集: traffic

### pred_len = 96

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| full_v2_fixed | 0.503044 | 0.006400 | 0.342836 | - | 8 | - | - |  |

### pred_len = 192

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| full_v2_fixed | 0.506568 | 0.004590 | 0.336603 | - | 8 | - | - |  |

## 数据集: weather

### pred_len = 96

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| full_v2_fixed | 0.145490 | 0.001366 | 0.193067 | - | 8 | - | - |  |

### pred_len = 192

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| full_v2_fixed | 0.192502 | 0.001691 | 0.235532 | - | 8 | - | - |  |

### pred_len = 336

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| full_v2_fixed | 0.224448 | 0.000737 | 0.263690 | - | 8 | - | - |  |

---

## full_v2 提升率汇总 (vs PatchTST, seed 配对)

| 数据集 | pred_len | 提升% mean | 提升% std | #seed | Wilcoxon p | Holm p | 显著 |
|--------|----------|-----------|-----------|------|-----------|--------|------|

## 平均提升率 (按数据集, 跨 horizon × seed, seed 配对)

| 数据集 | full_v2 提升% mean | #runs |
|--------|-------------------|-------|
