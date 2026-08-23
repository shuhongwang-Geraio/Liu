# CausalCIT 大规模实验报告 (full_v2 vs baselines)

> 生成时间: 2026-08-20 22:40:14
> 数据集: ['electricity', 'weather']
> 变体: ['capacity_match']
> 每个 (数据集, horizon) 下跨 seed 报告 mean±std MSE/MAE，以及 full_v2 相对 PatchTST 的提升%
> 显著性: seed 配对 Wilcoxon 符号秩检验 (双侧), 同组内 (同数据集×horizon) 跨变体 Holm 校正;
> n<5 对 seed 时不报 p 值 (功效不足). 已弃用方向不可辨的 t-test 报法 (P0-4).

## 数据集: electricity

### pred_len = 96

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| capacity_match | 0.160814 | 0.001964 | 0.254048 | - | 8 | - | - |  |

### pred_len = 192

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| capacity_match | 0.171853 | 0.001711 | 0.262972 | - | 5 | - | - |  |

## 数据集: weather

### pred_len = 96

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| capacity_match | 0.147850 | 0.001729 | 0.194039 | - | 7 | - | - |  |

### pred_len = 192

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| capacity_match | 0.191968 | 0.001745 | 0.234093 | - | 8 | - | - |  |

### pred_len = 336

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| capacity_match | 0.223927 | 0.001183 | 0.263386 | - | 8 | - | - |  |

---

## full_v2 提升率汇总 (vs PatchTST, seed 配对)

| 数据集 | pred_len | 提升% mean | 提升% std | #seed | Wilcoxon p | Holm p | 显著 |
|--------|----------|-----------|-----------|------|-----------|--------|------|

## 平均提升率 (按数据集, 跨 horizon × seed, seed 配对)

| 数据集 | full_v2 提升% mean | #runs |
|--------|-------------------|-------|
