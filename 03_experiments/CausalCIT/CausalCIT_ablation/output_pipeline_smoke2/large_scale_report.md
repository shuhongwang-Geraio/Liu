# CausalCIT 大规模实验报告 (full_v2 vs baselines)

> 生成时间: 2026-08-10 21:50:06
> 数据集: ['syn_ood']
> 变体: ['dlinear', 'itransformer', 'patchtst']
> 每个 (数据集, horizon) 下跨 seed 报告 mean±std MSE/MAE，以及 full_v2 相对 PatchTST 的提升%
> 显著性: seed 配对 Wilcoxon 符号秩检验 (双侧), 同组内 (同数据集×horizon) 跨变体 Holm 校正;
> n<5 对 seed 时不报 p 值 (功效不足). 已弃用方向不可辨的 t-test 报法 (P0-4).

## 数据集: syn_ood

### pred_len = 96

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| dlinear | 0.411147 | 0.000000 | 0.476184 | -26.12% | 1 | - | - |  |
| itransformer | 0.335830 | 0.000000 | 0.405108 | -3.02% | 1 | - | - |  |
| patchtst | 0.325996 | 0.000000 | 0.397129 | - | 1 | - | - |  |

### pred_len = 192

| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|---------|-------------------|-------|-----------|--------|------|
| dlinear | 0.510927 | 0.000000 | 0.544563 | -47.24% | 1 | - | - |  |
| itransformer | 0.354766 | 0.000000 | 0.426557 | -2.24% | 1 | - | - |  |
| patchtst | 0.347007 | 0.000000 | 0.421317 | - | 1 | - | - |  |

---

## full_v2 提升率汇总 (vs PatchTST, seed 配对)

| 数据集 | pred_len | 提升% mean | 提升% std | #seed | Wilcoxon p | Holm p | 显著 |
|--------|----------|-----------|-----------|------|-----------|--------|------|

## 平均提升率 (按数据集, 跨 horizon × seed, seed 配对)

| 数据集 | full_v2 提升% mean | #runs |
|--------|-------------------|-------|
