# PCD vs CausalCIT 对比报告

> 生成时间: 2026-08-10 21:19:56
> 数据集: ['syn_ood'] | 变体: ['patchtst', 'full_v2', 'pcd_gate'] | seeds: [42, 123] | epochs: 3
> 协议: seed 配对 Wilcoxon (n>=5), 组内 Holm 校正

## syn_ood

| 变体 | MSE mean | MSE std | MAE mean | vs PatchTST | vs full_v2 |
|------|---------|---------|---------|------------|------------|
| patchtst | 0.320446 | 0.001675 | 0.386153 | +0.00% (p=-, Holm=-) |
| full_v2 | 0.320754 | 0.000115 | 0.386089 | -0.10% (p=-, Holm=-) |
| pcd_gate | 0.320441 | 0.000138 | 0.385698 | +0.00% (p=-, Holm=-) |

**full_v2 vs pcd_gate**: full_v2 MSE=0.320754, pcd_gate MSE=0.320441, full_v2 相对 pcd_gate +0.10% (Wilcoxon p=nan)
