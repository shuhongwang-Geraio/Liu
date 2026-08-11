# PCD vs CausalCIT 对比报告

> 生成时间: 2026-08-11 13:45:49
> 数据集: ['syn_ood'] | 变体: ['patchtst', 'full_v2', 'pcd_gate'] | seeds: [42, 123, 2024, 5, 6] | epochs: 20
> 协议: seed 配对 Wilcoxon (n>=5), 组内 Holm 校正

## syn_ood

| 变体 | MSE mean | MSE std | MAE mean | vs PatchTST | vs full_v2 |
|------|---------|---------|---------|------------|------------|
| patchtst | 0.318576 | 0.000718 | 0.380114 | +0.00% (p=-, Holm=-) |
| full_v2 | 0.322037 | 0.000524 | 0.384597 | -1.09% (p=0.0625, Holm=0.1250) |
| pcd_gate | 0.322018 | 0.001108 | 0.384697 | -1.08% (p=0.0625, Holm=0.1250) |

**full_v2 vs pcd_gate**: full_v2 MSE=0.322037, pcd_gate MSE=0.322018, full_v2 相对 pcd_gate +0.01% (Wilcoxon p=1.0000)
