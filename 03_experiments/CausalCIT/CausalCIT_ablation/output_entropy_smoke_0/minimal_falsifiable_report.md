# 最小可证伪测试报告 (回应评审 re2 §6.1)

> 生成时间: 2026-08-10 16:54:27
> 数据集: syn_ood | 变体: ['patchtst', 'full_v2', 'full_v2_fixed', 'gate_prior_only'] | seed: [42]

说明: 本报告在同一份文件里对齐两条证据链 —— (A) MSE 是否有提升, (B) 提升是否伴随门控行为的合理变化 (非坍缩、不依赖测试batch组成)。只有当 full_v2 同时满足 (A) 显著优于 capacity_match/gate_prior_only 且 (B) 门控未坍缩、batch_dep_score 与 full_v2_fixed 差异可解释时，才能支持'因果稳定性门控'这个说法；否则应把结论降级。

## pred_len = 96

### (A) MSE 证据链: vs PatchTST

| 变体 | MSE mean | MSE std | #seed | 提升%(vs PatchTST) | Wilcoxon p | Holm p | 显著 |
|------|---------|---------|-------|--------------------|-----------|--------|------|
| patchtst | 0.325996 | 0.000000 | 1 | - | - | - |  |
| full_v2 | 0.325556 | 0.000000 | 1 | +0.13% | - | - |  |
| full_v2_fixed | 0.325526 | 0.000000 | 1 | +0.14% | - | - |  |
| gate_prior_only | 0.325548 | 0.000000 | 1 | +0.14% | - | - |  |

### (A') MSE 证据链: full_v2 vs 关键对照 (非 vs-PatchTST)

| 对照变体 | full_v2 MSE mean | 对照 MSE mean | full_v2提升% | #seed | Wilcoxon p | Holm p | 显著 |
|---------|-------------------|---------------|-------------|-------|-----------|--------|------|
| full_v2_fixed | 0.325556 | 0.325526 | -0.01% | 1 | - | - |  |
| gate_prior_only | 0.325556 | 0.325548 | -0.00% | 1 | - | - |  |

### (B) 门控行为证据链 (off_mean/off_std=坍缩检测; batch_dep_score=测试时是否依赖batch组成，越接近0越好)

| 变体 | off_mean | off_std | 坍缩? | batch_dep_score mean | batch_dep_score max | #seed |
|------|---------|---------|-------|----------------------|---------------------|-------|
| full_v2 | 0.1868 | 0.1619 | 否 | nan | nan | 1 |
| full_v2_fixed | 0.1359 | 0.0208 | 否 | nan | nan | 1 |
| gate_prior_only | 0.1756 | 0.0033 | 否 | nan | nan | 1 |
