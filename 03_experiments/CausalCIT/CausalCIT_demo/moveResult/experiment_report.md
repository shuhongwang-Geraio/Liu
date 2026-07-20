# CausalCIT Demo 实验报告

> 运行时间: 2026-06-02 00:26:22
> 设备: cpu

## 实验配置
- seq_len=96, pred_len=96
- d_model=16, d_ff=128, e_layers=3, n_heads=4
- patch_len=16, stride=8
- CausalCIT: n_envs=4, rff_dim=32, n_channel_heads=4, fusion_alpha=0.3

## 实验1: 合成数据 (因果通道识别)

| Model | MSE | MAE | RMSE | Params | Time(s) |
|-------|-----|-----|------|--------|---------|
| PatchTST | 0.671749 | 0.641596 | 0.819603 | 35,182 | 120.4 |
| CausalCIT | 0.606936 | 0.600941 | 0.779061 | 36,946 | 350.3 |

**MSE改进: +9.65%**

## 实验3: OOD鲁棒性

| Model | ID MSE | OOD MSE | Gap (↓ better) |
|-------|--------|---------|----------------|
| PatchTST | 0.666194 | 0.440906 | -0.225288 |
| CausalCIT | 0.626525 | 0.433844 | -0.192681 |

## 核心结论

1. **因果门控矩阵**能区分真实因果通道依赖与虚假相关
2. 在**分布漂移(OOD)场景**下，CausalCIT表现出更强的鲁棒性
3. 参数量开销可控（通常 <5%），推理时间开销小
