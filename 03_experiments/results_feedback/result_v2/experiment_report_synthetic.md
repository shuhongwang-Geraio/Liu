# CausalCIT 合成数据实验报告 (v2)

> 运行时间: 2026-06-04 20:25:05
> 设备: cuda

## 实验1: 增强版合成数据 (d_model=64, 50 epochs)

### 配置
- d_model=64, d_ff=256, e_layers=3, n_heads=4
- rff_dim=64, n_envs=4, fusion_alpha=0.3
- epochs=50, patience=10, 训练样本=8000

### 结果

| Model | MSE | MAE | RMSE | Params | Time(s) |
|-------|-----|-----|------|--------|---------|
| PatchTST | 0.484650 | 0.527967 | 0.696168 | 225,646 | 24.8 |
| CausalCIT | 0.494863 | 0.533836 | 0.703465 | 250,834 | 31.1 |

**MSE改进: -2.11%**

参数量开销: +11.2%

## 结论

⚠️ CausalCIT 在合成数据上未优于 PatchTST，MSE差异 -2.11%
- **需要排查**: HSIC环境划分、门控阈值、训练稳定性
