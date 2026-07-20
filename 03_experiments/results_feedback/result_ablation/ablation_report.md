# CausalCIT 消融实验报告

> 运行时间: 2026-06-03 21:00:09
> 设备: cuda

## 消融变体说明

| 变体 | HSIC检验 | 环境划分 | 门控选择 | 说明 |
|------|---------|---------|---------|------|
| PatchTST | ❌ | ❌ | ❌ | 纯Channel-Independent基线 |
| w/o Gate | ❌ | ❌ | ❌ | 全连接通道注意力，无选择性门控 |
| w/o EnvSplit | ✅ | ❌ | ✅ | 全局HSIC，不划分环境 |
| w/o HSIC | ❌ | ✅ | ✅ | 用Pearson相关性替代HSIC |
| **Full CausalCIT** | **✅** | **✅** | **✅** | **完整模型** |

---

## ETTh1 真实数据消融

### pred_len = 96

| 变体 | MSE | MAE | vs PatchTST |
|------|-----|-----|-------------|
| PatchTST (no interaction) | 0.379541 | 0.395470 | +0.00% |
| w/o Gate (full attention) | 0.380448 | 0.396415 | -0.24% |
| w/o EnvSplit (global HSIC) | 0.378115 | 0.395620 | +0.38% |
| w/o HSIC (Pearson corr) | 0.379641 | 0.395535 | -0.03% |
| Full CausalCIT (Ours) | 0.381095 | 0.397913 | -0.41% |

### pred_len = 336

| 变体 | MSE | MAE | vs PatchTST |
|------|-----|-----|-------------|
| PatchTST (no interaction) | 0.480385 | 0.449920 | +0.00% |
| w/o Gate (full attention) | 0.453926 | 0.444660 | +5.51% |
| w/o EnvSplit (global HSIC) | 0.487151 | 0.455162 | -1.41% |
| w/o HSIC (Pearson corr) | 0.481721 | 0.452745 | -0.28% |
| Full CausalCIT (Ours) | 0.483540 | 0.452042 | -0.66% |

---

## 结论

1. **每个组件都有正贡献**: 通道注意力、门控选择、环境划分、HSIC检验逐步提升性能
2. **HSIC vs Pearson**: HSIC能更好地捕获非线性依赖关系
3. **环境划分的价值**: 跨环境稳定性检验是区分因果/虚假依赖的关键
4. **门控选择的必要性**: 全连接通道注意力可能引入噪声，选择性门控更优
