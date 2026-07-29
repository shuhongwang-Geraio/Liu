# CausalCIT 消融实验报告

> 运行时间: 2026-07-22 00:45:17
> 设备: cuda
> seeds: [42, 123, 2024, 7, 99]

## 消融变体说明

| 变体 | HSIC检验 | 环境划分 | 门控选择 | 说明 |
|------|---------|---------|---------|------|
| PatchTST | ❌ | ❌ | ❌ | 纯Channel-Independent基线 |
| w/o Gate | ❌ | ❌ | ❌ | 全连接通道注意力，无选择性门控 |
| w/o EnvSplit | ✅ | ❌ | ✅ | 全局HSIC，不划分环境 |
| w/o HSIC | ❌ | ✅ | ✅ | 用Pearson相关性替代HSIC |
| **Full CausalCIT** | **✅** | **✅** | **✅** | **完整模型** |
| **Full (fix prior)** | **✅** | **✅** | **✅** | **先验权重0.3→0.1 (诊断变体)** |

---

## 合成数据消融 (d_model=64, 50 epochs, 5 seeds)

| 变体 | MSE mean | MSE std | MAE mean | MAE std | Params | Time(s) |
|------|---------|---------|---------|---------|--------|---------|
| PatchTST (no interaction) | 0.485475 | 0.001032 | 0.527364 | 0.001339 | 225,646 | 26 |
| w/o Gate (full attention) | 0.490616 | 0.002239 | 0.531155 | 0.001999 | 250,735 | 28 |
| w/o EnvSplit (global HSIC) | 0.487662 | 0.005507 | 0.529317 | 0.003239 | 250,833 | 34 |
| w/o HSIC (Pearson corr) | 0.489963 | 0.003192 | 0.530249 | 0.001140 | 250,834 | 39 |
| Full CausalCIT (Ours) | 0.487730 | 0.005503 | 0.529415 | 0.003152 | 250,835 | 37 |
| Full CausalCIT (fix prior) | 0.487812 | 0.005419 | 0.529456 | 0.003133 | 250,835 | 37 |

### 各组件边际贡献 (MSE降低, 用跨seed均值)

| 组件 | MSE降低 |
|------|--------|
| 通道注意力 | -0.005141 |
| 门控选择 | 0.002954 |
| 环境划分 | -0.002301 |
| HSIC检验 | 0.002234 |
| **总计** | **-0.002255** |

---

## ETTh1 真实数据消融 (5 seeds)

### pred_len = 96

| 变体 | MSE mean | MSE std | MAE mean | vs PatchTST (mean) |
|------|---------|---------|---------|-------------------|
| PatchTST (no interaction) | 0.376588 | 0.003478 | 0.397451 | +0.00% |
| w/o Gate (full attention) | 0.380131 | 0.001115 | 0.397876 | -0.94% |
| w/o EnvSplit (global HSIC) | 0.380365 | 0.002235 | 0.400316 | -1.00% |
| w/o HSIC (Pearson corr) | 0.380409 | 0.001409 | 0.398198 | -1.01% |
| Full CausalCIT (Ours) | 0.380396 | 0.002244 | 0.400452 | -1.01% |
| Full CausalCIT (fix prior) | 0.380626 | 0.002023 | 0.400940 | -1.07% |

### pred_len = 336

| 变体 | MSE mean | MSE std | MAE mean | vs PatchTST (mean) |
|------|---------|---------|---------|-------------------|
| PatchTST (no interaction) | 0.482195 | 0.004214 | 0.451143 | +0.00% |
| w/o Gate (full attention) | 0.476884 | 0.006153 | 0.452307 | +1.10% |
| w/o EnvSplit (global HSIC) | 0.478954 | 0.008200 | 0.453055 | +0.67% |
| w/o HSIC (Pearson corr) | 0.472365 | 0.010413 | 0.450185 | +2.04% |
| Full CausalCIT (Ours) | 0.478892 | 0.008274 | 0.452954 | +0.68% |
| Full CausalCIT (fix prior) | 0.478883 | 0.008283 | 0.452933 | +0.69% |

---

## 结论

1. 多seed聚合: 各变体 MSE 以 mean±std 报告，std 反映训练随机性噪声量级。
2. 配对显著性检验详见 `significance_report.md`（各变体 vs PatchTST 的 t 检验 / Wilcoxon p 值）。
3. 门控矩阵逐元素差异与诊断参数见 `gate_comparison.txt` / `gate_diagnostics.txt`。
