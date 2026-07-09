# CausalCIT Demo 实验结果目录

## 目录说明

本目录包含 CausalCIT（因果通道交互Transformer）的实验结果，用于迁移到服务器进行后续分析。

---

## 文件清单与来源

| 文件 | 来源 | 说明 |
|------|------|------|
| `experiment_report.md` | `../output/experiment_report.md` | 自动生成的实验报告，包含所有实验数据表格 |
| `synthetic_results.png` | `../output/synthetic_results.png` | 合成数据实验可视化（因果门控矩阵热力图等） |
| `ood_results.png` | `../output/ood_results.png` | OOD鲁棒性实验可视化（分布漂移场景对比） |
| `checkpoint.pth` | `../output/ckpt_causalcit_syn/checkpoint.pth` | CausalCIT模型在合成数据上的训练权重 |
| `cmd.txt` | 运行日志 | 完整的命令行输出记录 |

---

## 实验配置

| 参数 | 值 |
|------|-----|
| 序列长度 (seq_len) | 96 |
| 预测长度 (pred_len) | 96 |
| 模型维度 (d_model) | 16 |
| 前馈维度 (d_ff) | 128 |
| 编码器层数 (e_layers) | 3 |
| 注意力头数 (n_heads) | 4 |
| 环境数量 (n_envs) | 4 |
| 融合系数 (fusion_alpha) | 0.3 |

---

## 主要实验结果

### 1. 合成数据实验（因果通道识别）

| 模型 | MSE | MAE | 参数量 |
|------|-----|-----|--------|
| PatchTST | 0.6717 | 0.6416 | 35,182 |
| CausalCIT | **0.6069** | **0.6009** | 36,946 |

**结论**：CausalCIT MSE降低9.65%，参数量仅增加5%

### 2. OOD鲁棒性实验

| 模型 | ID MSE | OOD MSE | 鲁棒性差距 |
|------|--------|---------|------------|
| PatchTST | 0.6662 | 0.4409 | -0.2253 |
| CausalCIT | **0.6265** | **0.4338** | **-0.1927** |

**结论**：CausalCIT在分布漂移场景下表现更鲁棒

---

## 生成方式

本目录内容由以下命令生成：

```bash
# 运行完整demo（合成数据+OOD实验）
python run_demo.py

# 复制关键结果到moveResult
cp output/experiment_report.md moveResult/
cp output/synthetic_results.png moveResult/
cp output/ood_results.png moveResult/
cp output/ckpt_causalcit_syn/checkpoint.pth moveResult/
```

---

## 环境信息

- 运行设备：CPU（建议服务器使用GPU加速）
- Python版本：3.11.x
- PyTorch版本：2.0+
- 运行时间：2026-06-02 00:26:22

---

## 后续建议

1. 在服务器上使用GPU重新训练（`--device cuda`）
2. 运行真实数据实验（需ETTh1.csv）
3. 调整参数进行消融实验