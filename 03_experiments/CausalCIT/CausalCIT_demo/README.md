# CausalCIT Demo: 因果通道交互Transformer

> **Causal Channel Interaction Transformer** — 基于因果稳定性检验的自适应通道交互机制

## 核心思想

在PatchTST的Channel-Independent基础上，引入**因果稳定性门控**，选择性地对通道间进行交互建模：

- **稳定的通道依赖**（真实因果关系）→ 允许通道交互 → 提升预测精度
- **不稳定的通道依赖**（虚假相关）→ 阻断通道交互 → 避免引入噪声

## 创新点

| 对比维度 | PatchTST | Adapformer | CN (ICML'25) | **CausalCIT** |
|----------|----------|------------|--------------|---------------|
| 通道策略 | 完全独立 | 相关性度量 | 仿射区分 | **因果稳定性检验** |
| OOD考量 | ❌ | ❌ | ❌ | ✅ 核心目标 |
| 理论基础 | - | 统计相关 | 归一化理论 | **因果推断/稳定学习** |

## 项目结构

```
CausalCIT_demo/
├── models/
│   ├── layers.py           # 基础组件 (位置编码/RevIN/序列分解)
│   ├── patchtst.py         # PatchTST Baseline (自包含)
│   ├── causal_channel.py   # ★ 因果通道交互模块 (核心创新)
│   └── causalcit.py        # CausalCIT 完整模型
├── utils/
│   ├── data.py             # 数据加载 (ETT + 合成数据)
│   ├── metrics.py          # 评估指标
│   └── trainer.py          # 训练器
├── run_demo.py             # ★ 一键运行入口
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行全部实验（合成数据，无需额外数据文件）

```bash
python run_demo.py
```

### 3. 指定实验

```bash
# 仅合成数据实验（验证因果通道识别）
python run_demo.py --exp synthetic

# 仅OOD鲁棒性实验
python run_demo.py --exp ood

# 真实数据实验（需要ETTh1.csv）
python run_demo.py --exp real --data_path ./data/ETTh1.csv

# 全部实验（含真实数据）
python run_demo.py --exp all --use_real_data --data_path ./data/ETTh1.csv
```

### 4. 自定义参数

```bash
python run_demo.py \
    --seq_len 96 --pred_len 96 \
    --d_model 16 --d_ff 128 --e_layers 3 \
    --n_envs 4 --rff_dim 32 --fusion_alpha 0.3 \
    --epochs 30 --batch_size 32 --lr 0.001
```

## 实验说明

### 实验1: 合成数据 — 因果通道识别

构造含7个通道的合成时序数据：
- **Ch0-2**: 真实因果依赖（线性/非线性）
- **Ch3-4**: 虚假相关（分布漂移/混淆变量）
- **Ch5-6**: 独立噪声

**验证目标**：CausalCIT的门控矩阵应给因果通道对**高分**，虚假/独立通道对**低分**。

### 实验2: 真实数据 (ETTh1)

在标准benchmark上对比PatchTST vs CausalCIT的预测性能。

### 实验3: OOD鲁棒性

在分布漂移场景下对比两个模型：
- 训练集和测试集使用不同分布的数据
- **核心验证**：CausalCIT是否在虚假相关通道上表现更鲁棒

## 输出文件

运行后在 `./output/` 目录下生成：
- `synthetic_results.png` — 合成数据实验综合可视化
- `ood_results.png` — OOD鲁棒性实验可视化
- `real_data_results.png` — 真实数据实验可视化（如果运行）
- `experiment_report.md` — Markdown格式实验报告

## 技术原理

### 因果稳定性门控 (CausalStabilityGate)

```
输入时序 → 划分为N个"环境"(时间段)
         → 每个环境中计算通道对的HSIC依赖
         → 计算HSIC的跨环境变异系数(CV)
         → CV低 = 稳定依赖 = 因果 → 高门控
         → CV高 = 不稳定依赖 = 虚假 → 低门控
```

### 数据流

```
Input [B, L, C]
  → RevIN Norm
  → Patching [B, C, P_num, P_len]
  → CI-Encoder (PatchTST) [B, C, d_model, P_num]
  → ★ CausalChannelInteraction:
      ├─ Stability Gate → gate_matrix [B, C, C]
      ├─ Channel Pooling → [B, C, d_model]
      ├─ Gated Channel Attention → [B, C, d_model]
      └─ Adaptive Fusion (α blend)
  → Flatten Head → [B, C, pred_len]
  → RevIN Denorm
```

## 参考文献

- PatchTST (ICLR 2023): A Time Series is Worth 64 Words
- StableNet (CVPR 2021): Stable Learning via Sample Reweighting
- FOIL (ICML 2024): OOD Generalization for Time Series
- CN (ICML 2025): Channel Normalization for Time Series
- CGTFra (ICML 2026): Robust Inter-Series Dependency Modeling
