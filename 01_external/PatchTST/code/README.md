# PatchTST 论文复现

> **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**
>
> Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
>
> ICLR 2023 | IBM Research
>
> [[论文]](https://arxiv.org/abs/2211.14730) [[官方代码]](https://github.com/yuqinie98/PatchTST)

---

## 1. 论文简介

PatchTST 是一种基于 Transformer 的长期时间序列预测模型，提出了两个核心创新：

### Patching（分块机制）

将长度为 $L$ 的输入时间序列分割为若干长度为 $P$、步长为 $S$ 的 patches，作为 Transformer 的输入 tokens。

- **降低计算复杂度**：token 数量从 $L$ 降至约 $L/S$，自注意力复杂度平方级降低
- **保留局部语义**：每个 patch 包含连续的时间步信息，类似 NLP 中的"词"
- **更长回看窗口**：在相同计算预算下，可使用更长的历史输入

### Channel-Independence（通道独立）

- 每个变量（通道）独立通过同一个 Transformer 编码器
- 所有通道共享 embedding 和 Transformer 权重
- 避免了多变量之间复杂交互带来的过拟合问题

### 模型架构

```
Input [B, L, C] 
    → Permute [B, C, L] 
    → Patching [B, C, N, P] 
    → Linear Projection [B*C, N, D] 
    → + Positional Encoding 
    → Transformer Encoder (×layers) 
    → Flatten Head 
    → Output [B, T, C]
```

其中 B=batch, L=seq_len, C=channels, N=patch_num, P=patch_len, D=d_model, T=pred_len。

---

## 2. 项目结构

```
patchtst/
├── README.md                   # 项目说明（本文件）
├── 实验步骤.md                  # 详细复现步骤指南
├── requirements.txt            # Python 依赖
├── run_longExp.py              # 主入口文件（参数解析 + 启动训练/测试）
├── download_data.py            # 数据集自动下载脚本
│
├── models/
│   └── PatchTST.py            # 模型顶层封装（支持可选的序列分解）
│
├── layers/
│   ├── PatchTST_backbone.py   # 核心架构：RevIN → Patching → Encoder → Head
│   ├── PatchTST_layers.py     # 基础组件：位置编码、激活函数、序列分解
│   └── RevIN.py               # 可逆实例归一化（处理分布偏移）
│
├── data_provider/
│   ├── data_factory.py        # 数据工厂（根据配置创建 Dataset + DataLoader）
│   └── data_loader.py         # 数据集类（ETT_hour, ETT_minute, Custom, Pred）
│
├── exp/
│   ├── exp_basic.py           # 实验基类（设备管理）
│   └── exp_main.py            # 主实验类（训练循环、验证、测试、预测）
│
├── utils/
│   ├── tools.py               # 工具：EarlyStopping、学习率调整、可视化
│   ├── metrics.py             # 指标：MSE、MAE、RMSE、MAPE、RSE、CORR
│   └── timefeatures.py        # 时间特征编码（将日期转为数值特征）
│
├── scripts/PatchTST/          # Shell 训练脚本（Linux/Mac 用）
│   ├── etth1.sh
│   ├── etth2.sh
│   ├── ettm1.sh
│   ├── ettm2.sh
│   └── weather.sh
│
└── dataset/                   # 数据集存放目录
    ├── ETTh1.csv              # ✓ 已下载
    ├── ETTh2.csv              # ✓ 已下载
    ├── ETTm1.csv              # ✓ 已下载
    └── ETTm2.csv              # ✓ 已下载
```

---

## 3. 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 依赖列表

| 库 | 版本要求 | 用途 |
|----|---------|------|
| numpy | ≥1.21.0 | 数值计算 |
| pandas | ≥1.3.0 | 数据读取与时间处理 |
| torch | ≥1.9.0 | 深度学习框架 |
| matplotlib | ≥3.4.0 | 结果可视化 |
| scikit-learn | ≥0.24.0 | 数据标准化 |

### 硬件建议

- **GPU**（推荐）：RTX 3060 及以上，全部实验约 3-5 小时
- **CPU**：可运行，ETTh1 单配置约 20-30 分钟，全部实验约 24-50 小时

---

## 4. 数据准备

### 自动下载（ETT 数据集）

```bash
python download_data.py
```

### 手动下载（Weather / Electricity / Traffic）

从 Google Drive 下载后放入 `./dataset/` 目录：
> https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

### 支持的数据集

| 数据集 | 变量数 | 频率 | 数据量 | 说明 |
|--------|--------|------|--------|------|
| ETTh1 | 7 | 1h | 17,420 | 电力变压器油温（小时） |
| ETTh2 | 7 | 1h | 17,420 | 电力变压器油温（小时） |
| ETTm1 | 7 | 15min | 69,680 | 电力变压器油温（分钟） |
| ETTm2 | 7 | 15min | 69,680 | 电力变压器油温（分钟） |
| Weather | 21 | 10min | 52,696 | 气象站观测数据 |
| Electricity | 321 | 1h | 26,304 | 客户用电量 |
| Traffic | 862 | 1h | 17,544 | 道路传感器占用率 |

---

## 5. 快速开始

### 训练（以 ETTh1, pred_len=96 为例）

```powershell
python run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model PatchTST --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --patience 20 --itr 1 --batch_size 128 --learning_rate 0.0001 --lradj TST --pct_start 0.4 --num_workers 0
```

### 仅测试（加载已有模型）

```powershell
python run_longExp.py --is_training 0 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model PatchTST --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --num_workers 0
```

### Linux/Mac 批量运行

```bash
sh ./scripts/PatchTST/etth1.sh
```

---

## 6. 核心超参数说明

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--patch_len` | 16 | Patch 长度（论文核心参数） |
| `--stride` | 8 | Patch 步长（50% 重叠） |
| `--d_model` | 16/128 | Transformer 隐层维度 |
| `--d_ff` | 128/256 | FFN 中间层维度 |
| `--n_heads` | 4/16 | 注意力头数 |
| `--e_layers` | 3 | Encoder 层数 |
| `--revin` | 1 | 是否启用 RevIN 归一化 |
| `--individual` | 0 | 预测头是否每个变量独立 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seq_len` | 336 | 输入回看窗口长度 |
| `--pred_len` | 96 | 预测未来长度 |
| `--batch_size` | 128 | 批大小 |
| `--learning_rate` | 0.0001 | 学习率 |
| `--train_epochs` | 100 | 最大训练轮数 |
| `--patience` | 20 | 早停耐心值 |
| `--lradj` | TST | 学习率调度策略（OneCycleLR） |

### 各数据集推荐配置

| 数据集 | d_model | d_ff | n_heads | dropout |
|--------|---------|------|---------|---------|
| ETTh1/h2/m1/m2 | 16 | 128 | 4 | 0.3 |
| Weather | 128 | 256 | 16 | 0.2 |
| Electricity | 128 | 256 | 16 | 0.2 |
| Traffic | 128 | 256 | 16 | 0.2 |

---

## 7. 输出结果

### 结果文件

| 路径 | 内容 |
|------|------|
| `./result.txt` | 所有实验的 MSE/MAE/RSE 汇总 |
| `./checkpoints/` | 训练保存的最优模型权重 |
| `./test_results/` | 测试阶段的预测可视化 PDF |
| `./results/` | 预测值的 `.npy` 文件 |

### 论文报告的基准结果

| 数据集 | pred_len | MSE | MAE |
|--------|----------|-----|-----|
| ETTh1 | 96 | 0.370 | 0.400 |
| ETTh1 | 192 | 0.413 | 0.429 |
| ETTh1 | 336 | 0.422 | 0.440 |
| ETTh1 | 720 | 0.447 | 0.468 |
| ETTm1 | 96 | 0.293 | 0.346 |
| ETTm1 | 192 | 0.333 | 0.370 |
| ETTm1 | 336 | 0.369 | 0.392 |
| ETTm1 | 720 | 0.416 | 0.420 |
| Weather | 96 | 0.149 | 0.198 |
| Weather | 192 | 0.194 | 0.241 |
| Weather | 336 | 0.245 | 0.282 |
| Weather | 720 | 0.314 | 0.334 |

---

## 8. 方法细节

### 为什么叫"64 Words"？

以 `seq_len=512, patch_len=16, stride=8` 为例：
- Patch 数量 = (512 - 16) / 8 + 1 = **63**（加 padding 后为 **64**）
- 每个 patch 相当于 NLP 中的一个"词"（word/token）
- 即：一段时间序列仅需 64 个 tokens 就能表示，大幅降低自注意力的 $O(N^2)$ 开销

### RevIN（可逆实例归一化）

解决时间序列中的**分布偏移**问题：
1. 输入时：减均值、除标准差（归一化）
2. 输出时：乘标准差、加均值（反归一化）

使模型只需学习去除趋势后的模式，预测后再恢复原始尺度。

### Channel-Independence vs Channel-Mixing

论文实验表明，在长期预测任务中，通道独立（CI）优于通道混合（CM），原因包括：
- 避免多变量之间的虚假相关性
- 训练样本隐式增加（每个变量都是独立样本）
- 减少过拟合风险

---

## 9. 引用

```bibtex
@inproceedings{nie2023patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

---

## 10. 致谢

本项目参考了以下开源工作：
- [PatchTST 官方实现](https://github.com/yuqinie98/PatchTST)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)（代码框架）
- [RevIN](https://github.com/ts-kim/RevIN)（归一化模块）
- [Autoformer](https://github.com/thuml/Autoformer)（数据集）
