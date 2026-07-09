# CausalCIT 增强实验 v2

基于 `CausalCIT_demo` 的模型代码，运行增强版实验。

## 改进内容

1. **增大模型容量**: d_model 从 16 → 64，d_ff 从 128 → 256，rff_dim 从 32 → 64
2. **更长训练**: epochs 从 20 → 50，patience 从 5 → 10，训练样本从 5000 → 8000
3. **真实数据**: 在 ETTh1 (7变量) 和 Weather (21变量) 上全面对比
4. **多预测长度**: 96 / 192 / 336 / 720 四个标准预测长度

## 项目结构

```
CausalCIT_exp_v2/
├── run_enhanced.py    # 实验脚本（复用CausalCIT_demo的模型代码）
├── output/            # 实验输出
└── README.md
```

依赖关系：
- 模型代码：`../CausalCIT_demo/models/` 和 `../CausalCIT_demo/utils/`
- 数据集：`../patchtst/dataset/`（ETTh1.csv, weather.csv 等）

## 运行方式

```bash
# 全部实验（合成数据增强版 + 真实数据）
python run_enhanced.py

# 仅合成数据增强版（验证门控分化）
python run_enhanced.py --exp synthetic

# 仅真实数据（ETTh1 + Weather, 4个预测长度）
python run_enhanced.py --exp real

# 使用GPU
python run_enhanced.py --device cuda

# 指定数据集路径
python run_enhanced.py --dataset_dir /path/to/dataset
```

## 预计运行时间

| 实验 | CPU | GPU |
|------|-----|-----|
| 合成数据增强版 | ~30min | ~5min |
| ETTh1 (4个pred_len) | ~60min | ~10min |
| Weather (4个pred_len) | ~90min | ~15min |
| **全部** | **~3h** | **~30min** |

## 输出文件

- `enhanced_synthetic_results.png` — 增强版合成数据可视化（门控矩阵分化）
- `real_data_results.png` — 真实数据多预测长度对比图
- `experiment_report_v2.md` — 完整实验报告
