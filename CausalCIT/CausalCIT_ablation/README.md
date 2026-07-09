# CausalCIT 消融实验 (Ablation Study)

验证CausalCIT每个核心组件的贡献。

## 消融变体

```
Full CausalCIT = HSIC检验(A) + 环境划分(B) + 门控选择(C) + 通道注意力(D)

变体设计:
┌──────────────────┬──────┬──────┬──────┬──────┐
│ 变体             │ HSIC │ Env  │ Gate │ Attn │
├──────────────────┼──────┼──────┼──────┼──────┤
│ PatchTST         │  ❌  │  ❌  │  ❌  │  ❌  │  ← 纯CI基线
│ w/o Gate         │  ❌  │  ❌  │  ❌  │  ✅  │  ← +全连接注意力
│ w/o EnvSplit     │  ✅  │  ❌  │  ✅  │  ✅  │  ← +门控(全局HSIC)
│ w/o HSIC         │  ❌  │  ✅  │  ✅  │  ✅  │  ← +环境划分(Pearson)
│ Full CausalCIT   │  ✅  │  ✅  │  ✅  │  ✅  │  ← 完整模型
└──────────────────┴──────┴──────┴──────┴──────┘
```

## 运行方式

```bash
# 全部消融实验（合成数据 + ETTh1）
python run_ablation.py

# 仅合成数据消融
python run_ablation.py --exp synthetic

# 仅ETTh1消融
python run_ablation.py --exp real

# GPU加速
python run_ablation.py --device cuda
```

## 项目结构

```
CausalCIT_ablation/
├── models_ablation.py   # 消融变体模型定义
├── run_ablation.py      # 实验运行脚本
├── README.md
└── output/              # 运行后生成
```

依赖：
- `../CausalCIT_demo/models/` — 基础模型代码
- `../CausalCIT_demo/utils/` — 数据加载和训练工具
- `../patchtst/dataset/` — 数据集

## 输出

- `ablation_synthetic.png` — 合成数据消融可视化（门控矩阵 + 性能 + 组件贡献）
- `ablation_etth1.png` — ETTh1消融可视化
- `ablation_report.md` — 完整消融报告

## 预计运行时间

| 实验 | CPU | GPU |
|------|-----|-----|
| 合成数据 (5个变体 × 50 epochs) | ~2.5h | ~25min |
| ETTh1 (5个变体 × 2个pred_len × 30 epochs) | ~2h | ~20min |
| **全部** | **~4.5h** | **~45min** |
