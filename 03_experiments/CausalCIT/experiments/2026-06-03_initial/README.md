# 实验快照: 2026-06-03_initial

> CausalCIT 首次完整实验 — Demo + 增强v2 + 消融

## 运行环境

| 项目 | 详情 |
|------|------|
| 服务器 GPU | NVIDIA GeForce RTX 4090 |
| PyTorch | 2.5.1+cu121 |
| Python | 3.10.20 |
| CUDA | True |
| 关键依赖 | numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9 |

> 注: Demo 实验在 Windows CPU 上单独运行，不在此服务器环境内。

## 运行命令

```bash
# 增强实验 v2 (合成 + 真实数据)
cd CausalCIT_exp_v2 && python run_enhanced.py --device cuda

# 消融实验 (合成 + ETTh1)
cd CausalCIT_ablation && python run_ablation.py --device cuda
```

## 实验清单

| # | 实验 | 脚本 | 数据 | 状态 |
|---|------|------|------|------|
| 1 | Demo 合成 | `run_demo.py` | 合成数据 | ✅ |
| 2 | Demo OOD | `run_demo.py` | 合成数据 OOD | ✅ |
| 3 | v2 合成 (d=64) | `run_enhanced.py` | 合成数据 8000样本 | ✅ |
| 4 | v2 ETTh1 | `run_enhanced.py` | ETTh1, 4 pred_len | ✅ |
| 5 | v2 Weather | `run_enhanced.py` | Weather, 4 pred_len | ✅ |
| 6 | 消融 合成 | `run_ablation.py` | 合成, 5变体 | ✅ |
| 7 | 消融 ETTh1 | `run_ablation.py` | ETTh1, 5变体×2 pred_len | ✅ |

## 关键发现

### 正面
- **Demo (d_model=16)**: CausalCIT 比 PatchTST MSE 提升 **+9.65%**，OOD 鲁棒性更好
- **消融 ETTh1 pred336**: w/o Gate 变体比 PatchTST **+2.42%**，说明通道交互本身有价值

### 负面
- **v2 增强版 (d_model=64)**: CausalCIT 全面落后 PatchTST（合成 -2.11%，真实 0/8 胜）
- 消融合成数据: 5 个变体全部不如 PatchTST
- 全连接注意力(w/o Gate) 在部分场景反而优于带门控的 Full CausalCIT

### 核心假设
1. **门控太激进**: fusion_alpha=0.3 可能让 CI 分支权重过高
2. **大模型不需要**: d_model=64 的 PatchTST 已足够强，通道交互引入噪声
3. **HSIC 区分度不足**: 消融中 HSIC vs Pearson 无显著差异

## 下一步

- [ ] 调高 fusion_alpha (0.3 → 0.5 或 0.7)，让通道交互贡献更多
- [ ] 尝试 d_model=32 中间档，找到门控有效的模型规模
- [ ] 检查 HSIC 稳定性检验在合成数据上的区分度
- [ ] 考虑在 pred_len 较长时关闭门控（消融显示 pred336 下 w/o Gate 最优）

## 文件结构

```
2026-06-03_initial/
├── README.md              # 本文件
├── environment.txt         # 服务器环境
├── cmd_v2.txt             # 增强v2 完整输出
├── cmd_ablation.txt       # 消融实验 完整输出
├── demo/                  # Demo 结果 (CPU运行)
├── v2/                    # 增强v2 结果
└── ablation/              # 消融实验 结果
```
