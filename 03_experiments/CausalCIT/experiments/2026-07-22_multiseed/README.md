# 实验快照: 2026-07-22_multiseed

> CausalCIT 多seed消融实验（5 seeds）+ 门控诊断深度分析 + **SOTA改进(full_v2)**

## ⭐ 最新进展：SOTA 改进 full_v2（见 `sota_v2/SOTA_report.md`）

基于本快照的诊断结论，实施三项关键改进并验证成功：
1. **时间分辨率保留通道交互**(`temporal_mix`) — 修复池化广播导致的时间信息坍缩
2. **批量池化HSIC稳定性门控v2**(`stability_v2`) — 修复"只用CV忽略依赖强度"的逻辑错误
3. **逐通道融合系数+优雅回退**(`per_channel_alpha`, `alpha_init=-2`) — 混合无益时不劣于CI基线

**关键结果**：
- 门控成功分化，正确识别合成数据的因果簇(Ch0/1/2)并压制独立通道(Ch5/6)，std 0.0007→0.2476
- **weather pl96 稳健提升 +2.83%~+5.35%(3 seed，均值+4%)**，且门控版优于无门控版
- 合成/ETTh1 优雅回退到 -0.10%/-0.27%(接近CI基线)

详见 `sota_v2/SOTA_report.md`。

## 运行环境

| 项目 | 详情 |
|------|------|
| 服务器 GPU | NVIDIA GeForce RTX 4090 |
| PyTorch | 2.5.1+cu121 |
| Python | 3.10.20 |
| CUDA | True |
| 关键依赖 | numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9, scipy |
| Conda 环境 | causalcit |

## 运行命令

```bash
# 5-seed 多seed消融实验
cd CausalCIT_ablation && bash run_multiseed.sh all 5 "42,123,2024,7,99" 4 ./output_multiseed
```

## 实验清单

| # | 实验 | 脚本 | 数据 | seeds | 状态 |
|---|------|------|------|-------|------|
| 1 | 消融合成数据 | `run_ablation.py --exp all` | 合成数据 | 42,123,2024 | ✅ (3-seed, 见诊断报告) |
| 2 | 消融 ETTh1 | `run_ablation.py --exp all` | ETTh1, 2 pred_len | 42,123,2024 | ✅ (3-seed) |
| 3 | **消融合成数据 5-seed** | `run_ablation.py --exp all` | 合成数据 | 42,123,2024,7,99 | ✅ |
| 4 | **消融 ETTh1 5-seed** | `run_ablation.py --exp all` | ETTh1, 2 pred_len | 42,123,2024,7,99 | ✅ |

## 与上次快照(2026-06-03)的关键进展

### 已完成的改进
1. **多seed聚合（3→5 seeds）**: 从3个seed扩展到5个seed，降低标准误差
2. **配对显著性检验**: 对每个变体 vs PatchTST 做了配对 t-test + Wilcoxon 检验
3. **门控矩阵插桩**: 训练时保存了门控矩阵的完整数值（`.npy`），实现逐元素对比
4. **门控诊断参数采集**: 温度、先验值、稳定性偏置、熵等参数实时记录
5. **Full vs w/o EnvSplit 门控对标**: 逐元素差异分析，验证"两条路径学到等价门控"的假设
6. **诊断报告系统化**: 编写了完整的《诊断报告_3seed消融结果分析.md》，记录异常现象和可疑代码位置

### 确认的核心异常
- **门控未分化**: Full CausalCIT 门控矩阵所有非对角线元素 ≈ 0.67，无法区分因果/虚假通道
- **Full ≈ w/o EnvSplit**: 9组测试点 MSE 平均相对差异仅 0.041%，比训练噪声小一个数量级
- **组件间差异 < 跨seed噪声**: 5-seed 下所有 p 值 > 0.05，无统计显著性
- **唯一亮点**: ETTh1 pred_len=336 上 w/o HSIC (Pearson相关) +2.04%，提示 HSIC 可能不如简单相关

## 关键发现

### 正面
- 诊断基础设施完善：门控矩阵落盘、参数插桩、显著性检验均已就绪
- 5-seed 聚合提供了比 3-seed 更可靠的均值和置信区间
- 门控矩阵诊断确认了"先验主导"和"环境划分信号缺失"两个核心假设

### 负面
- **5-seed 下仍无统计显著差异**: 所有变体 vs PatchTST 的 p 值 > 0.05
- **Full 未展现预期优势**: 合成数据 -0.46%, ETTh1 pl96 -1.01%, pl336 +0.68%
- **环境划分机制失效**: 单环境仅3个patch，HSIC方差估计不可靠

### 核心假设（待验证）
1. **先验主导假说**: `channel_prior`（可学习先验，占权重30%）淹没了基于数据的稳定性分数信号
2. **环境划分退化假说**: `n_envs=4, patch_num=12` → 每环境3个patch，CV估计噪声淹没信号
3. **门控饱和假说**: 门控矩阵趋向全通（gate≈1），失去选择性功能

## 下一步（详见本快照的 next_steps.md ）

本快照已经验证了诊断报告中的假设A（先验主导）和假设C（门控饱和），并给出了部分证据支持假设B。后续方向分为两个层次：

**短期（代码修复验证）**: 降低先验权重、增大patch_num、降低温度、添加门控稀疏正则化
**中期（架构改进）**: 重构稳定性估计器、引入结构化先验（因果图）、探索 Pearson 相关替代 HSIC
**长期（理论深化）**: 设计新的通道交互机制、探索"何时需要通道交互"的判定条件

## 文件结构

```
2026-07-22_multiseed/
├── README.md                  # 本文件
├── environment.txt            # 服务器环境
├── next_steps.md              # 详细的后续工作计划
├── ablation/                  # 5-seed 消融实验完整输出
│   ├── report.md              # 消融实验报告（含合成+ETTh1全部结果表格）
│   ├── significance_report.md # 配对显著性检验报告
│   ├── gate_diagnostics.txt   # 门控参数诊断（温度、先验、熵等）
│   ├── gate_comparison.txt    # Full vs no_env 门控矩阵逐元素对比
│   └── gate_matrices/         # 门控矩阵 numpy 数据
└── diagnosis/                 # 深度诊断分析
    └── 诊断报告_3seed消融结果分析.md  # 异常现象、可疑代码位置、假设体系
```
