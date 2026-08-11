# P1 可视化升级 — 脚本说明与数据状态

> 更新: 2026-08-10

## 1. 本目录两个脚本

| 脚本 | 产出 | 数据来源 | 状态 |
|------|------|----------|------|
| `plot_bootstrap_ci.py` | 各数据集×horizon 提升率 bootstrap 95% CI 误差棒图 + 数值表 | `output_large_v2/results_shard*.csv` (720 结果) | ✅ 数据齐备, 可直接跑 |
| `plot_gate_heatmaps.py` | (a) 门控矩阵聚类热图 (b) 门控行为诊断图 | (a) `gates/*.npy` (b) `output_falsifiable_full/gate_diagnostics.json` | (a) 仅低维可跑 (b) ✅ 可直接跑 |
| `plot_gate_edge_boxplot.py` | 因果/虚假/独立边门控权重箱线图 + 汇总表 | `gates/*.npy` (门控矩阵 dump) | ✅ 可跑 (syn_ood 示例已出图) |
| `dump_gates_eval.py` | 从已有 checkpoint 提取门控矩阵 (eval, 不重训) | `*/checkpoint.pth` + 数据 | ✅ 可跑 (syn_ood 验证通过) |

## 2. 运行方法

```bash
# 1) 提升率 bootstrap CI 图 (论文主图候选)
python plot_bootstrap_ci.py --results_dir ./output_large_v2 --output ./output_large_v2

# 2) 门控行为诊断图 (回应 re2 §2.2 batch 依赖 bug 的可视化)
python plot_gate_heatmaps.py --diagnostics_json ./output_falsifiable_full/gate_diagnostics.json --output ./vis_output

# 3) 门控矩阵热图 (先用已有低维样例验证脚本)
python plot_gate_heatmaps.py --gates_dir ./output/gate_matrices --output ./vis_output

# 4) 因果/虚假/独立边门控权重箱线图 (论文素材)
python plot_gate_edge_boxplot.py --gates_dir ./output_pipeline_smoke/gates --output ./vis_output

# 5) 从已有 checkpoint 提取门控矩阵 (eval, 不重训) —— GPU 有 traffic checkpoint 后用
python dump_gates_eval.py --ckpt_dir <ckpt目录> --job_file <job文件> \
    --dataset_dir <csv目录> --output <gates输出目录>
```

## 3. 高维门控矩阵缺失 — 需要补做的事

**问题**: 论文核心图是 traffic(862ch)/electricity(321ch) 的 `full_v2_fixed` 门控矩阵聚类热图,
但 `run_large.py` 的 `_train_one` 只在 `cfg['n_vars'] <= 21` 时才 dump 门控矩阵
(见 `run_large.py` 第 ~285 行), 因此高维矩阵从未保存。当前 `output/gate_matrices/`
只有早期 ablation 的 (320, 7, 7) 低维样例。

**要做的事 (有 GPU + 有数据集时) —— 两条路都已备好代码 (2026-08-10)**:

方案 A — 重跑时顺带 dump (最可靠, 推荐配合 P0-1 使用):
```bash
# run_large.py gen 已支持 --dump_gates: 强制 n_vars>21 也保存门控矩阵
# traffic 862×862×fp32 ≈ 3MB/个, 8 seed × 2 horizon 可接受
python run_large.py gen --datasets traffic electricity --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 --dump_gates --output_dir ./output_large_v3
```

方案 B — 独立 eval dump (不重跑训练, 从已有 checkpoint 提取):
```bash
python dump_gates_eval.py --ckpt_dir <ckpt目录> --job_file <job文件> \
    --dataset_dir <csv目录> --output ./gates_eval
# 支持 --job_file 精确重建 model_kwargs (含敏感性参数); syn_ood 无需 dataset_dir
```

随后:
```bash
python plot_gate_heatmaps.py --gates_dir <dump目录> --output ./vis_output --subsample 50
python plot_gate_edge_boxplot.py --gates_dir <dump目录> --output ./vis_output
```

## 4. 统计口径备忘 (plot_bootstrap_ci.py)

- 提升率% (seed 配对) = (patchtst_mse - variant_mse) / patchtst_mse × 100
- CI = seed 级 bootstrap (以 seed 为采样单元, 保持配对结构), 2000 次, 2.5%/97.5% 分位
- 显著性 = seed 配对 Wilcoxon(双侧) + 同 (dataset, pred_len) 族内 Holm 校正
  (与 `run_large.py summarize` 口径一致; 图内 † 标记 Holm p<0.05)
- ⚠️ 注意: 当前 `output_large_v2` 的数字基于 2026-08-08 修复 spawn seed bug **之前**
  的协议 (P0-1 需重跑主表)。重跑后请用新 CSV 重新生成该图, 不要直接用于投稿。

## 5. 后续可选

- 高维矩阵的行聚类 (对 862 通道做 k-means / 层次聚类后按簇重排) —— 比等距子采样更能
  展示"因果簇 vs 虚假簇"结构, 是审稿人可能希望看到的图。
- `full_v2_fixed` 因果/虚假/独立边的箱线图: 需要合成数据真值边 (syn_ood) + 门控矩阵
  同时对齐, 属于"最小可证伪测试"的图补充。
