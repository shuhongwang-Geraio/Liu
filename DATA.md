# 数据位置说明 (DATA MAP)

本项目**不把大体积数据/产物放进 git**（GitHub 同步只传代码与结果报告）。
本文件说明各类数据"存在哪、多大、怎么拿"。克隆仓库后，按此处指引重新获取即可复现。

> 规则：单文件 ≤100MB 的已跟踪文件（如 PatchTST 的 `pred.npy`、早期 `gate_matrices`、
> 小 zip）保留在版本库；更大的（>100MB）一律 gitignore，本地保留、不进远程。

---

## 1. PatchTST 原始数据集 —— `01_external/PatchTST/code/dataset/`  (csv 已 gitignore)

- **内容**：电力/交通/汇率/气温/疾病等标准预测数据集：
  `electricity.csv`(92M)、`traffic.csv`(131M)、`ETTh1/2.csv`、`ETTm1/2.csv`、
  `exchange_rate.csv`、`ILI.csv`（含 `ILI.csv.bak` 备份，已忽略）。
- **用途**：CausalCIT 训练/评估用的基础数据（weather/electricity/exchange/traffic/ETT 等）。
- **重新获取**：从 PatchTST 官方仓库下载 `dataset/` 目录放入此处
  （https://github.com/yuqinie1998/PatchTST ）。

## 2. 实验输出 —— `03_experiments/CausalCIT/CausalCIT_ablation/output_*/`  (报告进 git，dump 忽略)

- **进 git 的**：各 `large_scale_report.md` / `gate_diagnostic_report.md` / `ablation_report.md` /
  `*_report.md` / `*.txt`（显著性、门控诊断等**阶段性成果**），以及少量 `.json` 任务规格、`.png` 图。
- **不进 git 的**：`gates/*.npy`（门控矩阵 dump）、`*.log`（GPU 日志）、`ckpt/`（checkpoint）——
  均可由 `dump_gates_eval.py` / 各 `run_*.py` 重新生成。
- 关键报告索引：
  - `output_synood/gate_diagnostic_report.md` — 门控结构识别诊断（直击审稿刀1，含 no_env 对照）
  - `output_synood/large_scale_report.md`、`output_controls/large_scale_report.md`、`output_large/large_scale_report.md` 等 — 8-seed 提升率汇总
  - `output_multiseed/significance_report.md` — Wilcoxon + Holm 显著性
  - `output_large_v2/large_scale_report.md` — 6 数据集 × 6 变体 × 8 seed 主表（**P0-1 重跑前数字**）
  - `output_falsifiable_full/gate_diagnostics.json` — traffic 门控诊断 80 条（full_v2_fixed 证据）
  - `output_large_v2/improvement_bootstrap_ci.{png,md}` — 提升率 bootstrap CI 误差棒图（旧数据）
  - `vis_output/` — 可视化产物：门控热图 / 诊断图 / 边箱线图（示例）
- **本地 smoke 产物**（2026-08-10/11, 无 GPU 验证用, 均基于 syn_ood 合成数据）：
  `output_pipeline_smoke{,2}/`（run_large 全流程）、`output_entropy_smoke_{0,1,2}/`（熵正则）、
  `output_baseline_smoke/`（DLinear/iTransformer 训练循环）、`output_fix_smoke/`（修复版 1-epoch）。
  这些是**协议验证**，数字不用于论文；正式数字以 P0-1 重跑 `output_large_v3/` 为准。

### ⚠️ 修复版协议（2026-08-11 起）

- `run_large.FULL_V2_KWARGS` 已启用 `rff_sigma_mode='median'` + `cka_normalize=True`
  （门 1 静态诊断修复，见 `docs/diagnostics/2026-08-11_gate_static_diagnosis.md`）。
- 仅影响 `full_v2` / `full_v2_fixed`；`capacity_match`/`gate_prior_only`/`no_env` 及
  baseline 不变。
- **修复版是新协议**：与 `output_large_v2` 旧数字**不可直接对比**；重跑请用新输出目录
  （建议 `output_large_v3/`）。

### 诊断与文档脚本（0 GPU, `CausalCIT_ablation/`）

- `diagnose_gate_static.py` — 门 1 静态诊断（RFF σ / HSIC 区分度 / cv 分解）
- `_verify_gate_fix.py` — 修 A+B 效果验证（fixed vs median+cka 对比）
- `dump_gates_eval.py` — 从 checkpoint 提取门控矩阵（eval, 不重训）
- `docs/` — 诊断报告（`diagnostics/`）、PCD 论证（`pcd/`）、表格/格式文档

## 3. OOD 数据切分 —— `03_experiments/CausalCIT/data_ood/`  (小，进 git)

- OOD 协议的训练/测试时段切分定义（时序漂移：早时段训练、晚时段测试，留 gap）。

## 4. 训练实验记录 —— `03_experiments/CausalCIT/experiments/`  (小，进 git)

- 各日期的实验日志与配置；`*.npy` 门控矩阵 / `*.pth` 检查点已 gitignore。

---

### 复现流程速记
```sh
# 1) 数据
#    PatchTST dataset/  按上方获取 (本地无 csv; GPU 机器需先下载)
# 2) 训练/评估 (CausalCIT_ablation)
cd 03_experiments/CausalCIT/CausalCIT_ablation
python run_large.py gen --datasets ... --variants ... --seeds ... --dump_gates --output_dir ./output_large_v3
python run_large.py run  --device cuda:0 --job_file ./output_large_v3/jobs_shard*.json --result_csv ...
python run_large.py summarize --output_dir ./output_large_v3
# 3) 报告/可视化
python plot_bootstrap_ci.py --results_dir ./output_large_v3          # 提升率 bootstrap CI 图
python plot_gate_heatmaps.py --gates_dir <gates> --output ./vis_output   # 高维聚类热图
python plot_gate_edge_boxplot.py --gates_dir <gates> --output ./vis_output  # 边箱线图
python dump_gates_eval.py --ckpt_dir <ckpt> --job_file <job> --dataset_dir <csv>  # 从 checkpoint 提取门控
```
