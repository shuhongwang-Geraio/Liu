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
- **不进 git 的**：`gates/*.npy`（门控矩阵 dump）、`*.log`（GPU 日志）—— 均可由
  `analyze_gates.py` / 各 `run_*.py` 重新生成。
- 关键报告索引：
  - `output_synood/gate_diagnostic_report.md` — 门控结构识别诊断（直击审稿刀1，含 no_env 对照）
  - `output_synood/large_scale_report.md`、`output_controls/large_scale_report.md`、`output_large/large_scale_report.md` 等 — 8-seed 提升率汇总
  - `output_multiseed/significance_report.md` — Wilcoxon + Holm 显著性

## 3. OOD 数据切分 —— `03_experiments/CausalCIT/data_ood/`  (小，进 git)

- OOD 协议的训练/测试时段切分定义（时序漂移：早时段训练、晚时段测试，留 gap）。

## 4. 训练实验记录 —— `03_experiments/CausalCIT/experiments/`  (小，进 git)

- 各日期的实验日志与配置；`*.npy` 门控矩阵 / `*.pth` 检查点已 gitignore。

---

### 复现流程速记
```sh
# 1) 数据
#    PatchTST dataset/  按上方获取
# 2) 训练/评估 (CausalCIT_ablation)
cd 03_experiments/CausalCIT/CausalCIT_ablation
python run_large.py ...          # 跑 OOD 实验
python analyze_gates.py --gates_dir ./output_synood/gates   # 生成门控诊断报告
```
