# output_synood —— 合成 OOD 门控诊断输出

本目录是 **门控结构识别诊断** 的结果（直接回应审稿刀1"门控是否干活"）。

## 进 git 的文件（阶段性成果）
- `gate_diagnostic_report.md` — 主报告：full_v2 vs no_env 对照、塌缩/因果-虚假分离、刀1 结论
- `large_scale_report.md` — 8-seed 提升率汇总（full_v2 vs PatchTST 基线）
- `improvement_heatmap.png` — 提升率热力图
- `results_shard*.csv` / `log_shard*.txt` / `jobs_shard*.json` — 各分片结果/日志/任务规格

## 不进 git 的文件（可由脚本重新生成）
- `gates/*.npy` — 8 seed × 2 horizon 的门控矩阵 dump（被 `*.npy` 规则忽略）
- `errors_shard*.csv` — 空的错误占位文件
- `ckpt/` — 训练检查点

## 复现
```sh
python analyze_gates.py --gates_dir ./output_synood/gates
```
