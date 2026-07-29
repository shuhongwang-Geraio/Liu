# experiments —— CausalCIT 训练实验记录

按日期组织的训练/评估实验产物。

- 进 git：日志（`*.txt`/`*.md`）、配置、小结果。
- 不进 git（被根 `.gitignore` 排除）：`gate_matrices/*.npy`、`ckpt/`、`*.pth`、`*.ckpt`、
  以及任何 `*.npy` 预测/门控 dump —— 均可由对应 `run_*.py` 重新生成。

子目录示例：
- `2026-06-03_initial/` — 最早的消融实验（含 `gate_matrices/` 门控 dump，已忽略）
- `2026-07-22_multiseed/` — 多 seed 复算
- `2026-07-23_largescale/` — 大规模 OOD 评测
