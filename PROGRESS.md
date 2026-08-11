# PROGRESS.md — CausalCIT 项目进度单一事实来源

> 按 research-org 规范维护。每次工作会话结束更新；详细待办命令见 `do.md`。
> 最近更新: 2026-08-11

## 项目目标

提出并验证一种用**跨环境稳定性**（而非相关性强度）决定时序通道交互的方法（CausalCIT，
PatchTST backbone + HSIC 稳定性门控），论证其对 OOD 泛化的价值。核心风险已从"范围窄"
收敛为"机制是否按设计运行"，当前主线是修复门 1 诊断出的三个根因（RFF 带宽 / HSIC 归一化 /
语义环境）并重跑验证。

## 当前状态

### 已完成
- [x] 8-seed 实验基础设施：`run_large.py`（gen/run/summarize）、spawn seed bug 修复（P0-1 前置）
- [x] baseline 接入：DLinear / iTransformer 已进 `create_ablation_model`（CPU 训练循环验证通过）
- [x] P0-2 统一门控坍缩判据（0.01，两脚本一致）
- [x] 门 1 静态诊断（2026-08-11）：确认根因 1/2/3（`docs/diagnostics/2026-08-11_gate_static_diagnosis.md`）
- [x] 修 A+B：RFF median heuristic 带宽 + HSIC CKA 归一化（`causal_channel.py`，默认关闭不破坏旧行为）
- [x] 修复版透传链路打通（`run_large.FULL_V2_KWARGS` → ... → `CausalStabilityGate`），1-epoch smoke 通过
- [x] PCD 初步发现：`full_v2 ≈ pcd_gate`，机制测试未通过（`docs/pcd/pcd_preliminary_findings.md`）
- [x] 文献/调研归档：`surveys/04_baseline_literature/`、外部材料补全至 01_external
- [x] 战略分析：`ideas/01_adaptive_channel/07_scope_and_publication_risk_analysis.md` + 新方向脑暴 `00_inbox/2026-08-11_new_directions.md`

### 未完成 / 有问题
- [ ] **GPU 验证靶场**（最高优先）：修复版（median+cka）8-seed 重跑 weather/electricity/traffic，
      看负收益是否翻正。修复版是新协议，与旧数字不可直接对比。
- [ ] 根因 3（语义环境切分）未修：`cv≈0.005`，稳定性项仍无信息（需时间戳/真实数据）
- [ ] P0-1 重跑主表（seed bug 修复后）尚未执行，旧数字均不可最终采信
- [ ] syn_ood 机制测试未通过（−1.21%）；PCD 与 full_v2 打平
- [ ] 高维门控矩阵（traffic/electricity）未 dump，聚类热图缺数据
- [ ] 遗留待清理：根目录 `research-org.zip`、空中文文件夹、`__pycache__`

## 下一步 (按优先级)

1. **GPU：验证靶场**（决策门 2 的判据）
   ```sh
   cd 03_experiments/CausalCIT/CausalCIT_ablation
   python run_large.py gen --datasets weather electricity traffic --variants full_v2_fixed \
       --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 --output_dir ./output_large_v3
   # run + summarize; 看 weather/electricity 负收益是否翻正
   ```
   负转正 → 抢救 CausalCIT 为主；仍负 → 转想法 1（跨环境风险厌恶 DRO）。
2. **GPU：P0-1 重跑主表**（6 数据集 × 8 seed，`output_large_v3`，记得 `--dump_gates`）
3. **0 GPU：修 C（语义环境切分）**——需要时间戳数据，先评估 weather/electricity 可行性
4. **0 GPU：想法 1（DRO 式目标函数）** 或 想法 2（可逆正交解耦）立项 —— 见 `00_inbox/2026-08-11_new_directions.md`
5. **0 GPU：写作**——差异论证（`surveys/04_baseline_literature/vs_difference_argument.md`）、论文章节草稿

## 备注

- 详细命令/参数/止损规则：`do.md`（决策门 §0.5、GPU 任务 §1、P2 §3）
- 历史诊断文档归档：`docs/diagnostics/`（2026-07-20 3seed 分析、2026-08-11 静态诊断）
- 旧"高维有效"结论（traffic +7.9% 等）在修复版重跑前**不构成最终判断**——很可能是
  d_model 混淆变量（见静态诊断）
- 若删除过时文档，请在本节记录（内容可从 git 恢复）
- **删除记录**：`problem.md` 已于 2026-08-11 删除（其内容三重冗余：已归档至
  `docs/diagnostics/2026-07-20_3seed_ablation_analysis.md` 与
  `docs/diagnostics/2026-08-11_gate_static_diagnosis.md`，结论见本文件"当前状态"；
  可从 git 恢复）
