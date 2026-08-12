# PROGRESS.md — CausalCIT 项目进度单一事实来源

> 按 research-org 规范维护。每次工作会话结束更新；详细待办命令见 `do.md`。
> 最近更新: 2026-08-12

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
- [x] **GPU 验证靶场（2026-08-11 夜 ~ 08-12）**：修复版 `full_v2_fixed`（median+cka）8-seed 重跑
      weather/electricity/traffic，输出 `output_large_v3`。**决策门 2 通过**：weather 旧负收益翻正
      (pl96 +2.7~+4.8%、pl192 +1.1~+1.4%)，electricity 提升加大 (pl96 +5.3~+8.3%)；weather pl336 仍轻微负
      (-1.2~-2.1%，符合长 horizon 退化假设)。注：旧 `output_large_v2` PatchTST 仅 3-seed，靶场为方向性验证，
      最终显著性由 P0-1 主表 (含 8-seed PatchTST) 给出。
- [x] **P0-1 重跑主表（2026-08-12 已启动，跑中）**：6 数据集 × 6 变体 × 8 seed + `--dump_gates`，
      816 job / 3 shard，3×RTX4090 并行，输出 `output_large_v3`。脚本 `_run_p0_main.sh`。
      已回传部分结果（见下节快照）。

### P0-1 部分结果快照（2026-08-12 16:31 服务器提交 b1403aa）
- 进度 **219/816 (27%)，errors=0**；旧 `output_large` 已归档为 `output_large_pre_fix_2026-07-22`。
- ⚠️ 服务器 commit message 为"跑完了"，但快照仅 27% —— **需核实服务器是否仍在跑**
  （本机无 `_DONE.txt`）；且执行者未按 `gpu_verification_task.md` §6 回传汇总（只 commit 结果文件）。
- ⚠️ 服务器实际变体与文档 §4 不符：快照含 `full_v2`/`no_gate`（无 `capacity_match`），
  变体构成 6 种（`patchtst full_v2 full_v2_fixed no_gate gate_prior_only no_env`），816 job 数与文档一致。
- **核心初步结论（8-seed，vs 已发表 PatchTST，需等主表内高维 patchtst 对照确认）**：
  - 高维三数据集 `full_v2_fixed` 全部 8-seed 完成，**全面翻正**：
    weather pl96 +5.2% / pl192 +5.0% / pl336 +1.0%；electricity pl96 +3.3% / pl192 +5.3%；
    traffic pl96 +8.0% / pl192 +12.1%（std 0.0008~0.007，8-seed 稳定）。
  - 低维 etth1（全 6 变体完成）确认旧规律：full_v2_fixed vs patchtst = −1.8% (pl96)，门控不占优（预期内）；
    `full_v2 ≈ full_v2_fixed`（0.38558 vs 0.38559，修复版在低维无影响）。
  - ettm1 pl96 有趣：`full_v2` 0.31466 < patchtst 0.31946（+1.5%，待 full_v2_fixed 完成复核）。
- [x] **0 GPU 批（2026-08-12）**：
      - 修 C 可行性评估：语义环境切分信息量 = 随机均分 **4–14×**（ETTh1 昼夜 13.7×、
        weather 季节 4.2×）→ **修 C 可行**（`assess_env_split.py` +
        `docs/diagnostics/2026-08-12_env_split_feasibility.md`）
      - 想法 1 立项：`ideas/04_dro_risk_aversion/`（00_spark + 01_proposal，DRO 式目标函数）
      - 方案 1 训练前适用性判据：`compute_pre_train_stats.py` + 4 数据集统计量（待 P0-1 对应）
      - 3b syn_ood 排查：方案文档 + `run_large.py` 新增 `--alpha_init/--fusion_alpha` 透传（CPU 验证过）
      - 方案 4 PCD 转资产：`vs_difference_argument.md` §3.1；论文草稿补 §2.8 适用性判据
      - 修 C 实施（代码就绪, CPU smoke 3/3）：语义环境切分管线全链路
        （data/causal_channel/causalcit/models_ablation/trainer/run_large），
        `--env_mode semantic` + `--env_scheme`；默认 `uniform`/`None` 保持旧行为
      - 想法 1 DRO 实现（代码就绪, CPU smoke 3/3）：`trainer.py` `risk_lambda` +
        按环境分组损失 `L=mean_e+λ·var_e`；`--risk_lambda`（syn_ood 无时间戳自动退化 ERM）
      - 清理：删除 `research-org.zip`、修正数据冗余（评估统一用 01_external 已有数据）

### 未完成 / 有问题
- [ ] 修 C **GPU 验证**：实现已就绪（CPU smoke 3/3），需 weather/electricity 上
      uniform vs semantic 对比（命令见 `do.md`「GPU 待跑」），实测 cv 是否提升、收益是否改善
- [ ] syn_ood 机制测试未通过（−1.21%）；PCD 与 full_v2 打平
- [ ] 高维门控矩阵（traffic/electricity）未 dump，聚类热图缺数据（P0-1 已带 --dump_gates，跑完可补）
- [ ] 遗留待清理：`__pycache__`（无害缓存，可择机统一清理；`research-org.zip` 与空中文文件夹已于 2026-08-12 清理）
- [ ] P0-1 主表剩余 ~597 job（ettm1/exchange/ili 全量 + 高维其余变体 + weather336 部分）；
      ⚠️ **需服务器确认是否仍在跑**（commit "跑完了" 与实际 27% 不符，见快照节）；
      完成后 `summarize` 生成最终报告（8-seed PatchTST 对照 + Wilcoxon 显著性 + 高维热图）

## 下一步 (按优先级)

1. **GPU：P0-1 重跑主表（跑中）**——完成后用 `summarize` 出 8-seed PatchTST 对照 + 显著性，
   重新生成 bootstrap CI 图与高维门控热图；用新数字替换所有草稿性能引用。
2. **GPU（P0-1 后，前置代码均已就绪并 CPU 验证）**：
   修 C 验证（`--env_mode semantic --env_scheme season`）、3b syn_ood 网格
   （`--alpha_init`/`--fusion_alpha`）、DRO λ 消融（`--risk_lambda ∈ {0,0.1,1}`）；
   完整命令见 `do.md`「GPU 待跑」。
3. **近 0 GPU**：服务器上对 traffic/electricity/ILI 补跑
   `assess_env_split.py` + `compute_pre_train_stats.py`。
4. **0 GPU（P0-1 出结果后）**：方案 1 训练前统计量 vs 增益符号对应（脚本 `compute_pre_train_stats.py` 已就绪）。
5. **0 GPU：想法 2（可逆解耦）对比评审**——决定与 04_dro_risk_aversion 的并行线优先级。
6. **0 GPU：写作**——差异论证（含 PCD 转资产 §3.1）、论文 §2.8 适用性判据已落地，待 P0-1 数字替换。

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
