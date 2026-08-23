# PROGRESS.md — CausalCIT 项目进度单一事实来源

> 按 research-org 规范维护。每次工作会话结束更新；详细待办命令见 `do.md`。
> 最近更新: 2026-08-23

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
- [x] **P0-1 重跑主表（2026-08-13 完成）**：6 数据集 × 6 变体 × 8 seed + `--dump_gates`，
      816 job / 3 shard，3×RTX4090 并行，输出 `output_large_v3/large_scale_report.md`。
      脚本 `_run_p0_main.sh`。2 个 traffic no_env seed 超时（已有 7/8，够用）。
      最终结论: traffic +8.81%、electricity +3.07% 显著优于 PatchTST (Holm p<0.05);
      etth1/weather/etm1 长程仍退化; weather pl96 修复版翻正 +2.76%* (旧版为负)。
      门控 dump 已产出待聚类热图分析。部分结果快照见下节（已过时，以最终报告为准）。

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
- [x] **0 GPU 批（2026-08-18, P0-1 完成后）**：
      - 方案 1 对应分析：`correspond_analysis.py` + `docs/diagnostics/2026-08-18_applicability_criterion.md`
        —— 依赖密度符号一致 **9/11**、稳定占比 8/11、语义信息量 5/11（无预测力）；horizon 效应独立（短正长负）
      - 修复版适用边界：`fixAB_boundary.py` —— weather (8/8, -3.25%) / electricity (8/8, -2.01%) 修复版有效；
        **traffic (0/8, +1.18%) 失效**（median 带宽 862 通道失真）
      - 3b 组合配置 CPU smoke 4/4（syn_ood + alpha_init/fusion_alpha 透传）
      - ili pl24 诚实讨论入论文 §2.7（-11.51% 显著负，低维小样本 regime）+ 修复版边界第 6 条
      - 服务器执行手册：`docs/server_tasks_2026-08-18.md`（高维热图 / 3b / 修 C / DRO / 补统计量 + 回传要求）

### 未完成 / 有问题
- [ ] 修 C **GPU 验证**：实现已就绪（CPU smoke 3/3），需 weather/electricity 上
      uniform vs semantic 对比（命令见 `docs/server_tasks_2026-08-18.md` §3）
- [ ] syn_ood 机制测试未通过（−1.21%）；3b 网格待跑（`--alpha_init`/`--fusion_alpha`，手册 §2）
- [ ] 高维门控聚类热图：dump 已产出（服务器 `output_large_v3/gates/`），
      需在服务器跑 `plot_gate_heatmaps.py`（手册 §1）后回传分析
- [ ] 方案 1 判据补数据点：服务器补 traffic/electricity/ILI 统计量（手册 §5），本机重跑对应
- [ ] DRO λ 消融（手册 §4）与想法 2 对比评审（决定并行线优先级）
- [ ] 遗留待清理：`__pycache__`（无害缓存，可择机统一清理；`research-org.zip` 与空中文文件夹已于 2026-08-12 清理）
- [x] P0-1 主表已完，`output_large_v3` 报告已生成（含 8-seed PatchTST 对照 + Holm 显著性）；高维门控 dump 已产出
- [x] ⭐ **GPU 第二轮回传综合 (2026-08-23, 5d81de0)**：
  - 修 C 语义切分 5/5 组均稍差 → 止损，改报负结果
  - DRO λ=0.1 弱正（weather pl192 -1.16% 单调），不作为主贡献
  - **syn_ood 修复版 +44.2~44.8% vs patchtst（旧版 +0.66%）→ 机制测试通过，OOD 直接证据** ⭐
  - 7 数据集判据验证：依赖密度/稳定占比 13/17（76%），ili 是 horizon 强反例
  - **高维门控诊断：full_v2_fixed 最佳平衡（off_std=0.062, batch_dep≈0），electricity 块对角结构 → 选到真实依赖** ⭐
  - 详见 `docs/post_run_analysis_2026-08-23.md`

## 剩余任务总清单（一次跑完，跑完前不回传）

### 服务器 GPU（一键脚本 `_run_all_remaining.sh`，跑完生成 `_ALL_DONE.txt` 才允许回传）

| 阶段 | 内容 | 目的 | 预计 |
|------|------|------|------|
| S1 | syn_ood 配对显著性（patchtst+full_v2_fixed，主表 8 seed） | +44% 升级为 Wilcoxon 显著 | ~30min |
| S2 | P1-2 baseline（dlinear+itransformer，6 数据集 × 8 seed） | 审稿 re2 必需对照 | 1-2 天 |
| S3 | P1-1 敏感性（traffic：n_envs 2/8，rff_dim 16/64） | 结论不依赖超参脆点 | 2-4h |
| S4 | P1-3 熵正则（traffic，ew 0.01/0.1） | 门控熵正则消融 | 1-2h |
| S5 | traffic 门控热图（子采样 50，dump 已有） | 高维门控结构可视化 | ~min |

> 不需要再跑（已结论）：DRO λ 配对（本机 p=0.195 不显著）、修 C semantic（已止损 5/5 组）。

### 0 GPU（本机，随时可做）
1. 论文 §2.7/§2.8 全面替换为 P0-1 + 第二轮数字；syn_ood +44% 写进摘要；
   整合 `docs/post_run_analysis_2026-08-23.md`
2. S1/S2 回传后：syn_ood 配对 Wilcoxon、baseline 对照表、敏感性表进论文
3. 想法 2（可逆解耦）对比评审——决定是否替代想法 1（DRO 弱正，想法 1 不优先）
4. traffic 热图回传后：聚类解读进论文

### 回传规则（重要）
- 服务器只提交一次：`_ALL_DONE.txt` 存在后，`git add -A && commit && push`；
  提交信息含各阶段完成计数（脚本已自动写入 `_ALL_DONE.txt`）
- 中途禁止单独回传（避免零碎交互）

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
