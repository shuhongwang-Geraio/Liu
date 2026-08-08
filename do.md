我把两处「必须要 GPU」的活拆开说清楚，包括**现在到底跑到哪一步了**（这点之前我漏了一个关键细节，刚才核对文件发现，需要纠正一下）。

## 1. 多数据集大规模跑——不是"新增"，是要用新协议**重跑**

现在 `output_large/large_scale_report.md` 里 electricity/etth1/ettm1/weather 的结果，是 **07-22 的老协议**：

```4:5:03_experiments/CausalCIT/CausalCIT_ablation/output_large/large_scale_report.md
> 数据集: ['electricity', 'etth1', 'ettm1', 'weather']
> 变体: ['full_v2', 'no_gate', 'patchtst']
```

问题：
- 只有 3 seed、3 个变体（没有 `capacity_match`/`gate_prior_only`/`full_v2_fixed`/`no_env` 这些评审 re2 要求的关键对照）
- 用的是已弃用的 t-test，没有 Holm 校正
- 效果本身也弱：etth1 三个 horizon 有两个是负的，weather 三个全负，只有 electricity/ettm1 的 pl96 有 +4~5%——这批结果早于 `entropy_weight` 死代码修复、`running_stats` batch依赖修复，**不能再拿来用**。

而 `output_falsifiable/large_scale_report.md` 是**新协议**（8 seed、5 变体全带、Wilcoxon+Holm）已经在 **traffic** 上跑完了，效果很好（pl96 +8.4%、pl192 +7.4%，Holm p<0.05，vs关键对照也显著）。

**要做的事**：用 `run_large.py` 新协议，对 electricity / etth1 / ettm1 / weather 重跑（8 seed），外加从未跑过的 exchange / ili。命令（有GPU后）：
```bash
python run_large.py gen --datasets weather etth1 ettm1 electricity exchange ili \
    --variants patchtst full_v2 full_v2_fixed capacity_match gate_prior_only no_env \
    --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 --output_dir ./output_large_v2
# 3张卡各跑一个 shard
CUDA_VISIBLE_DEVICES=0 python run_large.py run --device cuda:0 --job_file ./output_large_v2/jobs_shard0.json --result_csv ./output_large_v2/results_shard0.csv
...
python run_large.py summarize --output_dir ./output_large_v2
```
规模大致：6数据集×~2.5horizon×6变体×8seed ≈ 700+ 次训练，electricity(30ep,321变量)/traffic(30ep,862变量)最贵，weather/ETT较快，3卡并行估计数小时到一天量级。

## 2. 门控batch不变性检验——目前只有"意思到了"的CPU smoke test，没有真规模统计版本

`run_minimal_falsifiable.py` 现在跑出的 `output_falsifiable_smoketest/` 是 `--quick` 模式：

```213:217:03_experiments/CausalCIT/CausalCIT_ablation/run_minimal_falsifiable.py
    if args.quick:
        pls = pls[:1]
        args.seeds = args.seeds[:2]
        args.epochs = 2
```

即**只有1个pred_len、2个seed、2个epoch、用内置合成数据**——模型基本没训练到收敛，`batch_dep_score` 那组数字（0.015→0.000）只能证明"代码逻辑对了"，不能当正式结论用（`#seed`太少也做不了显著性检验）。

而且我之前核对 `output_falsifiable/gate_diagnostics.json` 发现它是空的 `[]`——因为那批数据是用 `run_large.py` 跑的（有 shard 文件为证），`run_large.py` 根本没有 `batch_invariance_check` 这个逻辑（那是 `run_minimal_falsifiable.py` 专属）。所以**目前完全没有 GPU 规模、真实数据、足量 seed 的 batch 不变性检验结果**。

**要做的事**：在真实数据集（推荐 traffic，评审指定的最小可证伪数据集）上用完整 epoch(30) + 8 seed 跑 `run_minimal_falsifiable.py`：
```bash
python run_minimal_falsifiable.py --dataset traffic \
    --seeds 42 123 2024 7 13 99 2023 31 --device cuda:0 \
    --output_dir ./output_falsifiable_full
```
这一步比第1项更贵，因为 `batch_invariance_check` 每个 (variant, seed, pred_len) 会额外做 `n_targets(5)×n_trials(3)=15` 次前向传播，且是单进程顺序跑完 6变体×8seed×2horizon=96 次训练（没有像 `run_large.py` 那样分shard并行）。可以考虑手动开多进程分批 `--seeds` 跑在不同GPU上，或者先加 `--skip_batch_invariance` 只拿MSE，再单独对 full_v2/full_v2_fixed 补一轮门控诊断。

---
两项都是脚本本身没问题、只是**规模/算力不够**，等有GPU环境随时可以直接跑上面的命令。

> ✅ **状态更新 (2026-08-08)**：以上两项均已完成 —— `output_large_v2/`（6数据集×6变体×8seed，720结果）与 `output_falsifiable_full/`（traffic 全规模门控诊断，80条）均已跑完并汇总进 `method_assessment.md`。

---

# 下一轮待办 (2026-08-08, 基于最新实验结果 method_assessment.md)

## 0. 当前定位（一句话）
CausalCIT 是**场景依赖的有效改进**，不是通用 SOTA：高维多通道（traffic +7.9%、electricity +3.9%，Holm p<0.05）显著，低维/长 horizon（ETTh1、ILI）不占优——符合方法假设（通道间因果依赖稀疏时门控退化为噪声）。论文正式采用 `full_v2_fixed`。

## P0 科学严谨性（审稿硬伤，最优先）
- [ ] **P0-1 重跑主表**：修复 run_large.py 的 spawn seed bug 后（见下），用 `full_v2_fixed` 重跑 6 数据集 × 8 seed 主表，确认 traffic/electricity 的显著提升在"seed 真正生效"协议下依然成立。这是论文主表可信度的基石。
- [ ] **P0-2 统一 collapsed 判据**：`analyze_gates.py`（std<0.01）与 `run_minimal_falsifiable.py`（std<1e-4）不一致，统一为同一常量。

## P1 论文内容补强
- [ ] **补 baseline**：目前只有 PatchTST。至少加 **iTransformer**（通道注意力相关）+ **DLinear**（强 CI 基线），可选 Crossformer。
- [ ] **敏感性分析**：n_envs(2/4/8)、rff_dim、prior_weight、temperature 各一组，证明结论不依赖超参脆点。
- [ ] **熵正则化实验**：`entropy_weight>0` 代码已接线（trainer + run_large --entropy_weight）但从未测过；gate_prior_only 坍缩/no_env 部分坍缩正是熵正则可对症的，值得在 traffic 上跑一组。
- [ ] **可视化升级**：traffic/electricity 高维门控矩阵聚类热图；full_v2_fixed 因果/虚假/独立边箱线图；提升率 bootstrap CI 误差棒图。

## P2 故事定位与诚实边界
- [ ] **OOD 结论谨慎处理**：现有 `output_ood_real` 中 learned_gate（纯容量）在 electricity_ood/traffic_ood 部分设置更强（traffic_ood pl96 +6.73% > full_v2 +5.20%）；`syn_ood` 上 full_v2 为 -1.21% 显著变差。**尚不能宣称"因果门控带来 OOD 鲁棒性"**——先排查 syn_ood 机制测试为何失败（spurious_strengths 配置 or 模型容量），否则此章会被审稿人反杀。
- [ ] **写作**：按"场景依赖有效改进 + 修复版批不变性 + 门控结构诊断"三条主线扩展 `method_assessment.md` 为论文核心章节，limitation 明确低维/长 horizon 边界。

## 本轮已完成（2026-08-08）
- [x] **代码审查**：全量语法编译 + lint 通过；`full_v2_fixed` 的 running_stats 修复、门控诊断插桩、seed 配对 Wilcoxon+Holm、高维 OOM 防护均确认正确。
- [x] **修复 run_large.py spawn seed bug**：`set_seed` 原只在主进程，spawn 子进程不继承 → seed 从未真正控制随机初始化。已在 `_train_one`/`_train_syn_ood` 内补 `set_seed(job['seed'])`。
- [x] **性能优化**（默认开关全关，不改变旧结果可复现性）：
  - `trainer.py`：`amp` 混合精度训练（仅 CUDA 生效，GradScaler + autocast，HSIC/门控仍在 causal_channel 内 fp32 保精度），预计提速 1.5-2x。
  - `causal_channel.py`：HSIC 的 for 循环合并为一次 `bmm`（峰值显存不变）；`compute_stability_score_v2` 显式 fp32 防 AMP 下 HSIC 精度损失；注意力门控改 5D 广播，省去 `[bs*patch_num,nv,nv]` 显式拷贝。
  - `data.py` + 两个 run 脚本：`get_dataloader` 支持 `pin_memory`，GPU 下自动开启。
  - 用法：`run_large.py run --amp ...` / `run_minimal_falsifiable.py --amp`。
- [ ] **注意**：`run_minimal_falsifiable.py` 仍是单进程串行（96 次训练不并行）——想提速可手动按 seed 分片开多进程（`--seeds 42 123 2024 7` 等跑在不同 GPU 上），或后续给脚本加 shard 逻辑。