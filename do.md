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