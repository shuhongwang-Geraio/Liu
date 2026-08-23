# 服务器任务执行手册 (2026-08-23 更新)

> **主入口: 一键脚本 `_run_all_remaining.sh`** (推荐, 跑完前不回传)。
> 前置: `git pull` 到最新。P0-1 已 `_DONE` (2026-08-13), 第二轮 (修 C/DRO/syn_ood 网格/
> 热图/统计量) 已回传并分析 (commit 4490d4e)。
> 数据路径: `/home/wangsh/workspace/Liu/01_external/PatchTST/code/dataset` (记作 `<DD>`)。

## 一键执行 (推荐)

```bash
cd <repo>/03_experiments/CausalCIT/CausalCIT_ablation
bash _run_all_remaining.sh        # 默认 REPO=/home/wangsh/workspace/Liu
```

- 依次执行 S1→S5 (见脚本头注释), 3 卡并行, 每阶段 `_STAGE_DONE` 断点续跑;
- **全部完成后才生成 `_ALL_DONE.txt`, 只有它存在才允许回传** (git add+commit+push);
- 中途断开直接重跑即可 (已完成的自动跳过)。

### 阶段清单

| 阶段 | 内容 | 预计 |
|------|------|------|
| S1 | **syn_ood 配对显著性** (patchtst+full_v2_fixed, 主表 8 seed) → 把 +44% 升级为 Wilcoxon 显著 | ~30min |
| S2 | **P1-2 baseline** (dlinear+itransformer, 6 数据集 × 8 seed, 审稿 re2 必需) | 1-2 天 |
| S3 | P1-1 敏感性 (traffic full_v2_fixed: n_envs 2/8, rff_dim 16/64) | 2-4h |
| S4 | P1-3 熵正则 (traffic, ew 0.01/0.1) | 1-2h |
| S5 | traffic 门控热图 (子采样 50, dump 已在服务器) | ~min |

> 注: DRO 配对显著性已在本机完成 (weather pl192 λ=0.1 vs 0, p=0.195 不显著, 无需再跑);
> 修 C semantic 已止损 (5/5 组 uniform 更优, 不再跑)。

## 手动执行 (不推荐, 仅调试用)

## 0. 先确认代码是最新

```bash
cd <repo>/03_experiments/CausalCIT/CausalCIT_ablation
git pull
python -c "import run_large; print('run_large OK')"   # 含 --env_mode/--risk_lambda 等新参数
```

## 1. 高维门控聚类热图 (P1 可视化, 数据已 dump 在服务器)

```bash
python plot_gate_heatmaps.py --gates_dir ./output_large_v3/gates \
    --output ./output_large_v3/vis_gates --subsample 50
# 若 gate_diagnostics.json 已生成:
python plot_gate_heatmaps.py --diagnostics_json ./output_large_v3/gate_diagnostics.json \
    --output ./output_large_v3/vis_gates
# 产出: traffic/electricity 门控热图 (子采样) + 诊断条形图 → 回传 git
```

## 2. 3b: syn_ood 识别-利用脱节网格 (8 配置 × 8 seed × 3 shard)

```bash
# 轮 1a: alpha_init 扫描 (fusion_alpha 保持默认 0.3)
for ai in -2.0 -0.5 0.0 1.0; do
  python run_large.py gen --datasets syn_ood --variants full_v2_fixed \
      --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 \
      --alpha_init $ai --output_dir ./output_synood_alpha_${ai//-/_}
done
# 轮 1b: fusion_alpha 扫描 (alpha_init 保持默认 -2.0)
for fa in 0.1 0.3 0.5 1.0; do
  python run_large.py gen --datasets syn_ood --variants full_v2_fixed \
      --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 \
      --fusion_alpha $fa --output_dir ./output_synood_fusion_${fa//./_}
done
# 每目录 3 shard, 3 卡并行:
#   python run_large.py run --device cuda:0 --job_file <dir>/jobs_shard0.json --result_csv <dir>/results_shard0.csv
#   (cuda:1 -> shard1, cuda:2 -> shard2)
# 判据: 任一配置 syn_ood 提升转正 -> 机制成立但利用不足; 全负 -> 配合构造侧审查
```

## 3. 修 C: 语义环境切分验证 (weather/electricity, uniform vs semantic)

```bash
# semantic (season), 与 output_large_v3 的 uniform 结果配对对比
python run_large.py gen --datasets weather electricity --variants full_v2_fixed \
    --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 \
    --env_mode semantic --env_scheme season --output_dir ./output_fixC_semantic \
    --dataset_dir <DD>
# run 3 shard (cuda:0/1/2) + summarize
# 对比对象: output_large_v3 (uniform, 已 _DONE)
# 判据: semantic 的 cv 提升 / 相对 uniform 不退化且 (期望) 短 horizon 增益改善
```

## 4. 想法 1: DRO λ 消融 (weather/electricity, capacity_match)

```bash
for lb in 0.0 0.1 1.0; do
  python run_large.py gen --datasets weather electricity --variants capacity_match \
      --seeds 42 123 2024 7 13 99 2023 31 --num_shards 3 \
      --env_scheme season --risk_lambda $lb --output_dir ./output_dro_lambda_${lb//./_} \
      --dataset_dir <DD>
done
# run + summarize
# 对照: capacity_match λ=0 (ERM) 与 P0-1 的 full_v2_fixed (0.14549/0.15946 等)
# 判据: 存在 λ*>0 显著优于 λ=0 -> 新主线; 全无 -> 止损转想法 2
```

## 5. 方案 1: 补训练前统计量 (近 0 GPU, 数据在服务器)

```bash
python compute_pre_train_stats.py --data <DD>/traffic.csv     --name traffic    --out _stats_traffic.json
python compute_pre_train_stats.py --data <DD>/electricity.csv --name electricity --out _stats_electricity.json
python compute_pre_train_stats.py --data <DD>/ILI.csv         --name ILI        --out _stats_ILI.json
# 回传 json 到本机后, 重跑:
#   python correspond_analysis.py  (7 数据集对应, 判据验证)
```

## 6. 回传要求 (遵守 gpu_verification_task.md §6)

每次提交: (a) results/errors csv; (b) `summarize` 汇总 md; (c) 说明剩余 job 与是否 `_DONE`;
不要只写 commit message "跑完了"。
