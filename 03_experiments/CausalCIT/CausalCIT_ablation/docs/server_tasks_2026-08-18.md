# 服务器任务执行手册 (2026-08-18)

> 前置: `git pull` 到最新 (含 6cf4d13 的修 C / DRO / 参数透传代码, 均已 CPU 验证)。
> P0-1 主表已 `_DONE` (2026-08-13), 以下任务可并行/排队执行。
> 所有命令在 `03_experiments/CausalCIT/CausalCIT_ablation/` 下执行。
> 数据路径: 服务器上 P0-1 使用的 dataset_dir (有 traffic/electricity/ILI csv 的目录),
> 记作 `<DD>`。

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
