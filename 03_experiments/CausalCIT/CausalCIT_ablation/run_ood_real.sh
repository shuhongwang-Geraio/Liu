#!/usr/bin/env bash
# 真实数据 OOD 实验: 8-seed 全量
# 数据集: traffic_ood / electricity_ood / weather_ood (时序漂移) + exchange (regime 漂移)
# 变体: patchtst / no_gate / full_v2 / learned_gate
# 在 3 张 GPU 上并行 (见 run_large.sh 的 3-shard 模式)
set -e
cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate causalcit

OUT=./output_ood_real
rm -rf "$OUT"
mkdir -p "$OUT"
NUM_SHARDS=3
export CIT_THREADS=8
JOB_TIMEOUT=3600
SEEDS="42 123 234 345 456 567 678 789"

echo "========== [1/3] 生成 jobs (8 seeds) =========="
python run_large.py gen \
    --datasets traffic_ood electricity_ood weather_ood exchange \
    --variants patchtst no_gate full_v2 learned_gate \
    --seeds $SEEDS --num_shards $NUM_SHARDS --output_dir "$OUT"

echo "========== [2/3] $NUM_SHARDS 张 GPU 并行训练 =========="
PIDS=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i nohup python -u run_large.py run \
        --device cuda:0 \
        --job_file "$OUT/jobs_shard$i.json" \
        --result_csv "$OUT/results_shard$i.csv" \
        --job_timeout $JOB_TIMEOUT \
        > "$OUT/log_shard$i.txt" 2>&1 &
    PIDS+=($!)
    echo "  启动 shard $i (PID ${PIDS[$i]}) on GPU $i"
done

echo "等待全部 shard 完成..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "全部 shard 训练完成."

echo "========== [3/3] 汇总 =========="
python -u run_large.py summarize --output_dir "$OUT"
echo "完成! 报告见 $OUT/large_scale_report.md"
