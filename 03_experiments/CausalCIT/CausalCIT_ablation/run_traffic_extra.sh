#!/usr/bin/env bash
# (b) 为 traffic_ood 补 5 个 seed (2025-2029) -> 共 8 seed, 强化显著性
# (c) 合并结果后重新 summarize, 重生成 output_controls/large_scale_report.md
# 用法: setsid bash run_traffic_extra.sh  (后台, 日志见 output_trafextra/launcher.log)
set -e
cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate causalcit

EXTRA=./output_trafextra
CTL=./output_controls
export CIT_THREADS=8
JOB_TIMEOUT=2400

mkdir -p "$EXTRA"

echo "[$(date +%H:%M:%S)] (b) gen traffic_ood 额外 5 seed (2025-2029) x4 variant x2 horizon"
python -u run_large.py gen --datasets traffic_ood \
    --variants patchtst capacity_match full_v2 gate_prior_only \
    --seeds 2025 2026 2027 2028 2029 \
    --num_shards 2 --output_dir "$EXTRA"

echo "[$(date +%H:%M:%S)] 在 GPU1/2 并行跑 2 个 shard (GPU0 被他人占用, 避开)"
PIDS=()
for i in 0 1; do
    GPU=$((i + 1))   # shard0->GPU1, shard1->GPU2
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u run_large.py run --device cuda:0 \
        --job_file "$EXTRA/jobs_shard$i.json" \
        --result_csv "$EXTRA/results_shard$i.csv" \
        --job_timeout $JOB_TIMEOUT \
        > "$EXTRA/log_shard$i.txt" 2>&1 &
    PIDS+=($!)
    echo "  shard$i -> GPU$GPU PID ${PIDS[$i]}"
done
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "[$(date +%H:%M:%S)] traffic extra 全部完成"

echo "[$(date +%H:%M:%S)] 合并结果到 $CTL/results_shard_traffic_extra.csv"
cat "$EXTRA/results_shard0.csv" > "$CTL/results_shard_traffic_extra.csv"
for i in 1; do
    tail -n +2 "$EXTRA/results_shard$i.csv" >> "$CTL/results_shard_traffic_extra.csv"
done

echo "[$(date +%H:%M:%S)] (c) 重新汇总 (含 traffic 8 seed) -> $CTL/large_scale_report.md"
python -u run_large.py summarize --output_dir "$CTL"
echo "[$(date +%H:%M:%S)] 完成"
