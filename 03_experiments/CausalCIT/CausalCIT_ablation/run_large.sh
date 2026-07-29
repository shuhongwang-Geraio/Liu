#!/usr/bin/env bash
# CausalCIT 大规模并行实验启动器
# 用法: bash run_large.sh
set -e
cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate causalcit

OUT=./output_large
mkdir -p "$OUT"
NUM_SHARDS=3
# 每 shard 进程限制 OpenMP/torch 线程数, 避免多进程各占满 32 核导致 GPU 饿死
export CIT_THREADS=8
# 单 job 超时秒数 (默认 2400=40min; 超时强杀并续跑)
JOB_TIMEOUT=2400

echo "========== [1/3] 生成 jobs + 装箱 =========="
python run_large.py gen --num_shards $NUM_SHARDS --output_dir "$OUT"

echo "========== [2/3] 在 $NUM_SHARDS 张 GPU 上并行训练 =========="
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

echo "========== [3/3] 汇总报告 =========="
python -u run_large.py summarize --output_dir "$OUT"

echo "完成! 报告见 $OUT/large_scale_report.md 与 $OUT/improvement_heatmap.png"
