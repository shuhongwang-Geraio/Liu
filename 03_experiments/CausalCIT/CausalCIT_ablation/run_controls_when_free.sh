#!/usr/bin/env bash
# 等待 GPU0 与 GPU2 同时空闲后, 自动跑 output_controls/ 的 96 个诊断 job (2 shard)
# 用法: setsid bash run_controls_when_free.sh  (后台运行, 日志见 output_controls/launcher.log)
set -e
cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate causalcit

OUT=./output_controls
export CIT_THREADS=8
JOB_TIMEOUT=2400
THRESH=2000                # MiB: 显存低于此值视为可跑(容纳 Sun ~800MiB 低占用)
GPU0=1                     # shard0 -> GPU1
GPU2=2                     # shard1 -> GPU2
WAIT_SLEEP=60              # 轮询间隔(秒)

mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] 等待 GPU${GPU0} 与 GPU${GPU2} 显存 < ${THRESH}MiB ..."

while true; do
    u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU0} | head -1 | tr -d ' ')
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU2} | head -1 | tr -d ' ')
    echo "[$(date +%H:%M:%S)] GPU${GPU0}=${u0}MiB  GPU${GPU2}=${u2}MiB"
    if [ -n "$u0" ] && [ -n "$u2" ] && [ "$u0" -lt "$THRESH" ] && [ "$u2" -lt "$THRESH" ]; then
        echo "[$(date +%H:%M:%S)] GPU${GPU0}/${GPU2} 均空闲, 启动训练"
        break
    fi
    sleep "$WAIT_SLEEP"
done

PIDS=()
CUDA_VISIBLE_DEVICES=${GPU0} nohup python -u run_large.py run \
    --device cuda:0 \
    --job_file "$OUT/jobs_shard0.json" \
    --result_csv "$OUT/results_shard0.csv" \
    --job_timeout $JOB_TIMEOUT \
    > "$OUT/log_shard0.txt" 2>&1 &
PIDS+=($!)
echo "[$(date +%H:%M:%S)] 启动 shard0 (GPU${GPU0}, PID ${PIDS[0]})"

CUDA_VISIBLE_DEVICES=${GPU2} nohup python -u run_large.py run \
    --device cuda:0 \
    --job_file "$OUT/jobs_shard1.json" \
    --result_csv "$OUT/results_shard1.csv" \
    --job_timeout $JOB_TIMEOUT \
    > "$OUT/log_shard1.txt" 2>&1 &
PIDS+=($!)
echo "[$(date +%H:%M:%S)] 启动 shard1 (GPU${GPU2}, PID ${PIDS[1]})"

echo "[$(date +%H:%M:%S)] 等待两个 shard 完成..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "[$(date +%H:%M:%S)] 两个 shard 训练完成."

echo "[$(date +%H:%M:%S)] 汇总报告..."
python -u run_large.py summarize --output_dir "$OUT"
echo "[$(date +%H:%M:%S)] 完成! 报告见 $OUT/large_scale_report.md"
