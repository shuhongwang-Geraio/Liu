#!/usr/bin/env bash
# Overnight 实验队列: 补齐评审 P0 所需实验, 趁 GPU 空闲自动串行跑两批.
#   Batch1: 合成 OOD (syn_ood + syn_ood_noise) x 5 变体 x 2 horizon x 8 seed -> ./output_synood
#           (答 P0-3 零结果缺口 + 直击刀1: full vs w/o EnvSplit 在已知结构数据上的差异)
#   Batch2: 时序漂移 OOD 补 seed (electricity_ood/exchange/weather_ood) x4 变体 -> ./output_controls
#           (答 P0-4: 扩到 8 seed 做 Wilcoxon; 结果并入 output_controls 得统一 8-seed 报告)
# 用法: setsid bash run_overnight_queue.sh > output_overnight_queue/launcher.log 2>&1 < /dev/null &

set +e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate causalcit
export CIT_THREADS=8
PY=/home/wangsh/miniconda3/envs/causalcit/bin/python
THRESH=3000            # MiB: 显存低于此视为可跑(容纳 Sun ~2200MiB 占用 + 我们的 job)
JOB_TIMEOUT=3600       # 单 job 超时(合成很快; electricity 重但 <2400s 已验证)
mkdir -p ./output_overnight_queue ./output_synood ./output_ood_extra_jobs

log(){ echo "[$(date +%H:%M:%S)] $*"; }

wait_gpu(){
  local gpu=$1
  while true; do
    local mem
    mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    mem=${mem:-99999}
    if [ "$mem" -lt "$THRESH" ]; then return 0; fi
    sleep 60
  done
}

# 在指定 GPU 上跑一个 shard (整函数在后台子 shell 里执行, 含 wait_gpu, 故多个 shard 可并行等待/启动)
run_one(){
  local outdir=$1 i=$2
  local GPU=$((i+1))
  wait_gpu $GPU
  log "$NAME shard$i -> GPU$GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u run_large.py run --device cuda:0 \
    --job_file "$outdir/jobs_shard$i.json" \
    --result_csv "$outdir/results_shard$i.csv" \
    --job_timeout $JOB_TIMEOUT > "$outdir/log_shard$i.txt" 2>&1
  log "$NAME shard$i 结束"
}

# ===================== Batch1: 合成 OOD =====================
NAME="Batch1"
log "===== $NAME: 合成 OOD (syn_ood / syn_ood_noise) ====="
$PY -u run_large.py gen --datasets syn_ood syn_ood_noise \
    --variants patchtst full_v2 capacity_match gate_prior_only no_env \
    --seeds 42 123 2024 2025 2026 2027 2028 2029 \
    --num_shards 2 --output_dir ./output_synood
run_one ./output_synood 0 & P0=$!
run_one ./output_synood 1 & P1=$!
wait $P0 $P1
$PY -u run_large.py summarize --output_dir ./output_synood
log "===== $NAME 完成 -> output_synood/large_scale_report.md ====="

# ===================== Batch2: 时序漂移 OOD 补 seed =====================
NAME="Batch2"
log "===== $NAME: 时序漂移 OOD 补 seed (electricity_ood/exchange/weather_ood) ====="
$PY -u run_large.py gen --datasets electricity_ood exchange weather_ood \
    --variants patchtst full_v2 capacity_match gate_prior_only \
    --seeds 2025 2026 2027 2028 2029 \
    --num_shards 2 --output_dir ./output_ood_extra_jobs
run_one ./output_ood_extra_jobs 0 & P0=$!
run_one ./output_ood_extra_jobs 1 & P1=$!
wait $P0 $P1
# 把 Batch2 结果并入 output_controls, 重写统一 8-seed 报告
$PY -u run_large.py summarize --output_dir ./output_controls
log "===== $NAME 完成 -> output_controls/large_scale_report.md (统一 8-seed) ====="
log "ALL DONE"
