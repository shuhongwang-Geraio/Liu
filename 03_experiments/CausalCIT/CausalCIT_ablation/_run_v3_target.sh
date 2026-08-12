#!/usr/bin/env bash
# 验证靶场: weather/electricity/traffic × full_v2_fixed × 8seed, 3卡并行
set -u
cd /home/wangsh/workspace/Liu/03_experiments/CausalCIT/CausalCIT_ablation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate causalcit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATASET_DIR=/home/wangsh/workspace/Liu/01_external/PatchTST/code/dataset

launch() {
  local shard=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu nohup python run_large.py run \
    --device cuda:0 --job_file ./output_large_v3/jobs_shard$shard.json \
    --result_csv ./output_large_v3/results_shard$shard.csv \
    --job_timeout 3600 > ./output_large_v3/run_shard$shard.log 2>&1 &
  echo "shard$shard -> GPU$gpu pid=$!"
}

launch 0 0
launch 1 1
launch 2 2
wait
echo "ALL SHARDS DONE at $(date)"
# 汇总
python run_large.py summarize --output_dir ./output_large_v3 --dataset_dir "$DATASET_DIR" \
  > ./output_large_v3/summarize.log 2>&1
echo "SUMMARIZE DONE at $(date)"
