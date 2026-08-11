#!/bin/bash
# do.md 两项 GPU 任务: 门控检验(任务2, GPU0) + 多数据集重跑(任务1, GPU1/GPU2)
set -u
cd /home/wangsh/workspace/Liu/03_experiments/CausalCIT/CausalCIT_ablation
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate causalcit 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- 任务 1: 多数据集大规模重跑 (新协议 8seed x 6变体) ----------
if [ ! -f ./output_large_v2/jobs_shard0.json ]; then
  python run_large.py gen \
    --datasets weather etth1 ettm1 electricity exchange ili \
    --variants patchtst full_v2 full_v2_fixed capacity_match gate_prior_only no_env \
    --seeds 42 123 2024 7 13 99 2023 31 \
    --num_shards 3 --output_dir ./output_large_v2
fi

# 3 卡并行跑 3 个 shard (GPU0 留给任务2, 这里只用 GPU1/GPU2 + 任务1本身还需1卡)
CUDA_VISIBLE_DEVICES=1 python run_large.py run --device cuda:0 \
  --job_file ./output_large_v2/jobs_shard0.json \
  --result_csv ./output_large_v2/results_shard0.csv \
  --job_timeout 21600 > ./output_large_v2/run_shard0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python run_large.py run --device cuda:0 \
  --job_file ./output_large_v2/jobs_shard1.json \
  --result_csv ./output_large_v2/results_shard1.csv \
  --job_timeout 21600 > ./output_large_v2/run_shard1.log 2>&1 &

# ---------- 任务 2: 门控 batch 不变性检验 (真实数据 traffic, 完整 30ep x 8seed) ----------
CUDA_VISIBLE_DEVICES=0 python run_minimal_falsifiable.py --dataset traffic \
  --dataset_dir ./01_external/PatchTST/code/dataset \
  --seeds 42 123 2024 7 13 99 2023 31 \
  --device cuda:0 \
  --output_dir ./output_falsifiable_full \
  > ./output_falsifiable_full/run.log 2>&1 &

wait
echo "ALL DONE"
# 任务1 第3个 shard 在 GPU 不够, 这里补跑 shard2 如果还没跑
if [ ! -f ./output_large_v2/results_shard2.csv ] || [ $(grep -c "MSE=" ./output_large_v2/results_shard2.csv 2>/dev/null) -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python run_large.py run --device cuda:0 \
    --job_file ./output_large_v2/jobs_shard2.json \
    --result_csv ./output_large_v2/results_shard2.csv \
    --job_timeout 21600 > ./output_large_v2/run_shard2.log 2>&1
fi
python run_large.py summarize --output_dir ./output_large_v2
