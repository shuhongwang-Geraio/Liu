#!/bin/bash
# 补跑 no_env (修复 OOM 后)
cd /home/wangsh/workspace/Liu/03_experiments/CausalCIT/CausalCIT_ablation
eval "$(conda shell.bash hook)"
conda activate causalcit

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CIT_THREADS=8

python -u run_large.py run \
    --device cuda:0 \
    --job_file ./output_falsifiable_noenv/jobs_shard0.json \
    --result_csv ./output_falsifiable_noenv/results_shard0.csv \
    --job_timeout 21600 \
    > ./output_falsifiable_noenv/run.log 2>&1
