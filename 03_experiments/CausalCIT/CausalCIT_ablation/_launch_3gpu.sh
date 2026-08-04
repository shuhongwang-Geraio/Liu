#!/bin/bash
# 3-GPU 并行跑 traffic falsifiable 测试
# 用法: bash _launch_3gpu.sh
# 查看进度: tail -f output_falsifiable/gpu0.log
eval "$(conda shell.bash hook)"
conda activate causalcit
cd /home/wangsh/workspace/Liu/03_experiments/CausalCIT/CausalCIT_ablation

echo "=== 启动 GPU0 ==="
CUDA_VISIBLE_DEVICES=0 CIT_THREADS=8 nohup python -u run_large.py run \
    --device cuda:0 --job_file ./output_falsifiable/jobs_shard0.json \
    --result_csv ./output_falsifiable/results_shard0.csv \
    --job_timeout 21600 \
    > ./output_falsifiable/gpu0.log 2>&1 &
PID0=$!

echo "=== 启动 GPU1 ==="
CUDA_VISIBLE_DEVICES=1 CIT_THREADS=8 nohup python -u run_large.py run \
    --device cuda:0 --job_file ./output_falsifiable/jobs_shard1.json \
    --result_csv ./output_falsifiable/results_shard1.csv \
    --job_timeout 21600 \
    > ./output_falsifiable/gpu1.log 2>&1 &
PID1=$!

echo "=== 启动 GPU2 ==="
CUDA_VISIBLE_DEVICES=2 CIT_THREADS=8 nohup python -u run_large.py run \
    --device cuda:0 --job_file ./output_falsifiable/jobs_shard2.json \
    --result_csv ./output_falsifiable/results_shard2.csv \
    --job_timeout 21600 \
    > ./output_falsifiable/gpu2.log 2>&1 &
PID2=$!

echo "PID: GPU0=$PID0 GPU1=$PID1 GPU2=$PID2"
echo "监控: tail -f output_falsifiable/gpu0.log"
echo "进度: grep '^\[' output_falsifiable/gpu0.log | tail -1"
echo "全部完成检查: grep -c 'shard 完成' output_falsifiable/gpu*.log"
