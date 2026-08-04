#!/bin/bash
# 后台启动 traffic 最小可证伪测试
set -e

# 必须 source conda 才能在脚本里激活
eval "$(conda shell.bash hook)"
conda activate causalcit

cd /home/wangsh/workspace/Liu/03_experiments/CausalCIT/CausalCIT_ablation
mkdir -p output_falsifiable

exec python -u run_minimal_falsifiable.py \
    --dataset traffic \
    --seeds 42 123 2024 7 13 99 2023 31 \
    --device cuda:0 \
    --output_dir ./output_falsifiable \
    > ./output_falsifiable/run.log 2>&1
