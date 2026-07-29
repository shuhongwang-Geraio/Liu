#!/bin/bash
# CausalCIT 多seed消融实验后台运行脚本
# 用法:
#   ./run_multiseed.sh [exp] [n_seeds] [seeds_csv] [n_envs] [output_dir]
# 示例:
#   ./run_multiseed.sh all 5 "" 4 ./output_multiseed
#   ./run_multiseed.sh all 5 "42,123,2024,7,99" 4 ./output_multiseed
#   ./run_multiseed.sh all 10 "" 4 ./output_multiseed        # 加大seed数提高检验功效
#   ./run_multiseed.sh real 5 "" 2 ./output_env2             # 环境划分对照(n_envs=2, seq_len=336)
set -e
cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate causalcit

EXP=${1:-all}
N_SEEDS=${2:-5}
SEEDS_CSV=${3:-""}
N_ENVS=${4:-4}
OUT=${5:-./output_multiseed}

mkdir -p "$OUT"
LOG="$OUT/run_multiseed.log"

# 构造 --seeds 参数 (若为逗号分隔列表)
SEEDS_ARG=""
if [ -n "$SEEDS_CSV" ]; then
    SEEDS_ARG="--seeds $(echo "$SEEDS_CSV" | tr ',' ' ')"
fi

echo ">>> 启动多seed消融: exp=$EXP n_seeds=$N_SEEDS seeds='$SEEDS_ARG' n_envs=$N_ENVS out=$OUT"
echo ">>> 日志: $LOG"

nohup python run_ablation.py \
    --exp "$EXP" \
    --n_seeds "$N_SEEDS" \
    $SEEDS_ARG \
    --n_envs "$N_ENVS" \
    --output_dir "$OUT" \
    > "$LOG" 2>&1 &

echo "PID=$!"
echo "查看进度: tail -f $LOG"
