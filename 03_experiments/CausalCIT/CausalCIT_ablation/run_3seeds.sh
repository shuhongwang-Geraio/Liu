#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate causalcit

cd /home/wangsh/workspace/Liu/03_experiments/CausalCIT/CausalCIT_ablation

echo "=========================================="
echo "  CausalCIT 多 Seed 消融实验"
echo "  开始时间: $(date)"
echo "  3 个 seed: 42, 123, 2024"
echo "=========================================="

for seed in 42 123 2024; do
    echo ""
    echo "=========================================="
    echo "  >>> Seed = $seed 开始: $(date)"
    echo "=========================================="
    python run_ablation.py --exp all --device cuda --seed $seed --output_dir ./output_seed${seed} 2>&1
    echo ""
    echo "  <<< Seed = $seed 完成: $(date)"
done

echo ""
echo "=========================================="
echo "  全部完成！"
echo "  结束时间: $(date)"
echo "  输出目录:"
echo "    ./output_seed42/ablation_report.md"
echo "    ./output_seed123/ablation_report.md"
echo "    ./output_seed2024/ablation_report.md"
echo "=========================================="
