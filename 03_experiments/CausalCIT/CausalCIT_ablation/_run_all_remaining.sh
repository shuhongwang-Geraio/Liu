#!/usr/bin/env bash
# =====================================================================
# 一键跑完所有剩余 GPU 任务 (2026-08-23)
#
# 用法: bash _run_all_remaining.sh [REPO] [DATASET_DIR]
#   默认 REPO=/home/wangsh/workspace/Liu
#   默认 DATASET_DIR=$REPO/01_external/PatchTST/code/dataset
#
# 阶段:
#   S1 syn_ood 配对显著性 (patchtst+full_v2_fixed, 主表8seed)   [关键, ~30min]
#   S2 P1-2 baseline (dlinear+itransformer, 6数据集×8seed)      [大, 1-2天]
#   S3 P1-1 敏感性 (traffic, n_envs 2/8, rff_dim 16/64)        [~2-4h]
#   S4 P1-3 熵正则 (traffic, ew 0.01/0.1)                      [~1-2h]
#   S5 traffic 门控热图 (子采样50, 服务器已有 dump)             [~min]
#
# 规则:
#   * 全部阶段完成后才生成 _ALL_DONE.txt —— 只有它存在才允许回传。
#   * 中途断开/报错可重跑: 已完成的阶段有 _STAGE_DONE 标记, 自动跳过 (断点续跑)。
#   * 每阶段进度见 _run_all.log。
# =====================================================================
set -u

REPO=${1:-/home/wangsh/workspace/Liu}
DATASET_DIR=${2:-$REPO/01_external/PatchTST/code/dataset}
SEEDS="42 123 2024 7 13 99 2023 31"
LOG=_run_all.log

cd "$REPO/03_experiments/CausalCIT/CausalCIT_ablation" || { echo "repo 路径错误"; exit 1; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate causalcit 2>/dev/null || true
git pull --ff-only || true
mkdir -p ./_run_all_logs

echo "=== [$(date)] 一键任务启动: REPO=$REPO DD=$DATASET_DIR ===" | tee -a $LOG

count_done() {
  local dir=$1 n=0
  for f in "$dir"/results_shard*.csv; do
    [ -f "$f" ] && n=$((n + $(($(wc -l < "$f") - 1))))
  done
  echo $n
}
count_total() {
  local dir=$1
  python -c "
import json,glob
print(sum(len(json.load(open(f,encoding='utf-8'))) for f in glob.glob('$dir/jobs_shard*.json')))
"
}
run_shards() {
  local dir=$1
  for gpu in 0 1 2; do
    [ -f "$dir/jobs_shard$gpu.json" ] || continue
    CUDA_VISIBLE_DEVICES=$gpu nohup python run_large.py run --device cuda:0 \
      --job_file "$dir/jobs_shard$gpu.json" --result_csv "$dir/results_shard$gpu.csv" \
      --job_timeout 7200 > "$dir/log_shard$gpu.txt" 2>&1 &
  done
  wait
  echo "    shard 并行完成 ($(date))" >> $LOG
}

# ============ S1: syn_ood 配对显著性 (关键) ============
S1=output_synood_paired
if [ ! -f "$S1/_STAGE_DONE" ]; then
  echo "[$(date)] S1: gen syn_ood (patchtst+full_v2_fixed, 主表8seed)" | tee -a $LOG
  python run_large.py gen --datasets syn_ood --variants patchtst full_v2_fixed \
      --seeds $SEEDS --num_shards 3 --output_dir ./$S1 || exit 1
  echo "[$(date)] S1: run 3 shard 并行" | tee -a $LOG
  run_shards ./$S1
  python run_large.py summarize --output_dir ./$S1 >> $LOG 2>&1
  touch "$S1/_STAGE_DONE"
  echo "[$(date)] S1 完成 ($(count_done ./$S1)/$(count_total ./$S1))" | tee -a $LOG
else
  echo "[$(date)] S1 已有 _STAGE_DONE, 跳过" | tee -a $LOG
fi

# ============ S2: P1-2 baseline (dlinear + itransformer) ============
S2=output_baselines
if [ ! -f "$S2/_STAGE_DONE" ]; then
  echo "[$(date)] S2: gen baselines (dlinear+itransformer, 6 数据集)" | tee -a $LOG
  python run_large.py gen --datasets etth1 ettm1 weather exchange traffic electricity \
      --variants dlinear itransformer --seeds $SEEDS --num_shards 3 \
      --output_dir ./$S2 --dataset_dir "$DATASET_DIR" || exit 1
  echo "[$(date)] S2: run (预计 1-2 天)" | tee -a $LOG
  run_shards ./$S2
  python run_large.py summarize --output_dir ./$S2 >> $LOG 2>&1
  touch "$S2/_STAGE_DONE"
  echo "[$(date)] S2 完成 ($(count_done ./$S2)/$(count_total ./$S2))" | tee -a $LOG
else
  echo "[$(date)] S2 已有 _STAGE_DONE, 跳过" | tee -a $LOG
fi

# ============ S3: P1-1 敏感性 (traffic full_v2_fixed) ============
S3_PREFIX=output_sensitivity
if [ ! -f "${S3_PREFIX}_done" ]; then
  echo "[$(date)] S3: gen+run 敏感性 4 配置 (n_envs 2/8, rff_dim 16/64)" | tee -a $LOG
  declare -A CFGS=(
    ["n_envs_2"]="--n_envs 2"
    ["n_envs_8"]="--n_envs 8"
    ["rff_dim_16"]="--rff_dim 16"
    ["rff_dim_64"]="--rff_dim 64"
  )
  for tag in n_envs_2 n_envs_8 rff_dim_16 rff_dim_64; do
    D="${S3_PREFIX}_${tag}"
    if [ -f "$D/_STAGE_DONE" ]; then
      echo "  S3 $tag 已完成, 跳过" | tee -a $LOG; continue
    fi
    echo "[$(date)] S3 $tag: ${CFGS[$tag]}" | tee -a $LOG
    python run_large.py gen --datasets traffic --variants full_v2_fixed --seeds $SEEDS \
        --num_shards 3 --output_dir ./$D --dataset_dir "$DATASET_DIR" ${CFGS[$tag]} || exit 1
    run_shards ./$D
    touch "$D/_STAGE_DONE"
    echo "  S3 $tag 完成 ($(count_done ./$D)/$(count_total ./$D))" | tee -a $LOG
  done
  touch "${S3_PREFIX}_done"
  echo "[$(date)] S3 全部完成" | tee -a $LOG
else
  echo "[$(date)] S3 已有完成标记, 跳过" | tee -a $LOG
fi

# ============ S4: P1-3 熵正则 (traffic) ============
S4_PREFIX=output_entropy
if [ ! -f "${S4_PREFIX}_done" ]; then
  echo "[$(date)] S4: gen+run 熵正则 (ew 0.01/0.1)" | tee -a $LOG
  for ew in 0.01 0.1; do
    D="${S4_PREFIX}_ew_${ew}"
    if [ -f "$D/_STAGE_DONE" ]; then
      echo "  S4 ew=$ew 已完成, 跳过" | tee -a $LOG; continue
    fi
    echo "[$(date)] S4 ew=$ew" | tee -a $LOG
    python run_large.py gen --datasets traffic --variants full_v2_fixed --seeds $SEEDS \
        --num_shards 3 --output_dir ./$D --dataset_dir "$DATASET_DIR" \
        --entropy_weight $ew || exit 1
    run_shards ./$D
    touch "$D/_STAGE_DONE"
    echo "  S4 ew=$ew 完成 ($(count_done ./$D)/$(count_total ./$D))" | tee -a $LOG
  done
  touch "${S4_PREFIX}_done"
  echo "[$(date)] S4 全部完成" | tee -a $LOG
else
  echo "[$(date)] S4 已有完成标记, 跳过" | tee -a $LOG
fi

# ============ S5: traffic 门控热图 (子采样) ============
echo "[$(date)] S5: traffic 门控热图 (子采样50)" | tee -a $LOG
python plot_gate_heatmaps.py --gates_dir ./output_large_v3/gates \
    --datasets traffic --subsample 50 --output ./output_large_v3/vis_gates_traffic \
    >> $LOG 2>&1 || echo "  [S5 警告] 热图脚本失败 (可能缺 traffic dump), 继续" | tee -a $LOG

# ============ 收尾: _ALL_DONE.txt ============
{
  echo "ALL STAGES DONE @ $(date)"
  echo "---"
  echo "S1 syn_ood_paired:        $(count_done ./$S1)/$(count_total ./$S1)"
  echo "S2 baselines:             $(count_done ./$S2)/$(count_total ./$S2)"
  for d in ${S3_PREFIX}_*; do
    [ -d "$d" ] && echo "S3 $d: $(count_done ./$d)/$(count_total ./$d)"
  done
  for d in ${S4_PREFIX}_*; do
    [ -d "$d" ] && echo "S4 $d: $(count_done ./$d)/$(count_total ./$d)"
  done
  echo "S5 traffic heatmaps:      done"
  echo "---"
  echo "回传前请检查: 各阶段计数是否满额; 然后 git add -A && git commit && git push"
} > _ALL_DONE.txt
echo "[$(date)] ============ 全部阶段完成, 已生成 _ALL_DONE.txt — 现在才允许回传 ============" | tee -a $LOG
