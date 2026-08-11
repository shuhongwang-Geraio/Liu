dim=32
model_name=UniTS_CM
wandb_mode=online
project_name=anomaly_detection_x32
exp_name=finetune_few_shot_anomaly_detection_pct05
random_port=$((RANDOM % 9000 + 1000))

ckpt_path=xxxxxx

torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  run.py \
  --fix_seed 2021 \
  --is_training 1 \
  --subsample_pct 0.05 \
  --model_id $exp_name \
  --pretrained_weight $ckpt_path \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $dim \
  --des 'Exp' \
  --itr 1 \
  --lradj finetune_anl \
  --learning_rate 5e-4 \
  --weight_decay 1e-3 \
  --train_epochs 10 \
  --batch_size 32 \
  --acc_it 32 \
  --dropout 0 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \
  --task_data_config_path data_provider/anomaly_detection.yaml