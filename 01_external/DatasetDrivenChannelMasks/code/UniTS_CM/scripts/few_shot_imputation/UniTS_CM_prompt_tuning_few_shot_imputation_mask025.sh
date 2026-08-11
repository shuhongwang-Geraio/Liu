dim=32
model_name=UniTS_CM
wandb_mode=online
project_name=fewshot_imputation_x32
exp_name=fewshot_imputation_prompt_tuning_mask025

ckpt_path=xxx
random_port=$((RANDOM % 9000 + 1000))

torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  run.py \
  --is_training 1 \
  --fix_seed 2021 \
  --model_id $exp_name \
  --subsample_pct 0.1 \
  --mask_rate 0.25 \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $dim \
  --des 'Exp' \
  --itr 1 \
  --prompt_tune_epoch 20 \
  --train_epochs 0 \
  --lradj prompt_tuning \
  --learning_rate 5e-3 \
  --weight_decay 0 \
  --batch_size 32 \
  --acc_it 32 \
  --clip_grad 1.0 \
  --dropout 0 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/imputation.yaml \