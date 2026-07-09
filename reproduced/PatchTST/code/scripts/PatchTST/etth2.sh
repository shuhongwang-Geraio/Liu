#!/bin/bash

# ETTh2 dataset - PatchTST/42
# Multivariate forecasting

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=PatchTST
seq_len=336

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_${seq_len}_${pred_len} \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 20 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --lradj 'TST' \
      --pct_start 0.4 >logs/LongForecasting/$model_name'_'ETTh2_${seq_len}_${pred_len}.log 2>&1
done
