#!/bin/bash

# Weather dataset - PatchTST/64
# Multivariate forecasting

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=PatchTST
seq_len=512

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path weather.csv \
      --model_id weather_${seq_len}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
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
      --pct_start 0.4 >logs/LongForecasting/$model_name'_'weather_${seq_len}_${pred_len}.log 2>&1
done
