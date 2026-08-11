export CUDA_VISIBLE_DEVICES=0

model_name=PRformer
seq_len=720
convWindows='6 24 48 144' 

echo weather
for pred_len in 96 192 336 720
do
    echo grid-parameters: pred_len:$pred_len  seq_len:$seq_len d_ff:$d_ff 
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_${seq_len}_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 3 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model 720\
      --d_ff 720\
      --itr 1 \
      --loss mae\
      --train_epochs 10\
      --patience 4\
      --batch_size 64 --learning_rate 0.0001\
      --convWindows $convWindows --rnnMixTemperature 0.002 --lradj type3
done > ./'PRformer_Weather'.log