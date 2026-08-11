export CUDA_VISIBLE_DEVICES=1

model_name=PRformer
seq_len=720

echo solar-energy
for pred_len in 96 192 336 720
do
    echo grid-parameters: pred_len:$pred_len
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/Solar/ \
      --data_path solar_AL.txt \
      --model_id solar_${seq_len}_$pred_len \
      --model $model_name \
      --data Solar \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --des 'Exp' \
      --d_model 512\
      --d_ff 512\
      --itr 1 \
      --loss mae\
      --train_epochs 30 --learning_rate 0.0001 --batch_size 64\
      --convWindows 6 24 48 144 --rnnMixTemperature 1
done > ./'PRformer_solar'.log