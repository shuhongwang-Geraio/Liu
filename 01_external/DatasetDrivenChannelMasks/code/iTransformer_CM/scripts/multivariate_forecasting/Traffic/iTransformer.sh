export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer_CM
ep=25
dataset=Traffic

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --dataset $dataset\
  --model $model_name \
  --train_epochs $ep\
  --init_alpha 2\
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 511\
  --d_ff 511 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
--is_training 1 \
--root_path ./dataset/traffic/ \
--data_path traffic.csv \
--model_id traffic_96_192 \
--dataset $dataset\
--model $model_name \
--train_epochs $ep\
--init_alpha 3\
--data custom \
--features M \
--seq_len 96 \
--pred_len 192 \
--e_layers 4 \
--enc_in 862 \
--dec_in 862 \
--c_out 862 \
--des 'Exp' \
--d_model 512 \
--d_ff 512 \
--batch_size 16 \
--learning_rate 0.001 \
--itr 1

python -u run.py \
--is_training 1 \
--root_path ./dataset/traffic/ \
--data_path traffic.csv \
--model_id traffic_96_336 \
--dataset $dataset\
--model $model_name \
--train_epochs $ep\
--init_alpha 3\
--data custom \
--features M \
--seq_len 96 \
--pred_len 336 \
--e_layers 4 \
--enc_in 862 \
--dec_in 862 \
--c_out 862 \
--des 'Exp' \
--d_model 512\
--d_ff 512 \
--batch_size 16 \
--learning_rate 0.001 \
--itr 1

python -u run.py \
--is_training 1 \
--root_path ./dataset/traffic/ \
--data_path traffic.csv \
--model_id traffic_96_720 \
--dataset $dataset\
--model $model_name \
--train_epochs $ep\
--init_alpha 3\
--data custom \
--features M \
--seq_len 96 \
--pred_len 720 \
--e_layers 4 \
--enc_in 862 \
--dec_in 862 \
--c_out 862 \
--des 'Exp' \
--d_model 512 \
--d_ff 512 \
--batch_size 16 \
--learning_rate 0.001\
--itr 1