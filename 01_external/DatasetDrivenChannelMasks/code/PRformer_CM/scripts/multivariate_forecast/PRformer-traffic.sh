export CUDA_VISIBLE_DEVICES=0

model_name=PRformer
seq_len=720
convWindows='24 48 72 96 144' 

echo traffic

for pred_len in  96 192 336 720 
do
    echo grid-parameters: seq_len:$seq_len pred_len:$pred_len 
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 4 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 520\
    --d_ff 520\
    --batch_size 16\
    --learning_rate 0.001\
    --itr 1 \
    --loss mae\
    --train_epochs 100\
    --patience 50\
    --convWindows $convWindows --rnnMixTemperature 0.002 --lradj type3
done  > ./'PRformer_Traffic'.log