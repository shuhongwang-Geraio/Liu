export CUDA_VISIBLE_DEVICES=0

model_name=PRformer
seq_len=720
convWindows='24 48 72 144' 

echo ETTh2
for pred_len in 96 192 336 720 
do
    echo grid-parameters: seq_len:$seq_len pred_len:$pred_len 
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv\
    --model_id ETTh2_multi_${seq_len}_${pred_len}\
    --model $model_name \
    --data ETTh2\
    --features M\
    --seq_len $seq_len\
    --label_len 48\
    --pred_len $pred_len\
    --e_layers 5\
    --d_layers 1\
    --factor 3\
    --enc_in 7\
    --dec_in 7\
    --c_out 7\
    --des 'Exp'\
    --d_model 720\
    --d_ff 720\
    --batch_size 256\
    --train_epochs 100\
    --patience 50\
    --learning_rate 0.0001\
    --itr 1 \
    --loss mae\
    --convWindows $convWindows --rnnMixTemperature 0.002\
    --lradj type3
done  > ./'PRformer-ETTh2'.log