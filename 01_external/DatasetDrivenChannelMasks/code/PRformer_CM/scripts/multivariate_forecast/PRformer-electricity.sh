export CUDA_VISIBLE_DEVICES=0

model_name=PRformer
seq_len=660
convWindows='22 44 66 88 132' # There are only 22 data records a day in electricity, missing data at 6, 7, 8 o 'clock

echo electricity
for pred_len in  96 192 336 720
do
    echo grid-parameters: seq_len:$seq_len pred_len:$pred_len 
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len ${seq_len} \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 660\
    --d_ff 660\
    --batch_size 16\
    --learning_rate 0.0005\
    --itr 1 \
    --loss mae\
    --train_epochs 100\
    --patience 50\
    --convWindows $convWindows --rnnMixTemperature 0.002 --lradj type3
done  > ./'PRformer-Electricity'.log