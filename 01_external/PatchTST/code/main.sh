@echo off
echo ===============================================
echo PatchTST 实验延迟执行脚本
echo 延迟时间: 1小时 (3600秒)
echo 当前时间: %date% %time%
echo ===============================================
echo.
echo 等待1小时后开始执行...
timeout /t 3600 /nobreak

echo.
echo ========== 开始执行实验 ==========
echo 当前时间: %date% %time%

call D:\Project\Miniconda\Scripts\activate.bat XiangMu_Liu

cd /d "c:\Users\RedMoon\Desktop\0519\刘\patchtst"

python run_longExp.py ^
  --is_training 1 ^
  --root_path ./dataset/ ^
  --data_path ETTh1.csv ^
  --model_id ETTh1_336_96 ^
  --model PatchTST ^
  --data ETTh1 ^
  --features M ^
  --seq_len 336 ^
  --pred_len 96 ^
  --enc_in 7 ^
  --e_layers 3 ^
  --n_heads 4 ^
  --d_model 16 ^
  --d_ff 128 ^
  --dropout 0.3 ^
  --fc_dropout 0.3 ^
  --head_dropout 0 ^
  --patch_len 16 ^
  --stride 8 ^
  --des Exp ^
  --train_epochs 100 ^
  --patience 20 ^
  --itr 1 ^
  --batch_size 128 ^
  --learning_rate 0.0001 ^
  --lradj TST ^
  --pct_start 0.4 ^
  --num_workers 0

echo.
echo ========== 实验完成 ==========
echo 结束时间: %date% %time%
pause