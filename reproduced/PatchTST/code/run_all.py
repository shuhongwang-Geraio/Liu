"""
PatchTST 全部实验一键运行脚本
论文: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (ICLR 2023)

运行方式:
    python run_all.py

所有结果会汇总到 ./result.txt 中
"""

import os
import subprocess
import time
import sys


# ============================================================
# 实验配置
# ============================================================

# 公共参数
COMMON_ARGS = {
    'is_training': 1,
    'root_path': './dataset/',
    'model': 'PatchTST',
    'features': 'M',
    'patch_len': 16,
    'stride': 8,
    'des': 'Exp',
    'train_epochs': 100,
    'patience': 20,
    'itr': 1,
    'batch_size': 128,
    'learning_rate': 0.0001,
    'lradj': 'TST',
    'pct_start': 0.4,
    'num_workers': 0,
}

# 各数据集的实验配置
EXPERIMENTS = [
    # ==================== ETTh1 ====================
    {
        'name': 'ETTh1',
        'data_path': 'ETTh1.csv',
        'data': 'ETTh1',
        'enc_in': 7,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'e_layers': 3,
        'n_heads': 4,
        'd_model': 16,
        'd_ff': 128,
        'dropout': 0.3,
        'fc_dropout': 0.3,
        'head_dropout': 0,
    },
    # ==================== ETTh2 ====================
    {
        'name': 'ETTh2',
        'data_path': 'ETTh2.csv',
        'data': 'ETTh2',
        'enc_in': 7,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'e_layers': 3,
        'n_heads': 4,
        'd_model': 16,
        'd_ff': 128,
        'dropout': 0.3,
        'fc_dropout': 0.3,
        'head_dropout': 0,
    },
    # ==================== ETTm1 ====================
    {
        'name': 'ETTm1',
        'data_path': 'ETTm1.csv',
        'data': 'ETTm1',
        'enc_in': 7,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'e_layers': 3,
        'n_heads': 4,
        'd_model': 16,
        'd_ff': 128,
        'dropout': 0.3,
        'fc_dropout': 0.3,
        'head_dropout': 0,
    },
    # ==================== ETTm2 ====================
    {
        'name': 'ETTm2',
        'data_path': 'ETTm2.csv',
        'data': 'ETTm2',
        'enc_in': 7,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'e_layers': 3,
        'n_heads': 4,
        'd_model': 16,
        'd_ff': 128,
        'dropout': 0.3,
        'fc_dropout': 0.3,
        'head_dropout': 0,
    },
    # ==================== Weather ====================
    # 注意: 需要先下载 weather.csv 到 ./dataset/ 目录
    {
        'name': 'Weather',
        'data_path': 'weather.csv',
        'data': 'custom',
        'enc_in': 21,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'e_layers': 3,
        'n_heads': 16,
        'd_model': 128,
        'd_ff': 256,
        'dropout': 0.2,
        'fc_dropout': 0.2,
        'head_dropout': 0,
    },
    # ==================== Electricity ====================
    # 注意: 需要先下载 electricity.csv 到 ./dataset/ 目录
    {
        'name': 'Electricity',
        'data_path': 'electricity.csv',
        'data': 'custom',
        'enc_in': 321,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'e_layers': 3,
        'n_heads': 16,
        'd_model': 128,
        'd_ff': 256,
        'dropout': 0.2,
        'fc_dropout': 0.2,
        'head_dropout': 0,
    },
    # ==================== Traffic ====================
    # 注意: 需要先下载 traffic.csv 到 ./dataset/ 目录
    {
        'name': 'Traffic',
        'data_path': 'traffic.csv',
        'data': 'custom',
        'enc_in': 862,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'e_layers': 3,
        'n_heads': 16,
        'd_model': 128,
        'd_ff': 256,
        'dropout': 0.2,
        'fc_dropout': 0.2,
        'head_dropout': 0,
    }
]


def build_command(exp_config, pred_len):
    """构建单次实验的命令行参数"""
    seq_len = exp_config['seq_len']
    model_id = f"{exp_config['name']}_{seq_len}_{pred_len}"

    cmd = [
        sys.executable, 'run_longExp.py',
        '--is_training', str(COMMON_ARGS['is_training']),
        '--root_path', COMMON_ARGS['root_path'],
        '--data_path', exp_config['data_path'],
        '--model_id', model_id,
        '--model', COMMON_ARGS['model'],
        '--data', exp_config['data'],
        '--features', COMMON_ARGS['features'],
        '--seq_len', str(seq_len),
        '--pred_len', str(pred_len),
        '--enc_in', str(exp_config['enc_in']),
        '--e_layers', str(exp_config['e_layers']),
        '--n_heads', str(exp_config['n_heads']),
        '--d_model', str(exp_config['d_model']),
        '--d_ff', str(exp_config['d_ff']),
        '--dropout', str(exp_config['dropout']),
        '--fc_dropout', str(exp_config['fc_dropout']),
        '--head_dropout', str(exp_config['head_dropout']),
        '--patch_len', str(COMMON_ARGS['patch_len']),
        '--stride', str(COMMON_ARGS['stride']),
        '--des', COMMON_ARGS['des'],
        '--train_epochs', str(COMMON_ARGS['train_epochs']),
        '--patience', str(COMMON_ARGS['patience']),
        '--itr', str(COMMON_ARGS['itr']),
        '--batch_size', str(COMMON_ARGS['batch_size']),
        '--learning_rate', str(COMMON_ARGS['learning_rate']),
        '--lradj', COMMON_ARGS['lradj'],
        '--pct_start', str(COMMON_ARGS['pct_start']),
        '--num_workers', str(COMMON_ARGS['num_workers']),
    ]
    return cmd


def run_experiment(exp_config, pred_len, exp_idx, total_exps):
    """运行单次实验"""
    model_id = f"{exp_config['name']}_{exp_config['seq_len']}_{pred_len}"
    
    print(f"\n{'='*70}")
    print(f"  [{exp_idx}/{total_exps}] {model_id}")
    print(f"  Dataset: {exp_config['name']} | pred_len: {pred_len}")
    print(f"  d_model: {exp_config['d_model']} | d_ff: {exp_config['d_ff']} | "
          f"n_heads: {exp_config['n_heads']} | e_layers: {exp_config['e_layers']}")
    print(f"{'='*70}")

    cmd = build_command(exp_config, pred_len)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n  [OK] {model_id} done in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    else:
        print(f"\n  [FAIL] {model_id} failed with return code {result.returncode}")
    
    return result.returncode == 0, elapsed


def main():
    # 计算总实验数
    total_exps = sum(len(exp['pred_lens']) for exp in EXPERIMENTS)
    
    print("="*70)
    print("  PatchTST 全部实验一键运行")
    print("  论文: A Time Series is Worth 64 Words (ICLR 2023)")
    print(f"  总计: {total_exps} 个实验")
    print("="*70)

    # 检查数据集是否存在
    missing = []
    for exp in EXPERIMENTS:
        data_file = os.path.join(COMMON_ARGS['root_path'], exp['data_path'])
        if not os.path.exists(data_file):
            missing.append(exp['data_path'])
    
    if missing:
        print(f"\n[WARNING] 以下数据集文件不存在，相关实验将跳过:")
        for f in missing:
            print(f"  - {COMMON_ARGS['root_path']}{f}")
        print()

    # 运行所有实验
    exp_idx = 0
    success_count = 0
    fail_count = 0
    total_time = 0
    results_summary = []

    overall_start = time.time()

    for exp_config in EXPERIMENTS:
        # 检查数据文件
        data_file = os.path.join(COMMON_ARGS['root_path'], exp_config['data_path'])
        if not os.path.exists(data_file):
            for pred_len in exp_config['pred_lens']:
                exp_idx += 1
                print(f"\n  [{exp_idx}/{total_exps}] SKIP {exp_config['name']}_{exp_config['seq_len']}_{pred_len} (data not found)")
                results_summary.append((exp_config['name'], pred_len, 'SKIP', 0))
            continue

        for pred_len in exp_config['pred_lens']:
            exp_idx += 1
            success, elapsed = run_experiment(exp_config, pred_len, exp_idx, total_exps)
            total_time += elapsed
            
            if success:
                success_count += 1
                results_summary.append((exp_config['name'], pred_len, 'OK', elapsed))
            else:
                fail_count += 1
                results_summary.append((exp_config['name'], pred_len, 'FAIL', elapsed))

    overall_elapsed = time.time() - overall_start

    # 打印总结
    print(f"\n\n{'='*70}")
    print(f"  ALL EXPERIMENTS FINISHED!")
    print(f"{'='*70}")
    print(f"  Success: {success_count}/{total_exps}")
    print(f"  Failed:  {fail_count}/{total_exps}")
    print(f"  Skipped: {total_exps - success_count - fail_count}/{total_exps}")
    print(f"  Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min, {overall_elapsed/3600:.2f}h)")
    print(f"{'='*70}")
    
    print(f"\n  Summary:")
    print(f"  {'Dataset':<15} {'pred_len':<10} {'Status':<8} {'Time':<10}")
    print(f"  {'-'*43}")
    for name, pred_len, status, elapsed in results_summary:
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        print(f"  {name:<15} {pred_len:<10} {status:<8} {time_str:<10}")

    print(f"\n  Results saved to: ./result.txt")
    print(f"  Checkpoints saved to: ./checkpoints/")
    print(f"  Visualizations saved to: ./test_results/")


if __name__ == '__main__':
    main()
