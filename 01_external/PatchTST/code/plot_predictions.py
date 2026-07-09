"""
PatchTST 预测结果可视化脚本

使用训练完成后保存在 ./results/ 下的 pred.npy 文件进行绘图
无需重新训练模型

使用方式:
    python plot_predictions.py

输出: ./figures/ 目录下的 PNG 图片
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

plt.switch_backend('agg')


# ============================================================
# 配置（和训练时保持一致）
# ============================================================

CONFIGS = [
    {
        'name': 'ETTh1_336_96',
        'dataset_class': Dataset_ETT_hour,
        'data_path': 'ETTh1.csv',
        'seq_len': 336,
        'pred_len': 96,
        'label_len': 48,
        'setting': 'ETTh1_336_96_PatchTST_ETTh1_ftM_sl336_ll48_pl96_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0',
    },
    {
        'name': 'ETTh1_336_192',
        'dataset_class': Dataset_ETT_hour,
        'data_path': 'ETTh1.csv',
        'seq_len': 336,
        'pred_len': 192,
        'label_len': 48,
        'setting': 'ETTh1_336_192_PatchTST_ETTh1_ftM_sl336_ll48_pl192_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0',
    },
    {
        'name': 'ETTh1_336_336',
        'dataset_class': Dataset_ETT_hour,
        'data_path': 'ETTh1.csv',
        'seq_len': 336,
        'pred_len': 336,
        'label_len': 48,
        'setting': 'ETTh1_336_336_PatchTST_ETTh1_ftM_sl336_ll48_pl336_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0',
    },
    {
        'name': 'ETTh1_336_720',
        'dataset_class': Dataset_ETT_hour,
        'data_path': 'ETTh1.csv',
        'seq_len': 336,
        'pred_len': 720,
        'label_len': 48,
        'setting': 'ETTh1_336_720_PatchTST_ETTh1_ftM_sl336_ll48_pl720_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0',
    },
    {
        'name': 'ETTh2_336_96',
        'dataset_class': Dataset_ETT_hour,
        'data_path': 'ETTh2.csv',
        'seq_len': 336,
        'pred_len': 96,
        'label_len': 48,
        'setting': 'ETTh2_336_96_PatchTST_ETTh2_ftM_sl336_ll48_pl96_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0',
    },
    {
        'name': 'ETTm1_336_96',
        'dataset_class': Dataset_ETT_minute,
        'data_path': 'ETTm1.csv',
        'seq_len': 336,
        'pred_len': 96,
        'label_len': 48,
        'setting': 'ETTm1_336_96_PatchTST_ETTm1_ftM_sl336_ll48_pl96_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0',
    },
    {
        'name': 'ETTm2_336_96',
        'dataset_class': Dataset_ETT_minute,
        'data_path': 'ETTm2.csv',
        'seq_len': 336,
        'pred_len': 96,
        'label_len': 48,
        'setting': 'ETTm2_336_96_PatchTST_ETTm2_ftM_sl336_ll48_pl96_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0',
    },
]


def find_available_results():
    """扫描 ./results/ 目录，找到所有可用的 pred.npy 文件"""
    results_dir = './results'
    available = []
    
    if not os.path.exists(results_dir):
        return available
    
    for setting_name in os.listdir(results_dir):
        pred_path = os.path.join(results_dir, setting_name, 'pred.npy')
        if os.path.exists(pred_path):
            available.append({
                'setting': setting_name,
                'pred_path': pred_path,
            })
    
    return available


def parse_setting_info(setting_name):
    """从 setting 名称解析出关键信息"""
    import re
    info = {}
    
    # 提取数据集名
    parts = setting_name.split('_')
    info['dataset'] = parts[0] if parts else 'Unknown'
    
    # 提取参数
    sl_match = re.search(r'_sl(\d+)_', setting_name)
    pl_match = re.search(r'_pl(\d+)_', setting_name)
    
    info['seq_len'] = int(sl_match.group(1)) if sl_match else 336
    info['pred_len'] = int(pl_match.group(1)) if pl_match else 96
    
    return info


def get_test_data(info):
    """加载测试集的原始数据用于绘制历史部分"""
    data_path_map = {
        'ETTh1': ('ETTh1.csv', Dataset_ETT_hour),
        'ETTh2': ('ETTh2.csv', Dataset_ETT_hour),
        'ETTm1': ('ETTm1.csv', Dataset_ETT_minute),
        'ETTm2': ('ETTm2.csv', Dataset_ETT_minute),
    }
    
    dataset_name = info['dataset']
    if dataset_name not in data_path_map:
        return None
    
    csv_file, DatasetClass = data_path_map[dataset_name]
    csv_path = os.path.join('./dataset/', csv_file)
    
    if not os.path.exists(csv_path):
        return None
    
    dataset = DatasetClass(
        root_path='./dataset/',
        flag='test',
        size=[info['seq_len'], 48, info['pred_len']],
        features='M',
        data_path=csv_file,
        target='OT',
        timeenc=1,
        freq='h'
    )
    return dataset


def plot_single_sample(input_seq, true_future, pred_future, seq_len, pred_len,
                       title, save_path, sample_idx=0):
    """绘制单个样本的预测图"""
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    total_len = seq_len + pred_len
    x_all = np.arange(total_len)
    x_history = np.arange(seq_len)
    x_future = np.arange(seq_len, total_len)
    
    # 绘制历史区域（浅蓝色背景）
    ax.axvspan(0, seq_len, alpha=0.06, color='blue', label='_nolegend_')
    # 绘制预测区域（浅红色背景）
    ax.axvspan(seq_len, total_len, alpha=0.06, color='red', label='_nolegend_')
    
    # 绘制分界线
    ax.axvline(x=seq_len, color='#333333', linestyle='--', linewidth=1.5, 
               label='Prediction Start', alpha=0.8)
    
    # 绘制历史真实值
    ax.plot(x_history, input_seq, color='#2196F3', linewidth=1.5, 
            label='History (Input)', alpha=0.9)
    
    # 绘制未来真实值
    ax.plot(x_future, true_future, color='#4CAF50', linewidth=2.0, 
            label='GroundTruth (Future)', alpha=0.9)
    
    # 绘制预测值
    ax.plot(x_future, pred_future, color='#F44336', linewidth=2.0, 
            label='Prediction', linestyle='-', alpha=0.9)
    
    # 连接历史和未来（让图看起来连续）
    ax.plot([seq_len - 1, seq_len], [input_seq[-1], true_future[0]], 
            color='#4CAF50', linewidth=1.5, alpha=0.5)
    ax.plot([seq_len - 1, seq_len], [input_seq[-1], pred_future[0]], 
            color='#F44336', linewidth=1.5, alpha=0.5)
    
    # 标注
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Value (Normalized)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.2)
    
    # 添加文字标注
    ax.text(seq_len * 0.5, ax.get_ylim()[1] * 0.95, 'History', 
            ha='center', fontsize=10, color='#1565C0', alpha=0.7)
    ax.text(seq_len + pred_len * 0.5, ax.get_ylim()[1] * 0.95, 'Forecast', 
            ha='center', fontsize=10, color='#C62828', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_multi_samples(dataset, preds, info, save_dir, num_samples=5):
    """绘制多个样本的预测图"""
    
    seq_len = info['seq_len']
    pred_len = info['pred_len']
    n_total = len(preds)
    
    # 选择均匀分布的样本
    indices = np.linspace(0, n_total - 1, num_samples, dtype=int)
    
    for idx, sample_idx in enumerate(indices):
        # 获取输入序列
        if dataset is not None and sample_idx < len(dataset):
            seq_x, seq_y, _, _ = dataset[sample_idx]
            input_seq = seq_x[:, -1]  # 最后一个变量
        else:
            # 如果没有原始数据，用预测的前面部分填充（标记为不可用）
            input_seq = np.zeros(seq_len)
        
        # 预测值和真实值
        pred_future = preds[sample_idx, :, -1]  # 最后一个变量
        
        # 从数据集获取真实未来值
        if dataset is not None and sample_idx < len(dataset):
            _, seq_y, _, _ = dataset[sample_idx]
            true_future = seq_y[-pred_len:, -1]
        else:
            true_future = np.zeros(pred_len)
        
        title = f"{info['dataset']} | seq_len={seq_len}, pred_len={pred_len} | Sample #{sample_idx}"
        filename = f"{info['dataset']}_pl{pred_len}_sample{idx}.png"
        save_path = os.path.join(save_dir, filename)
        
        plot_single_sample(input_seq, true_future, pred_future,
                          seq_len, pred_len, title, save_path, sample_idx)
    
    return len(indices)


def plot_overview(dataset, preds, info, save_dir):
    """绘制一张总览图：多个样本的子图拼接"""
    
    seq_len = info['seq_len']
    pred_len = info['pred_len']
    n_total = len(preds)
    
    num_samples = min(4, n_total)
    indices = np.linspace(0, n_total - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3.5 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for ax_idx, sample_idx in enumerate(indices):
        ax = axes[ax_idx]
        
        # 获取数据
        if dataset is not None and sample_idx < len(dataset):
            seq_x, seq_y, _, _ = dataset[sample_idx]
            input_seq = seq_x[:, -1]
            true_future = seq_y[-pred_len:, -1]
        else:
            input_seq = np.zeros(seq_len)
            true_future = np.zeros(pred_len)
        
        pred_future = preds[sample_idx, :, -1]
        
        total_len = seq_len + pred_len
        x_history = np.arange(seq_len)
        x_future = np.arange(seq_len, total_len)
        
        # 背景色区分
        ax.axvspan(0, seq_len, alpha=0.05, color='blue')
        ax.axvspan(seq_len, total_len, alpha=0.05, color='red')
        ax.axvline(x=seq_len, color='#333333', linestyle='--', linewidth=1.2, alpha=0.7)
        
        # 绘制
        ax.plot(x_history, input_seq, color='#2196F3', linewidth=1.2, alpha=0.8)
        ax.plot(x_future, true_future, color='#4CAF50', linewidth=1.8, label='GroundTruth')
        ax.plot(x_future, pred_future, color='#F44336', linewidth=1.8, label='Prediction')
        
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(f'Sample #{sample_idx}', fontsize=10, loc='left')
        ax.grid(alpha=0.2)
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Time Steps')
    fig.suptitle(f"{info['dataset']} | seq_len={seq_len}, pred_len={pred_len}", 
                 fontsize=13, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{info['dataset']}_pl{pred_len}_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def main():
    print("="*60)
    print("  PatchTST 预测结果可视化")
    print("="*60)
    
    # 创建输出目录
    fig_dir = './figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    # 扫描可用结果
    print("\n[1] 扫描已保存的预测结果...")
    available = find_available_results()
    
    if not available:
        print("  [ERROR] 没有找到任何预测结果!")
        print("  请确保 ./results/ 目录下有训练后保存的 pred.npy 文件")
        print("  或将服务器上的 ./results/ 文件夹拷贝到当前目录")
        return
    
    print(f"  找到 {len(available)} 个实验结果:")
    for item in available:
        print(f"    - {item['setting']}")
    
    # 逐个处理
    print(f"\n[2] 生成可视化图片 -> {os.path.abspath(fig_dir)}/")
    total_plots = 0
    
    for item in available:
        setting = item['setting']
        info = parse_setting_info(setting)
        
        print(f"\n  Processing: {setting}")
        print(f"    Dataset={info['dataset']}, seq_len={info['seq_len']}, pred_len={info['pred_len']}")
        
        # 加载预测结果
        preds = np.load(item['pred_path'])
        print(f"    pred.npy shape: {preds.shape}")
        
        # 加载测试数据集
        dataset = get_test_data(info)
        if dataset is None:
            print(f"    [WARN] 无法加载原始测试数据，将只绘制预测部分")
        
        # 绘制单样本图（5张）
        n = plot_multi_samples(dataset, preds, info, fig_dir, num_samples=5)
        total_plots += n
        print(f"    -> {n} single-sample plots saved")
        
        # 绘制总览图
        overview_path = plot_overview(dataset, preds, info, fig_dir)
        total_plots += 1
        print(f"    -> overview plot saved")
    
    print(f"\n{'='*60}")
    print(f"  Done! {total_plots} plots saved to: {os.path.abspath(fig_dir)}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
