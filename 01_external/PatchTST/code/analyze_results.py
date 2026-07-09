"""
PatchTST 实验结果分析与报告生成脚本

使用方式:
    python analyze_results.py

输入: ./result.txt (训练完成后自动生成的结果文件)
输出:
    - ./report/analysis_report.md   (Markdown 分析报告)
    - ./report/comparison_table.csv (对比表格 CSV)
    - ./report/mse_comparison.png   (MSE 对比柱状图)
    - ./report/mae_comparison.png   (MAE 对比柱状图)
    - ./report/pred_len_trend.png   (预测长度 vs 指标趋势图)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.switch_backend('agg')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)


# ============================================================
# 论文报告的基准结果 (Table 1 in paper)
# ============================================================
PAPER_RESULTS = {
    ('ETTh1', 96):  {'mse': 0.370, 'mae': 0.400},
    ('ETTh1', 192): {'mse': 0.413, 'mae': 0.429},
    ('ETTh1', 336): {'mse': 0.422, 'mae': 0.440},
    ('ETTh1', 720): {'mse': 0.447, 'mae': 0.468},
    ('ETTh2', 96):  {'mse': 0.274, 'mae': 0.337},
    ('ETTh2', 192): {'mse': 0.341, 'mae': 0.382},
    ('ETTh2', 336): {'mse': 0.329, 'mae': 0.384},
    ('ETTh2', 720): {'mse': 0.379, 'mae': 0.422},
    ('ETTm1', 96):  {'mse': 0.293, 'mae': 0.346},
    ('ETTm1', 192): {'mse': 0.333, 'mae': 0.370},
    ('ETTm1', 336): {'mse': 0.369, 'mae': 0.392},
    ('ETTm1', 720): {'mse': 0.416, 'mae': 0.420},
    ('ETTm2', 96):  {'mse': 0.166, 'mae': 0.256},
    ('ETTm2', 192): {'mse': 0.223, 'mae': 0.296},
    ('ETTm2', 336): {'mse': 0.274, 'mae': 0.329},
    ('ETTm2', 720): {'mse': 0.362, 'mae': 0.385},
    ('Weather', 96):  {'mse': 0.149, 'mae': 0.198},
    ('Weather', 192): {'mse': 0.194, 'mae': 0.241},
    ('Weather', 336): {'mse': 0.245, 'mae': 0.282},
    ('Weather', 720): {'mse': 0.314, 'mae': 0.334},
    ('Electricity', 96):  {'mse': 0.129, 'mae': 0.222},
    ('Electricity', 192): {'mse': 0.147, 'mae': 0.240},
    ('Electricity', 336): {'mse': 0.163, 'mae': 0.259},
    ('Electricity', 720): {'mse': 0.197, 'mae': 0.290},
    ('Traffic', 96):  {'mse': 0.360, 'mae': 0.249},
    ('Traffic', 192): {'mse': 0.379, 'mae': 0.256},
    ('Traffic', 336): {'mse': 0.392, 'mae': 0.264},
    ('Traffic', 720): {'mse': 0.432, 'mae': 0.286},
}


def parse_result_file(filepath='./result.txt'):
    """解析 result.txt 文件，提取所有实验结果"""
    results = []
    
    if not os.path.exists(filepath):
        print(f"[ERROR] 找不到结果文件: {filepath}")
        print("请确保已经运行过训练，或将服务器上的 result.txt 拷贝到当前目录")
        return results
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and 'mse:' in lines[i + 1] if i + 1 < len(lines) else False:
            setting = line
            metric_line = lines[i + 1].strip()
            
            # 解析 setting 字符串，提取数据集和预测长度
            dataset_name, pred_len = parse_setting(setting)
            
            # 解析指标
            mse_match = re.search(r'mse:([\d.]+)', metric_line)
            mae_match = re.search(r'mae:([\d.]+)', metric_line)
            rse_match = re.search(r'rse:([\d.]+)', metric_line)
            
            if mse_match and mae_match and dataset_name:
                results.append({
                    'setting': setting,
                    'dataset': dataset_name,
                    'pred_len': pred_len,
                    'mse': float(mse_match.group(1)),
                    'mae': float(mae_match.group(1)),
                    'rse': float(rse_match.group(1)) if rse_match else None,
                })
            i += 2
        else:
            i += 1
    
    return results


def parse_setting(setting):
    """从 setting 字符串中提取数据集名称和预测长度"""
    # 格式: ETTh1_336_96_PatchTST_ETTh1_ftM_sl336_ll48_pl96_dm16_...
    # 或: weather_336_96_PatchTST_custom_ftM_sl336_ll48_pl96_dm128_...
    
    # 提取 pred_len
    pl_match = re.search(r'_pl(\d+)_', setting)
    pred_len = int(pl_match.group(1)) if pl_match else None
    
    # 提取数据集名称
    dataset_name = None
    for name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'weather', 
                 'Electricity', 'electricity', 'Traffic', 'traffic']:
        if name in setting.split('_')[0] or name.lower() in setting.lower().split('_')[0]:
            dataset_name = name
            break
    
    # 规范化名称
    if dataset_name:
        name_map = {'weather': 'Weather', 'electricity': 'Electricity', 'traffic': 'Traffic'}
        dataset_name = name_map.get(dataset_name, dataset_name)
    
    return dataset_name, pred_len


def generate_comparison_table(results):
    """生成复现结果与论文对比表格"""
    table_data = []
    
    for r in results:
        key = (r['dataset'], r['pred_len'])
        paper = PAPER_RESULTS.get(key, {})
        
        row = {
            'dataset': r['dataset'],
            'pred_len': r['pred_len'],
            'mse_ours': r['mse'],
            'mae_ours': r['mae'],
            'mse_paper': paper.get('mse', None),
            'mae_paper': paper.get('mae', None),
        }
        
        if row['mse_paper']:
            row['mse_diff'] = r['mse'] - row['mse_paper']
            row['mse_diff_pct'] = (row['mse_diff'] / row['mse_paper']) * 100
        else:
            row['mse_diff'] = None
            row['mse_diff_pct'] = None
            
        if row['mae_paper']:
            row['mae_diff'] = r['mae'] - row['mae_paper']
            row['mae_diff_pct'] = (row['mae_diff'] / row['mae_paper']) * 100
        else:
            row['mae_diff'] = None
            row['mae_diff_pct'] = None
        
        table_data.append(row)
    
    return table_data


def plot_mse_comparison(table_data, save_path):
    """绘制 MSE 对比柱状图"""
    datasets = []
    ours_values = []
    paper_values = []
    labels = []
    
    for row in table_data:
        if row['mse_paper'] is not None:
            labels.append(f"{row['dataset']}\n{row['pred_len']}")
            ours_values.append(row['mse_ours'])
            paper_values.append(row['mse_paper'])
    
    if not labels:
        return
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 6))
    bars1 = ax.bar(x - width/2, ours_values, width, label='Ours (Reproduced)', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, paper_values, width, label='Paper (Reported)', color='#FF9800', alpha=0.8)
    
    ax.set_xlabel('Dataset / Prediction Length')
    ax.set_ylabel('MSE')
    ax.set_title('PatchTST Reproduction: MSE Comparison (Ours vs Paper)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_mae_comparison(table_data, save_path):
    """绘制 MAE 对比柱状图"""
    ours_values = []
    paper_values = []
    labels = []
    
    for row in table_data:
        if row['mae_paper'] is not None:
            labels.append(f"{row['dataset']}\n{row['pred_len']}")
            ours_values.append(row['mae_ours'])
            paper_values.append(row['mae_paper'])
    
    if not labels:
        return
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 6))
    bars1 = ax.bar(x - width/2, ours_values, width, label='Ours (Reproduced)', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x + width/2, paper_values, width, label='Paper (Reported)', color='#F44336', alpha=0.8)
    
    ax.set_xlabel('Dataset / Prediction Length')
    ax.set_ylabel('MAE')
    ax.set_title('PatchTST Reproduction: MAE Comparison (Ours vs Paper)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_pred_len_trend(results, save_path):
    """绘制预测长度 vs 指标的趋势图"""
    # 按数据集分组
    datasets = {}
    for r in results:
        if r['dataset'] not in datasets:
            datasets[r['dataset']] = {'pred_lens': [], 'mse': [], 'mae': []}
        datasets[r['dataset']]['pred_lens'].append(r['pred_len'])
        datasets[r['dataset']]['mse'].append(r['mse'])
        datasets[r['dataset']]['mae'].append(r['mae'])
    
    if not datasets:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#795548']
    
    for idx, (name, data) in enumerate(sorted(datasets.items())):
        # 排序
        sorted_pairs = sorted(zip(data['pred_lens'], data['mse'], data['mae']))
        pls = [p[0] for p in sorted_pairs]
        mses = [p[1] for p in sorted_pairs]
        maes = [p[2] for p in sorted_pairs]
        
        color = colors[idx % len(colors)]
        ax1.plot(pls, mses, 'o-', label=name, color=color, linewidth=2, markersize=6)
        ax2.plot(pls, maes, 's-', label=name, color=color, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Prediction Length')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE vs Prediction Length')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Prediction Length')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE vs Prediction Length')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def generate_markdown_report(results, table_data, report_path):
    """生成 Markdown 格式的分析报告"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# PatchTST 复现实验报告\n\n")
        f.write(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 论文信息
        f.write("## 1. 论文信息\n\n")
        f.write("- **标题**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers\n")
        f.write("- **会议**: ICLR 2023\n")
        f.write("- **作者**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam (IBM Research)\n\n")
        
        # 实验概述
        f.write("## 2. 实验概述\n\n")
        dataset_list = sorted(set(r['dataset'] for r in results))
        f.write(f"- **数据集**: {', '.join(dataset_list)}\n")
        f.write(f"- **实验总数**: {len(results)}\n")
        f.write(f"- **预测长度**: 96, 192, 336, 720\n")
        f.write(f"- **任务类型**: 多变量预测 (Multivariate)\n\n")
        
        # 结果对比表
        f.write("## 3. 复现结果与论文对比\n\n")
        f.write("| Dataset | Pred Len | MSE (Ours) | MSE (Paper) | Diff | MAE (Ours) | MAE (Paper) | Diff |\n")
        f.write("|---------|----------|-----------|-------------|------|-----------|-------------|------|\n")
        
        for row in table_data:
            mse_ours = f"{row['mse_ours']:.4f}"
            mae_ours = f"{row['mae_ours']:.4f}"
            
            if row['mse_paper']:
                mse_paper = f"{row['mse_paper']:.4f}"
                mse_diff = f"{row['mse_diff_pct']:+.1f}%"
            else:
                mse_paper = "-"
                mse_diff = "-"
                
            if row['mae_paper']:
                mae_paper = f"{row['mae_paper']:.4f}"
                mae_diff = f"{row['mae_diff_pct']:+.1f}%"
            else:
                mae_paper = "-"
                mae_diff = "-"
            
            f.write(f"| {row['dataset']} | {row['pred_len']} | {mse_ours} | {mse_paper} | {mse_diff} | {mae_ours} | {mae_paper} | {mae_diff} |\n")
        
        # 统计分析
        f.write("\n## 4. 统计分析\n\n")
        
        mse_diffs = [row['mse_diff_pct'] for row in table_data if row['mse_diff_pct'] is not None]
        mae_diffs = [row['mae_diff_pct'] for row in table_data if row['mae_diff_pct'] is not None]
        
        if mse_diffs:
            f.write("### MSE 偏差统计\n\n")
            f.write(f"- 平均偏差: {np.mean(mse_diffs):+.2f}%\n")
            f.write(f"- 最大偏差: {np.max(mse_diffs):+.2f}%\n")
            f.write(f"- 最小偏差: {np.min(mse_diffs):+.2f}%\n")
            f.write(f"- 标准差: {np.std(mse_diffs):.2f}%\n")
            
            within_5 = sum(1 for d in mse_diffs if abs(d) <= 5)
            f.write(f"- 偏差在 +/-5% 以内的比例: {within_5}/{len(mse_diffs)} ({within_5/len(mse_diffs)*100:.0f}%)\n\n")
        
        if mae_diffs:
            f.write("### MAE 偏差统计\n\n")
            f.write(f"- 平均偏差: {np.mean(mae_diffs):+.2f}%\n")
            f.write(f"- 最大偏差: {np.max(mae_diffs):+.2f}%\n")
            f.write(f"- 最小偏差: {np.min(mae_diffs):+.2f}%\n")
            f.write(f"- 标准差: {np.std(mae_diffs):.2f}%\n")
            
            within_5 = sum(1 for d in mae_diffs if abs(d) <= 5)
            f.write(f"- 偏差在 +/-5% 以内的比例: {within_5}/{len(mae_diffs)} ({within_5/len(mae_diffs)*100:.0f}%)\n\n")
        
        # 复现结论
        f.write("## 5. 复现结论\n\n")
        
        if mse_diffs:
            avg_mse_diff = abs(np.mean(mse_diffs))
            if avg_mse_diff <= 2:
                conclusion = "复现非常成功，结果与论文高度一致。"
            elif avg_mse_diff <= 5:
                conclusion = "复现成功，结果与论文基本一致，偏差在合理范围内。"
            elif avg_mse_diff <= 10:
                conclusion = "复现基本成功，存在一定偏差，可能由硬件/随机种子/库版本差异导致。"
            else:
                conclusion = "复现存在较大偏差，建议检查超参数配置和代码实现。"
            f.write(f"**{conclusion}**\n\n")
        
        f.write("偏差可能来源:\n")
        f.write("- 随机种子与硬件差异（CPU vs GPU 浮点精度）\n")
        f.write("- PyTorch/NumPy 版本差异\n")
        f.write("- 数据加载顺序差异\n\n")
        
        # 可视化说明
        f.write("## 6. 可视化\n\n")
        f.write("- `mse_comparison.png`: MSE 对比柱状图\n")
        f.write("- `mae_comparison.png`: MAE 对比柱状图\n")
        f.write("- `pred_len_trend.png`: 预测长度 vs 指标趋势图\n\n")
        
        # 各数据集详细分析
        f.write("## 7. 各数据集分析\n\n")
        for dataset in dataset_list:
            f.write(f"### {dataset}\n\n")
            ds_results = [r for r in results if r['dataset'] == dataset]
            ds_results.sort(key=lambda x: x['pred_len'])
            
            f.write(f"| pred_len | MSE | MAE | RSE |\n")
            f.write(f"|----------|-----|-----|-----|\n")
            for r in ds_results:
                rse_str = f"{r['rse']:.4f}" if r['rse'] else "-"
                f.write(f"| {r['pred_len']} | {r['mse']:.4f} | {r['mae']:.4f} | {rse_str} |\n")
            f.write("\n")
            
            # 分析趋势
            if len(ds_results) > 1:
                mse_trend = ds_results[-1]['mse'] - ds_results[0]['mse']
                f.write(f"- MSE 随预测长度增加而{'上升' if mse_trend > 0 else '下降'}，"
                        f"从 {ds_results[0]['mse']:.4f}（pred_len={ds_results[0]['pred_len']}）"
                        f"到 {ds_results[-1]['mse']:.4f}（pred_len={ds_results[-1]['pred_len']}）\n")
                f.write(f"- 增幅: {mse_trend/ds_results[0]['mse']*100:.1f}%\n\n")
    
    print(f"  [OK] Saved: {report_path}")


def save_csv(table_data, csv_path):
    """保存对比表格为 CSV"""
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Dataset,Pred_Len,MSE_Ours,MSE_Paper,MSE_Diff%,MAE_Ours,MAE_Paper,MAE_Diff%\n")
        for row in table_data:
            mse_paper = f"{row['mse_paper']:.4f}" if row['mse_paper'] else ""
            mae_paper = f"{row['mae_paper']:.4f}" if row['mae_paper'] else ""
            mse_diff = f"{row['mse_diff_pct']:.2f}" if row['mse_diff_pct'] is not None else ""
            mae_diff = f"{row['mae_diff_pct']:.2f}" if row['mae_diff_pct'] is not None else ""
            f.write(f"{row['dataset']},{row['pred_len']},{row['mse_ours']:.4f},{mse_paper},"
                    f"{mse_diff},{row['mae_ours']:.4f},{mae_paper},{mae_diff}\n")
    print(f"  [OK] Saved: {csv_path}")


def main():
    print("="*60)
    print("  PatchTST 实验结果分析与报告生成")
    print("="*60)
    
    # 创建报告目录
    report_dir = './report'
    os.makedirs(report_dir, exist_ok=True)
    
    # 解析结果
    print("\n[1/5] 解析实验结果...")
    results = parse_result_file('./result.txt')
    
    if not results:
        print("\n[ERROR] 没有找到有效的实验结果!")
        print("请确保 ./result.txt 文件存在且包含实验结果。")
        print("如果结果在服务器上，请将 result.txt 拷贝到当前目录。")
        return
    
    print(f"  找到 {len(results)} 条实验结果")
    for r in results:
        print(f"    - {r['dataset']} pred_len={r['pred_len']}: MSE={r['mse']:.4f}, MAE={r['mae']:.4f}")
    
    # 生成对比表
    print("\n[2/5] 生成对比表格...")
    table_data = generate_comparison_table(results)
    save_csv(table_data, os.path.join(report_dir, 'comparison_table.csv'))
    
    # 绘制可视化
    print("\n[3/5] 生成可视化图表...")
    plot_mse_comparison(table_data, os.path.join(report_dir, 'mse_comparison.png'))
    plot_mae_comparison(table_data, os.path.join(report_dir, 'mae_comparison.png'))
    plot_pred_len_trend(results, os.path.join(report_dir, 'pred_len_trend.png'))
    
    # 生成报告
    print("\n[4/5] 生成分析报告...")
    generate_markdown_report(results, table_data, os.path.join(report_dir, 'analysis_report.md'))
    
    # 打印摘要
    print("\n[5/5] 结果摘要:")
    print("-"*60)
    
    mse_diffs = [row['mse_diff_pct'] for row in table_data if row['mse_diff_pct'] is not None]
    if mse_diffs:
        print(f"  MSE 平均偏差: {np.mean(mse_diffs):+.2f}% (论文为基准)")
        print(f"  MAE 平均偏差: {np.mean([r['mae_diff_pct'] for r in table_data if r['mae_diff_pct'] is not None]):+.2f}%")
        
        avg_diff = abs(np.mean(mse_diffs))
        if avg_diff <= 5:
            print(f"\n  ** 复现成功! 偏差在合理范围内 (avg {avg_diff:.1f}%) **")
        else:
            print(f"\n  ** 存在一定偏差 (avg {avg_diff:.1f}%)，请检查配置 **")
    
    print(f"\n  报告已保存到: {os.path.abspath(report_dir)}/")
    print("="*60)


if __name__ == '__main__':
    main()
