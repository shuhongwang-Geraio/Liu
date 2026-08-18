"""
方案 1: 训练前适用性判据 — 统计量 vs P0-1 实测增益对应分析 (2026-08-18)
========================================================================

把 `compute_pre_train_stats.py` 的原始数据统计量与 P0-1 主表
(results_shard*.csv) 中 full_v2_fixed vs patchtst 的 8-seed 配对增益对应,
检验: 哪些训练前统计量能预测"修复版门控是否有效"。

用法: python correspond_analysis.py [--stats_dir .] [--results_dir output_large_v3]
"""

import argparse
import csv
import glob
import json
import os
import statistics

SEEDS = ['42', '123', '2024', '7', '13', '99', '2023', '31']


def load_stats(stats_dir):
    stats = {}
    for fp in glob.glob(os.path.join(stats_dir, '_stats_*.json')):
        d = json.load(open(fp, encoding='utf-8'))
        stats[d['dataset'].lower()] = d
    return stats


def load_gains(results_dir):
    """8-seed 配对增益: (patchtst_mse - fixed_mse)/patchtst_mse 均值与符号检验。"""
    res = {}
    for fp in glob.glob(os.path.join(results_dir, 'results_shard*.csv')):
        with open(fp, encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                key = (row['dataset'], row['pred_len'], row['variant'])
                res.setdefault(key, {})[row['seed']] = float(row['mse'])
    gains = {}
    for ds_pl_v, seeds in res.items():
        ds, pl, v = ds_pl_v
        if v != 'full_v2_fixed':
            continue
        base_key = (ds, pl, 'patchtst')
        if base_key not in res:
            continue
        common = [s for s in SEEDS if s in seeds and s in res[base_key]]
        if len(common) < 7:
            continue
        per = [(res[base_key][s] - seeds[s]) / res[base_key][s] * 100 for s in common]
        gains.setdefault((ds, pl), {})['gain_pct'] = statistics.mean(per)
        gains[(ds, pl)]['n'] = len(common)
        gains[(ds, pl)]['gain_min'] = min(per)
        gains[(ds, pl)]['gain_max'] = max(per)
    return gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats_dir', default='.')
    ap.add_argument('--results_dir', default='output_large_v3')
    args = ap.parse_args()

    stats = load_stats(args.stats_dir)
    gains = load_gains(args.results_dir)

    lines = []
    lines.append('=== 训练前统计量 vs P0-1 增益 (full_v2_fixed vs patchtst, 8-seed配对) ===')
    lines.append(f"{'dataset':10s} {'pl':>4s} {'gain%':>8s} {'n':>3s} | "
                 f"{'depDensity':>10s} {'envRatio(season)':>15s} {'stableFrac(season)':>19s} {'lag1':>7s}")
    rows = []
    for (ds, pl), g in sorted(gains.items()):
        s = stats.get(ds, {})
        rows.append((ds, pl, g, s))
        dep = s.get('dep_density_avg_abs_corr', float('nan'))
        er = s.get('env_ratio_season_over_rand', float('nan'))
        sf = s.get('stable_pair_frac_season', float('nan'))
        lag = s.get('lag1_autocorr', float('nan'))
        lines.append(f"{ds:10s} {pl:>4s} {g['gain_pct']:>+7.2f} {g['n']:>3d} | "
                     f"{dep:10.4f} {er:15.1f} {sf:19.3f} {lag:7.4f}")

    # 与各统计量的符号一致率 (增益>0 时统计量高/低)
    lines.append('\n=== 单因子符号一致率 (按 horizon 组) ===')
    for stat_key, stat_name, direction in [
        ('dep_density_avg_abs_corr', '依赖密度', 'high'),
        ('env_ratio_season_over_rand', '语义信息量(season)', 'high'),
        ('stable_pair_frac_season', '稳定通道占比(season)', 'high'),
    ]:
        agree, total = 0, 0
        for ds, pl, g, s in rows:
            if stat_key not in s or len(gains) == 0:
                continue
            total += 1
            v = s[stat_key]
            med = statistics.median([r[3][stat_key] for r in rows if stat_key in r[3]])
            pred_pos = v > med
            if (g['gain_pct'] > 0) == pred_pos:
                agree += 1
        lines.append(f"  {stat_name:16s} {agree}/{total} 组一致")

    out = '\n'.join(lines)
    print(out)
    with open('correspond_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print('\n[report saved] correspond_analysis_report.txt')


if __name__ == '__main__':
    main()
