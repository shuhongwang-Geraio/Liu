"""修 A+B (median带宽+CKA归一化) 在超高维的边界: full_v2 vs full_v2_fixed 8-seed 对比"""
import csv, glob, os, statistics

SEEDS = ['42', '123', '2024', '7', '13', '99', '2023', '31']
base = r'output_large_v3'

res = {}
for fp in glob.glob(os.path.join(base, 'results_shard*.csv')):
    with open(fp, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            res.setdefault((row['dataset'], row['pred_len'], row['variant']), {})[row['seed']] = float(row['mse'])

lines = []
lines.append('=== full_v2 vs full_v2_fixed (8-seed 配对, 修复版相对旧版 MSE 变化%) ===')
for key in sorted(res):
    ds, pl, v = key
    if v not in ('full_v2', 'full_v2_fixed'):
        continue
lines = [lines[0]]
groups = {}
for (ds, pl, v), seeds in res.items():
    if v in ('full_v2', 'full_v2_fixed'):
        groups.setdefault((ds, pl), {})[v] = seeds
for (ds, pl), vs in sorted(groups.items()):
    if 'full_v2' not in vs or 'full_v2_fixed' not in vs:
        continue
    common = [s for s in SEEDS if s in vs['full_v2'] and s in vs['full_v2_fixed']]
    if len(common) < 7:
        continue
    # 修复版相对旧版: 负值 = 修复版更好
    per = [(vs['full_v2_fixed'][s] - vs['full_v2'][s]) / vs['full_v2'][s] * 100 for s in common]
    m = statistics.mean(per)
    # 配对符号检验 (粗略)
    pos = sum(1 for p in per if p < 0)
    lines.append(f"{ds:12s} pl{pl:4s} 修复版相对旧版 {m:+.2f}% (fix<old 的 seed: {pos}/{len(common)})")
print('\n'.join(lines))
open('_fixAB_boundary.txt', 'w', encoding='utf-8').write('\n'.join(lines))
