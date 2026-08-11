"""
生成论文用图: 各数据集 × horizon 提升率 bootstrap CI 误差棒图 (full_v2 vs PatchTST)。

数据源:
    output_large_v2/results_shard*.csv  (6 数据集 × 6 变体 × 8 seed = 720 结果, 2026-08-06)
    (也可用 --results_dir 指向其它 run_large.py 的产物目录)

统计口径 (与 run_large.py summarize 一致):
    - 提升率% (seed 配对) = (patchtst_mse - variant_mse) / patchtst_mse * 100
    - CI: seed 级 bootstrap (以 seed 为采样单元, 保持配对结构), 2000 次, 2.5%/97.5% 分位
    - 显著性: seed 配对 Wilcoxon 符号秩检验(双侧) + 同 (dataset, pred_len) 族内 Holm 校正

产出:
    {output}/improvement_bootstrap_ci.png   误差棒图
    {output}/improvement_bootstrap_ci.md    每个 (dataset, pred_len) 的数值表

用法:
    python plot_bootstrap_ci.py --results_dir ./output_large_v2 --output ./output_large_v2
    python plot_bootstrap_ci.py --results_dir ./output_falsifiable --output ./output_falsifiable
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLUMNS = ['dataset', 'pred_len', 'variant', 'seed',
           'mse', 'mae', 'rmse', 'params', 'epochs', 'time']


def wilcoxon_paired(a, b):
    """seed 配对双侧 Wilcoxon; 返回 (p, n_pair), 配对不足时 p=nan。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) < 5:
        return float('nan'), int(min(len(a), len(b)))
    d = a - b
    d = d[d != 0]
    if len(d) < 5:
        return float('nan'), int(len(d))
    stat, p = stats.wilcoxon(d, alternative='two-sided')
    return float(p), int(len(d))


def holm_adjust(ps):
    """Holm-Bonferroni 校正 (升序 p -> 校正 p, 返回与原序同长的数组)。"""
    ps = np.asarray([p if not np.isnan(p) else 1.0 for p in ps], float)
    order = np.argsort(ps)
    ranked = np.argsort(order)
    m = len(ps)
    adj = np.zeros_like(ps)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, ps[idx] * (m - i))
    # Holm 单调性: 相邻校正值取 max (保持递增)
    for i in range(len(ps) - 1):
        if adj[order[i]] > adj[order[i + 1]]:
            adj[order[i + 1]] = adj[order[i]]
    return adj


def bootstrap_ci(improvements, n_boot=2000, seed=0, alpha=0.05):
    """以单条记录为单位重采样(记录=seed×变体配对后的提升率), 返回 (lo, hi, mean)。"""
    imp = np.asarray(improvements, float)
    rng = np.random.RandomState(seed)
    n = len(imp)
    if n < 2:
        return np.nan, np.nan, float(imp.mean()) if n else np.nan
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[i] = imp[idx].mean()
    return (np.percentile(boot, 100 * alpha / 2),
            np.percentile(boot, 100 * (1 - alpha / 2)),
            float(imp.mean()))


def load_results(results_dir):
    csvs = sorted(glob.glob(os.path.join(results_dir, 'results_shard*.csv')))
    if not csvs:
        sys.exit(f"未找到 results_shard*.csv in {results_dir}")
    frames = [pd.read_csv(c, header=0, names=COLUMNS) for c in csvs]
    df = pd.concat(frames, ignore_index=True)
    df['pred_len'] = df['pred_len'].astype(int)
    df['seed'] = df['seed'].astype(int)
    return df


def compute_stats(df, variant, base_variant='patchtst', n_boot=2000):
    """对每个 (dataset, pred_len): 配对提升率 + CI + (wilcoxon, holm) 族内校正。"""
    rows = []
    # 每个 dataset×pred_len 族内的全部非 base 变体 (用于 Holm 校正, 口径同 run_large)
    groups = df.groupby(['dataset', 'pred_len'])
    for (ds, pl), g in groups:
        base = g[g.variant == base_variant].set_index('seed')['mse']
        var = g[g.variant == variant].set_index('seed')['mse']
        if var.empty or base.empty:
            continue
        # 组内全部候选变体 -> Holm 校正族
        family_ps = []
        for cand in g.variant.unique():
            if cand == base_variant:
                continue
            c = g[g.variant == cand].set_index('seed')['mse']
            common = base.index.intersection(c.index)
            if len(common) >= 5:
                p, _ = wilcoxon_paired(c.loc[common].values, base.loc[common].values)
                family_ps.append((cand, p, c.loc[common].mean(), base.loc[common].mean()))
        if not family_ps:
            continue
        cands = [x[0] for x in family_ps]
        adj = holm_adjust([x[1] for x in family_ps])
        p_holm = dict(zip(cands, adj)).get(variant, np.nan)

        common = base.index.intersection(var.index)
        if len(common) < 5:
            continue
        vm = var.loc[common].values
        bm = base.loc[common].values
        imp = (bm - vm) / bm * 100.0
        lo, hi, mean = bootstrap_ci(imp, n_boot=n_boot)
        p, n_pair = wilcoxon_paired(vm, bm)
        rows.append(dict(dataset=ds, pred_len=pl, mean=mean, ci_lo=lo, ci_hi=hi,
                         p=p, p_holm=p_holm, n_pair=n_pair))
    return pd.DataFrame(rows)


def plot_ci(rows, variant, out_png):
    ds_order = ['traffic', 'electricity', 'ettm1', 'exchange', 'weather', 'etth1', 'ili']
    present = [d for d in ds_order if d in set(rows['dataset'])]
    present += [d for d in sorted(set(rows['dataset']) - set(ds_order))]
    n_ds = len(present)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.2 * max(n_ds, 1), 4.6), squeeze=False)
    palette = plt.cm.tab10
    for ax, ds in zip(axes[0], present):
        sub = rows[rows.dataset == ds].sort_values('pred_len')
        x = np.arange(len(sub))
        colors = [palette(i) for i in range(len(sub))]
        ax.bar(x, sub['mean'], yerr=[
            np.clip(sub['mean'] - sub['ci_lo'], 0, None),
            np.clip(sub['ci_hi'] - sub['mean'], 0, None)],
            capsize=4, color=colors, edgecolor='black', linewidth=0.5)
        for xi, r in zip(x, sub.itertuples()):
            sig = '†' if (not np.isnan(r.p_holm) and r.p_holm < 0.05) else ''
            ax.text(xi, r.mean + (0.3 if r.mean >= 0 else -1.0),
                    f"{r.mean:+.1f}%{sig}", ha='center', va='bottom', fontsize=8)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f"{ds}\n({n_vars_of(ds)} channels)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"pl{int(p)}" for p in sub['pred_len']], fontsize=8)
        ax.set_ylim(bottom=-8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('% improvement vs PatchTST' if ds == present[0] else '')
    fig.suptitle(f"{variant} vs PatchTST — improvement with bootstrap 95% CI (†: Holm p<0.05)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    print(f"图已保存: {out_png}")
    plt.close(fig)


def n_vars_of(ds):
    # 论文报告中已明确的通道数 (仅用于图标题注释)
    return {'traffic': 862, 'electricity': 321, 'ettm1': 7, 'exchange': 8,
            'weather': 21, 'etth1': 7, 'ili': 7}.get(ds, '?')


def write_md(rows, variant, out_md):
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(f"# {variant} vs PatchTST 提升率 (seed 配对 bootstrap 95% CI)\n\n")
        f.write("> 口径: 提升%=(patchtst_mse-variant_mse)/patchtst_mse×100; CI 为 seed 级 bootstrap(2000次); "
                "†=组内 Holm p<0.05; n_pair=有效配对 seed 数\n\n")
        f.write("| 数据集 | pred_len | 提升% mean | CI lo | CI hi | Wilcoxon p | Holm p | n_pair | 显著 |\n")
        f.write("|--------|----------|-----------|-------|-------|------------|--------|--------|------|\n")
        for r in rows.sort_values(['dataset', 'pred_len']).itertuples():
            sig = '†' if (not np.isnan(r.p_holm) and r.p_holm < 0.05) else ''
            f.write(f"| {r.dataset} | {int(r.pred_len)} | {r.mean:+.2f}% | {r.ci_lo:+.2f}% | "
                    f"{r.ci_hi:+.2f}% | {r.p:.4f} | {r.p_holm:.4f} | {r.n_pair} | {sig} |\n")
    print(f"表格已保存: {out_md}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='./output_large_v2')
    p.add_argument('--output', default=None)
    p.add_argument('--variant', default='full_v2')
    p.add_argument('--base_variant', default='patchtst')
    p.add_argument('--n_boot', type=int, default=2000)
    args = p.parse_args()
    out_dir = args.output or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    df = load_results(args.results_dir)
    print(f"已加载 {len(df)} 条结果 (数据集: {sorted(df.dataset.unique())})")
    rows = compute_stats(df, args.variant, args.base_variant, n_boot=args.n_boot)
    if rows.empty:
        sys.exit(f"变体 {args.variant} / 基线 {args.base_variant} 无有效配对数据")
    rows['pred_len'] = rows['pred_len'].astype(int)

    plot_ci(rows, args.variant, os.path.join(out_dir, 'improvement_bootstrap_ci.png'))
    write_md(rows, args.variant, os.path.join(out_dir, 'improvement_bootstrap_ci.md'))


if __name__ == '__main__':
    main()
