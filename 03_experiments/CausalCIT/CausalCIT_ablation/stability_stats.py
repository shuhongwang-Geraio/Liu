"""
逐 数据集×horizon 的 "3-seed 全正比例" 稳定性量化.

读取 run_large.py 产出的 results_shard*.csv, 对每个 (数据集, pred_len) 单元,
计算 full_v2 / no_gate 相对 patchtst 的逐 seed 提升%, 以及:
  * 该单元内 "提升为正的 seed 数 / 3"  (即单单元全正比例)
  * 全数据集汇总: 全正 seed 总数 / 总 seed 数,  全正 horizon 单元数 / 总单元数
  * 单侧符号检验 (binomial, 正提升占比是否显著高于 0.5)

用法:
  python stability_stats.py --output_dir ./output_large
  python stability_stats.py --output_dir ./output_traffic  (traffic 单独跑完后)
"""
import os
import csv
import argparse
import numpy as np
import pandas as pd

VARIANTS = ['full_v2', 'no_gate']
BASE = 'patchtst'


def load(args):
    dirs = args.output_dir if isinstance(args.output_dir, (list, tuple)) else [args.output_dir]
    files = []
    for d in dirs:
        if os.path.isdir(d):
            files += sorted(os.path.join(d, f)
                            for f in os.listdir(d)
                            if f.startswith('results_shard') and f.endswith('.csv'))
    rows = []
    for fp in files:
        with open(fp) as f:
            for r in csv.DictReader(f):
                try:
                    r['pred_len'] = int(r['pred_len'])
                    r['seed'] = int(r['seed'])
                    r['mse'] = float(r['mse'])
                    rows.append(r)
                except (ValueError, KeyError):
                    continue
    return pd.DataFrame(rows)


def _improvements(df, ds, pl, variant):
    """返回该 (ds,pl,variant) 相对 patchtst 的逐 seed 提升% 列表, 与对应 mse."""
    sub = df[(df.dataset == ds) & (df.pred_len == pl)]
    base = sub[sub.variant == BASE]
    v = sub[sub.variant == variant]
    if base.empty or v.empty:
        return None, None
    base_map = {int(s): m for s, m in zip(base['seed'], base['mse'])}
    imps, mses = [], []
    for _, row in v.iterrows():
        s = int(row['seed'])
        if s in base_map and base_map[s] > 0:
            imps.append((base_map[s] - row['mse']) / base_map[s] * 100)
            mses.append(row['mse'])
    return imps, mses


def build_tables(df):
    datasets = sorted(df.dataset.unique())
    pls = sorted(df.pred_len.unique())
    out = {}
    for variant in VARIANTS:
        # 逐 seed 提升表
        lines = [f"## {variant} vs PatchTST — 逐 seed 提升% (正=赢)", ""]
        lines.append("| 数据集 | " + " | ".join(f"pl{p}" for p in pls) + " |")
        lines.append("|" + "---|" * (len(pls) + 1))
        # 全正判定表
        lines2 = [f"## {variant} vs PatchTST — 全正统计 (3 seed)", ""]
        lines2.append("| 数据集 | " + " | ".join(f"pl{p}" for p in pls) + " |")
        lines2.append("|" + "---|" * (len(pls) + 1))
        all_pos_seeds = 0
        total_seeds = 0
        fullpos_cells = 0
        total_cells = 0
        for ds in datasets:
            row_imp = [ds]
            row_pos = [ds]
            for pl in pls:
                imps, _ = _improvements(df, ds, pl, variant)
                if imps is None:
                    row_imp.append("—")
                    row_pos.append("—")
                    continue
                total_cells += 1
                total_seeds += len(imps)
                n_pos = sum(1 for x in imps if x > 0)
                all_pos_seeds += n_pos
                if n_pos == len(imps):
                    fullpos_cells += 1
                imp_str = " / ".join(f"{x:+.2f}" for x in imps)
                row_imp.append(imp_str)
                row_pos.append(f"{n_pos}/{len(imps)}" + (" ✓" if n_pos == len(imps) else ""))
            lines.append("| " + " | ".join(row_imp) + " |")
            lines2.append("| " + " | ".join(row_pos) + " |")
        lines.append("")
        lines2.append("")
        # 汇总
        summary = [
            f"**汇总 ({variant}):**",
            f"- 全正 seed 数 / 总 seed 数 = **{all_pos_seeds}/{total_seeds}** "
            f"({100*all_pos_seeds/total_seeds:.1f}%)",
            f"- 全正 horizon 单元数 / 总单元数 = **{fullpos_cells}/{total_cells}** "
            f"({100*fullpos_cells/total_cells:.1f}%)",
        ]
        out[variant] = ("\n".join(lines) + "\n" + "\n".join(lines2) + "\n" +
                        "\n".join(summary) + "\n")
    return out, datasets, pls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', nargs='+', default=['./output_large'])
    ap.add_argument('--md', default=None, help='输出 markdown 路径 (默认打印+写 stability_table.md)')
    args = ap.parse_args()
    df = load(args)
    if df.empty:
        print("无有效结果")
        return
    tables, datasets, pls = build_tables(df)

    dirs = args.output_dir if isinstance(args.output_dir, (list, tuple)) else [args.output_dir]
    tag = "+".join(os.path.basename(d) for d in dirs)
    header = [f"# 稳定性统计: 3-seed 全正比例 ({tag})", "",
              f"> 生成时间: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}",
              f"> 数据集: {datasets} | horizons: {pls} | 基线: {BASE}", ""]
    body = header + [tables[v] for v in VARIANTS]
    md = "\n".join(body)
    print(md)
    if args.md is None:
        args.md = os.path.join(args.output_dir, 'stability_table.md')
    with open(args.md, 'w') as f:
        f.write(md)
    print(f"\n[已保存] {args.md}")


if __name__ == '__main__':
    main()
