"""
因果 / 虚假 / 独立边门控权重箱线图 (P1 可视化升级, 论文素材)。

输入: 门控矩阵 dump 目录 (glob *.npy, 每个文件为 [*, n, n] 或 [n, n],
      命名 gate_{ds}_pl{pl}_{variant}_s{seed}.npy, 由 run_large._train_one
      / _train_syn_ood 自动 dump)。
边定义 (syn_ood 真值, 与 analyze_gates.py 一致):
  - 因果边 (跨环境稳定): Ch1<-Ch0, Ch2<-Ch0
  - 虚假边 (跨环境漂移): Ch3/Ch4 相关的 confound 边
  - 独立边: 与独立通道 Ch5/Ch6 相关的其余非对角边
输出: {output}/gate_edge_boxplot.png + {output}/gate_edge_boxplot.md

用法:
  python plot_gate_edge_boxplot.py --gates_dir ./output_pipeline_smoke/gates --output ./vis_output
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SYN_N = 7
TRUE_EDGES = {(1, 0), (2, 0)}
SPURIOUS_EDGES = {(3, 0), (0, 3), (4, 0), (0, 4), (1, 3), (2, 3), (1, 4), (2, 4)}


def load_matrix(path):
    arr = np.load(path)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.mean(axis=0)
    raise ValueError(f"{path}: 维度 {arr.ndim} 不支持")


def edge_vectors(mat):
    n = mat.shape[0]
    true_v = [mat[i, j] for (i, j) in TRUE_EDGES]
    spur_v = [mat[i, j] for (i, j) in SPURIOUS_EDGES]
    indep_v = [mat[i, j] for i in range(n) for j in range(n)
               if i != j and (i, j) not in TRUE_EDGES and (i, j) not in SPURIOUS_EDGES
               and (i in (5, 6) or j in (5, 6))]
    return {'causal': true_v, 'spurious': spur_v, 'independent': indep_v}


def collect(gates_dir):
    """返回 {variant: {'causal': [...], 'spurious': [...], 'independent': [...]}} (跨 seed/pl 聚合)。"""
    out = {}
    for fp in sorted(glob.glob(os.path.join(gates_dir, '*.npy'))):
        name = os.path.splitext(os.path.basename(fp))[0]
        # gate_{ds}_pl{pl}_{variant}_s{seed}
        try:
            _, ds, pl_part, variant = name.split('_', 3)
            variant = variant.rsplit('_s', 1)[0]
        except ValueError:
            variant = name
        mat = load_matrix(fp)
        vecs = edge_vectors(mat)
        d = out.setdefault(variant, {'causal': [], 'spurious': [], 'independent': []})
        for k, v in vecs.items():
            d[k].extend(v)
    return out


def plot(data, out_png, out_md):
    variants = sorted(data.keys())
    kinds = ['causal', 'spurious', 'independent']
    colors = {'causal': '#2ecc71', 'spurious': '#e74c3c', 'independent': '#95a5a6'}

    fig, ax = plt.subplots(figsize=(1.8 * len(variants) + 2, 5))
    positions, labels, x = [], [], 0
    for vi, v in enumerate(variants):
        for ki, k in enumerate(kinds):
            vals = data[v][k]
            if not vals:
                continue
            bp = ax.boxplot(vals, positions=[x], widths=0.55, patch_artist=True,
                            showfliers=False)
            bp['boxes'][0].set_facecolor(colors[k])
            bp['boxes'][0].set_alpha(0.75)
            labels.append(f"{v}\n{k}")
            x += 1
        x += 0.6
    ax.set_xticks(range(0, len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('gate weight (off-diagonal)')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_title('Gate weight by edge type (causal vs spurious vs independent)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# 门控边权重汇总 (causal / spurious / independent)\n\n")
        f.write("| 变体 | 边类型 | 均值 | 中位数 | std | 条数 |\n")
        f.write("|------|--------|------|--------|-----|------|\n")
        for v in variants:
            for k in kinds:
                vals = np.array(data[v][k])
                if len(vals) == 0:
                    continue
                f.write(f"| {v} | {k} | {vals.mean():.4f} | {np.median(vals):.4f} "
                        f"| {vals.std():.4f} | {len(vals)} |\n")
        # 分离度
        f.write("\n## 分离度 (causal_mean - spurious_mean)\n\n")
        for v in variants:
            c = np.mean(data[v]['causal']) if data[v]['causal'] else np.nan
            s = np.mean(data[v]['spurious']) if data[v]['spurious'] else np.nan
            f.write(f"- {v}: {c - s:+.4f}\n")
    print(f"箱线图已保存: {out_png}")
    print(f"汇总表已保存: {out_md}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gates_dir', required=True)
    p.add_argument('--output', default='./vis_output')
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)
    data = collect(args.gates_dir)
    if not data:
        print(f"警告: {args.gates_dir} 下无 *.npy 门控矩阵。\n"
              f"  syn_ood 的门控矩阵由 run_large 在训练时自动 dump (n_vars<=21);\n"
              f"  traffic/electricity 高维矩阵获取方法见 plot_visualization_README.md")
        return
    plot(data, os.path.join(args.output, 'gate_edge_boxplot.png'),
         os.path.join(args.output, 'gate_edge_boxplot.md'))


if __name__ == '__main__':
    main()
