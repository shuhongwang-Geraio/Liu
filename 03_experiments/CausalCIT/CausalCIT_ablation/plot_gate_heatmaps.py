"""
门控矩阵可视化 (P1 可视化升级):
  (1) 门控矩阵聚类热图: 输入 gates/*.npy (每个文件为 [*, n, n] 或 [n, n] 的门控矩阵),
      低维(n<=21)直接画; 高维(n>21, 如 traffic 862)自动通道子采样 + 行均值聚合。
  (2) 门控行为诊断图: 读 gate_diagnostics.json (off_std / collapsed / batch_dep_score),
      跨变体分组条形图 —— 回应评审 re2 §2.2 的 batch 依赖 bug 的直接可视化。

当前数据状态 (2026-08-10):
  - 已有: output/gate_matrices/*.npy  (早期 ablation, 320×7×7, 低维 etth1/syn)
          output_falsifiable_full/gate_diagnostics.json  (traffic 全规模, 80 条)
  - 缺失: traffic(862ch)/electricity(321ch) 的 full_v2/full_v2_fixed 门控矩阵 dump
          (run_large.py 仅在 n_vars<=21 时保存 gates/, 见 run_large.py _train_one)
  → 高维热图部分当前只能跑低维样例; 跑高维矩阵的步骤见 plot_visualization_README.md

用法:
  python plot_gate_heatmaps.py --gates_dir ./output/gate_matrices --output ./vis_output
  python plot_gate_heatmaps.py --diagnostics_json ./output_falsifiable_full/gate_diagnostics.json --output ./vis_output
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_matrix(path):
    """加载门控矩阵, 统一为 2D [n, n]; 3D [*, n, n] 取时间维平均。"""
    arr = np.load(path)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.mean(axis=0)
    raise ValueError(f"{path}: 不支持维度 {arr.ndim}")


def plot_heatmap(mat, title, out_png, subsample=50):
    """画门控矩阵热图; 高维(n>subsample)时等距抽通道, 保持视觉可读。"""
    n = mat.shape[0]
    if n > subsample:
        idx = np.linspace(0, n - 1, subsample).astype(int)
        mat_show = mat[np.ix_(idx, idx)]
        note = f" (子采样 {subsample}/{n} 通道)"
    else:
        mat_show = mat
        note = ""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat_show, cmap='viridis', aspect='auto')
    ax.set_title(f"{title}{note}\noff-diag std={np.std(mat[np.eye(n, dtype=bool) == False]):.4f}",
                 fontsize=10)
    ax.set_xlabel('source channel')
    ax.set_ylabel('target channel')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  热图已保存: {out_png}")


def plot_diagnostics(json_path, out_dir):
    """跨变体门控行为图: off_std 均值 + batch_dep_score 均值 (含 collapsed 标注)。"""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    if not data:
        print("gate_diagnostics.json 为空")
        return
    import pandas as pd
    df = pd.DataFrame(data)
    # 每 (pred_len, variant) 跨 seed 聚合
    df['off_std'] = df['off_std'].astype(float)
    df['batch_dep_score_mean'] = df['batch_dep_score_mean'].astype(float)
    for pl, sub in df.groupby('pred_len'):
        agg = sub.groupby('variant').agg(
            off_std=('off_std', 'mean'),
            bd_mean=('batch_dep_score_mean', 'mean'),
            collapsed_frac=('collapsed', 'mean'),
        ).sort_index()
        x = np.arange(len(agg))
        w = 0.36
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.bar(x - w / 2, agg['off_std'], w, label='off_std (larger = more structured)', color='#2ecc71')
        ax1.bar(x + w / 2, agg['bd_mean'], w, label='batch_dep_score (lower is better)', color='#e74c3c')
        for xi, (_, r) in zip(x, agg.iterrows()):
            ax1.text(xi, max(r['off_std'], r['bd_mean']) + 0.01,
                     f"collapse {r['collapsed_frac']:.0%}", ha='center', fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(agg.index, rotation=15, ha='right')
        ax1.set_ylabel('score')
        ax1.set_title(f"Gate diagnostics (pred_len={int(pl)}, mean over seeds)")
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"gate_diagnostics_pl{int(pl)}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  诊断图已保存: {out_png}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gates_dir', default=None, help='门控矩阵 npy dump 目录 (glob *.npy)')
    p.add_argument('--diagnostics_json', default=None, help='gate_diagnostics.json 路径')
    p.add_argument('--output', default='./vis_output')
    p.add_argument('--subsample', type=int, default=50, help='高维通道子采样上限')
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.gates_dir:
        files = sorted(glob.glob(os.path.join(args.gates_dir, '*.npy')))
        if not files:
            print(f"警告: {args.gates_dir} 下没有 *.npy。\n"
                  f"  当前只 dump 了低维(<=21通道)门控矩阵 (见 output/gate_matrices/)。\n"
                  f"  traffic/electricity 高维矩阵的获取步骤见 plot_visualization_README.md。")
        for fp in files:
            try:
                mat = load_matrix(fp)
                name = os.path.splitext(os.path.basename(fp))[0]
                plot_heatmap(mat, name, os.path.join(args.output, f"heatmap_{name}.png"),
                             subsample=args.subsample)
            except Exception as e:
                print(f"  跳过 {fp}: {e}")

    if args.diagnostics_json:
        plot_diagnostics(args.diagnostics_json, args.output)

    if not args.gates_dir and not args.diagnostics_json:
        p.print_help()


if __name__ == '__main__':
    main()
