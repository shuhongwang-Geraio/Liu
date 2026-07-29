"""
CausalCIT 消融实验

验证每个组件的贡献:
  1. Full CausalCIT          — HSIC + 环境划分 + 门控 (完整模型)
  2. w/o HSIC (NoHSIC)       — 用Pearson相关性替代HSIC
  3. w/o EnvSplit (NoEnv)    — 不划分环境，全局计算HSIC
  4. w/o Gate (NoGate)       — 去掉门控，全连接通道注意力
  5. PatchTST                — 纯CI基线 (无通道交互)
  6. Full CausalCIT (fix)    — 降低先验权重(prior_weight=0.1), 诊断"先验主导"假设

用法:
    python run_ablation.py                     # 合成数据上消融
    python run_ablation.py --exp real          # ETTh1上消融
    python run_ablation.py --exp all           # 全部
    python run_ablation.py --device cuda       # GPU
    python run_ablation.py --n_seeds 5         # 多seed聚合 + 配对显著性检验
    python run_ablation.py --seeds 42 123 2024 7 99   # 显式指定seed列表
    python run_ablation.py --n_envs 2 --seq_len 336    # 加大环境划分粒度对照(假设B)
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # CausalCIT/
DEMO_DIR = os.path.join(PROJECT_DIR, 'CausalCIT_demo')

# 数据集目录: 优先 patchtst/dataset/，回退到几个常见位置
_DEFAULT_PATHS = [
    os.path.join(PROJECT_DIR, 'patchtst', 'dataset'),
    os.path.join(PROJECT_DIR, 'data'),
    os.path.join(os.path.dirname(PROJECT_DIR), 'patchtst', 'dataset'),
    # 新目录结构: 外部数据在 01_external/PatchTST/code/dataset/
    os.path.join(os.path.dirname(os.path.dirname(PROJECT_DIR)), '01_external', 'PatchTST', 'code', 'dataset'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(PROJECT_DIR))), '01_external', 'PatchTST', 'code', 'dataset'),
]
DATASET_DIR = _DEFAULT_PATHS[0]
for _dp in _DEFAULT_PATHS:
    if os.path.isdir(_dp) and os.listdir(_dp):
        DATASET_DIR = _dp
        break

sys.path.insert(0, DEMO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.data import SyntheticCausalDataset, ETTDataset, get_dataloader
from utils.trainer import Trainer
from utils.metrics import metric
from models_ablation import create_ablation_model


VARIANTS = [
    ('patchtst',  'PatchTST\n(no interaction)'),
    ('no_gate',   'w/o Gate\n(full attention)'),
    ('no_env',    'w/o EnvSplit\n(global HSIC)'),
    ('no_hsic',   'w/o HSIC\n(Pearson corr)'),
    ('full',      'Full CausalCIT\n(Ours)'),
    ('full_fix',  'Full CausalCIT\n(fix prior)'),
]

COLORS = {
    'patchtst': '#95a5a6',
    'no_gate':  '#f39c12',
    'no_env':   '#9b59b6',
    'no_hsic':  '#3498db',
    'full':     '#e74c3c',
    'full_fix': '#2ecc71',
}


def set_seed(seed):
    """固定随机种子：保证不同变体在相同初始化/相同DataLoader shuffle顺序下对比，
    否则消融实验的差异会被"训练随机性噪声"淹没（历史上曾出现同一变体两次跑出
    +5.51% vs -0.99%这种反直觉波动，根源就是没有固定种子）。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def resolve_seeds(args):
    """解析多seed实验的seed列表。显式 --seeds 优先于 --n_seeds。"""
    if args.seeds:
        return [int(s) for s in args.seeds]
    return list(range(int(args.n_seeds)))


# ============================================================
# 门控矩阵 / 诊断参数 插桩辅助函数
# ============================================================

def _collect_gate_matrices(model, test_loader, device, max_batches=10):
    """在测试集前 max_batches 个 batch 上采样门控矩阵，返回 [N, nvars, nvars] 或 None。"""
    if not hasattr(model, 'get_gate_matrix'):
        return None
    model.eval()
    gms = []
    with torch.no_grad():
        for i, (bx, _) in enumerate(test_loader):
            if i >= max_batches:
                break
            _ = model(bx.to(device))
            gm = model.get_gate_matrix()
            if gm is not None:
                gms.append(gm.detach().cpu().numpy())
    if not gms:
        return None
    return np.concatenate(gms, axis=0)


def _collect_diagnostics(model, variant_key):
    """返回门控相关可学习参数诊断字典，或 None（无门控变体）。"""
    if hasattr(model, 'get_diagnostics'):
        try:
            return model.get_diagnostics()
        except Exception as e:
            print(f"  [warn] get_diagnostics({variant_key}) 失败: {e}")
            return None
    return None


def _aggregate_and_save_gates(gates_all, output_dir):
    """跨seed聚合门控矩阵，np.save 各变体均值矩阵，并生成 full vs no_env 对比文件。"""
    os.makedirs(output_dir, exist_ok=True)
    seeds = list(gates_all.keys())
    variants = gates_all[seeds[0]].keys()
    mean_gate = {}
    for v in variants:
        arrs = [gates_all[s][v] for s in seeds if gates_all[s].get(v) is not None]
        if arrs:
            m = np.mean([a.mean(axis=0) for a in arrs], axis=0)
            mean_gate[v] = m
            np.save(os.path.join(output_dir, f'gate_{v}.npy'), m)
    _save_gate_comparison(mean_gate, output_dir)
    return mean_gate


def _save_gate_comparison(gate_matrices, output_dir):
    """生成 full vs no_env 门控矩阵逐元素差异文件（验证假设A/C：先验主导 / 门控饱和）。"""
    path = os.path.join(output_dir, 'gate_comparison.txt')
    lines = []
    lines.append("=== Gate Matrix Comparison: full vs no_env (跨seed均值) ===")
    lines.append("")
    if 'full' in gate_matrices and 'no_env' in gate_matrices:
        gf = gate_matrices['full']
        gn = gate_matrices['no_env']
        diff = np.abs(gf - gn)
        lines.append(f"mean gate (full)    : {gf.mean():.4f}")
        lines.append(f"mean gate (no_env)  : {gn.mean():.4f}")
        lines.append(f"max  |diff|          : {diff.max():.4f}")
        lines.append(f"mean |diff|          : {diff.mean():.4f}")
        lines.append(f"fraction gate≈1 (full)  : {float((gf > 0.99).mean()):.4f}  (饱和检查, 假设C)")
        lines.append(f"fraction gate≈1 (no_env): {float((gn > 0.99).mean()):.4f}")
        lines.append("")
        lines.append("full gate matrix (mean, row=query channel):")
        lines.append(np.array2string(gf, precision=3))
        lines.append("")
        lines.append("no_env gate matrix (mean, row=query channel):")
        lines.append(np.array2string(gn, precision=3))
        lines.append("")
        if diff.mean() < 1e-3:
            lines.append("⚠ full 与 no_env 门控矩阵几乎逐点重合 -> 两条路径可能学到等价门控（诊断报告异常2）")
    else:
        lines.append("full/no_env gate matrices 不可用，无法对比。")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  门控对比已保存: {path}")


def _aggregate_and_save_diagnostics(diag_all, output_dir):
    """跨seed聚合诊断参数（channel_prior/stability_bias/temperature/alpha），落盘并对比 full vs no_env。"""
    os.makedirs(output_dir, exist_ok=True)
    seeds = list(diag_all.keys())
    variants = diag_all[seeds[0]].keys()
    mean_diag = {}
    for v in variants:
        dicts = [diag_all[s][v] for s in seeds if diag_all[s].get(v) is not None]
        if not dicts:
            continue
        agg = {}
        for k in dicts[0]:
            vals = [d[k] for d in dicts if k in d]
            if vals and all(isinstance(x, (int, float)) for x in vals):
                agg[k] = float(np.mean(vals))
            else:
                agg[k] = dicts[0][k]
        mean_diag[v] = agg

    lines = ["=== Gate Diagnostics (跨seed均值) ===", ""]
    for v in variants:
        if v in mean_diag:
            lines.append(f"[{v}]")
            for k, val in mean_diag[v].items():
                lines.append(f"  {k}: {val}")
            lines.append("")
    path = os.path.join(output_dir, 'gate_diagnostics.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    # 打印 full vs no_env 对比（验证假设A：先验主导）
    if 'full' in mean_diag and 'no_env' in mean_diag:
        print("\n--- 诊断参数对比 full vs no_env (跨seed均值) ---")
        for k in mean_diag['full']:
            if k in mean_diag['no_env']:
                print(f"  {k}: full={mean_diag['full'][k]}  no_env={mean_diag['no_env'][k]}")
    print(f"  诊断参数已保存: {path}")
    return mean_diag


# ============================================================
# 合成数据消融
# ============================================================

def run_synthetic_ablation(args, seed):
    print_header(f"消融实验: 合成数据 (seed={seed}, d_model=64, 50 epochs)")
    device = args.device

    train_set = SyntheticCausalDataset(n_samples=8000, seq_len=args.seq_len,
                                       pred_len=args.pred_len, flag='train')
    val_set = SyntheticCausalDataset(n_samples=2000, seq_len=args.seq_len,
                                     pred_len=args.pred_len, flag='val')
    test_set = SyntheticCausalDataset(n_samples=3000, seq_len=args.seq_len,
                                      pred_len=args.pred_len, flag='test')

    train_loader = get_dataloader(train_set, batch_size=args.batch_size)
    val_loader = get_dataloader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_set, batch_size=args.batch_size, shuffle=False)

    n_vars = train_set.n_vars
    common_kwargs = dict(
        enc_in=n_vars, seq_len=args.seq_len, pred_len=args.pred_len,
        e_layers=3, n_heads=4, d_model=64, d_ff=256,
        dropout=0.2, fc_dropout=0.2,
        patch_len=args.patch_len, stride=args.stride, padding_patch='end',
        n_channel_heads=4, n_envs=args.n_envs, rff_dim=64,
        channel_dropout=0.1, fusion_alpha=0.3,
    )

    results = {}
    gate_matrices = {}
    diagnostics = {}

    for variant_key, variant_label in VARIANTS:
        label_short = variant_label.replace('\n', ' ')
        print(f"\n--- {label_short} ---")
        set_seed(seed)  # 每个变体训练前重置种子，保证初始化/shuffle顺序一致，公平对比
        model = create_ablation_model(variant_key, **common_kwargs)
        params = count_params(model)
        print(f"  参数量: {params:,}")

        trainer = Trainer(model, device=device)
        save_dir = os.path.join(args.output_dir, f'ckpt_syn_{variant_key}_s{seed}')
        hist = trainer.train(train_loader, val_loader, epochs=50,
                             lr=args.lr, patience=10, save_dir=save_dir)
        res = trainer.test(test_loader)
        results[variant_key] = {
            'mse': res['mse'], 'mae': res['mae'], 'rmse': res['rmse'],
            'params': params, 'time': hist['total_time'],
            'preds': res['preds'], 'trues': res['trues'],
        }
        print(f"  MSE: {res['mse']:.6f} | MAE: {res['mae']:.6f} | Time: {hist['total_time']:.0f}s")

        # 提取门控矩阵 + 诊断参数
        gate_matrices[variant_key] = _collect_gate_matrices(model, test_loader, device)
        diagnostics[variant_key] = _collect_diagnostics(model, variant_key)

    _plot_synthetic_ablation(results, gate_matrices, train_set.channel_labels, args, seed)
    return results, gate_matrices, diagnostics


def run_synthetic_multiseed(args):
    """多seed合成数据消融：循环seed，聚合门控矩阵与诊断参数后落盘。"""
    seeds = resolve_seeds(args)
    syn_all = {}
    gates_all = {}
    diag_all = {}
    for seed in seeds:
        r, g, d = run_synthetic_ablation(args, seed)
        syn_all[seed] = r
        gates_all[seed] = g
        diag_all[seed] = d
    _aggregate_and_save_gates(gates_all, args.output_dir)
    _aggregate_and_save_diagnostics(diag_all, args.output_dir)
    return syn_all


def _plot_synthetic_ablation(results, gate_matrices, channel_labels, args, seed=None):
    n_var = len(VARIANTS)
    # 行1三等分、行2二等分的跨列边界，随变体数自适应
    a = max(1, round(n_var / 3))
    b = max(a + 1, round(2 * n_var / 3))
    half = max(1, round(n_var / 2))
    fig = plt.figure(figsize=(4.0 * n_var, 16))
    gs = plt.GridSpec(3, n_var, figure=fig, hspace=0.45, wspace=0.35)

    short_ch = ['Base', 'C1', 'C2', 'S1', 'S2', 'I1', 'I2']

    # Row 0: 门控矩阵对比 (每个变体一个)
    for idx, (variant_key, variant_label) in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[0, idx])
        if variant_key in gate_matrices and gate_matrices[variant_key] is not None:
            gm = gate_matrices[variant_key].mean(axis=0)
            im = ax.imshow(gm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(len(short_ch)))
            ax.set_yticks(range(len(short_ch)))
            ax.set_xticklabels(short_ch, fontsize=7, rotation=45)
            ax.set_yticklabels(short_ch, fontsize=7)
            for i in range(len(short_ch)):
                for j in range(len(short_ch)):
                    c = 'black' if gm[i,j] > 0.4 else 'white'
                    ax.text(j, i, f'{gm[i,j]:.2f}', ha='center', va='center',
                           fontsize=6, color=c)
        else:
            ax.text(0.5, 0.5, 'N/A\n(No Gate)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        label_line1 = variant_label.split('\n')[0]
        ax.set_title(label_line1, fontsize=10, fontweight='bold',
                    color=COLORS[variant_key])

    # Row 1 Left: MSE条形图
    ax_mse = fig.add_subplot(gs[1, 0:a])
    names = [vl.replace('\n', ' ') for _, vl in VARIANTS]
    mse_vals = [results[vk]['mse'] for vk, _ in VARIANTS]
    colors = [COLORS[vk] for vk, _ in VARIANTS]
    bars = ax_mse.barh(range(len(names)), mse_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax_mse.set_yticks(range(len(names)))
    ax_mse.set_yticklabels(names, fontsize=9)
    ax_mse.set_xlabel('MSE (↓ better)')
    ax_mse.set_title('MSE Comparison', fontsize=12, fontweight='bold')
    ax_mse.invert_yaxis()
    for bar, v in zip(bars, mse_vals):
        ax_mse.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

    # Row 1 Middle: MAE条形图
    ax_mae = fig.add_subplot(gs[1, a:b])
    mae_vals = [results[vk]['mae'] for vk, _ in VARIANTS]
    bars = ax_mae.barh(range(len(names)), mae_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax_mae.set_yticks(range(len(names)))
    ax_mae.set_yticklabels(names, fontsize=9)
    ax_mae.set_xlabel('MAE (↓ better)')
    ax_mae.set_title('MAE Comparison', fontsize=12, fontweight='bold')
    ax_mae.invert_yaxis()
    for bar, v in zip(bars, mae_vals):
        ax_mae.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

    # Row 1 Right: 组件贡献分析
    ax_contrib = fig.add_subplot(gs[1, b:n_var])
    base_mse = results['patchtst']['mse']
    full_mse = results['full']['mse']
    total_gain = base_mse - full_mse

    components = {
        'Channel Attn\n(+NoGate)': base_mse - results['no_gate']['mse'],
        '+Gate\n(+HSIC Global)': results['no_gate']['mse'] - results['no_env']['mse'],
        '+EnvSplit\n(w/ Pearson)': results['no_env']['mse'] - results['no_hsic']['mse'],
        '+HSIC\n(Full Model)': results['no_hsic']['mse'] - results['full']['mse'],
    }
    comp_names = list(components.keys())
    comp_vals = list(components.values())
    comp_colors = ['#f39c12', '#9b59b6', '#3498db', '#e74c3c']
    bars = ax_contrib.barh(range(len(comp_names)), comp_vals,
                          color=comp_colors, edgecolor='black', linewidth=0.5)
    ax_contrib.set_yticks(range(len(comp_names)))
    ax_contrib.set_yticklabels(comp_names, fontsize=8)
    ax_contrib.set_xlabel('MSE Reduction')
    ax_contrib.set_title(f'Component Contribution\n(Total: {total_gain:.4f})', fontsize=10)
    ax_contrib.invert_yaxis()
    ax_contrib.axvline(x=0, color='gray', linewidth=0.8)

    # Row 2: 按通道类型分析OOD误差
    ax_ch = fig.add_subplot(gs[2, 0:half])
    ch_groups = {
        'Causal (Ch0-2)': [0, 1, 2],
        'Spurious (Ch3-4)': [3, 4],
        'Independent (Ch5-6)': [5, 6],
    }
    x_pos = np.arange(len(ch_groups))
    width = 0.13
    for i, (vk, vl) in enumerate(VARIANTS):
        preds = results[vk]['preds']
        trues = results[vk]['trues']
        group_mse = []
        for gname, ch_ids in ch_groups.items():
            ch_mse = np.mean([(preds[:,:,c] - trues[:,:,c])**2 for c in ch_ids])
            group_mse.append(ch_mse)
        label = vl.split('\n')[0]
        ax_ch.bar(x_pos + i * width, group_mse, width, label=label,
                 color=COLORS[vk], edgecolor='black', linewidth=0.3)
    ax_ch.set_xticks(x_pos + width * 2.5)
    ax_ch.set_xticklabels(list(ch_groups.keys()), fontsize=10)
    ax_ch.set_ylabel('MSE')
    ax_ch.set_title('Per-Channel-Type MSE Analysis', fontsize=12, fontweight='bold')
    ax_ch.legend(fontsize=8, loc='upper left')

    # Row 2 Right: 参数量与时间
    ax_eff = fig.add_subplot(gs[2, half:n_var])
    params = [results[vk]['params'] for vk, _ in VARIANTS]
    times = [results[vk]['time'] for vk, _ in VARIANTS]
    mses = [results[vk]['mse'] for vk, _ in VARIANTS]
    scatter = ax_eff.scatter(params, mses, c=[COLORS[vk] for vk, _ in VARIANTS],
                            s=[max(80, t*0.8) for t in times],
                            edgecolors='black', linewidth=0.5, zorder=5)
    for i, (vk, vl) in enumerate(VARIANTS):
        label = vl.split('\n')[0]
        ax_eff.annotate(label, (params[i], mses[i]),
                       textcoords="offset points", xytext=(8, 5),
                       fontsize=8, color=COLORS[vk], fontweight='bold')
    ax_eff.set_xlabel('Parameters')
    ax_eff.set_ylabel('MSE (↓ better)')
    ax_eff.set_title('Efficiency: Params vs MSE\n(bubble size = training time)', fontsize=11)
    ax_eff.grid(True, alpha=0.3)

    seed_tag = f" (seed={seed})" if seed is not None else ""
    plt.suptitle('CausalCIT Ablation Study — Synthetic Data\n'
                 'Each component contributes to the final performance' + seed_tag,
                 fontsize=15, fontweight='bold', y=1.01)
    save_path = os.path.join(args.output_dir, 'ablation_synthetic.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  可视化已保存: {save_path}")


# ============================================================
# 真实数据消融 (ETTh1)
# ============================================================

def run_real_ablation(args, seed):
    print_header(f"消融实验: ETTh1 真实数据 (seed={seed})")
    device = args.device

    data_path = os.path.join(args.dataset_dir, 'ETTh1.csv')
    if not os.path.exists(data_path):
        print(f"  ⚠ 数据不存在: {data_path}, 跳过")
        return None

    pred_lens = [96, 336]
    all_results = {}
    gate_matrices_all = {}
    diagnostics_all = {}

    for pred_len in pred_lens:
        print(f"\n  ▶ pred_len = {pred_len}")

        train_set = ETTDataset(data_path, seq_len=args.seq_len, pred_len=pred_len, flag='train')
        val_set = ETTDataset(data_path, seq_len=args.seq_len, pred_len=pred_len, flag='val')
        test_set = ETTDataset(data_path, seq_len=args.seq_len, pred_len=pred_len, flag='test')

        train_loader = get_dataloader(train_set, batch_size=args.batch_size)
        val_loader = get_dataloader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = get_dataloader(test_set, batch_size=args.batch_size, shuffle=False)

        common_kwargs = dict(
            enc_in=7, seq_len=args.seq_len, pred_len=pred_len,
            e_layers=3, n_heads=4, d_model=32, d_ff=128,
            dropout=0.3, fc_dropout=0.3,
            patch_len=args.patch_len, stride=args.stride, padding_patch='end',
            n_channel_heads=4, n_envs=args.n_envs, rff_dim=32,
            channel_dropout=0.1, fusion_alpha=0.3,
        )

        pl_results = {}
        pl_gates = {}
        pl_diag = {}
        for variant_key, variant_label in VARIANTS:
            label_short = variant_label.replace('\n', ' ')
            set_seed(seed)  # 每个变体训练前重置种子，保证初始化/shuffle顺序一致，公平对比
            model = create_ablation_model(variant_key, **common_kwargs)
            trainer = Trainer(model, device=device)
            save_dir = os.path.join(args.output_dir, f'ckpt_etth1_{variant_key}_pl{pred_len}_s{seed}')
            hist = trainer.train(train_loader, val_loader, epochs=30,
                                 lr=0.001, patience=7, save_dir=save_dir)
            res = trainer.test(test_loader)
            pl_results[variant_key] = {
                'mse': res['mse'], 'mae': res['mae'],
                'params': count_params(model), 'time': hist['total_time'],
            }
            print(f"    {label_short:30s}  MSE={res['mse']:.6f}  MAE={res['mae']:.6f}")
            # 提取门控矩阵 + 诊断参数
            pl_gates[variant_key] = _collect_gate_matrices(model, test_loader, device)
            pl_diag[variant_key] = _collect_diagnostics(model, variant_key)

        all_results[pred_len] = pl_results
        gate_matrices_all[pred_len] = pl_gates
        diagnostics_all[pred_len] = pl_diag

    if all_results:
        _plot_real_ablation(all_results, args, seed)
    return all_results, gate_matrices_all, diagnostics_all


def run_real_multiseed(args):
    """多seed真实数据消融：循环seed，按 pred_len 聚合门控矩阵与诊断参数后落盘。"""
    seeds = resolve_seeds(args)
    real_all = {}
    gates_all = {}
    diag_all = {}
    for seed in seeds:
        r, g, d = run_real_ablation(args, seed)
        if r is None:
            continue
        real_all[seed] = r
        gates_all[seed] = g
        diag_all[seed] = d
    if not real_all:
        return None
    pred_lens = sorted(next(iter(real_all.values())).keys())
    for pl in pred_lens:
        pl_gates = {s: gates_all[s][pl] for s in real_all}
        _aggregate_and_save_gates(pl_gates, os.path.join(args.output_dir, f'etth1_pl{pl}'))
        pl_diag = {s: diag_all[s][pl] for s in real_all}
        _aggregate_and_save_diagnostics(pl_diag, os.path.join(args.output_dir, f'etth1_pl{pl}'))
    return real_all


def _plot_real_ablation(all_results, args, seed=None):
    pred_lens = sorted(all_results.keys())
    fig, axes = plt.subplots(1, len(pred_lens) + 1, figsize=(7 * (len(pred_lens) + 1), 6))
    if len(pred_lens) + 1 == 1:
        axes = [axes]

    for idx, pl in enumerate(pred_lens):
        ax = axes[idx]
        names = [vl.split('\n')[0] for _, vl in VARIANTS]
        mse_vals = [all_results[pl][vk]['mse'] for vk, _ in VARIANTS]
        colors = [COLORS[vk] for vk, _ in VARIANTS]
        bars = ax.bar(range(len(names)), mse_vals, color=colors,
                     edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=15, ha='right')
        ax.set_ylabel('MSE')
        ax.set_title(f'ETTh1 pred_len={pl}', fontsize=12, fontweight='bold')
        for bar, v in zip(bars, mse_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

    # 最后一列: 改进率汇总
    ax = axes[-1]
    for i, (vk, vl) in enumerate(VARIANTS):
        if vk == 'patchtst': continue
        improvements = []
        for pl in pred_lens:
            base = all_results[pl]['patchtst']['mse']
            curr = all_results[pl][vk]['mse']
            improvements.append((base - curr) / base * 100)
        label = vl.split('\n')[0]
        ax.plot(pred_lens, improvements, 'o-', label=label,
               color=COLORS[vk], linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Prediction Length')
    ax.set_ylabel('MSE Improvement vs PatchTST (%)')
    ax.set_title('Improvement Over Baseline', fontsize=12, fontweight='bold')
    ax.set_xticks(pred_lens)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    seed_tag = f" (seed={seed})" if seed is not None else ""
    plt.suptitle('CausalCIT Ablation Study — ETTh1 Real Data' + seed_tag,
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'ablation_etth1.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  可视化已保存: {save_path}")


# ============================================================
# 报告生成
# ============================================================

def _mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def generate_report(syn_all, real_all, args):
    report = []
    report.append("# CausalCIT 消融实验报告")
    report.append(f"\n> 运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"> 设备: {args.device}")
    report.append(f"> seeds: {resolve_seeds(args)}")
    report.append("")

    report.append("## 消融变体说明")
    report.append("")
    report.append("| 变体 | HSIC检验 | 环境划分 | 门控选择 | 说明 |")
    report.append("|------|---------|---------|---------|------|")
    report.append("| PatchTST | ❌ | ❌ | ❌ | 纯Channel-Independent基线 |")
    report.append("| w/o Gate | ❌ | ❌ | ❌ | 全连接通道注意力，无选择性门控 |")
    report.append("| w/o EnvSplit | ✅ | ❌ | ✅ | 全局HSIC，不划分环境 |")
    report.append("| w/o HSIC | ❌ | ✅ | ✅ | 用Pearson相关性替代HSIC |")
    report.append("| **Full CausalCIT** | **✅** | **✅** | **✅** | **完整模型** |")
    report.append("| **Full (fix prior)** | **✅** | **✅** | **✅** | **先验权重0.3→0.1 (诊断变体)** |")
    report.append("")

    if syn_all:
        seeds = list(syn_all.keys())
        # 绘图用第一个seed的 preds/trues
        syn_first = syn_all[seeds[0]]
        report.append("---")
        report.append("")
        report.append(f"## 合成数据消融 (d_model=64, 50 epochs, {len(seeds)} seeds)")
        report.append("")
        report.append("| 变体 | MSE mean | MSE std | MAE mean | MAE std | Params | Time(s) |")
        report.append("|------|---------|---------|---------|---------|--------|---------|")
        for vk, vl in VARIANTS:
            mse_m, mse_s = _mean_std([syn_all[s][vk]['mse'] for s in seeds])
            mae_m, mae_s = _mean_std([syn_all[s][vk]['mae'] for s in seeds])
            r0 = syn_first[vk]
            label = vl.replace('\n', ' ')
            report.append(f"| {label} | {mse_m:.6f} | {mse_s:.6f} | {mae_m:.6f} | {mae_s:.6f} | "
                         f"{r0['params']:,} | {r0['time']:.0f} |")
        report.append("")

        base_m, base_s = _mean_std([syn_all[s]['patchtst']['mse'] for s in seeds])
        full_m, full_s = _mean_std([syn_all[s]['full']['mse'] for s in seeds])
        total = base_m - full_m
        report.append("### 各组件边际贡献 (MSE降低, 用跨seed均值)")
        report.append("")
        report.append("| 组件 | MSE降低 |")
        report.append("|------|--------|")
        report.append(f"| 通道注意力 | {base_m - _mean_std([syn_all[s]['no_gate']['mse'] for s in seeds])[0]:.6f} |")
        report.append(f"| 门控选择 | {_mean_std([syn_all[s]['no_gate']['mse'] for s in seeds])[0] - _mean_std([syn_all[s]['no_env']['mse'] for s in seeds])[0]:.6f} |")
        report.append(f"| 环境划分 | {_mean_std([syn_all[s]['no_env']['mse'] for s in seeds])[0] - _mean_std([syn_all[s]['no_hsic']['mse'] for s in seeds])[0]:.6f} |")
        report.append(f"| HSIC检验 | {_mean_std([syn_all[s]['no_hsic']['mse'] for s in seeds])[0] - full_m:.6f} |")
        report.append(f"| **总计** | **{total:.6f}** |")
        report.append("")

    if real_all:
        seeds = list(real_all.keys())
        report.append("---")
        report.append("")
        report.append(f"## ETTh1 真实数据消融 ({len(seeds)} seeds)")
        report.append("")
        for pl in sorted(real_all[seeds[0]].keys()):
            report.append(f"### pred_len = {pl}")
            report.append("")
            report.append("| 变体 | MSE mean | MSE std | MAE mean | vs PatchTST (mean) |")
            report.append("|------|---------|---------|---------|-------------------|")
            base_m, base_s = _mean_std([real_all[s][pl]['patchtst']['mse'] for s in seeds])
            for vk, vl in VARIANTS:
                mse_m, mse_s = _mean_std([real_all[s][pl][vk]['mse'] for s in seeds])
                mae_m, _ = _mean_std([real_all[s][pl][vk]['mae'] for s in seeds])
                delta = (base_m - mse_m) / base_m * 100
                label = vl.replace('\n', ' ')
                report.append(f"| {label} | {mse_m:.6f} | {mse_s:.6f} | {mae_m:.6f} | {delta:+.2f}% |")
            report.append("")

    report.append("---")
    report.append("")
    report.append("## 结论")
    report.append("")
    report.append("1. 多seed聚合: 各变体 MSE 以 mean±std 报告，std 反映训练随机性噪声量级。")
    report.append("2. 配对显著性检验详见 `significance_report.md`（各变体 vs PatchTST 的 t 检验 / Wilcoxon p 值）。")
    report.append("3. 门控矩阵逐元素差异与诊断参数见 `gate_comparison.txt` / `gate_diagnostics.txt`。")
    report.append("")

    report_path = os.path.join(args.output_dir, 'ablation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n  报告已保存: {report_path}")


# ============================================================
# 多seed聚合 + 配对显著性检验
# ============================================================

def aggregate_and_test(syn_all, real_all, args):
    """计算各变体 vs PatchTST 的跨seed提升率，用 scipy 配对 t 检验 / Wilcoxon 检验。"""
    try:
        from scipy import stats
    except ImportError:
        print("\n  ⚠ 未安装 scipy，跳过显著性检验（pip install scipy 后重跑）。")
        return

    lines = ["# CausalCIT 多seed聚合与配对显著性检验", ""]
    lines.append(f"> seeds: {resolve_seeds(args)}")
    lines.append(f"> 检验方法: scipy.stats.ttest_rel (配对t检验) / wilcoxon (配对符号秩检验)")
    lines.append("> 解读: 提升率% = (base_mse - variant_mse) / base_mse * 100；p<0.05 表示显著优于基线")
    lines.append("")

    def _section(title, results_all):
        out = [f"## {title}", ""]
        out.append("| 变体 | mean MSE | std MSE | 提升% mean | 提升% std | t-test p | Wilcoxon p |")
        out.append("|------|---------|---------|-----------|-----------|----------|------------|")
        seeds = list(results_all.keys())
        base = [results_all[s]['patchtst']['mse'] for s in seeds]
        for vk, vl in VARIANTS:
            if vk == 'patchtst':
                continue
            curr = [results_all[s][vk]['mse'] for s in seeds]
            imp = [(b - c) / b * 100 for b, c in zip(base, curr)]
            imp_m, imp_s = _mean_std(imp)
            try:
                t_stat, p_t = stats.ttest_rel(base, curr)
            except Exception:
                p_t = float('nan')
            try:
                w_stat, p_w = stats.wilcoxon(base, curr)
            except Exception:
                p_w = float('nan')
            label = vl.replace('\n', ' ')
            out.append(f"| {label} | {np.mean(curr):.6f} | {np.std(curr):.6f} | "
                       f"{imp_m:+.2f} | {imp_s:.2f} | {p_t:.4f} | {p_w:.4f} |")
        out.append("")
        return out

    if syn_all:
        lines += _section("合成数据 (配对检验: 各变体 vs PatchTST)", syn_all)
    if real_all:
        seeds = list(real_all.keys())
        for pl in sorted(real_all[seeds[0]].keys()):
            pl_all = {s: real_all[s][pl] for s in seeds}
            lines += _section(f"ETTh1 pred_len={pl} (配对检验 vs PatchTST)", pl_all)

    lines.append("## 结论")
    lines.append("")
    lines.append("- 若某变体提升% mean>0 但 p 值不显著，说明效应被训练噪声淹没（诊断报告异常1）。")
    lines.append("- 若 Full 与 w/o EnvSplit 在多个 pred_len 上均不显著且提升%接近，印证两条路径学到等价门控（诊断报告异常2）。")
    lines.append("- full_fix 与 full 的差异反映先验权重(0.3→0.1)的影响（诊断报告假设A）。")
    lines.append("")

    path = os.path.join(args.output_dir, 'significance_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  显著性检验报告已保存: {path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='CausalCIT Ablation Study')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'synthetic', 'real'])
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='(已废弃) 全局随机种子；多seed模式下由 --n_seeds / --seeds 控制')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='环境划分数量；控制实验旋钮: 增大 seq_len 并减小 n_envs 可加大 patch_num，'
                             '拉开 Full 与 w/o EnvSplit 差异 (诊断报告假设B)')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='多seed实验的seed数量，用于聚合 mean±std 与配对显著性检验')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='显式指定seed列表 (如 42 123 2024 7 99)，优先于 --n_seeds')
    args = parser.parse_args()
    if args.dataset_dir is None:
        args.dataset_dir = DATASET_DIR

    # 自动创建数据集目录
    os.makedirs(args.dataset_dir, exist_ok=True)

    # 检查真实数据是否存在
    if args.exp in ['all', 'real']:
        _missing = []
        for _fn in ['ETTh1.csv']:
            if not os.path.exists(os.path.join(args.dataset_dir, _fn)):
                _missing.append(_fn)
        if _missing:
            print(f"\n  ⚠ 数据集缺失: {', '.join(_missing)}")
            print(f"  存放位置: {args.dataset_dir}/")
            print(f"  下载方式: cd {PROJECT_DIR} && python download_data.py")
            print(f"  或手动放入后重新运行。合成数据消融实验不受影响。\n")

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    seeds = resolve_seeds(args)
    print("=" * 72)
    print("  CausalCIT 消融实验 (Ablation Study)")
    print("=" * 72)
    print(f"  设备: {args.device}")
    print(f"  变体数: {len(VARIANTS)}")
    for vk, vl in VARIANTS:
        print(f"    - {vk}: {vl.replace(chr(10), ' ')}")
    print(f"  seeds: {seeds}")
    print(f"  n_envs: {args.n_envs}")

    syn_all = None
    real_all = None

    if args.exp in ['all', 'synthetic']:
        syn_all = run_synthetic_multiseed(args)

    if args.exp in ['all', 'real']:
        real_all = run_real_multiseed(args)

    generate_report(syn_all, real_all, args)
    aggregate_and_test(syn_all, real_all, args)

    print("\n" + "=" * 72)
    print("  消融实验完成！")
    print("=" * 72)
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
