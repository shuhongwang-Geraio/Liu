"""
PCD 静态相关掩码 vs CausalCIT HSIC 稳定性门控 —— 最小可证伪对比实验

研究问题:
  通道交互增益的来源是"静态相关强度"(PCD, ICASSP'26) 还是
  "跨环境因果稳定性"(CausalCIT)? 
  若 OOD(虚假相关强度随环境漂移) 下 pcd_gate 明显劣于 full_v2,
  则证明 HSIC 稳定性门控有 PCD 不具备的增量价值。

协议(与主表完全一致):
  - 变体: patchtst(CI基线) / full_v2(HSIC稳定性门控) / pcd_gate(静态相关掩码)
  - seed 配对 Wilcoxon + Holm 校正; n_seed>=5 才报 p 值
  - 数据集: syn_ood(受控虚假相关漂移) / etth1(真实低维7通道)
  - 每个 (数据集, 变体) 跨 seed 报 mean±std MSE/MAE

用法:
  python run_pcd_compare.py --datasets syn_ood etth1 --seeds 42 123 2024 --epochs 50
  python run_pcd_compare.py --datasets etth1 --seeds 42 123 2024 5 6 --epochs 50
"""

import os
import sys
import argparse
import time
import csv
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_DEMO_DIR = os.path.join(_PROJECT_DIR, 'CausalCIT_demo')
sys.path.insert(0, _DEMO_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from utils.data import ETTDataset, SyntheticOODDataset, get_dataloader
from utils.trainer import Trainer
from models_ablation import create_ablation_model


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 与 run_large.py 保持一致的超参
DATASET_CFG = {
    'syn_ood': dict(n_vars=7, d_model=64, d_ff=256, batch_size=32, epochs=50, patience=8, pred_lens=[96]),
    'etth1':   dict(n_vars=7, d_model=32, d_ff=128, batch_size=32, epochs=50, patience=8, pred_lens=[96]),
}


def resolve_dataset_dir():
    # 项目根: Liu/ (03_experiments 的上一级)
    root = os.path.dirname(os.path.dirname(_PROJECT_DIR))
    candidates = [
        os.path.join(root, '01_external', 'PatchTST', 'code', 'dataset'),
        os.path.join(root, '01_external', 'Crossformer', 'code', 'datasets'),
        os.path.join(root, '01_external', 'TimeDRL', 'code', 'dataset', 'forecasting', 'ETT-small'),
        os.path.join(root, '01_external', 'DLinear', 'code', 'dataset'),
        os.path.join(_PROJECT_DIR, 'data'),
        os.path.join(_PROJECT_DIR, 'patchtst', 'dataset'),
    ]
    for p in candidates:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'ETTh1.csv')):
            return p
    return candidates[0]


def compute_corr_matrix(dataset):
    """从训练集数据计算 Pearson 相关矩阵 |R| (数据集级, PCD 协议)."""
    data = dataset.data  # [N, n_vars]
    mu = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    zn = (data - mu) / std
    corr = (zn.T @ zn) / zn.shape[0]
    return torch.tensor(np.abs(corr), dtype=torch.float32)


def make_datasets(ds, seq_len, pred_len, dataset_dir):
    if ds == 'syn_ood':
        # 训练/验证 regime='train'(稳定虚假相关强度), 测试 regime='test'(漂移/反转)
        base = dict(seed=0, spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                    test_spurious_strengths=(0.05, -0.2, 0.1, -0.05))
        train = SyntheticOODDataset(seq_len=seq_len, pred_len=pred_len, flag='train', regime='train', **base)
        val = SyntheticOODDataset(seq_len=seq_len, pred_len=pred_len, flag='val', regime='train', **base)
        test = SyntheticOODDataset(seq_len=seq_len, pred_len=pred_len, flag='test', regime='test', **base)
        return train, val, test
    else:
        data_path = os.path.join(dataset_dir, 'ETTh1.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'ETTh1.csv not found in {dataset_dir}')
        return (ETTDataset(data_path, seq_len=seq_len, pred_len=pred_len, flag='train'),
                ETTDataset(data_path, seq_len=seq_len, pred_len=pred_len, flag='val'),
                ETTDataset(data_path, seq_len=seq_len, pred_len=pred_len, flag='test'))


def build_kwargs(ds, pl, variant, dataset_dir):
    cfg = DATASET_CFG[ds]
    seq_len = 96 if pl <= 192 else 336
    base = dict(enc_in=cfg['n_vars'], seq_len=seq_len, pred_len=pl,
                e_layers=3, n_heads=4, d_model=cfg['d_model'], d_ff=cfg['d_ff'],
                dropout=0.2, fc_dropout=0.2, patch_len=16, stride=8, padding_patch='end',
                n_channel_heads=4, n_envs=4, rff_dim=32, channel_dropout=0.1,
                fusion_alpha=0.3)
    if variant == 'full_v2':
        base.update(prior_weight=0.05, temperature=0.5, alpha_init=-2.0)
    elif variant in ('pcd_gate',):
        base.update(prior_weight=0.05, temperature=0.5, alpha_init=-2.0)
    return base


def train_one(ds, pl, variant, seed, dataset_dir, epochs, device):
    set_seed(seed)
    cfg = DATASET_CFG[ds]
    seq_len = 96 if pl <= 192 else 336
    train_set, val_set, test_set = make_datasets(ds, seq_len, pl, dataset_dir)
    pin = device.startswith('cuda')
    train_loader = get_dataloader(train_set, batch_size=cfg['batch_size'], pin_memory=pin)
    val_loader = get_dataloader(val_set, batch_size=cfg['batch_size'], shuffle=False, pin_memory=pin)
    test_loader = get_dataloader(test_set, batch_size=cfg['batch_size'], shuffle=False, pin_memory=pin)

    model = create_ablation_model(variant, **build_kwargs(ds, pl, variant, dataset_dir))
    # PCD 变体: 训练前注入数据集级相关矩阵
    if variant == 'pcd_gate':
        corr = compute_corr_matrix(train_set)
        ci = model.model.causal_channel if hasattr(model, 'model') else model
        ci.set_corr_matrix(corr)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = Trainer(model, device=device)
    hist = trainer.train(train_loader, val_loader, epochs=epochs, lr=0.001,
                         patience=cfg['patience'],
                         save_dir=os.path.join('output_pcd', 'ckpt', f'{ds}_pl{pl}_{variant}_s{seed}'))
    res = trainer.test(test_loader)
    return dict(mse=float(res['mse']), mae=float(res['mae']), rmse=float(res['rmse']),
                params=params, epochs=hist['epochs_trained'])


def wilcoxon_paired(base_mse, var_mse):
    from scipy import stats
    b = np.array([base_mse[s] for s in base_mse if s in var_mse])
    v = np.array([var_mse[s] for s in base_mse if s in var_mse])
    if len(b) < 5:
        return float('nan')
    try:
        _, p = stats.wilcoxon(b, v)
        return float(p)
    except Exception:
        return float('nan')


def holm_adjust(pvals):
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = len(idx)
    adj = [float('nan')] * len(pvals)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['syn_ood', 'etth1'])
    ap.add_argument('--seeds', nargs='+', default=['42', '123', '2024', '5', '6'], type=int)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output', default='output_pcd')
    args = ap.parse_args()

    dataset_dir = resolve_dataset_dir()
    print(f'数据目录: {dataset_dir} | 设备: {args.device}')
    os.makedirs(args.output, exist_ok=True)

    variants = ['patchtst', 'full_v2', 'pcd_gate']
    results = {}
    csv_path = os.path.join(args.output, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'pred_len', 'variant', 'seed', 'mse', 'mae', 'rmse', 'params', 'epochs', 'time'])
        for ds in args.datasets:
            for pl in DATASET_CFG[ds]['pred_lens']:
                for v in variants:
                    for s in args.seeds:
                        t0 = time.time()
                        r = train_one(ds, pl, v, s, dataset_dir, args.epochs, args.device)
                        dt = time.time() - t0
                        w.writerow([ds, pl, v, s, f"{r['mse']:.6f}", f"{r['mae']:.6f}",
                                    f"{r['rmse']:.6f}", r['params'], r['epochs'], f"{dt:.1f}"])
                        f.flush()
                        print(f"[{ds} pl{pl} {v} s{s}] MSE={r['mse']:.5f} MAE={r['mae']:.5f} ({dt:.0f}s)", flush=True)
    print(f'结果已保存: {csv_path}')

    # ---- 汇总 + 统计 ----
    import pandas as pd
    df = pd.read_csv(csv_path)
    lines = [f'# PCD vs CausalCIT 对比报告', '',
             f'> 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'> 数据集: {args.datasets} | 变体: {variants} | seeds: {args.seeds} | epochs: {args.epochs}',
             f'> 协议: seed 配对 Wilcoxon (n>=5), 组内 Holm 校正', '']
    for ds in args.datasets:
        lines.append(f'## {ds}')
        lines.append('')
        lines.append('| 变体 | MSE mean | MSE std | MAE mean | vs PatchTST | vs full_v2 |')
        lines.append('|------|---------|---------|---------|------------|------------|')
        sub = df[df.dataset == ds]
        base = sub[sub.variant == 'patchtst'].set_index('seed')['mse']
        fv = sub[sub.variant == 'full_v2'].set_index('seed')['mse']
        row_cache = []
        for v in variants:
            sv = sub[sub.variant == v]
            mse_m, mse_s = sv['mse'].mean(), sv['mse'].std()
            mae_m = sv['mae'].mean()
            bm = base.mean()
            imp = f"{(bm - mse_m) / bm * 100:+.2f}%" if bm > 0 else '-'
            if v != 'patchtst':
                p_w = wilcoxon_paired(base.to_dict(), sv.set_index('seed')['mse'].to_dict())
            else:
                p_w = float('nan')
            row_cache.append([v, mse_m, mse_s, mae_m, imp, p_w])
        holm = holm_adjust([r[5] for r in row_cache])
        for r, ph in zip(row_cache, holm):
            v, mse_m, mse_s, mae_m, imp, p_w = r
            p_str = f'{p_w:.4f}' if not np.isnan(p_w) else '-'
            ph_str = f'{ph:.4f}' if not np.isnan(ph) else '-'
            sig = '*' if (not np.isnan(ph) and ph < 0.05) else ''
            lines.append(f'| {v} | {mse_m:.6f} | {mse_s:.6f} | {mae_m:.6f} | {imp} (p={p_str}, Holm={ph_str}){sig} |')
        # full_v2 vs pcd_gate 关键对照
        lines.append('')
        p_fv_pcd = wilcoxon_paired(fv.to_dict(), sub[sub.variant == 'pcd_gate'].set_index('seed')['mse'].to_dict())
        fv_m = fv.mean(); pcd_m = sub[sub.variant == 'pcd_gate']['mse'].mean()
        lines.append(f'**full_v2 vs pcd_gate**: full_v2 MSE={fv_m:.6f}, pcd_gate MSE={pcd_m:.6f}, '
                     f'full_v2 相对 pcd_gate {(fv_m - pcd_m) / pcd_m * 100:+.2f}% '
                     f'(Wilcoxon p={p_fv_pcd:.4f})')
        lines.append('')
    out_path = os.path.join(args.output, 'pcd_vs_causalcit_report.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'报告已保存: {out_path}')


if __name__ == '__main__':
    main()
