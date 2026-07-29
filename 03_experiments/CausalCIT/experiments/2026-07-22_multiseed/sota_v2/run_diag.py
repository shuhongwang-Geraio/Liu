"""
CausalCIT 快速诊断脚本 (run_diag.py)

用途: 快速迭代验证架构改进方向，不走 run_ablation.py 的完整6变体+多seed+绘图管线。
聚焦对比少量变体 (默认 patchtst / full / full_v2)，打印:
  - MSE / MAE / 相对PatchTST提升%
  - 门控矩阵统计 (非对角线均值/标准差/饱和比例) —— 验证"门控是否分化"
  - 合成数据上打印完整门控均值矩阵 (查看 Ch1←Ch0, Ch2←Ch0 因果对是否被识别)

用法:
    python run_diag.py --data syn --epochs 20 --seed 42
    python run_diag.py --data etth1 --pred_len 96 --epochs 20
    python run_diag.py --data syn --variants patchtst,full,full_v2 \
        --prior_weight 0.05 --temperature 0.5 --entropy_weight 0.01
"""

import os
import sys
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEMO_DIR = os.path.join(PROJECT_DIR, 'CausalCIT_demo')

_DEFAULT_PATHS = [
    os.path.join(PROJECT_DIR, 'patchtst', 'dataset'),
    os.path.join(PROJECT_DIR, 'data'),
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
from models_ablation import create_ablation_model


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_gate(model, loader, device, max_batches=10):
    if not hasattr(model, 'get_gate_matrix'):
        return None
    model.eval()
    gms = []
    with torch.no_grad():
        for i, (bx, _) in enumerate(loader):
            if i >= max_batches:
                break
            _ = model(bx.to(device))
            gm = model.get_gate_matrix()
            if gm is not None:
                gms.append(gm.detach().cpu().numpy())
    if not gms:
        return None
    return np.concatenate(gms, axis=0)


def gate_stats(gm):
    """gm: [N, nvars, nvars] -> 非对角线统计字典"""
    m = gm.mean(axis=0)
    nv = m.shape[0]
    eye = np.eye(nv, dtype=bool)
    off = m[~eye]
    return dict(off_mean=float(off.mean()), off_std=float(off.std()),
                off_min=float(off.min()), off_max=float(off.max()),
                frac_low=float((off < 0.3).mean()), frac_high=float((off > 0.7).mean()),
                mean_matrix=m)


def build_kwargs(args, n_vars, pred_len, d_model, d_ff):
    # 注意: 不在此处放 prior_weight/temperature，避免污染 full 基线。
    # 这两个改进参数仅在 run_one 中对 full_v2/full_fix 注入。
    kw = dict(
        enc_in=n_vars, seq_len=args.seq_len, pred_len=pred_len,
        e_layers=3, n_heads=4, d_model=d_model, d_ff=d_ff,
        dropout=args.dropout, fc_dropout=args.dropout,
        patch_len=args.patch_len, stride=args.stride, padding_patch='end',
        n_channel_heads=4, n_envs=args.n_envs, rff_dim=64,
        channel_dropout=0.1, fusion_alpha=0.3,
    )
    return kw


def run_one(variant, kwargs, train_loader, val_loader, test_loader, args, tag):
    set_seed(args.seed)
    # 仅对改进变体注入 prior_weight/temperature，保持 full/patchtst 为干净基线
    vk = dict(kwargs)
    if variant in ('full_v2', 'full_fix'):
        vk['prior_weight'] = args.prior_weight
        vk['temperature'] = args.temperature
    model = create_ablation_model(variant, **vk)
    params = count_params(model)
    trainer = Trainer(model, device=args.device)
    save_dir = os.path.join(args.output_dir, f'ckpt_{tag}_{variant}')
    ew = args.entropy_weight if variant in ('full', 'full_v2', 'full_fix') else 0.0
    hist = trainer.train(train_loader, val_loader, epochs=args.epochs,
                         lr=args.lr, patience=args.patience, save_dir=save_dir,
                         entropy_weight=ew)
    res = trainer.test(test_loader)
    gm = collect_gate(model, test_loader, args.device)
    diag = model.get_diagnostics() if hasattr(model, 'get_diagnostics') else None
    return {
        'mse': res['mse'], 'mae': res['mae'], 'params': params,
        'time': hist['total_time'], 'gate': gm, 'diag': diag,
    }


def print_comparison(results, variants, title, show_matrix=False, channel_labels=None):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    base_mse = results['patchtst']['mse'] if 'patchtst' in results else None
    header = f"{'variant':12s} {'MSE':>10s} {'MAE':>10s} {'vs PT%':>8s} {'params':>10s} {'gate_off_mean':>13s} {'gate_off_std':>12s}"
    print(header)
    print("-" * 78)
    for v in variants:
        r = results[v]
        vs = f"{(base_mse - r['mse'])/base_mse*100:+.2f}" if base_mse else "  -"
        gs = gate_stats(r['gate']) if r['gate'] is not None else None
        gm_mean = f"{gs['off_mean']:.4f}" if gs else "   -"
        gm_std = f"{gs['off_std']:.4f}" if gs else "   -"
        print(f"{v:12s} {r['mse']:10.6f} {r['mae']:10.6f} {vs:>8s} {r['params']:10,d} {gm_mean:>13s} {gm_std:>12s}")
    print("-" * 78)

    # 门控分化诊断
    for v in variants:
        r = results[v]
        if r['gate'] is None:
            continue
        gs = gate_stats(r['gate'])
        print(f"\n[{v}] 门控非对角线: mean={gs['off_mean']:.4f} std={gs['off_std']:.4f} "
              f"min={gs['off_min']:.4f} max={gs['off_max']:.4f} "
              f"frac<0.3={gs['frac_low']:.2f} frac>0.7={gs['frac_high']:.2f}")
        if r['diag']:
            d = r['diag']
            keys = ['prior_weight', 'temperature', 'stability_bias',
                    'channel_prior_sig_mean', 'last_entropy']
            dd = {k: d[k] for k in keys if k in d}
            print(f"     diag: {dd}")
        if show_matrix:
            labels = channel_labels if channel_labels else [str(i) for i in range(gs['mean_matrix'].shape[0])]
            print(f"     门控均值矩阵 (row=query, 通道: {labels}):")
            print(np.array2string(gs['mean_matrix'], precision=3, prefix='       '))


def run_synthetic(args):
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
    kwargs = build_kwargs(args, n_vars, args.pred_len, d_model=64, d_ff=256)

    labels = getattr(train_set, 'channel_labels', None)
    short = ['Base', 'C1', 'C2', 'S1', 'S2', 'I1', 'I2']

    results = {}
    for v in args.variants:
        print(f"\n>>> [synthetic] 训练变体: {v}")
        results[v] = run_one(v, kwargs, train_loader, val_loader, test_loader, args, tag='syn')
    print_comparison(results, args.variants, 'DIAG 合成数据对比',
                     show_matrix=True, channel_labels=short)
    return results


_CSV_MAP = {
    'etth1': 'ETTh1.csv', 'etth2': 'ETTh2.csv',
    'ettm1': 'ETTm1.csv', 'ettm2': 'ETTm2.csv',
    'weather': 'weather.csv', 'electricity': 'electricity.csv',
    'traffic': 'traffic.csv',
}


def run_csv(args, name):
    import pandas as pd
    csv = _CSV_MAP.get(name, name)
    data_path = os.path.join(args.dataset_dir, csv)
    if not os.path.exists(data_path):
        print(f"  ⚠ 数据不存在: {data_path}")
        return None
    n_vars = pd.read_csv(data_path, nrows=1).shape[1] - 1  # 去date列
    train_set = ETTDataset(data_path, seq_len=args.seq_len, pred_len=args.pred_len, flag='train')
    val_set = ETTDataset(data_path, seq_len=args.seq_len, pred_len=args.pred_len, flag='val')
    test_set = ETTDataset(data_path, seq_len=args.seq_len, pred_len=args.pred_len, flag='test')
    train_loader = get_dataloader(train_set, batch_size=args.batch_size)
    val_loader = get_dataloader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_set, batch_size=args.batch_size, shuffle=False)

    # 高维数据用更小的 d_model 控制显存/时间
    dm = 32 if n_vars <= 21 else 16
    kwargs = build_kwargs(args, n_vars, args.pred_len, d_model=dm, d_ff=dm * 4)

    results = {}
    for v in args.variants:
        print(f"\n>>> [{name} pl{args.pred_len} nvars={n_vars}] 训练变体: {v}")
        results[v] = run_one(v, kwargs, train_loader, val_loader, test_loader, args, tag=f'{name}_pl{args.pred_len}')
    print_comparison(results, args.variants, f'DIAG {name} pred_len={args.pred_len} nvars={n_vars} 对比',
                     show_matrix=(n_vars <= 21))
    return results


def parse_args():
    p = argparse.ArgumentParser(description='CausalCIT 快速诊断')
    p.add_argument('--data', type=str, default='syn',
                   help="syn / etth1 / etth2 / ettm1 / ettm2 / weather / electricity / traffic / both")
    p.add_argument('--variants', type=str, default='patchtst,full,full_v2')
    p.add_argument('--output_dir', type=str, default='./output_diag')
    p.add_argument('--dataset_dir', type=str, default=None)
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--n_envs', type=int, default=4)
    p.add_argument('--prior_weight', type=float, default=0.05)
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--entropy_weight', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.dataset_dir is None:
        args.dataset_dir = DATASET_DIR
    args.variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 78)
    print("  CausalCIT 快速诊断 (run_diag.py)")
    print("=" * 78)
    print(f"  data={args.data}  variants={args.variants}  device={args.device}")
    print(f"  seq_len={args.seq_len} pred_len={args.pred_len} n_envs={args.n_envs} epochs={args.epochs}")
    print(f"  prior_weight={args.prior_weight} temperature={args.temperature} entropy_weight={args.entropy_weight}")

    if args.data == 'syn':
        run_synthetic(args)
    elif args.data == 'both':
        run_synthetic(args)
        run_csv(args, 'etth1')
    else:
        run_csv(args, args.data)


if __name__ == '__main__':
    main()
