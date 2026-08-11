"""
PCD 对比实验 — 从已有 checkpoint 做 eval 补全 (不重训)。

背景: output_pcd/ckpt 已有部分 50-epoch 的完整 checkpoint
      (full_v2 s42/s123, pcd_gate s42/s123, patchtst s42/s123/s2024/s5),
      但没有对应的 results.csv 汇总。本脚本:
  1) 从这些 ckpt 加载模型, 在 syn_ood test 集 (OOD: 虚假相关反转) 上评估;
  2) 对缺失的 (variant, seed) 组合执行短训练补全 (用较小 epochs, 标注非正式);
  3) 汇总为与 run_pcd_compare.py 相同格式的 results.csv + 报告。

用法:
  python pcd_eval_from_ckpt.py --epochs 20 --output output_pcd_full
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CausalCIT_demo'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data import SyntheticOODDataset, get_dataloader
from utils.trainer import Trainer
from models_ablation import create_ablation_model

import run_pcd_compare as pcd  # 复用 DATASET_CFG / build_kwargs / compute_corr_matrix / make_datasets

CKPT_DIR = os.path.join('output_pcd', 'ckpt')
DS, PL = 'syn_ood', 96
VARIANTS = ['patchtst', 'full_v2', 'pcd_gate']
SEEDS = [42, 123, 2024, 5, 6]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_ckpt(ds, pl, variant, seed, device):
    """从 ckpt 加载并评估; ckpt 缺失返回 None。"""
    ckpt = os.path.join(CKPT_DIR, f'{ds}_pl{pl}_{variant}_s{seed}', 'checkpoint.pth')
    if not os.path.exists(ckpt):
        return None
    kw = pcd.build_kwargs(ds, pl, variant, pcd.resolve_dataset_dir())
    model = create_ablation_model(variant, **kw)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    if variant == 'pcd_gate':
        train_set, _, _ = pcd.make_datasets(ds, 96, pl, pcd.resolve_dataset_dir())
        corr = pcd.compute_corr_matrix(train_set)
        ci = model.model.causal_channel if hasattr(model, 'model') else model
        ci.set_corr_matrix(corr)
    _, _, test_set = pcd.make_datasets(ds, 96, pl, pcd.resolve_dataset_dir())
    loader = get_dataloader(test_set, batch_size=32, shuffle=False, pin_memory=False)
    trainer = Trainer(model, device=device)
    res = trainer.test(loader)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(mse=float(res['mse']), mae=float(res['mae']), rmse=float(res['rmse']),
                params=params, epochs=None, time=0.0, from_ckpt=True)


def train_missing(ds, pl, variant, seed, epochs, device):
    """训练缺失组合 (短 epoch, 非正式, 用于流程验证)。"""
    set_seed(seed)
    cfg = pcd.DATASET_CFG[ds]
    train_set, val_set, test_set = pcd.make_datasets(ds, 96, pl, pcd.resolve_dataset_dir())
    tl = get_dataloader(train_set, batch_size=cfg['batch_size'], pin_memory=False)
    vl = get_dataloader(val_set, batch_size=cfg['batch_size'], shuffle=False, pin_memory=False)
    te = get_dataloader(test_set, batch_size=cfg['batch_size'], shuffle=False, pin_memory=False)
    model = create_ablation_model(variant, **pcd.build_kwargs(ds, pl, variant, pcd.resolve_dataset_dir()))
    if variant == 'pcd_gate':
        corr = pcd.compute_corr_matrix(train_set)
        ci = model.model.causal_channel if hasattr(model, 'model') else model
        ci.set_corr_matrix(corr)
    trainer = Trainer(model, device=device)
    hist = trainer.train(tl, vl, epochs=epochs, lr=0.001, patience=cfg['patience'],
                         save_dir=os.path.join(CKPT_DIR, f'{ds}_pl{pl}_{variant}_s{seed}'))
    res = trainer.test(te)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(mse=float(res['mse']), mae=float(res['mae']), rmse=float(res['rmse']),
                params=params, epochs=hist['epochs_trained'], time=0.0, from_ckpt=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20, help='缺失组合的补跑轮数(非正式)')
    ap.add_argument('--output', default='output_pcd_full')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    csv_path = os.path.join(args.output, 'results.csv')
    rows = []
    for v in VARIANTS:
        for s in SEEDS:
            t0 = time.time()
            r = eval_ckpt(DS, PL, v, s, args.device)
            if r is None:
                print(f'[eval] {v} s{s} ckpt缺失 -> 短训 {args.epochs} epoch', flush=True)
                r = train_missing(DS, PL, v, s, args.epochs, args.device)
                tag = 'trained'
            else:
                print(f'[eval] {v} s{s} 从ckpt评估', flush=True)
                tag = 'ckpt'
            r['time'] = time.time() - t0
            r['tag'] = tag
            rows.append(r)
            print(f'    MSE={r["mse"]:.5f} MAE={r["mae"]:.5f} [{tag}]', flush=True)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'pred_len', 'variant', 'seed', 'mse', 'mae', 'rmse',
                    'params', 'epochs', 'time', 'tag'])
        for r, (v, s) in zip(rows, [(vv, ss) for vv in VARIANTS for ss in SEEDS]):
            w.writerow([DS, PL, v, s, f"{r['mse']:.6f}", f"{r['mae']:.6f}",
                        f"{r['rmse']:.6f}", r['params'],
                        r['epochs'] if r['epochs'] else '', f"{r['time']:.1f}", r['tag']])
    print(f'结果已保存: {csv_path}')

    # 汇总报告
    import pandas as pd
    df = pd.read_csv(csv_path)
    lines = [f'# PCD vs CausalCIT 对比报告 (从 ckpt 补全)', '',
             f'> 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'> 数据集: [syn_ood] | 变体: {VARIANTS} | seeds: {SEEDS}',
             f'> 说明: ckpt=从已存 50-epoch checkpoint 评估; trained=本次短训(非正式)', '']
    base = df[df.variant == 'patchtst'].set_index('seed')['mse']
    for v in VARIANTS:
        sv = df[df.variant == v]
        m = sv['mse'].mean(); sd = sv['mse'].std()
        imp = f"{(base.mean() - m) / base.mean() * 100:+.2f}%" if base.mean() > 0 else '-'
        tags = dict(zip(sv.seed, sv.tag))
        lines.append(f'| {v} | {m:.6f} | {sd:.6f} | {sv["mae"].mean():.6f} | {imp} | {tags} |')
    fv = df[df.variant == 'full_v2'].set_index('seed')['mse']
    pcdv = df[df.variant == 'pcd_gate'].set_index('seed')['mse']
    lines.append('')
    lines.append(f'**full_v2 vs pcd_gate**: full_v2 {fv.mean():.6f} vs pcd_gate {pcdv.mean():.6f} '
                 f'= {(fv.mean() - pcdv.mean()) / pcdv.mean():+.2%} (负值=full_v2更优)')
    out_path = os.path.join(args.output, 'pcd_vs_causalcit_report.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'报告已保存: {out_path}')


if __name__ == '__main__':
    main()
