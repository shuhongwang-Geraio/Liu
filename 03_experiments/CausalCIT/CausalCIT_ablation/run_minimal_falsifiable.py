"""
最小可证伪测试 (回应评审 re2 §6.1 第1/4/5条)

背景:
  评审 re2 的核心质疑不是"数据不够漂亮"，而是三处代码层面的问题:
    (a) entropy_weight 从未被传给 trainer.train()，一直是死代码；
    (b) full_v2 的 stability_v2=True 门控把 batch 维一起池化估 HSIC，
        导致同一测试样本换一批"batch同伴"，门控矩阵/预测结果会变；
    (c) capacity_match/gate_prior_only 这两个"关键对照"从未真正做过
        vs-full_v2 的配对显著性检验(只做过 vs-PatchTST)。
  这三处已经在 causal_channel.py / models_ablation.py / run_large.py 里修好
  (running_stats 开关 + entropy_weight 接线 + 关键对照 Holm 校正)。

本脚本做什么:
  在单一数据集(默认 traffic)上，用完全相同的代码路径、完全相同的 8 个 seed，
  训练 patchtst / full_v2 / full_v2_fixed / capacity_match / gate_prior_only / no_env，
  在**同一份报告**里同时给出两条证据链:
    (A) MSE 证据链: mean±std, 配对 Wilcoxon + Holm 校正
        (vs-PatchTST, 以及 full_v2 vs 关键对照)
    (B) 门控行为证据链:
        - off_mean/off_std: 门控是否坍缩成常数 (对真实数据集也适用，不需要
          ground-truth因果标签; 因果分离度 off_std/causal_minus_spurious 这类
          需要已知因果边的指标，只有 syn_ood 合成数据才能算，这里不强行套用)
        - batch_dep_score: 同一个测试样本换不同 batch 同伴，门控矩阵变化幅度
          (直接检验 (b) 的 bug 是否被 running_stats 修复 —— 这是本脚本的核心新增)

如果对齐两条证据链后，full_v2 相对 capacity_match/gate_prior_only 的提升仍然
不显著，或 full_v2_fixed 把 full_v2 的提升"修没了"，这就是评审要的"可证伪"结果，
应据此把结论降级为"稳定性正则化的通道注意力"，而不是强行论证"因果机制"。

用法 (有 GPU 的机器上):
  python run_minimal_falsifiable.py --dataset traffic \
      --seeds 42 123 2024 7 13 99 2023 31 --device cuda:0 \
      --output_dir ./output_falsifiable

没有 GPU 时的 sanity check (验证代码本身没有 bug, 用内置合成数据, 不需要外部csv):
  python run_minimal_falsifiable.py --dataset syn_ood --device cpu \
      --seeds 42 123 --quick --output_dir ./output_falsifiable_smoketest
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEMO_DIR = os.path.join(PROJECT_DIR, 'CausalCIT_demo')
sys.path.insert(0, DEMO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.data import ETTDataset, SyntheticOODDataset, get_dataloader
from utils.trainer import Trainer
from models_ablation import create_ablation_model

from run_large import (
    dataset_config, seq_for_pl, resolve_dataset_dir, DATASET_CSV,
    build_kwargs, set_seed, _mean_std, _wilcoxon_paired, _holm_adjust,
)

DEFAULT_VARIANTS = ['patchtst', 'full_v2', 'full_v2_fixed',
                    'capacity_match', 'gate_prior_only', 'no_env']
DEFAULT_SEEDS = [42, 123, 2024, 7, 13, 99, 2023, 31]


# ============================================================
# 训练单个 (variant, seed, pred_len) —— 与 run_large._train_one 同一套
# 建模/数据/训练代码路径，但额外返回训练好的 model + test_set 供门控诊断使用。
# ============================================================
def train_one(ds, pl, variant, seed, dataset_dir, device, epochs=None,
              entropy_weight=0.0, amp=False):
    set_seed(seed)
    cfg = dataset_config(ds)
    job = build_kwargs(ds, pl, variant, seed, dataset_dir, entropy_weight=entropy_weight)
    seq_len = job['model_kwargs']['seq_len']
    ep = epochs if epochs is not None else job['epochs']

    if ds in ('syn_ood', 'syn_ood_noise'):
        tr = dict(regime='train', seed=seed,
                  spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                  test_spurious_strengths=(0.05, -0.2, 0.1, -0.05),
                  train_noise=0.05, test_noise=0.05)
        te = dict(regime='test', seed=seed,
                  spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                  test_spurious_strengths=(0.05, -0.2, 0.1, -0.05),
                  train_noise=0.05, test_noise=0.05)
        train_set = SyntheticOODDataset(seq_len=seq_len, pred_len=pl, flag='train', **tr)
        val_set = SyntheticOODDataset(seq_len=seq_len, pred_len=pl, flag='val', **tr)
        test_set = SyntheticOODDataset(seq_len=seq_len, pred_len=pl, flag='test', **te)
    else:
        csv_name = DATASET_CSV.get(ds)
        data_path = os.path.join(dataset_dir, csv_name)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"数据集缺失: {data_path}\n"
                f"(本机没有该csv是预期的 —— 请把本脚本随 03_experiments/CausalCIT/ "
                f"一起搬到有GPU且已放好 dataset/ 目录的机器上运行)")
        train_set = ETTDataset(data_path, seq_len=seq_len, pred_len=pl, flag='train')
        val_set = ETTDataset(data_path, seq_len=seq_len, pred_len=pl, flag='val')
        test_set = ETTDataset(data_path, seq_len=seq_len, pred_len=pl, flag='test')

    pin = device.startswith('cuda')
    train_loader = get_dataloader(train_set, batch_size=job['batch_size'], pin_memory=pin)
    val_loader = get_dataloader(val_set, batch_size=job['batch_size'], shuffle=False, pin_memory=pin)
    test_loader = get_dataloader(test_set, batch_size=job['batch_size'], shuffle=False, pin_memory=pin)

    model = create_ablation_model(variant, **job['model_kwargs'])
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer = Trainer(model, device=device)
    save_dir = os.path.join('.', '_falsifiable_ckpt', f"{ds}_pl{pl}_{variant}_s{seed}")
    hist = trainer.train(train_loader, val_loader, epochs=ep, lr=0.001,
                         patience=job['patience'], save_dir=save_dir,
                         entropy_weight=entropy_weight, amp=amp)
    res = trainer.test(test_loader)
    return dict(mse=float(res['mse']), mae=float(res['mae']), rmse=float(res['rmse']),
               params=params, epochs=hist['epochs_trained'],
               model=trainer.model, test_set=test_set, batch_size=job['batch_size'])


# ============================================================
# 门控行为证据链
# ============================================================
def gate_collapse_check(model, test_set, device, batch_size=32, max_batches=5):
    """off_mean/off_std: 门控是否坍缩成常数。不需要因果ground truth，适用于任何数据集。"""
    if not hasattr(model, 'get_gate_matrix'):
        return None
    model.eval()
    loader = get_dataloader(test_set, batch_size=batch_size, shuffle=False)
    vals = []
    with torch.no_grad():
        for i, (bx, _by) in enumerate(loader):
            if i >= max_batches:
                break
            bx = bx.to(device)
            try:
                model(bx)
                gm = model.get_gate_matrix()
            except Exception:
                return None
            if gm is None:
                return None
            g = gm[0].detach().cpu().numpy()
            n = g.shape[0]
            off = g[~np.eye(n, dtype=bool)]
            vals.append(off)
    if not vals:
        return None
    allv = np.concatenate(vals)
    return dict(off_mean=float(allv.mean()), off_std=float(allv.std()),
               off_min=float(allv.min()), off_max=float(allv.max()),
               collapsed=bool(allv.std() < 1e-4))


def batch_invariance_check(model, test_set, device, batch_size=32,
                           n_targets=5, n_trials=3, seed=0):
    """
    直接检验评审 re2 §2.2 的 bug 是否被修复: 同一个测试样本，换不同的
    "batch同伴"，看门控矩阵是否发生变化。
      - running_stats=False (旧 full_v2): 预期 batch_dep_score 明显 > 0 (bug存在)
      - running_stats=True  (full_v2_fixed) + eval(): 预期 batch_dep_score ≈ 0
      - capacity_match/gate_prior_only/no_env 本身不做 batch 池化，预期天然 ≈ 0，
        可作为"无bug"的对照基准，帮助判断 full_v2 的分数是否真的异常。
    """
    if not hasattr(model, 'get_gate_matrix'):
        return None
    model.eval()
    n = len(test_set)
    if n < batch_size + 1:
        return None
    rng = np.random.RandomState(seed)
    target_idxs = rng.choice(n, size=min(n_targets, n), replace=False)
    devs = []
    with torch.no_grad():
        for ti in target_idxs:
            x0, _ = test_set[int(ti)]
            x0 = x0.unsqueeze(0)
            gates = []
            for _trial in range(n_trials):
                pool = [i for i in range(n) if i != ti]
                others_idx = rng.choice(pool, size=min(batch_size - 1, len(pool)),
                                        replace=False)
                xs = [x0] + [test_set[int(i)][0].unsqueeze(0) for i in others_idx]
                batch = torch.cat(xs, dim=0).to(device)
                try:
                    model(batch)
                    gm = model.get_gate_matrix()
                except Exception:
                    return None
                if gm is None:
                    return None
                gates.append(gm[0].detach().cpu().numpy())
            for a in range(len(gates)):
                for b in range(a + 1, len(gates)):
                    denom = (np.mean(np.abs(gates[a])) + np.mean(np.abs(gates[b]))) / 2 + 1e-8
                    devs.append(float(np.mean(np.abs(gates[a] - gates[b])) / denom))
    if not devs:
        return None
    return dict(batch_dep_score_mean=float(np.mean(devs)),
               batch_dep_score_max=float(np.max(devs)),
               n_targets=int(len(target_idxs)), n_trials=n_trials)


# ============================================================
# 主流程: 训练所有 (variant, seed, pred_len) 并汇总成一份对齐报告
# ============================================================
def run(args):
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    cfg = dataset_config(args.dataset)
    pls = args.pred_lens or cfg['pred_lens']
    if args.quick:
        pls = pls[:1]
        args.seeds = args.seeds[:2]
        args.epochs = 2
        args.skip_batch_invariance = False  # smoke test 也要跑一下诊断逻辑

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []          # MSE 结果
    gate_rows = []      # 门控诊断结果 (仅门控相关变体)
    n_total = len(pls) * len(args.variants) * len(args.seeds)
    done = 0
    t0 = time.time()

    for pl in pls:
        for variant in args.variants:
            for seed in args.seeds:
                done += 1
                print(f"[{done}/{n_total}] {args.dataset} pl={pl} variant={variant} "
                     f"seed={seed} ...", flush=True)
                try:
                    out = train_one(args.dataset, pl, variant, seed, dataset_dir,
                                    args.device, epochs=args.epochs,
                                    entropy_weight=args.entropy_weight,
                                    amp=args.amp)
                except Exception as e:
                    print(f"  [err] {e!r}")
                    rows.append(dict(dataset=args.dataset, pred_len=pl, variant=variant,
                                     seed=seed, mse=float('nan'), mae=float('nan'),
                                     rmse=float('nan'), error=repr(e)))
                    continue
                rows.append(dict(dataset=args.dataset, pred_len=pl, variant=variant,
                                 seed=seed, mse=out['mse'], mae=out['mae'],
                                 rmse=out['rmse'], params=out['params'],
                                 epochs=out['epochs']))
                print(f"  mse={out['mse']:.6f} mae={out['mae']:.6f} "
                     f"params={out['params']} epochs={out['epochs']}")

                # 门控行为证据链: 只对有门控的变体做，patchtst 没有门控自动跳过
                gc = gate_collapse_check(out['model'], out['test_set'], args.device,
                                         batch_size=min(out['batch_size'], 32))
                bd = None
                if not args.skip_batch_invariance:
                    bd = batch_invariance_check(out['model'], out['test_set'], args.device,
                                                batch_size=max(out['batch_size'], 8),
                                                seed=seed)
                if gc is not None or bd is not None:
                    grow = dict(dataset=args.dataset, pred_len=pl, variant=variant, seed=seed)
                    if gc is not None:
                        grow.update(gc)
                    if bd is not None:
                        grow.update(bd)
                    gate_rows.append(grow)

                # 增量保存，防止长任务中断丢结果
                with open(os.path.join(args.output_dir, 'mse_results.json'), 'w') as f:
                    json.dump(rows, f, indent=2)
                with open(os.path.join(args.output_dir, 'gate_diagnostics.json'), 'w') as f:
                    json.dump(gate_rows, f, indent=2)

    print(f"全部完成, 用时 {time.time() - t0:.0f}s")
    write_report(args, pls, rows, gate_rows)


def write_report(args, pls, rows, gate_rows):
    import pandas as pd
    df = pd.DataFrame([r for r in rows if not np.isnan(r.get('mse', float('nan')))])
    lines = ["# 最小可证伪测试报告 (回应评审 re2 §6.1)", ""]
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 数据集: {args.dataset} | 变体: {args.variants} | seed: {args.seeds}")
    lines.append("")
    lines.append("说明: 本报告在同一份文件里对齐两条证据链 —— (A) MSE 是否有提升, "
                "(B) 提升是否伴随门控行为的合理变化 (非坍缩、不依赖测试batch组成)。"
                "只有当 full_v2 同时满足 (A) 显著优于 capacity_match/gate_prior_only "
                "且 (B) 门控未坍缩、batch_dep_score 与 full_v2_fixed 差异可解释时，"
                "才能支持'因果稳定性门控'这个说法；否则应把结论降级。")
    lines.append("")

    if df.empty:
        lines.append("**没有任何成功完成的run，无法生成MSE表格 —— 请检查上面打印的 [err] 信息。**")
    else:
        for pl in pls:
            sub = df[df.pred_len == pl]
            if sub.empty:
                continue
            lines.append(f"## pred_len = {pl}")
            lines.append("")
            lines.append("### (A) MSE 证据链: vs PatchTST")
            lines.append("")
            lines.append("| 变体 | MSE mean | MSE std | #seed | 提升%(vs PatchTST) | Wilcoxon p | Holm p | 显著 |")
            lines.append("|------|---------|---------|-------|--------------------|-----------|--------|------|")
            base = sub[sub.variant == 'patchtst']
            base_m = base['mse'].mean() if not base.empty else None
            row_cache = []
            for v in args.variants:
                sv = sub[sub.variant == v]
                if sv.empty:
                    continue
                mse_m, mse_s = _mean_std(sv['mse'].tolist())
                n_seed = sv['seed'].nunique()
                imp_str = (f"{(base_m - mse_m) / base_m * 100:+.2f}%"
                          if base_m and v != 'patchtst' else "-")
                p_w = float('nan')
                if v != 'patchtst' and not base.empty:
                    p_w, _ = _wilcoxon_paired(base, sv)
                row_cache.append([v, mse_m, mse_s, n_seed, imp_str, p_w])
            holm = _holm_adjust([r[5] for r in row_cache])
            for r, ph in zip(row_cache, holm):
                v, mse_m, mse_s, n_seed, imp_str, p_w = r
                p_str = f"{p_w:.4f}" if not np.isnan(p_w) else "-"
                ph_str = f"{ph:.4f}" if not np.isnan(ph) else "-"
                sig = "*" if (not np.isnan(ph) and ph < 0.05) else ""
                lines.append(f"| {v} | {mse_m:.6f} | {mse_s:.6f} | {n_seed} | "
                            f"{imp_str} | {p_str} | {ph_str} | {sig} |")
            lines.append("")

            lines.append("### (A') MSE 证据链: full_v2 vs 关键对照 (非 vs-PatchTST)")
            lines.append("")
            lines.append("| 对照变体 | full_v2 MSE mean | 对照 MSE mean | full_v2提升% | #seed | Wilcoxon p | Holm p | 显著 |")
            lines.append("|---------|-------------------|---------------|-------------|-------|-----------|--------|------|")
            fv = sub[sub.variant == 'full_v2']
            key_targets = [v for v in args.variants
                          if v not in ('full_v2', 'patchtst') and not sub[sub.variant == v].empty]
            if fv.empty or not key_targets:
                lines.append("| (跳过：full_v2 或关键对照缺数据) | - | - | - | - | - | - | - |")
            else:
                kc = []
                for v in key_targets:
                    sv = sub[sub.variant == v]
                    p_w, n_pair = _wilcoxon_paired(sv, fv)
                    fvm, svm = fv['mse'].mean(), sv['mse'].mean()
                    imp = (svm - fvm) / svm * 100 if svm > 0 else float('nan')
                    kc.append([v, fvm, svm, imp, n_pair, p_w])
                holm_kc = _holm_adjust([r[5] for r in kc])
                for r, ph in zip(kc, holm_kc):
                    v, fvm, svm, imp, n_pair, p_w = r
                    p_str = f"{p_w:.4f}" if not np.isnan(p_w) else "-"
                    ph_str = f"{ph:.4f}" if not np.isnan(ph) else "-"
                    sig = "*" if (not np.isnan(ph) and ph < 0.05) else ""
                    lines.append(f"| {v} | {fvm:.6f} | {svm:.6f} | {imp:+.2f}% | {n_pair} | "
                                f"{p_str} | {ph_str} | {sig} |")
            lines.append("")

            # (B) 门控行为证据链
            gdf_rows = [g for g in gate_rows if g['dataset'] == args.dataset and g['pred_len'] == pl]
            if gdf_rows:
                gdf = pd.DataFrame(gdf_rows)
                lines.append("### (B) 门控行为证据链 (off_mean/off_std=坍缩检测; "
                            "batch_dep_score=测试时是否依赖batch组成，越接近0越好)")
                lines.append("")
                lines.append("| 变体 | off_mean | off_std | 坍缩? | batch_dep_score mean | batch_dep_score max | #seed |")
                lines.append("|------|---------|---------|-------|----------------------|---------------------|-------|")
                for v in args.variants:
                    gv = gdf[gdf.variant == v]
                    if gv.empty:
                        continue
                    off_m = gv['off_mean'].mean() if 'off_mean' in gv else float('nan')
                    off_s = gv['off_std'].mean() if 'off_std' in gv else float('nan')
                    collapsed = bool(gv['collapsed'].any()) if 'collapsed' in gv else False
                    bd_m = gv['batch_dep_score_mean'].mean() if 'batch_dep_score_mean' in gv else float('nan')
                    bd_x = gv['batch_dep_score_max'].max() if 'batch_dep_score_max' in gv else float('nan')
                    n_seed = gv['seed'].nunique()
                    lines.append(f"| {v} | {off_m:.4f} | {off_s:.4f} | "
                                f"{'是' if collapsed else '否'} | {bd_m:.4f} | {bd_x:.4f} | {n_seed} |")
                lines.append("")
            else:
                lines.append("### (B) 门控行为证据链")
                lines.append("")
                lines.append("(无门控诊断数据，可能所有变体训练均失败或未启用诊断)")
                lines.append("")

    out_path = os.path.join(args.output_dir, 'minimal_falsifiable_report.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"报告已保存: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='traffic',
                   help="默认 traffic (评审建议的最小可证伪数据集); "
                        "没有GPU时可用 'syn_ood' 做本地sanity check(内置合成数据,不需要csv)")
    p.add_argument('--dataset_dir', default=None)
    p.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS)
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--pred_lens', nargs='+', type=int, default=None,
                   help='默认用该数据集的标准 pred_lens (dataset_config)')
    p.add_argument('--epochs', type=int, default=None,
                   help='默认用该数据集的标准 epochs; 传入以覆盖(如 --quick sanity check)')
    p.add_argument('--entropy_weight', type=float, default=0.0,
                   help='门控熵正则化系数(回应§2.3), 默认0=不启用')
    p.add_argument('--amp', action='store_true',
                   help='混合精度训练(仅CUDA生效), 通常提速1.5-2x; HSIC/门控仍走fp32保精度')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output_dir', default='./output_falsifiable')
    p.add_argument('--skip_batch_invariance', action='store_true',
                   help='batch不变性检验较慢(每个run额外做 n_targets*n_trials 次前向), '
                        '大数据集/GPU紧张时可先跳过，只看MSE+坍缩检测')
    p.add_argument('--quick', action='store_true',
                   help='本地sanity check: 只跑1个pred_len/2个seed/2个epoch, '
                        '验证代码没有bug (配合 --dataset syn_ood --device cpu)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
