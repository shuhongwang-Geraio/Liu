"""
从已训好的 checkpoint 提取门控矩阵 (eval, 不重训) —— 供高维聚类热图 / 边箱线图使用。

背景 (plot_visualization_README.md §3):
  run_large 默认只在 n_vars<=21 时 dump 门控矩阵, traffic(862)/electricity(321)
  的高维矩阵从未保存。本脚本直接加载 checkpoint + test 数据前向一次, 取
  model.get_gate_matrix() 保存为 gates/ 下的 npy, 供 plot_gate_heatmaps.py /
  plot_gate_edge_boxplot.py 消费。

用法:
  # 有 job 文件时 (推荐): 用 job 里的 model_kwargs 精确重建模型 (含敏感性参数)
  python dump_gates_eval.py --ckpt_dir ./output_large_v3/ckpt \
      --job_file ./output_large_v3/jobs_shard0.json --dataset_dir <csv目录> \
      --output ./output_large_v3/gates

  # 无 job 文件: 用标准参数重建 (dataset_config 默认 + full_v2 超参)
  python dump_gates_eval.py --ckpt_dir ./output_falsifiable_ckpt \
      --output ./gates_eval

说明:
  - checkpoint 目录名必须为 {ds}_pl{pl}_{variant}_s{seed}/checkpoint.pth
    (与 run_large._train_one/_train_syn_ood 的 save_dir 一致)。
  - 真实数据集需要 --dataset_dir (csv); syn_ood/syn_ood_noise 用内置合成数据。
"""

import os
import glob
import json
import argparse
import numpy as np
import torch

import run_large as rl  # 复用 dataset_config / seq_for_pl / FULL_V2_KWARGS (并设置 sys.path)

from utils.data import ETTDataset, SyntheticOODDataset, get_dataloader
from models_ablation import create_ablation_model


def parse_ckpt_name(name):
    """syn_ood_pl96_full_v2_s42 -> (ds, pl, variant, seed)"""
    ds = name.split('_pl')[0]
    rest = name.split('_pl', 1)[1]
    pl = int(rest.split('_')[0])
    vseed = rest.split('_', 1)[1]
    variant, seed = vseed.rsplit('_s', 1)
    return ds, pl, variant, int(seed)


def build_test_loader(ds, pl, seed, dataset_dir):
    cfg = rl.dataset_config(ds)
    seq_len = rl.seq_for_pl(pl)
    if ds in ('syn_ood', 'syn_ood_noise'):
        te = dict(regime='test', seed=seed,
                  spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                  test_spurious_strengths=(0.05, -0.2, 0.1, -0.05),
                  train_noise=0.05, test_noise=0.05)
        test_set = SyntheticOODDataset(seq_len=seq_len, pred_len=pl, flag='test', **te)
    else:
        csv_name = rl.DATASET_CSV.get(ds)
        if csv_name is None or dataset_dir is None or not os.path.exists(
                os.path.join(dataset_dir, csv_name)):
            raise FileNotFoundError(
                f"{ds}: 需要 --dataset_dir 下的 {csv_name} (真实数据集), 本地若无 csv 无法 eval")
        data_path = os.path.join(dataset_dir, csv_name)
        test_set = ETTDataset(data_path, seq_len=seq_len, pred_len=pl, flag='test')
    return get_dataloader(test_set, batch_size=cfg['batch_size'], shuffle=False,
                          pin_memory=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', required=True, help='含 */checkpoint.pth 的目录')
    p.add_argument('--output', default=None, help='npy 输出目录 (默认 {ckpt_dir}/../gates_eval)')
    p.add_argument('--job_file', default=None,
                   help='run_large gen 的 job 文件, 提供精确 model_kwargs (推荐)')
    p.add_argument('--dataset_dir', default=None, help='真实数据集 csv 目录 (syn_ood 可省)')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    # 加载 job 文件 (若提供) -> {(ds, pl, variant, seed): model_kwargs}
    jobs = []
    if args.job_file:
        with open(args.job_file) as f:
            for sh in json.load(f):
                jobs.extend(sh if isinstance(sh, list) else [sh])
    job_map = {(j['dataset'], j['pred_len'], j['variant'], j['seed']): j
               for j in jobs}

    out_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt_dir)), 'gates_eval')
    os.makedirs(out_dir, exist_ok=True)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, '*/checkpoint.pth')))
    if not ckpts:
        print(f"警告: {args.ckpt_dir} 下没有 */checkpoint.pth")
        return
    print(f"共 {len(ckpts)} 个 checkpoint, 输出到 {out_dir}")

    n_saved = 0
    for ckpt in ckpts:
        name = os.path.basename(os.path.dirname(ckpt))
        try:
            ds, pl, variant, seed = parse_ckpt_name(name)
        except Exception as e:
            print(f"  跳过 {name}: 目录名无法解析 ({e})")
            continue
        cfg = rl.dataset_config(ds)
        seq_len = rl.seq_for_pl(pl)
        job = job_map.get((ds, pl, variant, seed))
        if job is not None:
            kw = dict(job['model_kwargs'])  # 精确重建 (含敏感性覆盖)
            kw.setdefault('alpha_init', -2.0)
        else:
            kw = dict(enc_in=cfg['n_vars'], seq_len=seq_len, pred_len=pl,
                      e_layers=3, n_heads=4, d_model=cfg['d_model'],
                      d_ff=cfg['d_ff'], dropout=0.2, fc_dropout=0.2,
                      patch_len=16, stride=8, padding_patch='end',
                      n_channel_heads=4, n_envs=4, rff_dim=32,
                      channel_dropout=0.1, fusion_alpha=0.3,
                      prior_weight=0.05, temperature=0.5, alpha_init=-2.0)
        try:
            model = create_ablation_model(variant, **kw)
            sd = torch.load(ckpt, map_location=args.device, weights_only=True)
            model.load_state_dict(sd)
            model.to(args.device).eval()
        except Exception as e:
            print(f"  跳过 {name}: 模型重建/加载失败 ({e})")
            continue
        if not hasattr(model, 'get_gate_matrix'):
            print(f"  跳过 {name}: 变体 {variant} 无门控矩阵")
            continue
        try:
            loader = build_test_loader(ds, pl, seed, args.dataset_dir)
            with torch.no_grad():
                gm = None
                for xb, _yb in loader:
                    _ = model(xb.to(args.device))
                    gm = model.get_gate_matrix()
                    if gm is not None:
                        break
            if gm is None:
                print(f"  跳过 {name}: get_gate_matrix() 返回 None")
                continue
            npy = os.path.join(out_dir, f'gate_{ds}_pl{pl}_{variant}_s{seed}.npy')
            np.save(npy, gm.detach().cpu().numpy())
            print(f"  保存 {os.path.basename(npy)}  {tuple(gm.shape)}")
            n_saved += 1
        except Exception as e:
            print(f"  跳过 {name}: {e}")
    print(f"完成, 共保存 {n_saved} 个门控矩阵")


if __name__ == '__main__':
    main()
