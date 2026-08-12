"""
CausalCIT 大规模并行实验 (full_v2 正式对标)

跨 多数据集 × 多horizon × 多变体 × 多seed 的系统性实验，利用多 GPU 并行。

设计要点 (踩坑后修正):
  * 每个 shard 进程显式限制 OpenMP/torch 线程数 (CIT_THREADS, 默认 8),
    避免多进程各占满 32 核导致 GPU 喂数据主线程被饿死 (GPU 空转、进程 CPU 自旋)。
  * 单个 job 在独立 spawn 子进程中执行, 由 --job_timeout 秒超时保护:
    卡死/超时被强杀后记录错误并继续下一个 job (断点续跑不受影响)。
  * gen 贪心装箱保证各 shard 总代价均衡; 每个 shard 内部按代价升序排列,
    让快的数据集 (weather/ETT) 先出结果。

用法:
  # 1) 生成 3 个 shard (默认 datasets: weather/etth1/ettm1/electricity)
  python run_large.py gen --num_shards 3 --output_dir ./output_large

  # 2) 在 3 张 GPU 上并行跑 (见 run_large.sh)
  CUDA_VISIBLE_DEVICES=0 CIT_THREADS=8 python -u run_large.py run --device cuda:0 \
      --job_file ./output_large/jobs_shard0.json --result_csv ./output_large/results_shard0.csv &

  # 3) 汇总
  python run_large.py summarize --output_dir ./output_large
"""

import os

# ---- 必须在 import torch 之前设置线程数, 否则 OpenMP 不生效 ----
_CIT_THREADS = os.environ.get('CIT_THREADS', '8')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, _CIT_THREADS)

import sys
import json
import time
import csv
import argparse
import multiprocessing as mp
import numpy as np
import torch
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEMO_DIR = os.path.join(PROJECT_DIR, 'CausalCIT_demo')

sys.path.insert(0, DEMO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.data import ETTDataset, SyntheticOODDataset, TemporalOODDataset, get_dataloader
from utils.trainer import Trainer
from models_ablation import create_ablation_model

# 解析线程数 (供 worker 内 set_num_threads 使用)
try:
    THREADS = int(_CIT_THREADS)
except ValueError:
    THREADS = 8

DATASET_CSV = {
    'weather': 'weather.csv',
    'etth1': 'ETTh1.csv',
    'ettm1': 'ETTm1.csv',
    'electricity': 'electricity.csv',
    'traffic': 'traffic.csv',
    'exchange': 'exchange_rate.csv',
    'ili': 'ILI.csv',
    # 时序漂移 OOD 变体 (复用基础 csv)
    'traffic_ood': 'traffic.csv',
    'electricity_ood': 'electricity.csv',
    'weather_ood': 'weather.csv',
}

_DEFAULT_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(PROJECT_DIR)), '01_external', 'PatchTST', 'code', 'dataset'),
    os.path.join(PROJECT_DIR, 'patchtst', 'dataset'),
    os.path.join(PROJECT_DIR, 'data'),
]


def resolve_dataset_dir(explicit=None):
    if explicit:
        return explicit
    for p in _DEFAULT_PATHS:
        if os.path.isdir(p) and os.listdir(p):
            return p
    return _DEFAULT_PATHS[0]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据集配置: 维度 / 模型规模 / batch / 训练预算
# ============================================================
def dataset_config(name):
    C = {
        'weather':     dict(n_vars=21,  d_model=64, d_ff=256, batch_size=32, epochs=50, patience=8, pred_lens=[96, 192, 336]),
        'etth1':       dict(n_vars=7,   d_model=32, d_ff=128, batch_size=32, epochs=50, patience=8, pred_lens=[96, 192, 336]),
        'ettm1':       dict(n_vars=7,   d_model=32, d_ff=128, batch_size=32, epochs=50, patience=8, pred_lens=[96, 192, 336]),
        'electricity': dict(n_vars=321, d_model=32, d_ff=128, batch_size=16, epochs=30, patience=8, pred_lens=[96, 192]),
        'traffic':     dict(n_vars=862, d_model=16, d_ff=64,  batch_size=8,  epochs=30, patience=8, pred_lens=[96, 192]),
        'syn_ood':      dict(n_vars=7,   d_model=64, d_ff=256, batch_size=32, epochs=50, patience=8, pred_lens=[96, 192]),
        'syn_ood_noise':dict(n_vars=7,   d_model=64, d_ff=256, batch_size=32, epochs=50, patience=8, pred_lens=[96, 192]),
        # 真实数据集 OOD
        'exchange':     dict(n_vars=8,   d_model=32, d_ff=128, batch_size=32, epochs=50, patience=8, pred_lens=[96, 192]),
        'ili':          dict(n_vars=7,   d_model=32, d_ff=128, batch_size=32, epochs=50, patience=8, pred_lens=[24, 48]),
        # 现有数据时序漂移 OOD (复用基础csv, 显式拉开训练/测试时段)
        'traffic_ood':     dict(n_vars=862, d_model=16, d_ff=64,  batch_size=8,  epochs=30, patience=8, pred_lens=[96, 192]),
        'electricity_ood': dict(n_vars=321, d_model=32, d_ff=128, batch_size=16, epochs=30, patience=8, pred_lens=[96, 192]),
        'weather_ood':     dict(n_vars=21,  d_model=64, d_ff=256, batch_size=32, epochs=50, patience=8, pred_lens=[96, 192]),
    }
    if name not in C:
        raise ValueError(f"未知数据集 {name}; 可选: {list(C.keys())}")
    return C[name]


def seq_for_pl(pl):
    # PatchTST 标准协议: 短horizon用96, 长horizon用336 上下文
    return 96 if pl <= 192 else 336


# full_v2 的改进超参 (经诊断验证的默认值)
# 2026-08-11 门1静态诊断修复: rff_sigma_mode='median' + cka_normalize=True
# (见 causal_channel.py 修 A+B)。仅影响 full_v2/full_v2_fixed。
# 注意: P0-1 重跑后旧报告数字不可直接对比 (这是修复版协议)。
FULL_V2_KWARGS = dict(prior_weight=0.05, temperature=0.5,
                       alpha_init=-2.0,
                       rff_sigma_mode='median', cka_normalize=True)


def build_kwargs(ds, pl, variant, seed, dataset_dir, entropy_weight=0.0,
                 n_envs=None, rff_dim=None, prior_weight=None, temperature=None,
                 alpha_init=None, fusion_alpha=None,
                 env_mode=None, env_scheme=None, risk_lambda=0.0,
                 epochs=None, dump_gates=False):
    """构造单个 job。

    敏感性分析支持 (P1): 通过 n_envs/rff_dim/prior_weight/temperature 覆盖
    模型默认超参 (默认 None = 保持 build_kwargs 内的默认值, 不影响旧结果可复现性)。
    """
    cfg = dataset_config(ds)
    n_vars = cfg['n_vars']
    seq_len = seq_for_pl(pl)
    base = dict(
        enc_in=n_vars, seq_len=seq_len, pred_len=pl,
        e_layers=3, n_heads=4, d_model=cfg['d_model'], d_ff=cfg['d_ff'],
        dropout=0.2, fc_dropout=0.2,
        patch_len=16, stride=8, padding_patch='end',
        n_channel_heads=4, n_envs=4, rff_dim=32,
        channel_dropout=0.1, fusion_alpha=0.3,
    )
    if n_envs is not None:
        base['n_envs'] = n_envs
    if rff_dim is not None:
        base['rff_dim'] = rff_dim
    if fusion_alpha is not None:
        base['fusion_alpha'] = fusion_alpha
    job = dict(
        dataset=ds, pred_len=pl, variant=variant, seed=seed,
        n_vars=n_vars, epochs=epochs if epochs is not None else cfg['epochs'],
        patience=cfg['patience'],
        batch_size=cfg['batch_size'], dataset_dir=dataset_dir,
        model_kwargs=base,
        # 回应评审re2 §2.3/§6.1: entropy_weight 之前从未被传给 trainer.train()，
        # 一直是死代码。这里显式写进 job，由 _train_one/_train_syn_ood 传下去。
        # 默认 0.0，不影响旧结果的可复现性；测试该正则化效果时用 --entropy_weight>0。
        entropy_weight=entropy_weight,
        # P1 可视化: 默认只在 n_vars<=21 时 dump 门控矩阵; --dump_gates 强制
        # 高维 (traffic 862 / electricity 321) 也 dump, 供聚类热图/边箱线图使用。
        dump_gates=dump_gates,
        # 修 C / 想法 1 (2026-08-12): env_scheme 数据层语义切分 (None=旧行为);
        # risk_lambda>0 启用 DRO 式风险厌恶损失 (Trainer 层)。
        env_scheme=env_scheme,
        risk_lambda=risk_lambda,
    )
    # 变体专属超参: 敏感性分析时用显式传入值覆盖默认, 否则保持原默认
    if variant in ('full_v2', 'full_v2_fixed'):
        base.update(FULL_V2_KWARGS)
        if prior_weight is not None:
            base['prior_weight'] = prior_weight
        if temperature is not None:
            base['temperature'] = temperature
        if alpha_init is not None:
            base['alpha_init'] = alpha_init
        if env_mode is not None:
            base['env_mode'] = env_mode
    elif variant in ('learned_gate', 'capacity_match', 'gate_prior_only'):
        # capacity_match(答刀2): 同参数规模的纯学习通道注意力, 无因果稳定性逻辑
        # gate_prior_only(答刀1): 与 full_v2 同结构但剥离稳定性/HSIC 信号
        base.update(dict(prior_weight=0.05, alpha_init=-2.0))
        if prior_weight is not None:
            base['prior_weight'] = prior_weight
    elif variant == 'no_env':
        # 直击评审刀1 (原指控: full vs w/o EnvSplit 仅差 0.04%):
        # 与 full_v2 同结构/同 prior_weight, 但全局HSIC(不划分环境).
        # 只传 prior_weight —— NoEnv_ChannelInteraction.__init__ 不接受
        # temperature/alpha_init 等 full_v2 专属参数, 传了会报非法 kwarg.
        base.update(dict(prior_weight=0.05))
        if prior_weight is not None:
            base['prior_weight'] = prior_weight
    return job


# ============================================================
# gen: 生成 jobs + 贪心装箱
# ============================================================
def est_cost(job):
    cfg = dataset_config(job['dataset'])
    # 粗略代价: epochs × n_vars × 数据规模因子
    factor = {'weather': 1.0, 'etth1': 0.4, 'ettm1': 0.4,
              'electricity': 1.6, 'traffic': 2.2,
              'syn_ood': 0.3, 'syn_ood_noise': 0.3,
              'exchange': 0.4, 'ili': 0.4,
              'traffic_ood': 2.2, 'electricity_ood': 1.6, 'weather_ood': 1.0}.get(job['dataset'], 1.0)
    return job['epochs'] * job['n_vars'] * factor


def gen_jobs(args):
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    variants = args.variants
    seeds = [int(s) for s in args.seeds]
    ew = getattr(args, 'entropy_weight', 0.0)
    # P1 敏感性分析: 覆盖超参 (None = 用模型默认)
    n_envs = getattr(args, 'n_envs', None)
    rff_dim = getattr(args, 'rff_dim', None)
    p_weight = getattr(args, 'prior_weight', None)
    temp = getattr(args, 'temperature', None)
    a_init = getattr(args, 'alpha_init', None)
    f_alpha = getattr(args, 'fusion_alpha', None)
    env_mode = getattr(args, 'env_mode', None)
    env_scheme = getattr(args, 'env_scheme', None)
    risk_lambda = getattr(args, 'risk_lambda', 0.0)
    epochs = getattr(args, 'epochs', None)
    dump_gates = getattr(args, 'dump_gates', False)
    jobs = []
    for ds in args.datasets:
        cfg = dataset_config(ds)
        for pl in cfg['pred_lens']:
            for v in variants:
                for s in seeds:
                    jobs.append(build_kwargs(ds, pl, v, s, dataset_dir, entropy_weight=ew,
                                             n_envs=n_envs, rff_dim=rff_dim,
                                             prior_weight=p_weight, temperature=temp,
                                             alpha_init=a_init, fusion_alpha=f_alpha,
                                             env_mode=env_mode, env_scheme=env_scheme,
                                             risk_lambda=risk_lambda,
                                             epochs=epochs, dump_gates=dump_gates))
    # 贪心装箱: 按代价降序分配到当前总代价最小的 shard (保证各 shard 总代价均衡)
    jobs_sorted = sorted(jobs, key=est_cost, reverse=True)
    shards = [[] for _ in range(args.num_shards)]
    shard_cost = [0.0] * args.num_shards
    for j in jobs_sorted:
        i = int(np.argmin(shard_cost))
        shards[i].append(j)
        shard_cost[i] += est_cost(j)
    # 每个 shard 内部按代价升序排列 -> 快的数据集先出结果
    for sh in shards:
        sh.sort(key=est_cost)
    os.makedirs(args.output_dir, exist_ok=True)
    for i, sh in enumerate(shards):
        path = os.path.join(args.output_dir, f'jobs_shard{i}.json')
        with open(path, 'w') as f:
            json.dump(sh, f)
        print(f"  shard {i}: {len(sh)} jobs, est_cost={shard_cost[i]:.0f}")
    print(f"总计 {len(jobs)} 个 job, 分配到 {args.num_shards} 个 shard")
    return shards


# ============================================================
# run: 执行单个 shard (单 job 超时保护)
# ============================================================
def _job_done_key(row):
    return (row['dataset'], int(row['pred_len']), row['variant'], int(row['seed']))


def _train_one(job, device, out_dir):
    # 限制本进程线程数, 避免多 shard 超订阅饿死 GPU 主线程
    torch.set_num_threads(THREADS)
    # 修复(2026-08-08): job 在 spawn 子进程中执行, 不继承主进程的随机状态。
    # 之前 set_seed 只写在主进程 run_jobs() 里, spawn 子进程里的训练实际用的是
    # 系统熵随机的初始化 —— seed 参数从未真正控制随机性, 破坏"同seed配对"的可复现性。
    # 这里必须在子进程内、每次训练前重新 set_seed, 配对比较才严格有效。
    set_seed(job['seed'])
    ds = job['dataset']
    pl = job['pred_len']
    variant = job['variant']
    cfg = dataset_config(ds)
    if ds in ('syn_ood', 'syn_ood_noise'):
        return _train_syn_ood(job, device, out_dir)
    csv_name = DATASET_CSV.get(ds)
    if csv_name is None or not os.path.exists(os.path.join(job['dataset_dir'], csv_name)):
        raise FileNotFoundError(f"数据集缺失: {os.path.join(job['dataset_dir'], csv_name)}")
    data_path = os.path.join(job['dataset_dir'], csv_name)
    seq_len = job['model_kwargs']['seq_len']

    # 修 C (2026-08-12): env_scheme 非 None 时 Dataset 返回 (x, y, env_label) 三元组,
    # 驱动语义环境切分 (season/daynight/tod); None = 旧行为二元组。
    env_scheme = job.get('env_scheme')
    if ds.endswith('_ood'):
        # 显式时序漂移: 训练=早期时段, 测试=晚期时段, 之间留 gap 最大化分布漂移
        ood_kwargs = dict(train_frac=0.5, val_frac=0.1, test_frac=0.25, gap_frac=0.15)
        def mk(f):
            return TemporalOODDataset(data_path, seq_len=seq_len, pred_len=pl,
                                      flag=f, env_scheme=env_scheme, **ood_kwargs)
    else:
        def mk(f):
            return ETTDataset(data_path, seq_len=seq_len, pred_len=pl, flag=f,
                              env_scheme=env_scheme)

    pin = device.startswith('cuda')   # GPU 下 pin_memory 加快 host->device 拷贝
    train_set = mk('train')
    val_set = mk('val')
    test_set = mk('test')
    train_loader = get_dataloader(train_set, batch_size=job['batch_size'], pin_memory=pin)
    val_loader = get_dataloader(val_set, batch_size=job['batch_size'], shuffle=False, pin_memory=pin)
    test_loader = get_dataloader(test_set, batch_size=job['batch_size'], shuffle=False, pin_memory=pin)

    model = create_ablation_model(variant, **job['model_kwargs'])
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer = Trainer(model, device=device)
    save_dir = os.path.join(out_dir, 'ckpt', f"{ds}_pl{pl}_{variant}_s{job['seed']}")
    hist = trainer.train(train_loader, val_loader, epochs=job['epochs'],
                         lr=0.001, patience=job['patience'], save_dir=save_dir,
                         entropy_weight=job.get('entropy_weight', 0.0),
                         amp=job.get('amp', False),
                         risk_lambda=job.get('risk_lambda', 0.0))
    res = trainer.test(test_loader)
    # 保存门控矩阵 (用于分析): 默认仅低维; --dump_gates 强制高维 (traffic/electricity)
    try:
        if hasattr(model, 'get_gate_matrix') and (cfg['n_vars'] <= 21 or job.get('dump_gates')) and variant in ('full_v2', 'full_v2_fixed'):
            gm = model.get_gate_matrix()
            if gm is not None:
                gdir = os.path.join(out_dir, 'gates')
                os.makedirs(gdir, exist_ok=True)
                np.save(os.path.join(gdir, f'gate_{ds}_pl{pl}_s{job["seed"]}.npy'),
                        gm.detach().cpu().numpy())
    except Exception:
        pass
    return dict(mse=float(res['mse']), mae=float(res['mae']), rmse=float(res['rmse']),
                params=params, epochs=hist['epochs_trained'])


def _train_syn_ood(job, device, out_dir):
    """合成 OOD 训练: train/val 用 regime='train', test 用 regime='test' (漂移)."""
    torch.set_num_threads(THREADS)
    set_seed(job['seed'])  # 同 _train_one: spawn 子进程内必须重设种子
    ds = job['dataset']
    pl = job['pred_len']
    variant = job['variant']
    cfg = dataset_config(ds)
    seq_len = job['model_kwargs']['seq_len']
    seed = job['seed']

    if ds == 'syn_ood':
        # 机制测试: 虚假通道强度跨环境变化, 训练->测试漂移(弱化/反转)
        tr = dict(regime='train', seed=seed,
                  spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                  test_spurious_strengths=(0.05, -0.2, 0.1, -0.05),
                  train_noise=0.05, test_noise=0.05)
        te = dict(regime='test', seed=seed,
                  spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                  test_spurious_strengths=(0.05, -0.2, 0.1, -0.05),
                  train_noise=0.05, test_noise=0.05)
    else:  # syn_ood_noise: 仅噪声水平漂移 (协变量漂移鲁棒性)
        tr = dict(regime='train', seed=seed,
                  spurious_strengths=(0.6, 0.6, 0.6, 0.6),
                  test_spurious_strengths=(0.6, 0.6, 0.6, 0.6),
                  train_noise=0.05, test_noise=0.6)
        te = dict(regime='test', seed=seed,
                  spurious_strengths=(0.6, 0.6, 0.6, 0.6),
                  test_spurious_strengths=(0.6, 0.6, 0.6, 0.6),
                  train_noise=0.05, test_noise=0.6)

    train_set = SyntheticOODDataset(seq_len=seq_len, pred_len=pl, flag='train', **tr)
    val_set = SyntheticOODDataset(seq_len=seq_len, pred_len=pl, flag='val', **tr)
    test_set = SyntheticOODDataset(seq_len=seq_len, pred_len=pl, flag='test', **te)
    pin = device.startswith('cuda')
    train_loader = get_dataloader(train_set, batch_size=job['batch_size'], pin_memory=pin)
    val_loader = get_dataloader(val_set, batch_size=job['batch_size'], shuffle=False, pin_memory=pin)
    test_loader = get_dataloader(test_set, batch_size=job['batch_size'], shuffle=False, pin_memory=pin)

    model = create_ablation_model(variant, **job['model_kwargs'])
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer = Trainer(model, device=device)
    save_dir = os.path.join(out_dir, 'ckpt', f"{ds}_pl{pl}_{variant}_s{seed}")
    hist = trainer.train(train_loader, val_loader, epochs=job['epochs'],
                         lr=0.001, patience=job['patience'], save_dir=save_dir,
                         entropy_weight=job.get('entropy_weight', 0.0),
                         amp=job.get('amp', False),
                         risk_lambda=job.get('risk_lambda', 0.0))
    res = trainer.test(test_loader)
    try:
        if hasattr(model, 'get_gate_matrix') and (cfg['n_vars'] <= 21 or job.get('dump_gates')) and variant in ('full_v2', 'full_v2_fixed', 'learned_gate', 'gate_prior_only', 'capacity_match', 'no_env'):
            gm = model.get_gate_matrix()
            if gm is not None:
                gdir = os.path.join(out_dir, 'gates')
                os.makedirs(gdir, exist_ok=True)
                np.save(os.path.join(gdir, f'gate_{ds}_pl{pl}_{variant}_s{seed}.npy'),
                        gm.detach().cpu().numpy())
    except Exception:
        pass
    return dict(mse=float(res['mse']), mae=float(res['mae']), rmse=float(res['rmse']),
                params=params, epochs=hist['epochs_trained'])


def _worker(job, device, out_dir, q):
    """spawn 子进程入口 (必须是模块级函数才能被 pickle)。"""
    try:
        res = _train_one(job, device, out_dir)
        q.put(('ok', res))
    except Exception as e:
        q.put(('err', repr(e)))


def _run_job_with_timeout(job, device, out_dir, timeout):
    """在独立 spawn 子进程中跑单个 job; 超时则强杀并返回 ('timeout', None)。"""
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(job, device, out_dir, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        try:
            p.join(5)
        except Exception:
            pass
        return ('timeout', None)
    if not q.empty():
        return q.get()
    return ('err', 'no-result')


def run_jobs(args):
    device = args.device
    timeout = args.job_timeout
    with open(args.job_file) as f:
        jobs = json.load(f)
    done = set()
    if os.path.exists(args.result_csv):
        with open(args.result_csv) as f:
            for row in csv.DictReader(f):
                done.add(_job_done_key(row))
    out_dir = os.path.dirname(os.path.abspath(args.result_csv))
    os.makedirs(out_dir, exist_ok=True)
    err_path = os.path.join(out_dir, 'errors_' + os.path.basename(args.result_csv).replace('results_', ''))
    ferr = open(err_path, 'a')
    fout = open(args.result_csv, 'a', newline='')
    writer = csv.writer(fout)
    if not done:
        writer.writerow(['dataset', 'pred_len', 'variant', 'seed',
                         'mse', 'mae', 'rmse', 'params', 'epochs', 'time'])

    total = len(jobs)
    finished = len(done)
    for idx, job in enumerate(jobs):
        key = (job['dataset'], int(job['pred_len']), job['variant'], int(job['seed']))
        tag = f"{job['dataset']} pl{job['pred_len']} {job['variant']} s{job['seed']}"
        if key in done:
            print(f"  [skip] {tag} (已完成)", flush=True)
            continue
        job['amp'] = bool(getattr(args, 'amp', False))  # 透传到 spawn 子进程内
        set_seed(job['seed'])
        t0 = time.time()
        status, payload = _run_job_with_timeout(job, device, out_dir, timeout)
        dt = time.time() - t0
        if status == 'ok':
            res = payload
            writer.writerow([job['dataset'], job['pred_len'], job['variant'], job['seed'],
                             f"{res['mse']:.6f}", f"{res['mae']:.6f}", f"{res['rmse']:.6f}",
                             res['params'], res['epochs'], f"{dt:.1f}"])
            fout.flush()
            finished += 1
            print(f"  [{finished}/{total}] {tag} -> MSE={res['mse']:.5f} "
                  f"MAE={res['mae']:.5f} ({dt:.0f}s)", flush=True)
        else:
            msg = payload if status == 'err' else f'timeout>{timeout}s'
            print(f"  [FAIL] {tag} -> {status}: {msg} ({dt:.0f}s)", flush=True)
            ferr.write(f"{time.strftime('%H:%M:%S')} {tag} {status}: {msg}\n")
            ferr.flush()
    fout.close()
    ferr.close()
    print(f"shard 完成: {args.job_file} ({finished}/{total} 个结果, 失败见 {err_path})", flush=True)


# ============================================================
# summarize: 聚合 + 报告 + 显著性检验
# ============================================================
def _mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def _paired_by_seed(base_df, var_df, col='mse'):
    """按 seed 对齐两组结果 (P0-4 修复: 原实现按行序配对, seed 顺序不同即错配).
    重复 seed 取均值; 返回 (base_vals, var_vals, common_seeds)."""
    b = base_df.groupby('seed')[col].mean()
    v = var_df.groupby('seed')[col].mean()
    common = sorted(set(b.index) & set(v.index))
    if not common:
        return None, None, []
    return b.loc[common].values, v.loc[common].values, common


def _wilcoxon_paired(base_df, var_df, col='mse'):
    """seed 配对 Wilcoxon 符号秩检验. 返回 (p, n_pairs); 失败返回 (nan, 0)."""
    try:
        from scipy import stats
    except ImportError:
        return float('nan'), 0
    b, v, seeds = _paired_by_seed(base_df, var_df, col)
    if b is None or len(seeds) < 5:
        return float('nan'), 0 if b is None else len(seeds)
    try:
        _, p = stats.wilcoxon(b, v)
        return float(p), len(seeds)
    except Exception:
        return float('nan'), len(seeds)


def _holm_adjust(pvals):
    """Holm-Bonferroni 校正. 输入原始 p 列表 (可含 nan), 返回同长度校正后 p."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = len(idx)
    adj = [float('nan')] * len(pvals)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: pvals[i])
    running_max = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running_max = max(running_max, val)  # 保证单调
        adj[i] = running_max
    return adj


def summarize(args):
    files = sorted([os.path.join(args.output_dir, f)
                    for f in os.listdir(args.output_dir) if f.startswith('results_shard') and f.endswith('.csv')])
    if not files:
        print("未找到 results_shard*.csv")
        return
    rows = []
    for fp in files:
        with open(fp) as f:
            for r in csv.DictReader(f):
                try:
                    r['pred_len'] = int(r['pred_len'])
                    r['seed'] = int(r['seed'])
                    r['mse'] = float(r['mse'])
                    r['mae'] = float(r['mae'])
                    r['rmse'] = float(r['rmse'])
                    rows.append(r)
                except (ValueError, KeyError):
                    continue  # 跳过解析失败的脏行
    if not rows:
        print("无有效结果行")
        return
    df = pd.DataFrame(rows)
    print(f"聚合 {len(df)} 行结果, 变体: {sorted(df.variant.unique())}")

    variants = sorted(df.variant.unique())
    datasets = sorted(df.dataset.unique())
    pls = sorted(df.pred_len.unique())

    lines = ["# CausalCIT 大规模实验报告 (full_v2 vs baselines)", ""]
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 数据集: {datasets}")
    lines.append(f"> 变体: {variants}")
    lines.append(f"> 每个 (数据集, horizon) 下跨 seed 报告 mean±std MSE/MAE，以及 full_v2 相对 PatchTST 的提升%")
    lines.append(f"> 显著性: seed 配对 Wilcoxon 符号秩检验 (双侧), 同组内 (同数据集×horizon) 跨变体 Holm 校正;")
    lines.append(f"> n<5 对 seed 时不报 p 值 (功效不足). 已弃用方向不可辨的 t-test 报法 (P0-4).")
    lines.append("")

    for ds in datasets:
        lines.append(f"## 数据集: {ds}")
        lines.append("")
        for pl in pls:
            sub = df[(df.dataset == ds) & (df.pred_len == pl)]
            if sub.empty:
                continue
            lines.append(f"### pred_len = {pl}")
            lines.append("")
            lines.append("| 变体 | MSE mean | MSE std | MAE mean | 提升% (vs PatchTST) | #seed | Wilcoxon p | Holm p | 显著 |")
            lines.append("|------|---------|---------|---------|-------------------|-------|-----------|--------|------|")
            base = sub[sub.variant == 'patchtst']
            base_m = base['mse'].mean() if not base.empty else None
            # 先收集本组所有变体的原始 p, 再做组内 Holm 校正
            row_cache = []
            for v in variants:
                sv = sub[sub.variant == v]
                if sv.empty:
                    continue
                mse_m, mse_s = _mean_std(sv['mse'].tolist())
                mae_m, _ = _mean_std(sv['mae'].tolist())
                n_seed = sv['seed'].nunique()
                if base_m is not None and base_m > 0 and v != 'patchtst':
                    imp_str = f"{(base_m - mse_m) / base_m * 100:+.2f}%"
                else:
                    imp_str = "-"
                p_w, n_pair = (float('nan'), 0)
                if v != 'patchtst' and not base.empty:
                    p_w, n_pair = _wilcoxon_paired(base, sv)
                row_cache.append([v, mse_m, mse_s, mae_m, imp_str, n_seed, p_w])
            holm = _holm_adjust([r[6] for r in row_cache])
            for r, ph in zip(row_cache, holm):
                v, mse_m, mse_s, mae_m, imp_str, n_seed, p_w = r
                p_str = f"{p_w:.4f}" if not np.isnan(p_w) else "-"
                ph_str = f"{ph:.4f}" if not np.isnan(ph) else "-"
                sig = "*" if (not np.isnan(ph) and ph < 0.05) else ""
                lines.append(f"| {v} | {mse_m:.6f} | {mse_s:.6f} | {mae_m:.6f} | "
                             f"{imp_str} | {n_seed} | {p_str} | {ph_str} | {sig} |")
            lines.append("")

            # 回应评审re2 §6.1第5条: 之前只做了 vs-PatchTST 的显著性检验，
            # capacity_match/gate_prior_only 这两个"关键对照"(证明提升不是单纯参数量
            # 或纯学习门控带来的)从未真正被检验过。这里单独起一个 Holm 校正族:
            # full_v2 vs {capacity_match, gate_prior_only, no_env, full_v2_fixed}。
            # 注意: 'learned_gate' 与 'capacity_match' 实现完全相同(见 models_ablation.py
            # 注释), 故不重复纳入此族, 避免虚增比较数/重复计数同一条证据。
            key_targets = [v for v in ('capacity_match', 'gate_prior_only', 'no_env', 'full_v2_fixed')
                          if v in variants and v != 'full_v2']
            fv = sub[sub.variant == 'full_v2']
            if not fv.empty and key_targets:
                lines.append("**关键对照显著性 (full_v2 vs 容量匹配/去因果信号对照, 非vs-PatchTST):**")
                lines.append("")
                lines.append("| 对照变体 | full_v2 MSE mean | 对照 MSE mean | full_v2提升% | #seed | Wilcoxon p | Holm p | 显著 |")
                lines.append("|---------|-------------------|---------------|-------------|-------|-----------|--------|------|")
                kc = []
                for v in key_targets:
                    sv = sub[sub.variant == v]
                    if sv.empty:
                        continue
                    p_w, n_pair = _wilcoxon_paired(sv, fv)  # base=对照, var=full_v2
                    fvm = fv['mse'].mean()
                    svm = sv['mse'].mean()
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

    lines.append("---")
    lines.append("")
    lines.append("## full_v2 提升率汇总 (vs PatchTST, seed 配对)")
    lines.append("")
    lines.append("| 数据集 | pred_len | 提升% mean | 提升% std | #seed | Wilcoxon p | Holm p | 显著 |")
    lines.append("|--------|----------|-----------|-----------|------|-----------|--------|------|")
    sum_rows = []
    for ds in datasets:
        for pl in pls:
            sub = df[(df.dataset == ds) & (df.pred_len == pl)]
            b = sub[sub.variant == 'patchtst']
            f = sub[sub.variant == 'full_v2']
            if b.empty or f.empty:
                continue
            bv, fv, seeds = _paired_by_seed(b, f)
            if bv is None:
                continue
            imps = [(bv[i] - fv[i]) / bv[i] * 100 for i in range(len(seeds))]
            im, isd = _mean_std(imps)
            p_w, _ = _wilcoxon_paired(b, f)
            sum_rows.append([ds, pl, im, isd, len(seeds), p_w])
    # Holm 按数据集分族 (每族 = 该数据集的各 horizon)。
    # 理由: n=8 seed 时 Wilcoxon 最小 p=0.0078, 若跨全部 数据集×horizon 全局校正
    # (m=8) 则最小校正 p=0.0625, 结构上不可能显著 —— 校正族应与假设范围一致
    # ("full_v2 在数据集 X 上优于基线" 是按数据集提出的假设)。
    holm_sum = [float('nan')] * len(sum_rows)
    for ds in {r[0] for r in sum_rows}:
        fam = [i for i, r in enumerate(sum_rows) if r[0] == ds]
        adj = _holm_adjust([sum_rows[i][5] for i in fam])
        for i, a in zip(fam, adj):
            holm_sum[i] = a
    for r, ph in zip(sum_rows, holm_sum):
        ds, pl, im, isd, n, p_w = r
        p_str = f"{p_w:.4f}" if not np.isnan(p_w) else "-"
        ph_str = f"{ph:.4f}" if not np.isnan(ph) else "-"
        sig = "*" if (not np.isnan(ph) and ph < 0.05) else ""
        lines.append(f"| {ds} | {pl} | {im:+.2f}% | {isd:.2f}% | {n} | {p_str} | {ph_str} | {sig} |")
    lines.append("")

    lines.append("## 平均提升率 (按数据集, 跨 horizon × seed, seed 配对)")
    lines.append("")
    lines.append("| 数据集 | full_v2 提升% mean | #runs |")
    lines.append("|--------|-------------------|-------|")
    for ds in datasets:
        sub = df[(df.dataset == ds)]
        b = sub[sub.variant == 'patchtst']
        f = sub[sub.variant == 'full_v2']
        if b.empty or f.empty:
            continue
        imps = []
        for (pl, sd), grp in f.groupby(['pred_len', 'seed']):
            bm = b[(b.pred_len == pl) & (b.seed == sd)]['mse']
            if not bm.empty:
                imps.append((bm.mean() - grp['mse'].mean()) / bm.mean() * 100)
        if imps:
            im, _ = _mean_std(imps)
            lines.append(f"| {ds} | {im:+.2f}% | {len(imps)} |")
    lines.append("")

    out_path = os.path.join(args.output_dir, 'large_scale_report.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"报告已保存: {out_path}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        mat = np.full((len(datasets), len(pls)), np.nan)
        for di, ds in enumerate(datasets):
            for pi, pl in enumerate(pls):
                sub = df[(df.dataset == ds) & (df.pred_len == pl)]
                b = sub[sub.variant == 'patchtst']
                f = sub[sub.variant == 'full_v2']
                if b.empty or f.empty:
                    continue
                bv, fv, seeds = _paired_by_seed(b, f)
                if bv is None:
                    continue
                mat[di, pi] = np.mean([(bv[i] - fv[i]) / bv[i] * 100
                                       for i in range(len(seeds))])
        plt.figure(figsize=(1.2 * len(pls) + 2, 0.8 * len(datasets) + 2))
        im = plt.imshow(mat, cmap='RdYlGn', vmin=-5, vmax=8, aspect='auto')
        plt.colorbar(im, label='Improvement vs PatchTST (%)')
        plt.xticks(range(len(pls)), [str(p) for p in pls])
        plt.yticks(range(len(datasets)), datasets)
        for di in range(len(datasets)):
            for pi in range(len(pls)):
                v = mat[di, pi]
                if not np.isnan(v):
                    plt.text(pi, di, f"{v:+.1f}", ha='center', va='center',
                             fontsize=9, color='black' if -2 < v < 5 else 'white')
        plt.xlabel('Prediction Length')
        plt.ylabel('Dataset')
        plt.title('CausalCIT full_v2 Improvement (%) over PatchTST')
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, 'improvement_heatmap.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"热图已保存: {fig_path}")
    except Exception as e:
        print(f"  [warn] 画图失败: {e}")


# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    g = sub.add_parser('gen')
    g.add_argument('--datasets', nargs='+',
                   default=['weather', 'etth1', 'ettm1', 'electricity'])
    g.add_argument('--variants', nargs='+', default=['patchtst', 'no_gate', 'full_v2'])
    g.add_argument('--seeds', nargs='+', default=['42', '123', '2024'])
    g.add_argument('--num_shards', type=int, default=3)
    g.add_argument('--output_dir', default='./output_large')
    g.add_argument('--dataset_dir', default=None)
    g.add_argument('--entropy_weight', type=float, default=0.0,
                   help='门控熵正则化系数(回应评审re2 §2.3, 默认0=旧行为不变)')
    # P1 敏感性分析: 覆盖 full_v2 关键超参, 证明结论不依赖超参脆点 (均默认 None=不变)
    g.add_argument('--n_envs', type=int, default=None,
                   help='覆盖环境切分数 n_envs (敏感性分析, 默认4)')
    g.add_argument('--rff_dim', type=int, default=None,
                   help='覆盖 RFF 特征维度 rff_dim (敏感性分析, 默认32)')
    g.add_argument('--prior_weight', type=float, default=None,
                   help='覆盖先验权重 prior_weight (敏感性分析, 默认0.05)')
    g.add_argument('--temperature', type=float, default=None,
                   help='覆盖门控温度 temperature (敏感性分析, 默认0.5)')
    g.add_argument('--alpha_init', type=float, default=None,
                   help='覆盖融合系数初始值 alpha_init (syn_ood 识别-利用脱节排查, 默认-2.0)')
    g.add_argument('--fusion_alpha', type=float, default=None,
                   help='覆盖通道混合残差权重 fusion_alpha (syn_ood 排查, 默认0.3)')
    g.add_argument('--env_mode', type=str, default=None,
                   choices=['uniform', 'semantic'],
                   help='修 C: 环境切分模式 (uniform=旧行为patch均分; semantic=按时间戳语义分组)')
    g.add_argument('--env_scheme', type=str, default=None,
                   choices=['season', 'daynight', 'tod', 'wd'],
                   help='修 C: 语义环境方案 (semantic 模式需要, 默认 season)')
    g.add_argument('--risk_lambda', type=float, default=0.0,
                   help='想法1 DRO: 跨环境风险厌恶系数 (0=ERM; >0 用 L=mean_e+λ*var_e)')
    g.add_argument('--epochs', type=int, default=None,
                   help='覆盖训练轮数 (默认用 dataset_config; 本地 smoke/调试时设小值)')
    g.add_argument('--dump_gates', action='store_true',
                   help='强制 dump 门控矩阵 (默认仅 n_vars<=21; 用于 traffic/electricity 高维热图)')

    r = sub.add_parser('run')
    r.add_argument('--device', default='cuda:0')
    r.add_argument('--job_file', required=True)
    r.add_argument('--result_csv', required=True)
    r.add_argument('--job_timeout', type=float, default=2400,
                   help='单个 job 超时秒数 (默认 2400=40min); 超时强杀并续跑')
    r.add_argument('--amp', action='store_true',
                   help='混合精度训练(仅CUDA生效), 通常提速1.5-2x; HSIC/门控仍走fp32保精度')

    s = sub.add_parser('summarize')
    s.add_argument('--output_dir', default='./output_large')
    s.add_argument('--dataset_dir', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == 'gen':
        gen_jobs(args)
    elif args.cmd == 'run':
        run_jobs(args)
    elif args.cmd == 'summarize':
        summarize(args)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
