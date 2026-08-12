"""
训练前适用性判据: 原始数据统计量 (0 GPU, 2026-08-12)
=====================================================

背景 (07_scope_and_publication_risk_analysis.md 方案 1)
-------------------------------------------------------
把"范围窄"从 B 类(无解释)变成 A 类(有原则、可预测):
只从原始数据计算统计量, 与 7 数据集 × horizon 的实测增益做对应。
若某统计量能预测增益正负号 → "训练前即可计算的适用性判据"。

本脚本对每个数据集 (需有 date 列) 计算:
  1. 通道间依赖密度        avg|corr| (全时段平均绝对相关)
  2. 稳定性信号           跨语义环境(season)相关矩阵的离散度 CV
  3. 稳定通道对占比        跨环境极差小且依赖强的通道对比例
  4. 语义环境信息量       (season/daynight) 组间相关差异 / 随机均分对照
  5. 长程记忆             lag-1 自相关均值
  6. 维度                 C, N

输出: stdout 表格 + --out 指定 json。

用法:
  python compute_pre_train_stats.py --data <path> --name <name> [--freq h|15min|10min] [--out stats.json]
"""

import argparse
import json
import numpy as np
import pandas as pd


def load_time_series(path):
    df = pd.read_csv(path)
    dt = pd.to_datetime(df.iloc[:, 0])
    X = df.iloc[:, 1:].values.astype(np.float64)
    return dt, X


def corr_matrix(X, subsample=20000):
    """内存友好: 若样本过多, 均匀子采样再算相关。"""
    n = len(X)
    if n > subsample:
        idx = np.linspace(0, n - 1, subsample).astype(int)
        X = X[idx]
    return np.corrcoef(X, rowvar=False)


def env_corrs(X, labels, min_n=200):
    corrs = {}
    for lab in sorted(set(labels)):
        mask = labels == lab
        if mask.sum() < min_n:
            continue
        corrs[lab] = corr_matrix(X[mask])
    return corrs


def frob_per_pair(A, B):
    C = A.shape[0]
    return np.linalg.norm(A - B, ord='fro') / (C * (C - 1)) * 2.0


def pair_diff(corrs):
    labs = list(corrs.keys())
    if len(labs) < 2:
        return 0.0
    ds = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            ds.append(frob_per_pair(corrs[labs[i]], corrs[labs[j]]))
    return float(np.mean(ds))


def rand_split_diff(X, n_envs=4, n_trials=5, subsample=20000):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    ds = []
    for _ in range(n_trials):
        perm = rng.permutation(idx)
        groups = np.array_split(perm, n_envs)
        corrs = {}
        for g, sub in enumerate(groups):
            if len(sub) < 200:
                continue
            s = sub
            if len(s) > subsample:
                s = s[np.linspace(0, len(s) - 1, subsample).astype(int)]
            corrs[g] = np.corrcoef(X[s], rowvar=False)
        ds.append(pair_diff(corrs))
    return float(np.mean(ds))


def stable_pair_frac(corrs, dep_thr=0.1, range_thr=0.15):
    """稳定通道对占比: 平均依赖强 且 跨环境波动小 的通道对比例。"""
    labs = list(corrs.keys())
    if len(labs) < 2:
        return float('nan')
    stack = np.stack([corrs[l] for l in labs], axis=0)  # [E, C, C]
    r_mean = np.abs(stack).mean(axis=0)
    r_range = np.abs(stack).max(axis=0) - np.abs(stack).min(axis=0)
    C = r_mean.shape[0]
    off = ~np.eye(C, dtype=bool)
    dense = r_mean > dep_thr
    stable = r_range < range_thr
    both = dense & stable & off
    return float(both.sum() / off.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    dt, X = load_time_series(args.data)
    C, N = X.shape[1], len(X)

    R_all = corr_matrix(X)
    off = ~np.eye(C, dtype=bool)
    dep_density = float(np.abs(R_all[off]).mean())
    lag1 = float(np.corrcoef(X[:-1].ravel(), X[1:].ravel())[0, 1])

    season = (dt.dt.month % 12 // 3)
    daynight = ((dt.dt.hour < 6) | (dt.dt.hour >= 18)).astype(int)

    c_season = env_corrs(X, season)
    c_daynight = env_corrs(X, daynight)

    diff_season = pair_diff(c_season)
    diff_daynight = pair_diff(c_daynight)
    rnd = rand_split_diff(X)

    stable_frac_season = stable_pair_frac(c_season)
    stable_frac_daynight = stable_pair_frac(c_daynight)

    stats = {
        'dataset': args.name, 'C': C, 'N': N,
        'dep_density_avg_abs_corr': round(dep_density, 5),
        'lag1_autocorr': round(lag1, 5),
        'env_info_season_pair_diff': round(diff_season, 5),
        'env_info_daynight_pair_diff': round(diff_daynight, 5),
        'env_info_rand_split_ref': round(rnd, 5),
        'env_ratio_season_over_rand': round(diff_season / max(rnd, 1e-9), 2),
        'env_ratio_daynight_over_rand': round(diff_daynight / max(rnd, 1e-9), 2),
        'stable_pair_frac_season': round(stable_frac_season, 4),
        'stable_pair_frac_daynight': round(stable_frac_daynight, 4),
    }

    print(f"\n=== {args.name}  C={C} N={N}")
    for k, v in stats.items():
        if k not in ('dataset', 'C', 'N'):
            print(f"  {k:32s} {v}")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n[stats saved] {args.out}")


if __name__ == '__main__':
    main()
