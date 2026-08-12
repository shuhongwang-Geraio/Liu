"""
语义环境切分可行性评估 (0 GPU, 2026-08-12)
=============================================

背景
----
根因 3 (门 1 静态诊断): CausalStabilityGate 的"环境"= 窗口内均匀切分
(env_size = patch_num // n_envs), 不反映真实机制变化, 导致 cv≈0.005,
跨环境稳定性项 (1/(1+cv)) 无信息。修 C 的候选方案是"语义环境切分":
用时间戳 (季节 / 工作日vs周末 / 一天时段) 定义环境。

本脚本回答一个前置问题: 语义环境切分在真实数据上是否真的更有信息?
方法
----
对每个数据集, 用时间戳解析出语义标签, 按标签分组计算组内通道相关矩阵
(Pearson, 作为依赖强度的可解释代理), 度量"环境间相关矩阵差异"; 并与
当前方法(随机均分)的组间差异对照。

判据
----
- 若 语义环境组间差异 >> 随机均分组间差异 (数量级/显著):
    语义环境有信息, 修 C 可行 (稳定性项将获得真实信号)。
- 若 两者都接近 0 或无差异:
    修 C 无意义 (稳定性项在任何切分下都无信息), 应转向想法 1 (DRO) 等。

用法
----
python assess_env_split.py --data <path> --name <dataset_name> [--freq h|10min]
"""

import argparse
import os
import numpy as np
import pandas as pd

ENV_SCHEMES = {
    'wd':  lambda dt: (dt.dt.dayofweek >= 5).astype(int),          # 工作日/周末
    'season': lambda dt: (dt.dt.month % 12 // 3),                  # 0=冬 1=春 2=夏 3=秋
    'tod':  lambda dt: (dt.dt.hour // 6),                          # 0-6/6-12/12-18/18-24
    'daynight': lambda dt: ((dt.dt.hour < 6) | (dt.dt.hour >= 18)).astype(int),  # 昼夜
}


def load_data(path):
    df = pd.read_csv(path)
    time_col = df.columns[0]
    dt = pd.to_datetime(df[time_col])
    X = df.iloc[:, 1:].values.astype(np.float64)
    return dt, X


def env_corr_matrices(X, labels):
    """按标签分组, 返回 组间通道相关矩阵 {label: [C,C]} (未中心化的协方差相关)。"""
    corrs = {}
    for lab in sorted(set(labels)):
        mask = labels == lab
        if mask.sum() < 50:
            continue
        sub = X[mask]
        corr = np.corrcoef(sub, rowvar=False)
        corrs[lab] = corr
    return corrs


def pair_diff(corrs):
    """平均成对 Frobenius 距离 (归一化到每通道对), 度量环境间依赖结构差异。"""
    labs = list(corrs.keys())
    if len(labs) < 2:
        return 0.0
    C = corrs[labs[0]].shape[0]
    diffs = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            d = np.linalg.norm(corrs[labs[i]] - corrs[labs[j]], ord='fro')
            diffs.append(d)
    # 归一化: 每对通道的平均 |r_a - r_b|, 并对角置零 (对角恒为1无信息)
    per_pair = np.mean(diffs) / (C * (C - 1)) * 2.0
    return float(per_pair)


def rand_split_diff(X, n_envs, n_trials=5):
    """当前方法的对照: 把时间行随机/均匀切成 n_envs 份, 度量组间差异。"""
    n = len(X)
    idx = np.arange(n)
    diffs = []
    rng = np.random.RandomState(0)
    for _ in range(n_trials):
        perm = rng.permutation(idx)
        groups = np.array_split(perm, n_envs)
        corrs = {}
        for g, sub_idx in enumerate(groups):
            sub = X[sub_idx]
            if len(sub) < 50:
                continue
            corrs[g] = np.corrcoef(sub, rowvar=False)
        diffs.append(pair_diff(corrs))
    return float(np.mean(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--n_envs', type=int, default=4)
    args = ap.parse_args()

    dt, X = load_data(args.data)
    C = X.shape[1]
    n = len(X)
    print(f"\n=== {args.name} ===  C={C}, N={n}")
    print(f"{'scheme':10s} {'#envs':>5s} {'组间|r_a-r_b|mean':>20s}  (语义环境)")
    results = {}
    for name, fn in ENV_SCHEMES.items():
        labels = fn(dt)
        corrs = env_corr_matrices(X, labels)
        d = pair_diff(corrs)
        results[name] = d
        print(f"{name:10s} {len(set(labels)):>5d} {d:20.5f}")
    # 随机均分对照 (当前方法的信息量上限)
    rnd = rand_split_diff(X, args.n_envs)
    print(f"\n随机均分对照 (当前方法, n_envs={args.n_envs}, 5 trials): 组间差异 = {rnd:.5f}")
    print(f"语义最大 vs 随机: {max(results.values()):.5f} vs {rnd:.5f}  "
          f"(ratio = {max(results.values())/max(rnd,1e-9):.1f}x)")
    best = max(results, key=results.get)
    verdict = "语义切分有信息 → 修 C 可行" if max(results.values()) > 3 * rnd else \
              ("语义切分信息有限 → 修 C 收益存疑" if max(results.values()) > rnd else
               "语义切分无信息 → 修 C 无意义, 转想法 1")
    print(f"判定: {verdict} (最佳方案: {best})")


if __name__ == '__main__':
    main()
