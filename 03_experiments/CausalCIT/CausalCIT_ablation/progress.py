"""
CausalCIT run_large 进度查看器 (只读, 不触碰任何运行进程)。

统计:
  - 总 job 数 / 各 shard 已完成数 (通过 ckpt 目录匹配)
  - 各数据集 × 变体 × seed 的完成矩阵
  - 结果 CSV 写入行数

用法:
  python progress.py
  python progress.py --output_dir ./output_large
"""
import os
import json
import argparse


def job_signature(j):
    """从 job dict 生成与 ckpt 目录名对应的标识。"""
    return f"{j['dataset']}_pl{j['pred_len']}_{j['variant']}_s{j['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="./output_large")
    args = ap.parse_args()
    OUT = args.output_dir

    ckpt_root = os.path.join(OUT, "ckpt")
    done_sigs = set()
    if os.path.isdir(ckpt_root):
        for d in os.listdir(ckpt_root):
            if os.path.isdir(os.path.join(ckpt_root, d)):
                # ckpt 目录名形如 dataset_pl96_variant_s42[_extra]
                # 取前4段 (dataset_pl{variant}_s{seed}) 作为签名
                parts = d.split("_")
                if len(parts) >= 4:
                    done_sigs.add("_".join(parts[:4]))

    total = 0
    per_shard = []
    matrix = {}  # (dataset, variant) -> {seed: done_bool}
    for i in range(3):
        jf = os.path.join(OUT, f"jobs_shard{i}.json")
        if not os.path.exists(jf):
            per_shard.append((0, 0))
            continue
        jobs = json.load(open(jf))
        n = len(jobs)
        done = 0
        for j in jobs:
            sig = job_signature(j)
            ok = sig in done_sigs
            if ok:
                done += 1
            key = (j["dataset"], j["variant"])
            matrix.setdefault(key, {})
            matrix[key][j["seed"]] = ok
        total += n
        per_shard.append((done, n))

    overall_done = sum(d for d, _ in per_shard)
    print("=" * 60)
    print(f"CausalCIT 大规模实验进度  (output_dir={OUT})")
    print("=" * 60)
    print(f"总计 job: {total}  |  已完成: {overall_done}  |  进度: {overall_done/total*100:.1f}%")
    print("-" * 60)
    for i, (d, n) in enumerate(per_shard):
        print(f"  shard{i}: {d}/{n}  ({d/n*100:.0f}%)")
    print("-" * 60)
    print("完成矩阵 (●=完成 ○=未完成):")
    for key in sorted(matrix.keys()):
        ds, var = key
        seeds = sorted(matrix[key].keys())
        cells = " ".join(f"s{s}:{'●' if matrix[key][s] else '○'}" for s in seeds)
        print(f"  {ds:12s} {var:10s} {cells}")
    print("=" * 60)


if __name__ == "__main__":
    main()
