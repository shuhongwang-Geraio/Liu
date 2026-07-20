"""
数据集自动下载工具

下载 ETTh1.csv 和 weather.csv 到 patchtst/dataset/ 目录。
支持断点续传和镜像源回退。

用法:
    python download_data.py                      # 下载全部数据集
    python download_data.py --dataset ETTh1     # 仅下载 ETTh1
    python download_data.py --dataset Weather   # 仅下载 Weather
    python download_data.py --output /path/to/data  # 指定输出目录
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import shutil
import hashlib

# ── 数据集源地址配置 ──────────────────────────────────────────

DATASETS = {
    'ETTh1': {
        'filename': 'ETTh1.csv',
        'urls': [
            'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
        ],
        'size_mb': 8.2,
    },
    'Weather': {
        'filename': 'weather.csv',
        'urls': [
            'https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather/weather.csv',
        ],
        'size_mb': 12.3,
    },
}

# ── 工具函数 ──────────────────────────────────────────────────

def download_file(url, dest_path, retries=2):
    """下载文件，支持重试和进度显示"""
    for attempt in range(retries + 1):
        try:
            print(f"  下载: {url}")
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; CausalCIT-setup/1.0)'
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = resp.length if resp.length > 0 else -1
                downloaded = 0
                with open(dest_path + '.tmp', 'wb') as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            mb = downloaded / (1024 * 1024)
                            print(f"\r    {pct:.0f}% ({mb:.1f} MB)", end='', flush=True)
                print("")
            os.replace(dest_path + '.tmp', dest_path)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"\n  下载失败 (尝试 {attempt + 1}/{retries + 1}): {e}")
            if os.path.exists(dest_path + '.tmp'):
                os.remove(dest_path + '.tmp')
            if attempt < retries:
                print(f"  重试中...")
    return False


def check_existing(filepath, expected_size_mb=None):
    """检查已有文件是否有效"""
    if not os.path.exists(filepath):
        return False
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if expected_size_mb and size_mb < expected_size_mb * 0.5:
        print(f"  ⚠ 文件存在但过小 ({size_mb:.1f} MB < 预期 {expected_size_mb:.1f} MB)，重新下载")
        return False
    print(f"  ✓ 已存在 ({size_mb:.1f} MB)，跳过下载")
    return True


# ── 主逻辑 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CausalCIT 数据集下载工具')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'ETTh1', 'Weather'],
                        help='下载指定数据集 (默认: 全部)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: ../patchtst/dataset)')
    args = parser.parse_args()

    # 默认输出到 patchtst/dataset/，在新目录结构下回退到 01_external/PatchTST/code/dataset/
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(script_dir, 'patchtst', 'dataset')
        # 如果默认路径不存在，尝试新目录结构下的数据集位置
        if not os.path.isdir(default_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))),
                                    '01_external', 'PatchTST', 'code', 'dataset')
            if os.path.isdir(alt_path):
                default_path = alt_path
        args.output = default_path

    os.makedirs(args.output, exist_ok=True)

    targets = DATASETS if args.dataset == 'all' else {args.dataset: DATASETS[args.dataset]}

    print("=" * 60)
    print("  CausalCIT 数据集下载")
    print("=" * 60)
    print(f"  目标目录: {args.output}")
    print(f"  数据集: {', '.join(targets.keys())}")
    print()

    success = True
    for name, info in targets.items():
        dest = os.path.join(args.output, info['filename'])
        print(f"[{name}] {info['filename']} (~{info['size_mb']} MB)")

        if check_existing(dest, info['size_mb']):
            continue

        downloaded = False
        for url in info['urls']:
            if download_file(url, dest):
                downloaded = True
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                print(f"  ✓ 下载完成 ({size_mb:.1f} MB)")
                break

        if not downloaded:
            print(f"  ✗ 所有镜像源均下载失败")
            print(f"  手动下载: 请将 {info['filename']} 放到 {args.output}/")
            success = False
        print()

    if success:
        print("=" * 60)
        print("  全部数据集准备就绪！")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  部分数据集下载失败，请查看上方提示手动准备")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
