"""
数据集下载脚本
从 GitHub 上下载 ETT 数据集用于 PatchTST 实验

数据集来源: https://github.com/zhouhaoyi/ETDataset
"""

import os
import urllib.request
import sys


def download_file(url, save_path):
    """下载文件并显示进度"""
    print(f"正在下载: {os.path.basename(save_path)}")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"  [OK] 下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"  [FAIL] 下载失败: {e}")
        return False


def main():
    # 创建数据集目录
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    # ETT 数据集 URL（来自 ETDataset 仓库）
    base_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
    
    ett_files = [
        "ETTh1.csv",
        "ETTh2.csv",
        "ETTm1.csv",
        "ETTm2.csv",
    ]

    print("=" * 60)
    print("PatchTST 数据集下载工具")
    print("=" * 60)
    print(f"\n数据将保存到: {dataset_dir}\n")

    success_count = 0
    for filename in ett_files:
        url = base_url + filename
        save_path = os.path.join(dataset_dir, filename)
        
        if os.path.exists(save_path):
            print(f"  - {filename} 已存在，跳过")
            success_count += 1
            continue
            
        if download_file(url, save_path):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"完成! 成功下载 {success_count}/{len(ett_files)} 个文件")
    print(f"{'=' * 60}")
    
    print("\n注意: Weather、Electricity、Traffic 等数据集需要从以下地址手动下载:")
    print("https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy")
    print("\n下载后将 CSV 文件放入 ./dataset/ 目录即可")


if __name__ == '__main__':
    main()
