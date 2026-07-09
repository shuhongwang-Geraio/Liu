"""
数据加载工具
支持: ETTh1/ETTh2/ETTm1/ETTm2 + 合成数据（含虚假相关通道）
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class ETTDataset(Dataset):
    """ETT数据集加载器"""
    def __init__(self, data_path, seq_len=96, pred_len=96, flag='train',
                 scale=True, freq='h'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        df = pd.read_csv(data_path)
        cols = df.columns[1:]  # 去掉date列
        df_data = df[cols].values.astype(np.float32)

        # ETT标准划分
        n = len(df_data)
        if 'ETTh' in data_path:
            borders = [
                [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24],       # train end
                [12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24,  # val end
                 12 * 30 * 24 + 8 * 30 * 24],
            ]
            border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ETTm' in data_path:
            border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            # 通用7:1:2划分
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            border1s = [0, train_end - seq_len, val_end - seq_len]
            border2s = [train_end, val_end, n]

        border1 = border1s[self.set_type]
        border2 = min(border2s[self.set_type], n)

        # 标准化（只用训练集统计量）
        self.scaler = StandardScaler()
        if scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            df_data = self.scaler.transform(df_data)

        self.data = df_data[border1:border2]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data[s_begin:s_end]       # [seq_len, n_vars]
        seq_y = self.data[s_end:r_end]         # [pred_len, n_vars]
        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class SyntheticCausalDataset(Dataset):
    """合成数据集：含真实因果通道和虚假相关通道

    用于验证CausalCIT能否正确识别因果vs虚假相关:
    - Channel 0: 基础信号 (AR过程)
    - Channel 1: 因果依赖于Ch0 (稳定关系)
    - Channel 2: 因果依赖于Ch0 (稳定关系)
    - Channel 3: 虚假相关——仅在前半段与Ch0相关，后半段关系改变
    - Channel 4: 虚假相关——受隐变量confounding
    - Channel 5: 独立噪声 (与其他通道无真实关系)
    - Channel 6: 独立噪声
    """
    def __init__(self, n_samples=10000, seq_len=96, pred_len=96, flag='train',
                 n_vars=7, noise_level=0.1, seed=42):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        rng = np.random.RandomState(seed + {'train': 0, 'val': 1, 'test': 2}[flag])

        total_len = n_samples + seq_len + pred_len
        data = np.zeros((total_len, n_vars), dtype=np.float32)

        # Ch0: AR(1) 基础信号
        for t in range(1, total_len):
            data[t, 0] = 0.8 * data[t-1, 0] + rng.randn() * 0.5
            # 加入周期性
            data[t, 0] += 0.3 * np.sin(2 * np.pi * t / 96)

        # Ch1: 稳定因果 (线性依赖Ch0，延迟1步)
        for t in range(1, total_len):
            data[t, 1] = 0.6 * data[t-1, 0] + 0.3 * data[t-1, 1] + rng.randn() * noise_level

        # Ch2: 稳定因果 (非线性依赖Ch0)
        for t in range(1, total_len):
            data[t, 2] = 0.5 * np.tanh(data[t-1, 0]) + 0.4 * data[t-1, 2] + rng.randn() * noise_level

        # Ch3: 虚假相关 (关系随时间变化 - 分布漂移)
        midpoint = total_len // 2
        for t in range(1, total_len):
            if t < midpoint:
                data[t, 3] = 0.7 * data[t-1, 0] + rng.randn() * noise_level  # 前半段高相关
            else:
                data[t, 3] = -0.3 * data[t-1, 0] + 0.5 * rng.randn()         # 后半段关系反转

        # Ch4: 虚假相关 (受隐变量confounding)
        hidden = np.cumsum(rng.randn(total_len) * 0.3)
        for t in range(1, total_len):
            data[t, 4] = 0.5 * hidden[t] + rng.randn() * noise_level
        # 也给Ch0加入hidden的影响（制造confounding）
        data[:, 0] += 0.2 * hidden

        # Ch5, Ch6: 独立噪声
        data[:, 5] = np.cumsum(rng.randn(total_len) * 0.3)
        data[:, 6] = rng.randn(total_len) * 0.5
        for t in range(1, total_len):
            data[t, 6] += 0.4 * data[t-1, 6]

        # 标准化
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        self.data = (data - mean) / std
        self.n_vars = n_vars

        # 通道标签（用于分析）
        self.channel_labels = [
            'Ch0:Base(AR)',
            'Ch1:Causal(linear)',
            'Ch2:Causal(nonlinear)',
            'Ch3:Spurious(shift)',
            'Ch4:Spurious(confound)',
            'Ch5:Independent',
            'Ch6:Independent'
        ]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[s_end:r_end]
        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32))


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True)
