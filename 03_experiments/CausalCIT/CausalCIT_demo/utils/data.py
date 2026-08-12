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


# 语义环境切分方案 (修 C / 想法 1 DRO 共用, 2026-08-12)
# 依据: assess_env_split.py 可行性评估 —— season/daynight/tod 有信息 (4-14x vs 随机均分),
#       wd (工作日/周末) 单独无信息 (1.2-1.9x), 保留但默认不推荐单独使用。
ENV_SCHEMES = {
    'season':   lambda dt: (dt.dt.month % 12 // 3),                                  # 0=冬 1=春 2=夏 3=秋
    'daynight': lambda dt: ((dt.dt.hour < 6) | (dt.dt.hour >= 18)).astype(int),      # 0=昼 1=夜
    'tod':      lambda dt: (dt.dt.hour // 6),                                        # 0-6/6-12/12-18/18-24
    'wd':       lambda dt: (dt.dt.dayofweek >= 5).astype(int),                       # 0=工作日 1=周末
}


def _build_env_labels(df, border1, border2, env_scheme):
    """从时间戳列解析语义环境标签, 并按 [border1, border2) 切分 (与数据行对齐)。

    返回 np.int64 数组, 长度 = border2-border1; border1/border2 为 None 时取全量。
    """
    if env_scheme is None:
        return None
    if env_scheme not in ENV_SCHEMES:
        raise ValueError(f"未知 env_scheme: {env_scheme} (可选: {list(ENV_SCHEMES)})")
    dt = pd.to_datetime(df.iloc[:, 0])
    labels = ENV_SCHEMES[env_scheme](dt)
    b1 = border1 if border1 is not None else 0
    b2 = border2 if border2 is not None else len(labels)
    return labels[b1:b2].to_numpy(dtype=np.int64)


class ETTDataset(Dataset):
    """ETT数据集加载器"""
    def __init__(self, data_path, seq_len=96, pred_len=96, flag='train',
                 scale=True, freq='h', env_scheme=None):
        """env_scheme (修 C / DRO): None=返回(x,y) 保持旧行为;
        否则返回 (x, y, env_label), 语义环境标签由时间戳解析
        (season/daynight/tod/wd, 见模块级 ENV_SCHEMES)。"""
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.env_scheme = env_scheme

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
        self.env_labels = _build_env_labels(df, border1, border2, env_scheme)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data[s_begin:s_end]       # [seq_len, n_vars]
        seq_y = self.data[s_end:r_end]         # [pred_len, n_vars]
        if self.env_labels is not None:
            return (torch.tensor(seq_x, dtype=torch.float32),
                    torch.tensor(seq_y, dtype=torch.float32),
                    int(self.env_labels[s_begin]))
        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class TemporalOODDataset(Dataset):
    """真实数据时序漂移 OOD: 训练用早期时段, 测试用晚期时段, 中间留 gap 最大化分布漂移.

    与 ETTDataset 的区别: 不按 7:1:2 紧挨切分, 而是 train=[0, train_frac],
    val 紧接 train 之后, test=[1-test_frac, n], 二者之间留一段 *未使用* 的 gap
    区域 (真实世界后期分布), 从而显式构造 "训练期 -> 测试期" 的时序分布漂移.
    归一化仍只用训练统计量, 以暴露漂移 (标准 OOD 协议).
    """

    def __init__(self, data_path, seq_len=96, pred_len=96, flag='train',
                 scale=True, freq='h', train_frac=0.5, val_frac=0.1,
                 test_frac=0.25, gap_frac=0.15, env_scheme=None):
        """env_scheme (修 C / DRO): None=返回(x,y); 否则返回 (x,y,env_label)。"""
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.env_scheme = env_scheme

        df = pd.read_csv(data_path)
        cols = df.columns[1:]  # 去掉date列
        df_data = df[cols].values.astype(np.float32)

        n = len(df_data)
        train_end = int(n * train_frac)
        test_start = n - int(n * test_frac)
        val_len = int(n * val_frac)
        val_start = train_end
        val_end = min(train_end + val_len, test_start - int(n * gap_frac))

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        border1s = [0, val_start, test_start]
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
        self.env_labels = _build_env_labels(df, border1, border2, env_scheme)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[s_end:r_end]
        if self.env_labels is not None:
            return (torch.tensor(seq_x, dtype=torch.float32),
                    torch.tensor(seq_y, dtype=torch.float32),
                    int(self.env_labels[s_begin]))
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


class SyntheticOODDataset(Dataset):
    """受控 OOD 合成数据 (IRM 风格): 验证因果稳定性门控在分布漂移下的鲁棒性.

    机制:
      - Ch0: AR 基础信号
      - Ch1, Ch2: 稳定因果 (跨时间/环境不变) -> 不变预测子依据
      - Ch3: 虚假相关, 与 Ch0 的相关 *强度随数据环境变化* (同号不同强度).
             因此跨随机环境 HSIC 稳定性低 -> 因果门控应下压该边;
             纯容量模型 (learned_gate) 拟合边际相关并过拟合训练环境.
      - Ch4: 受隐变量 confound
      - Ch5, Ch6: 独立噪声
    分布漂移: regime='train' 用训练强度集, regime='test' 用漂移(弱化/反转)强度集.
    归一化: 始终用 train 统计量, 以暴露分布漂移 (标准 OOD 协议).
    """

    def __init__(self, n_samples=10000, seq_len=96, pred_len=96, flag='train',
                 n_vars=7, seed=42, regime='train',
                 spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                 test_spurious_strengths=(0.05, -0.2, 0.1, -0.05),
                 train_noise=0.1, test_noise=0.1, n_data_envs=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = 7  # 固定 7 维合成
        self.regime = regime
        # 参考 train 统计量 (用于全 split 统一归一化)
        ref = self._generate('train', seed, spurious_strengths, train_noise,
                             n_data_envs, n_samples, seq_len, pred_len)
        self.mean = ref.mean(axis=0)
        self.std = ref.std(axis=0) + 1e-8
        # 生成本实例数据
        strengths = spurious_strengths if regime == 'train' else test_spurious_strengths
        noise = train_noise if regime == 'train' else test_noise
        flag_off = {'train': 0, 'val': 1, 'test': 2}[flag]
        data = self._generate(regime, seed + flag_off, strengths, noise,
                              n_data_envs, n_samples, seq_len, pred_len)
        self.data = (data - self.mean) / self.std
        self.channel_labels = ['Ch0:Base', 'Ch1:Causal', 'Ch2:Causal',
                               'Ch3:Spurious(env)', 'Ch4:Confound',
                               'Ch5:Indep', 'Ch6:Indep']

    @staticmethod
    def _generate(regime, seed, strengths, noise, n_data_envs,
                  n_samples, seq_len, pred_len):
        rng = np.random.RandomState(seed)
        total_len = n_samples + seq_len + pred_len
        data = np.zeros((total_len, 7), dtype=np.float32)
        # Ch0: AR(1) 基础信号
        for t in range(1, total_len):
            data[t, 0] = 0.8 * data[t - 1, 0] + rng.randn() * 0.5
            data[t, 0] += 0.3 * np.sin(2 * np.pi * t / 96)
        # 数据环境标签: 平滑过程 -> 每个 batch 混合多环境 (暴露跨环境不稳定性)
        env_phase = np.sin(2 * np.pi * np.arange(total_len) / 40.0)
        env_idx = ((env_phase * 0.5 + 0.5) * n_data_envs).astype(int) % n_data_envs
        # Ch1/Ch2: 稳定因果 (强耦合, 不随环境变) -> 正确的跨通道选择应明显获益
        for t in range(1, total_len):
            data[t, 1] = 0.9 * data[t - 1, 0] + 0.4 * data[t - 1, 1] + rng.randn() * noise
            data[t, 2] = 0.8 * np.tanh(data[t - 1, 0]) + 0.4 * data[t - 1, 2] + rng.randn() * noise
            # Ch3: 虚假相关, 强度随数据环境变化
            data[t, 3] = strengths[env_idx[t]] * data[t - 1, 0] + rng.randn() * noise
        # Ch4: 受隐变量 confound
        hidden = np.cumsum(rng.randn(total_len) * 0.3)
        for t in range(1, total_len):
            data[t, 4] = 0.5 * hidden[t] + rng.randn() * noise
        data[:, 0] += 0.2 * hidden
        # Ch5, Ch6: 独立噪声
        data[:, 5] = np.cumsum(rng.randn(total_len) * 0.3)
        data[:, 6] = rng.randn(total_len) * 0.5
        for t in range(1, total_len):
            data[t, 6] += 0.4 * data[t - 1, 6]
        return data

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s = index
        e = s + self.seq_len
        r = e + self.pred_len
        return (torch.tensor(self.data[s:e], dtype=torch.float32),
                torch.tensor(self.data[e:r], dtype=torch.float32))


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False):
    """num_workers/pin_memory 默认关闭, 保证与旧结果完全一致;
    在 GPU 训练时可传 num_workers>0/pin_memory=True 加快数据管线。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
