"""
CausalCIT 消融实验变体模型

完整CausalCIT的三大核心组件:
  A. HSIC稳定性检验 (CausalStabilityGate)
  B. 环境划分 (n_envs个时间段)
  C. 选择性通道注意力 (门控调制)

消融变体:
  1. Full CausalCIT          — A+B+C (完整模型)
  2. w/o HSIC (NoHSIC)       — 去掉A，用简单相关性替代HSIC
  3. w/o EnvSplit (NoEnv)    — 去掉B，不划分环境，全局计算HSIC
  4. w/o Gate (NoGate)       — 去掉C，所有通道全连接注意力(无门控)
  5. w/o All = PatchTST      — 去掉A+B+C，退化为纯PatchTST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys, os

# 复用CausalCIT_demo的基础组件
DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CausalCIT_demo')
sys.path.insert(0, DEMO_DIR)

from models.layers import RevIN, series_decomp
from models.patchtst import TSTiEncoder, Flatten_Head, PatchTST
from models.causal_channel import (
    RFFKernel, CausalStabilityGate, CausalChannelAttention, CausalChannelInteraction
)
from models.causalcit import CausalCIT


# ============================================================
# 变体1: w/o HSIC — 用Pearson相关性替代HSIC
# ============================================================

class CorrelationGate(nn.Module):
    """用简单Pearson相关性替代HSIC的门控"""
    def __init__(self, n_vars, d_model, n_envs=4, **kwargs):
        super().__init__()
        self.n_vars = n_vars
        self.n_envs = n_envs
        self.channel_prior = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.stability_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [bs, nvars, patch_num, d_model]
        bs, nvars, patch_num, d_model = x.shape

        env_size = patch_num // self.n_envs
        if env_size < 2:
            gate = torch.ones(bs, nvars, nvars, device=x.device)
            eye = torch.eye(nvars, device=x.device).unsqueeze(0)
            return gate * (1 - eye) + eye

        x_trunc = x[:, :, :self.n_envs * env_size, :]
        x_envs = x_trunc.reshape(bs, nvars, self.n_envs, env_size, d_model)
        # 对d_model取均值得到标量表示
        x_scalar = x_envs.mean(dim=-1)  # [bs, nvars, n_envs, env_size]

        # 每个环境内计算Pearson相关系数
        corr_per_env = []
        for e in range(self.n_envs):
            xe = x_scalar[:, :, e, :]  # [bs, nvars, env_size]
            xe_centered = xe - xe.mean(dim=-1, keepdim=True)
            xe_norm = xe_centered / (xe_centered.norm(dim=-1, keepdim=True) + 1e-8)
            corr = torch.bmm(xe_norm, xe_norm.transpose(1, 2))  # [bs, nvars, nvars]
            corr_per_env.append(corr.abs())

        corr_stack = torch.stack(corr_per_env, dim=-1)  # [bs, nv, nv, n_envs]
        corr_mean = corr_stack.mean(dim=-1).clamp(min=1e-8)
        corr_std = corr_stack.std(dim=-1)
        cv = corr_std / corr_mean
        stability = 1.0 / (1.0 + cv + self.stability_bias.abs())

        prior = torch.sigmoid(self.channel_prior)
        stability = stability * 0.7 + prior.unsqueeze(0) * 0.3
        gate = self.gate_mlp(stability.unsqueeze(-1)).squeeze(-1)

        eye = torch.eye(self.n_vars, device=x.device).unsqueeze(0)
        gate = gate * (1 - eye) + eye
        return gate


class NoHSIC_ChannelInteraction(nn.Module):
    """变体: 用Pearson相关性替代HSIC"""
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3):
        super().__init__()
        self.stability_gate = CorrelationGate(n_vars, d_model, n_envs=n_envs)
        self.channel_attn = CausalChannelAttention(d_model, n_heads, n_vars, dropout)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.alpha = nn.Parameter(torch.tensor(fusion_alpha))

    def forward(self, x):
        bs, nvars, d_model, patch_num = x.shape
        x_for_gate = x.permute(0, 1, 3, 2)
        gate_matrix = self.stability_gate(x_for_gate)
        x_pooled = x.mean(dim=-1)
        x_channel = self.channel_attn(x_pooled, gate_matrix)
        x_proj = self.fusion_proj(x_channel).unsqueeze(-1).expand_as(x)
        alpha = torch.sigmoid(self.alpha)
        out = (1 - alpha) * x + alpha * x_proj
        return out, gate_matrix


# ============================================================
# 变体2: w/o EnvSplit — 不划分环境，全局HSIC
# ============================================================

class GlobalHSICGate(nn.Module):
    """不划分环境，在整个序列上计算单一HSIC"""
    def __init__(self, n_vars, d_model, rff_dim=32, **kwargs):
        super().__init__()
        self.n_vars = n_vars
        self.rff_kernel = RFFKernel(d_model, rff_dim)
        self.channel_prior = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [bs, nvars, patch_num, d_model]
        bs, nvars, patch_num, d_model = x.shape

        # 全局RFF（不划分环境）
        x_flat = x.reshape(-1, d_model)
        z_flat = self.rff_kernel(x_flat)
        rff_dim = z_flat.shape[-1]
        z = z_flat.reshape(bs, nvars, patch_num, rff_dim)
        z_centered = z - z.mean(dim=2, keepdim=True)

        # 全局HSIC: 对整个patch序列计算
        zi = z_centered.unsqueeze(2)  # [bs, nv, 1, patch_num, rff]
        zj = z_centered.unsqueeze(1)  # [bs, 1, nv, patch_num, rff]
        cross_diag = (zi * zj).mean(dim=3)  # [bs, nv, nv, rff]
        hsic_global = (cross_diag ** 2).sum(dim=-1)  # [bs, nv, nv]

        # 没有跨环境稳定性概念，直接用HSIC值做门控
        hsic_norm = hsic_global / (hsic_global.max() + 1e-8)

        prior = torch.sigmoid(self.channel_prior)
        score = hsic_norm * 0.7 + prior.unsqueeze(0) * 0.3
        gate = self.gate_mlp(score.unsqueeze(-1)).squeeze(-1)

        eye = torch.eye(self.n_vars, device=x.device).unsqueeze(0)
        gate = gate * (1 - eye) + eye
        return gate


class NoEnv_ChannelInteraction(nn.Module):
    """变体: 不划分环境，全局HSIC"""
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3):
        super().__init__()
        self.stability_gate = GlobalHSICGate(n_vars, d_model, rff_dim=rff_dim)
        self.channel_attn = CausalChannelAttention(d_model, n_heads, n_vars, dropout)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.alpha = nn.Parameter(torch.tensor(fusion_alpha))

    def forward(self, x):
        bs, nvars, d_model, patch_num = x.shape
        x_for_gate = x.permute(0, 1, 3, 2)
        gate_matrix = self.stability_gate(x_for_gate)
        x_pooled = x.mean(dim=-1)
        x_channel = self.channel_attn(x_pooled, gate_matrix)
        x_proj = self.fusion_proj(x_channel).unsqueeze(-1).expand_as(x)
        alpha = torch.sigmoid(self.alpha)
        out = (1 - alpha) * x + alpha * x_proj
        return out, gate_matrix


# ============================================================
# 变体3: w/o Gate — 全连接通道注意力，无门控
# ============================================================

class NoGate_ChannelInteraction(nn.Module):
    """变体: 所有通道全连接注意力，不做门控选择"""
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3):
        super().__init__()
        self.n_vars = n_vars
        # 标准多头注意力（无门控mask）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.alpha = nn.Parameter(torch.tensor(fusion_alpha))

    def forward(self, x):
        bs, nvars, d_model, patch_num = x.shape
        x_pooled = x.mean(dim=-1)  # [bs, nvars, d_model]
        residual = x_pooled

        Q = self.W_Q(x_pooled).view(bs, nvars, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x_pooled).view(bs, nvars, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x_pooled).view(bs, nvars, self.n_heads, self.d_k).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # 无门控mask，所有通道全连接
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(bs, nvars, d_model)
        out = self.W_O(out)
        out = self.norm(residual + self.dropout(out))

        x_proj = self.fusion_proj(out).unsqueeze(-1).expand_as(x)
        alpha_val = torch.sigmoid(self.alpha)
        out_final = (1 - alpha_val) * x + alpha_val * x_proj

        # 返回全1门控矩阵（方便统一接口）
        gate_matrix = torch.ones(bs, nvars, nvars, device=x.device)
        return out_final, gate_matrix


# ============================================================
# 通用消融变体骨干
# ============================================================

class AblationBackbone(nn.Module):
    """通用消融骨干：接受不同的通道交互模块"""
    def __init__(self, c_in, context_window, target_window, patch_len, stride,
                 channel_interaction_cls, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, dropout=0., padding_patch=None,
                 individual=False, revin=True, affine=True, subtract_last=False,
                 n_channel_heads=4, n_envs=4, rff_dim=32,
                 channel_dropout=0.1, fusion_alpha=0.3, **kwargs):
        super().__init__()
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = TSTiEncoder(
            c_in, patch_num=patch_num, patch_len=patch_len,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, act="gelu", res_attention=True,
            pe='zeros', learn_pe=True, **kwargs
        )

        self.causal_channel = channel_interaction_cls(
            n_vars=c_in, d_model=d_model, patch_num=patch_num,
            n_heads=n_channel_heads, n_envs=n_envs, rff_dim=rff_dim,
            dropout=channel_dropout, fusion_alpha=fusion_alpha,
        )

        self.head_nf = d_model * patch_num
        self.head = Flatten_Head(individual, c_in, self.head_nf,
                                 target_window, head_dropout=0)
        self.last_gate_matrix = None

    def forward(self, z):
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)
        z, gate_matrix = self.causal_channel(z)
        self.last_gate_matrix = gate_matrix.detach()
        z = self.head(z)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def get_gate_matrix(self):
        return self.last_gate_matrix


class AblationModel(nn.Module):
    """通用消融模型包装"""
    def __init__(self, channel_interaction_cls, enc_in, seq_len, pred_len,
                 e_layers=3, n_heads=4, d_model=64, d_ff=256,
                 dropout=0.2, patch_len=16, stride=8, padding_patch='end',
                 n_channel_heads=4, n_envs=4, rff_dim=64,
                 channel_dropout=0.1, fusion_alpha=0.3, **kwargs):
        super().__init__()
        self.model = AblationBackbone(
            c_in=enc_in, context_window=seq_len, target_window=pred_len,
            patch_len=patch_len, stride=stride,
            channel_interaction_cls=channel_interaction_cls,
            n_layers=e_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, padding_patch=padding_patch,
            n_channel_heads=n_channel_heads, n_envs=n_envs, rff_dim=rff_dim,
            channel_dropout=channel_dropout, fusion_alpha=fusion_alpha,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = x.permute(0, 2, 1)
        return x

    def get_gate_matrix(self):
        return self.model.get_gate_matrix()


# ============================================================
# 工厂函数: 创建各变体
# ============================================================

def create_ablation_model(variant, **kwargs):
    """
    variant:
        'full'     — 完整CausalCIT (A+B+C)
        'no_hsic'  — 去掉HSIC，用Pearson相关性
        'no_env'   — 去掉环境划分，全局HSIC
        'no_gate'  — 去掉门控，全连接通道注意力
        'patchtst' — 纯PatchTST (无通道交互)
    """
    if variant == 'full':
        return CausalCIT(**kwargs)
    elif variant == 'no_hsic':
        return AblationModel(NoHSIC_ChannelInteraction, **kwargs)
    elif variant == 'no_env':
        return AblationModel(NoEnv_ChannelInteraction, **kwargs)
    elif variant == 'no_gate':
        return AblationModel(NoGate_ChannelInteraction, **kwargs)
    elif variant == 'patchtst':
        # 只取PatchTST需要的参数
        pt_keys = ['enc_in', 'seq_len', 'pred_len', 'e_layers', 'n_heads',
                    'd_model', 'd_ff', 'dropout', 'fc_dropout', 'patch_len',
                    'stride', 'padding_patch']
        pt_kwargs = {k: v for k, v in kwargs.items() if k in pt_keys}
        if 'fc_dropout' not in pt_kwargs:
            pt_kwargs['fc_dropout'] = pt_kwargs.get('dropout', 0.2)
        return PatchTST(**pt_kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
