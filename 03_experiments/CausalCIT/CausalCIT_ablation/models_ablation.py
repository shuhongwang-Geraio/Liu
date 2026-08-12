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
import inspect
import sys, os

# 复用CausalCIT_demo的基础组件
DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CausalCIT_demo')
sys.path.insert(0, DEMO_DIR)

from models.layers import RevIN, series_decomp
from models.patchtst import TSTiEncoder, Flatten_Head, PatchTST
from models.causal_channel import (
    RFFKernel, CausalStabilityGate, CausalChannelAttention,
    CausalChannelAttentionTemporal, CausalChannelInteraction
)
from models.causalcit import CausalCIT


# ============================================================
# 变体1: w/o HSIC — 用Pearson相关性替代HSIC
# ============================================================

class CorrelationGate(nn.Module):
    """用简单Pearson相关性替代HSIC的门控"""
    def __init__(self, n_vars, d_model, n_envs=4, prior_weight: float = 0.3, **kwargs):
        super().__init__()
        self.n_vars = n_vars
        self.n_envs = n_envs
        self.prior_weight = prior_weight
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
        stability = stability * (1 - self.prior_weight) + prior.unsqueeze(0) * self.prior_weight
        gate = self.gate_mlp(stability.unsqueeze(-1)).squeeze(-1)

        eye = torch.eye(self.n_vars, device=x.device).unsqueeze(0)
        gate = gate * (1 - eye) + eye
        return gate

    def get_diagnostics(self):
        """返回门控相关可学习参数诊断信息，供消融可观测性插桩使用。"""
        prior_sig = torch.sigmoid(self.channel_prior).detach()
        return {
            'gate_type': 'CorrelationGate',
            'channel_prior_sig_mean': float(prior_sig.mean()),
            'channel_prior_sig_min': float(prior_sig.min()),
            'channel_prior_sig_max': float(prior_sig.max()),
            'prior_weight': float(self.prior_weight),
        }


class NoHSIC_ChannelInteraction(nn.Module):
    """变体: 用Pearson相关性替代HSIC"""
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.3):
        super().__init__()
        self.prior_weight = prior_weight
        self.stability_gate = CorrelationGate(n_vars, d_model, n_envs=n_envs,
                                              prior_weight=prior_weight)
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

    def get_diagnostics(self):
        return self.stability_gate.get_diagnostics()


# ============================================================
# 变体2: w/o EnvSplit — 不划分环境，全局HSIC
# ============================================================

class GlobalHSICGate(nn.Module):
    """不划分环境，在整个序列上计算单一HSIC"""
    def __init__(self, n_vars, d_model, rff_dim=32, prior_weight: float = 0.3, **kwargs):
        super().__init__()
        self.n_vars = n_vars
        self.prior_weight = prior_weight
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
        z_centered = z - z.mean(dim=2, keepdim=True)  # [bs, nv, patch_num, rff]

        # 全局HSIC: 对整个patch序列计算。
        # 修复(回应评审re2 OOM): 原实现用 unsqueeze(1)/unsqueeze(2) 一次性
        # 物化 [bs, nv, nv, patch_num, rff], 对 traffic(nv=862) 需 ~8.5GB 临时显存,
        # 在 4090 上直接 OOM。改为逐通道块(chunk)计算 hsic_global[i,j],
        # 峰值显存降到 O(chunk × patch_num × rff), 与通道数无关。
        # hsic_global[i,j] = || <z_i, z_j>_patch ||^2 (线性核RFF近似),
        # = sum_r ( sum_t z[i,t,r]*z[j,t,r] )^2 / patch_num^2。
        hsic_global = torch.empty(bs, nvars, nvars, device=x.device, dtype=z_centered.dtype)
        chunk = 64  # 每块的i通道数, 调小可进一步降低峰值显存
        for i0 in range(0, nvars, chunk):
            i1 = min(i0 + chunk, nvars)
            zi = z_centered[:, i0:i1, :, :]                 # [bs, ci, patch_num, rff]
            # 与所有j通道做内积: [bs, ci, nv, patch_num]
            inner = (zi.unsqueeze(2) * z_centered.unsqueeze(1)).sum(dim=-1)
            hsic_block = (inner ** 2).mean(dim=-1) / patch_num  # [bs, ci, nv]
            hsic_global[:, i0:i1, :] = hsic_block

        # 没有跨环境稳定性概念，直接用HSIC值做门控
        hsic_norm = hsic_global / (hsic_global.max() + 1e-8)

        prior = torch.sigmoid(self.channel_prior)
        score = hsic_norm * (1 - self.prior_weight) + prior.unsqueeze(0) * self.prior_weight
        gate = self.gate_mlp(score.unsqueeze(-1)).squeeze(-1)

        eye = torch.eye(self.n_vars, device=x.device).unsqueeze(0)
        gate = gate * (1 - eye) + eye
        return gate

    def get_diagnostics(self):
        """返回门控相关可学习参数诊断信息，供消融可观测性插桩使用。"""
        prior_sig = torch.sigmoid(self.channel_prior).detach()
        return {
            'gate_type': 'GlobalHSICGate',
            'channel_prior_sig_mean': float(prior_sig.mean()),
            'channel_prior_sig_min': float(prior_sig.min()),
            'channel_prior_sig_max': float(prior_sig.max()),
            'prior_weight': float(self.prior_weight),
        }


class NoEnv_ChannelInteraction(nn.Module):
    """变体: 不划分环境，全局HSIC"""
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.3):
        super().__init__()
        self.prior_weight = prior_weight
        self.stability_gate = GlobalHSICGate(n_vars, d_model, rff_dim=rff_dim,
                                             prior_weight=prior_weight)
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

    def get_diagnostics(self):
        return self.stability_gate.get_diagnostics()


# ============================================================
# 变体3: w/o Gate — 全连接通道注意力，无门控
# ============================================================

class NoGate_ChannelInteraction(nn.Module):
    """变体: 所有通道全连接注意力，不做门控选择"""
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.3):
        super().__init__()
        self.n_vars = n_vars
        self.prior_weight = prior_weight
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

    def get_diagnostics(self):
        return None


# ============================================================
# 变体4: 控制容量对照 — 纯可学习门控 (LearnedGate)
# 与 full_v2 参数规模完全匹配(含 N×N 可学习门控矩阵 channel_prior),
# 但门控完全由学习矩阵决定, 不经任何因果/稳定性逻辑(HSIC).
# 目的: 隔离 "因果稳定性逻辑" vs "可学习容量" —— 若它也能赢, 则增益来自容量/过拟合.
# ============================================================

class PureLearnedGate(nn.Module):
    """纯可学习 N×N 通道门控矩阵, 不走任何因果/稳定性逻辑。
    参数规模与 full_v2 的 channel_prior 完全匹配 (N×N), 用于容量控制对照。"""
    def __init__(self, n_vars, d_model, prior_weight: float = 0.05,
                 temperature: float = 1.0, **kwargs):
        super().__init__()
        self.n_vars = n_vars
        self.prior_weight = prior_weight
        self.channel_prior = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.temperature = temperature

    def forward(self, x):
        # x: [bs, nvars, patch_num, d_model] (未使用, 门控纯学习)
        bs = x.shape[0]
        gate = torch.sigmoid(self.prior_weight * self.channel_prior / self.temperature)
        eye = torch.eye(self.n_vars, device=gate.device)
        gate = gate * (1 - eye) + eye
        return gate.unsqueeze(0).expand(bs, self.n_vars, self.n_vars)

    def get_diagnostics(self):
        return {'gate_type': 'PureLearnedGate',
                'channel_prior_sig_mean': float(torch.sigmoid(self.channel_prior).mean()),
                'prior_weight': float(self.prior_weight)}


class LearnedGate_ChannelInteraction(nn.Module):
    """容量匹配对照: 与 full_v2 相同通道注意力骨架 + 相同规模可学习门控,
    但门控由纯学习矩阵决定(无 HSIC 因果约束)。"""
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.05,
                 temporal_mix: bool = True, alpha_init: float = -2.0, **kwargs):
        super().__init__()
        self.stability_gate = PureLearnedGate(n_vars, d_model, prior_weight=prior_weight)
        if temporal_mix:
            self.channel_attn = CausalChannelAttentionTemporal(d_model, n_heads, n_vars, dropout)
        else:
            self.channel_attn = CausalChannelAttention(d_model, n_heads, n_vars, dropout)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.alpha = nn.Parameter(torch.full((n_vars,), float(alpha_init)))

    def forward(self, x):
        bs, nvars, d_model, patch_num = x.shape
        x_for_gate = x.permute(0, 1, 3, 2)
        gate_matrix = self.stability_gate(x_for_gate)
        alpha_vec = torch.sigmoid(self.alpha).view(1, nvars, 1, 1)
        if isinstance(self.channel_attn, CausalChannelAttentionTemporal):
            x_channel = self.channel_attn(x, gate_matrix)
            x_ch = x_channel.permute(0, 1, 3, 2)
            x_ch = self.fusion_proj(x_ch).permute(0, 1, 3, 2)
            out = (1 - alpha_vec) * x + alpha_vec * x_ch
        else:
            x_pooled = x.mean(dim=-1)
            x_channel = self.channel_attn(x_pooled, gate_matrix)
            x_channel_proj = self.fusion_proj(x_channel).unsqueeze(-1).expand_as(x)
            out = (1 - alpha_vec) * x + alpha_vec * x_channel_proj
        return out, gate_matrix

    def get_diagnostics(self):
        return self.stability_gate.get_diagnostics()


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
                 channel_dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.3,
                 env_mode: str = 'uniform', **kwargs):
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

        ci_kwargs = dict(n_vars=c_in, d_model=d_model, patch_num=patch_num,
                         n_heads=n_channel_heads, n_envs=n_envs, rff_dim=rff_dim,
                         dropout=channel_dropout, fusion_alpha=fusion_alpha,
                         prior_weight=prior_weight)
        # 修 C (2026-08-12): 仅把 env_mode 传给支持它的交互类
        # (CausalChannelInteraction 系); 其余变体 (NoEnv/NoGate/LearnedGate 等) 不接受。
        if 'env_mode' in inspect.signature(channel_interaction_cls).parameters:
            ci_kwargs['env_mode'] = env_mode
        self.causal_channel = channel_interaction_cls(**ci_kwargs)
        # 语义模式 forward 需要接收 env_labels (仅 CausalChannelInteraction 支持)
        self._ci_accepts_env = ('env_labels' in
                                inspect.signature(self.causal_channel.forward).parameters)

        self.head_nf = d_model * patch_num
        self.head = Flatten_Head(individual, c_in, self.head_nf,
                                 target_window, head_dropout=0)
        self.last_gate_matrix = None

    def forward(self, z, env_labels=None):
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)
        if env_labels is not None and self._ci_accepts_env:
            z, gate_matrix = self.causal_channel(z, env_labels)
        else:
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

    def get_diagnostics(self):
        if hasattr(self.causal_channel, 'get_diagnostics'):
            return self.causal_channel.get_diagnostics()
        return None

    def get_gate_entropy(self):
        """P1: 熵正则接口。转发给通道交互模块 (各 ChannelInteraction 若实现
        get_last_entropy 则返回门控熵, 否则返回 None 由 trainer 跳过)。
        修复(2026-08-10): 之前 AblationModel/AblationBackbone 均无此方法,
        导致走 AblationModel 的变体 (gate_prior_only 等) 永远不触发熵正则。"""
        if hasattr(self.causal_channel, 'get_last_entropy'):
            return self.causal_channel.get_last_entropy()
        return None


class PriorOnly_ChannelInteraction(nn.Module):
    """诊断对照 (gate_prior_only): 与 full_v2 完全一致的门控结构
    (gate_mlp + 可学习温度 + 通道先验 + temporal_mix + per_channel_alpha),
    但 gate 只吃 channel_prior, 稳定性/HSIC 信号被完全剥离
    (CausalStabilityGate(prior_only=True)). 用于验证 full_v2 的提升是否
    真正来自因果稳定性信号, 而非 gate 结构/参数本身 (回应评审刀1)。"""

    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4, rff_dim=32,
                 dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.05,
                 temperature: float = 0.5, temporal_mix: bool = True,
                 stability_v2: bool = True, per_channel_alpha: bool = True,
                 alpha_init: float = -2.0):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.patch_num = patch_num
        self.fusion_alpha = fusion_alpha
        self.prior_weight = prior_weight
        self.temporal_mix = temporal_mix
        self.per_channel_alpha = per_channel_alpha
        self.stability_gate = CausalStabilityGate(
            n_vars=n_vars, d_model=d_model, n_envs=n_envs, rff_dim=rff_dim,
            prior_weight=prior_weight, temperature=temperature,
            stability_v2=stability_v2, prior_only=True,
        )
        if temporal_mix:
            self.channel_attn = CausalChannelAttentionTemporal(
                d_model=d_model, n_heads=n_heads, n_vars=n_vars, dropout=dropout)
        else:
            self.channel_attn = CausalChannelAttention(
                d_model=d_model, n_heads=n_heads, n_vars=n_vars, dropout=dropout)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        if per_channel_alpha:
            init = alpha_init if alpha_init is not None else -2.0
            self.alpha = nn.Parameter(torch.full((n_vars,), float(init)))
        else:
            init = alpha_init if alpha_init is not None else fusion_alpha
            self.alpha = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        bs, nvars, d_model, patch_num = x.shape
        x_for_gate = x.permute(0, 1, 3, 2)   # [bs, nvars, patch_num, d_model]
        gate_matrix = self.stability_gate(x_for_gate)
        self.last_gate_matrix = gate_matrix.detach()
        if self.per_channel_alpha:
            alpha_vec = torch.sigmoid(self.alpha).view(1, nvars, 1, 1)
        else:
            alpha_vec = torch.sigmoid(self.alpha)
        if self.temporal_mix:
            x_channel = self.channel_attn(x, gate_matrix)         # [bs, nvars, d_model, patch_num]
            x_ch = x_channel.permute(0, 1, 3, 2)                  # [bs, nvars, patch_num, d_model]
            x_ch = self.fusion_proj(x_ch).permute(0, 1, 3, 2)     # 回到 [bs, nvars, d_model, patch_num]
            out = (1 - alpha_vec) * x + alpha_vec * x_ch
        else:
            x_pooled = x.mean(dim=-1)             # [bs, nvars, d_model]
            x_channel = self.channel_attn(x_pooled, gate_matrix)
            x_channel_proj = self.fusion_proj(x_channel)
            x_channel_expand = x_channel_proj.unsqueeze(-1).expand_as(x)
            out = (1 - alpha_vec) * x + alpha_vec * x_channel_expand
        return out, gate_matrix

    def get_gate_matrix(self):
        return self.last_gate_matrix

    def get_last_entropy(self):
        return self.stability_gate.last_entropy

    def get_diagnostics(self):
        return self.stability_gate.get_diagnostics()


class PCD_ChannelInteraction(nn.Module):
    """PCD (ICASSP'26, Dataset-Driven Channel Masks) 静态相关掩码门控对照变体.

    与 full_v2 完全相同的通道注意力骨架 (CausalChannelAttentionTemporal +
    fusion_proj + per_channel_alpha 优雅回退), 唯一差异在门控来源:
      - full_v2  : 跨环境 HSIC 稳定性 (输入相关, 识别"强且稳定"的依赖)
      - pcd_gate : 数据集级静态 Pearson 相关掩码 (不随输入变化, 无稳定性概念)
    M = sigmoid(exp(scale_log) * (R - R.mean()) + beta),  R = |Pearson corr|.

    用途: 最小可证伪测试 —— 若静态相关掩码也获得与 full_v2 相同增益,
    则因果稳定性信号没有增量价值; 若在 OOD (虚假相关强度随环境漂移) 下
    pcd_gate 明显更差, 则证明"跨环境稳定性"是必要成分。
    """
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.05,
                 temperature: float = 0.5, temporal_mix: bool = True,
                 alpha_init: float = -2.0, **kwargs):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.patch_num = patch_num
        self.prior_weight = prior_weight
        self.temperature = temperature
        self.temporal_mix = temporal_mix
        # 数据集级静态相关掩码 (初始为单位阵, 训练前由 set_corr_matrix 注入)
        self.register_buffer('R', torch.eye(n_vars))
        self.register_buffer('_corr_ready', torch.tensor(0))
        self.scale_log = nn.Parameter(torch.tensor(0.0))  # scale = exp(scale_log), 原论文 alpha_ds
        self.beta = nn.Parameter(torch.tensor(0.0))
        if temporal_mix:
            self.channel_attn = CausalChannelAttentionTemporal(
                d_model=d_model, n_heads=n_heads, n_vars=n_vars, dropout=dropout)
        else:
            self.channel_attn = CausalChannelAttention(
                d_model=d_model, n_heads=n_heads, n_vars=n_vars, dropout=dropout)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        init = alpha_init if alpha_init is not None else -2.0
        self.alpha = nn.Parameter(torch.full((n_vars,), float(init)))

    def set_corr_matrix(self, corr):
        """注入数据集级相关矩阵 (训练前调用). corr: [nvars, nvars] 绝对相关值."""
        self.R = corr.abs().detach().float()
        self._corr_ready = torch.tensor(1)

    def forward(self, x):
        bs, nvars, d_model, patch_num = x.shape
        R = self.R.to(x.device) if bool(self._corr_ready) else torch.eye(nvars, device=x.device)
        R_bar = R - R.mean()
        M = torch.sigmoid(torch.exp(self.scale_log) * R_bar + self.beta)
        eye = torch.eye(nvars, device=x.device)
        M = M * (1 - eye) + eye
        gate_matrix = M.unsqueeze(0).expand(bs, nvars, nvars)
        alpha_vec = torch.sigmoid(self.alpha).view(1, nvars, 1, 1)
        if self.temporal_mix:
            x_channel = self.channel_attn(x, gate_matrix)         # [bs, nvars, d_model, patch_num]
            x_ch = x_channel.permute(0, 1, 3, 2)
            x_ch = self.fusion_proj(x_ch).permute(0, 1, 3, 2)
            out = (1 - alpha_vec) * x + alpha_vec * x_ch
        else:
            x_pooled = x.mean(dim=-1)
            x_channel = self.channel_attn(x_pooled, gate_matrix)
            x_channel_proj = self.fusion_proj(x_channel).unsqueeze(-1).expand_as(x)
            out = (1 - alpha_vec) * x + alpha_vec * x_channel_proj
        self.last_gate_matrix = gate_matrix.detach()
        return out, gate_matrix

    def get_gate_matrix(self):
        return self.last_gate_matrix

    def get_diagnostics(self):
        return {'gate_type': 'PCDStaticMask',
                'scale': float(torch.exp(self.scale_log).detach()),
                'beta': float(self.beta.detach()),
                'corr_ready': bool(self._corr_ready)}


class AblationModel(nn.Module):
    """通用消融模型包装"""
    def __init__(self, channel_interaction_cls, enc_in, seq_len, pred_len,
                 e_layers=3, n_heads=4, d_model=64, d_ff=256,
                 dropout=0.2, patch_len=16, stride=8, padding_patch='end',
                 n_channel_heads=4, n_envs=4, rff_dim=64,
                 channel_dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.3,
                 env_mode: str = 'uniform', **kwargs):
        super().__init__()
        self.model = AblationBackbone(
            c_in=enc_in, context_window=seq_len, target_window=pred_len,
            patch_len=patch_len, stride=stride,
            channel_interaction_cls=channel_interaction_cls,
            n_layers=e_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout, padding_patch=padding_patch,
            n_channel_heads=n_channel_heads, n_envs=n_envs, rff_dim=rff_dim,
            channel_dropout=channel_dropout, fusion_alpha=fusion_alpha,
            prior_weight=prior_weight, env_mode=env_mode,
        )

    def forward(self, x, env_labels=None):
        x = x.permute(0, 2, 1)
        x = self.model(x, env_labels)
        x = x.permute(0, 2, 1)
        return x

    def get_gate_matrix(self):
        return self.model.get_gate_matrix()

    def get_diagnostics(self):
        return self.model.get_diagnostics()

    def get_gate_entropy(self):
        """P1: 熵正则接口, 转发给 backbone (同上, 2026-08-10 修复)。"""
        return self.model.get_gate_entropy()


# ============================================================
# 新增 baseline: DLinear (AAAI 2023) / iTransformer (ICLR 2024)
# 与 PatchTST/AblationModel 同一外层接口: forward(x) 输入 [bs, seq_len, nvars],
# 输出 [bs, pred_len, nvars] (与 dataloader 的 batch_x/batch_y 对齐)。
# 多余 kwargs 由 **kwargs 吸收, 以便 run_large.build_kwargs 公共参数直接透传。
# ============================================================

class DLinear(nn.Module):
    """DLinear (cure-lab/LTSF-Linear): 移动平均分解趋势/季节, 各接线性层。
    参考官方 DLinear.py, 自包含实现 (复用 models.layers.series_decomp)。"""
    def __init__(self, enc_in, seq_len, pred_len, dropout=0.2, individual=False,
                 kernel_size=25, **kwargs):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.decomp = series_decomp(kernel_size)
        if individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):  # x: [bs, seq_len, nvars]
        seasonal, trend = self.decomp(x)     # [bs, seq_len, nvars] × 2
        seasonal = seasonal.permute(0, 2, 1)  # [bs, nvars, seq_len]
        trend = trend.permute(0, 2, 1)
        if self.individual:
            out_s = torch.stack([self.Linear_Seasonal[i](seasonal[:, i])
                                 for i in range(self.enc_in)], dim=1)
            out_t = torch.stack([self.Linear_Trend[i](trend[:, i])
                                 for i in range(self.enc_in)], dim=1)
        else:
            out_s = self.Linear_Seasonal(seasonal)  # [bs, nvars, pred_len]
            out_t = self.Linear_Trend(trend)
        out = out_s + out_t
        return out.permute(0, 2, 1)  # [bs, pred_len, nvars]


class iTransformerModel(nn.Module):
    """iTransformer (thuml/iTransformer, ICLR 2024) 适配版: 倒置 Transformer,
    以变量为 token 做通道维度自注意力。参考官方实现, 自包含 (无时间戳 embedding,
    因为本 pipeline 数据不含 time-mark; 仅编码器 + 投影头)。
    输入 [bs, seq_len, nvars], 输出 [bs, pred_len, nvars]。
    """
    def __init__(self, enc_in, seq_len, pred_len, e_layers=3, n_heads=4,
                 d_model=64, d_ff=256, dropout=0.2, use_norm=True, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        # 倒置嵌入: 每变量整个回看窗口 -> d_model token
        self.enc_embedding = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x):  # x: [bs, seq_len, nvars]
        x = x.permute(0, 2, 1)              # [bs, nvars, seq_len]
        if self.use_norm:  # Non-stationary Transformer 式实例归一化
            means = x.mean(dim=-1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        enc_out = self.enc_embedding(x)     # [bs, nvars, d_model]
        enc_out = self.encoder(enc_out)     # [bs, nvars, d_model] (通道维注意力)
        dec_out = self.projector(enc_out)   # [bs, nvars, pred_len]
        if self.use_norm:
            dec_out = dec_out * stdev + means
        return dec_out.permute(0, 2, 1)     # [bs, pred_len, nvars]


# ============================================================
# 工厂函数: 创建各变体
# ============================================================

def create_ablation_model(variant, **kwargs):
    """
    variant:
        'full'          — 完整CausalCIT (A+B+C)
        'full_fix'      — 完整CausalCIT, 降低先验权重(prior_weight=0.1), 用于诊断先验主导假设
        'full_v2'       — SOTA改进版 (temporal_mix + stability_v2 batch池化门控 + per_channel_alpha)
        'full_v2_fixed' — full_v2 + running_stats修复(门控不再依赖测试batch组成), 用于A/B对照
        'no_hsic'       — 去掉HSIC，用Pearson相关性
        'no_env'        — 去掉环境划分，全局HSIC
        'no_gate'       — 去掉门控，全连接通道注意力
        'learned_gate'  — 与'capacity_match'实现完全相同(见下方注释), 不要重复计入统计检验
        'gate_prior_only' — 与full_v2门控结构相同但剥离HSIC/稳定性信号的诊断对照
        'capacity_match'  — 与full_v2容量匹配的纯学习门控对照(无因果/稳定性逻辑)
        'patchtst'      — 纯PatchTST (无通道交互)
    """
    if variant == 'full':
        return CausalCIT(**kwargs)
    elif variant == 'full_fix':
        return CausalCIT(**{**kwargs, 'prior_weight': 0.1})
    elif variant == 'full_v2':
        # SOTA改进版: 时间分辨率保留的通道交互 + 批量池化HSIC稳定性门控(v2)
        #             + 低先验权重 + 温度锐化门控
        v2 = {k: v for k, v in kwargs.items()
              if k not in ('prior_weight', 'temperature', 'temporal_mix', 'stability_v2',
                           'per_channel_alpha', 'alpha_init', 'running_stats')}
        return CausalCIT(prior_weight=kwargs.get('prior_weight', 0.05),
                         temperature=kwargs.get('temperature', 0.5),
                         temporal_mix=True, stability_v2=True,
                         per_channel_alpha=True, alpha_init=kwargs.get('alpha_init', -2.0),
                         running_stats=kwargs.get('running_stats', False),
                         **v2)
    elif variant == 'full_v2_fixed':
        # 回应评审 re2 §2.2/§6.1: 与 full_v2 完全一致，唯一区别是
        # stability_v2 门控改用 running_stats (BatchNorm式EMA population统计量)，
        # 消除"测试预测依赖测试batch组成"的问题。用于在同一协议下与 full_v2 直接对照，
        # 判断该bug修复前后 MSE/门控行为是否发生实质变化。
        v2 = {k: v for k, v in kwargs.items()
              if k not in ('prior_weight', 'temperature', 'temporal_mix', 'stability_v2',
                           'per_channel_alpha', 'alpha_init', 'running_stats')}
        return CausalCIT(prior_weight=kwargs.get('prior_weight', 0.05),
                         temperature=kwargs.get('temperature', 0.5),
                         temporal_mix=True, stability_v2=True,
                         per_channel_alpha=True, alpha_init=kwargs.get('alpha_init', -2.0),
                         running_stats=True,
                         **v2)
    elif variant == 'no_hsic':
        return AblationModel(NoHSIC_ChannelInteraction, **kwargs)
    elif variant == 'no_env':
        return AblationModel(NoEnv_ChannelInteraction, **kwargs)
    elif variant == 'no_gate':
        return AblationModel(NoGate_ChannelInteraction, **kwargs)
    elif variant == 'learned_gate':
        # 注意(回应评审re2 §2.4): 'learned_gate' 与下面的 'capacity_match' 是
        # 完全相同的实现 (同一个类 LearnedGate_ChannelInteraction, 同样的kwargs)，
        # 两者不构成两条独立证据。报告/统计检验中不要把它们当作两个不同的对照组
        # 重复计入 —— 只保留其中一个(建议用语义更明确的 'capacity_match')。
        return AblationModel(LearnedGate_ChannelInteraction,
                             **{k: v for k, v in kwargs.items()})
    elif variant == 'gate_prior_only':
        # 诊断对照 (回应评审刀1): 与 full_v2 完全一致的门控结构
        # (gate_mlp + 可学习温度 + 通道先验 + temporal_mix + per_channel_alpha),
        # 但剥离稳定性/HSIC 信号 (CausalStabilityGate(prior_only=True)),
        # 用以验证 full_v2 的提升是否真来自因果稳定性信号。
        return AblationModel(PriorOnly_ChannelInteraction,
                             **{k: v for k, v in kwargs.items()})
    elif variant == 'capacity_match':
        # 容量匹配对照 (回应评审刀2): 与 full_v2 同参数规模的标准通道注意力,
        # 门控由纯学习矩阵决定, 不经任何因果/稳定性(HSIC)逻辑。
        # (与 'learned_gate' 实现完全相同，见上方注释 —— 只是这里是本方法族的"正式"命名)
        return AblationModel(LearnedGate_ChannelInteraction,
                             **{k: v for k, v in kwargs.items()})
    elif variant == 'pcd_gate':
        # PCD (ICASSP'26) 静态相关掩码对照: 数据集级 |Pearson corr| 掩码门控,
        # 无跨环境稳定性/HSIC. 需在训练前由 run_pcd_compare 注入 corr 矩阵.
        return AblationModel(PCD_ChannelInteraction,
                             **{k: v for k, v in kwargs.items()})
    elif variant == 'dlinear':
        # 强 CI 基线 (AAAI 2023): 分解线性, 无通道交互 (回应审稿"补 baseline")
        dl_keys = ['enc_in', 'seq_len', 'pred_len', 'dropout']
        dl_kwargs = {k: v for k, v in kwargs.items() if k in dl_keys}
        return DLinear(**dl_kwargs)
    elif variant == 'itransformer':
        # 通道注意力基线 (ICLR 2024): 倒置 Transformer (回应审稿"补 baseline")
        it_keys = ['enc_in', 'seq_len', 'pred_len', 'e_layers', 'n_heads',
                   'd_model', 'd_ff', 'dropout']
        it_kwargs = {k: v for k, v in kwargs.items() if k in it_keys}
        return iTransformerModel(**it_kwargs)
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
