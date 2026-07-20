"""
CausalChannel: 因果稳定性检验驱动的通道交互模块

核心创新:
1. 将时间序列的不同时间段视为不同"环境"
2. 使用RFF近似的HSIC检验通道间依赖在不同环境下的稳定性
3. 仅对通过稳定性检验的通道对施加交叉注意力，其余保持独立

与现有工作的区别:
- Adapformer: 基于相关性强度 -> 本方法基于因果稳定性
- CGTFra: 基于信息瓶颈对齐 -> 本方法基于跨环境HSIC一致性
- CN: 仿射变换区分通道 -> 本方法动态门控通道交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RFFKernel(nn.Module):
    """Random Fourier Features (RFF) 核近似
    将O(n^2)的核矩阵计算降低为O(nD)
    参考: Rahimi & Recht, NeurIPS 2007
    """
    def __init__(self, input_dim, rff_dim=64, sigma=1.0):
        super().__init__()
        self.rff_dim = rff_dim
        self.register_buffer('W', torch.randn(input_dim, rff_dim) / sigma)
        self.register_buffer('b', torch.rand(rff_dim) * 2 * math.pi)

    def forward(self, x):
        # x: [batch, features] -> [batch, rff_dim]
        proj = x @ self.W + self.b
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(proj)


class CausalStabilityGate(nn.Module):
    """因果稳定性门控模块

    将时间序列按时间段划分为多个"环境"，
    检测每对通道间的HSIC依赖在不同环境下是否稳定。

    稳定的通道依赖 → 高门控权重 → 允许通道交互
    不稳定的通道依赖（虚假相关）→ 低门控权重 → 保持通道独立
    """
    def __init__(self, n_vars, d_model, n_envs=4, rff_dim=32,
                 stability_threshold=0.1, temperature=1.0, learn_temperature=True):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_envs = n_envs
        self.rff_dim = rff_dim
        self.rff_kernel = RFFKernel(d_model, rff_dim)
        self.stability_bias = nn.Parameter(torch.zeros(1))
        self.channel_prior = nn.Parameter(torch.zeros(n_vars, n_vars))
        # gate_mlp 输出原始logit（不再内置Sigmoid），配合可学习温度缩放后再做sigmoid。
        # P1优化(温度): T越小 -> sigmoid(logit/T)越陡峭，门控趋向果断的0/1判断；
        #              T越大 -> 门控越平滑保守。初始T=temperature，训练中自适应调节。
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(),
            nn.Linear(16, 1)
        )
        if learn_temperature:
            self.temperature_param = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer('temperature_param', torch.tensor(float(temperature)))
        # P1优化(熵正则化): 记录最近一次forward的门控熵（不含对角线），
        # 供Trainer作为辅助loss加入，鼓励gate远离0.5附近的模糊区间，做出更果断的选择。
        self.last_entropy = None

    def compute_stability_score(self, x):
        """计算通道对的跨环境稳定性分数
        x: [bs, nvars, patch_num, d_model]
        returns: [bs, nvars, nvars]
        """
        bs, nvars, patch_num, d_model = x.shape
        env_size = patch_num // self.n_envs
        if env_size < 2:
            return torch.ones(bs, nvars, nvars, device=x.device)

        x_trunc = x[:, :, :self.n_envs * env_size, :]
        x_envs = x_trunc.reshape(bs, nvars, self.n_envs, env_size, d_model)

        x_flat = x_envs.reshape(-1, d_model)
        z_flat = self.rff_kernel(x_flat)
        z = z_flat.reshape(bs, nvars, self.n_envs, env_size, self.rff_dim)
        z_centered = z - z.mean(dim=3, keepdim=True)

        zi_expand = z_centered.unsqueeze(2)  # [bs, nv, 1, n_envs, env_size, rff]
        zj_expand = z_centered.unsqueeze(1)  # [bs, 1, nv, n_envs, env_size, rff]
        cross_diag = (zi_expand * zj_expand).mean(dim=4)  # [bs, nv, nv, n_envs, rff]
        hsic_per_env = (cross_diag ** 2).sum(dim=-1)       # [bs, nv, nv, n_envs]

        hsic_mean = hsic_per_env.mean(dim=-1).clamp(min=1e-8)
        hsic_std = hsic_per_env.std(dim=-1)
        cv = hsic_std / hsic_mean
        stability = 1.0 / (1.0 + cv + self.stability_bias.abs())
        return stability

    def forward(self, x):
        """
        x: [bs, nvars, patch_num, d_model]
        returns: gate_matrix [bs, nvars, nvars] ∈ [0,1]
        """
        stability = self.compute_stability_score(x)
        prior = torch.sigmoid(self.channel_prior)
        stability = stability * 0.7 + prior.unsqueeze(0) * 0.3
        logit = self.gate_mlp(stability.unsqueeze(-1)).squeeze(-1)
        # 温度缩放：clamp防止T退化为0/负数导致数值爆炸或方向反转
        temp = self.temperature_param.clamp(min=0.1, max=10.0)
        gate = torch.sigmoid(logit / temp)
        eye = torch.eye(self.n_vars, device=x.device).unsqueeze(0)
        gate = gate * (1 - eye) + eye

        # 熵正则化统计：只统计非对角线（真实待判断的通道对），
        # 越靠近0.5熵越大(模糊)，越靠近0/1熵越小(果断)。
        off_diag_mask = (1 - eye).bool().expand_as(gate)
        p = gate[off_diag_mask].clamp(min=1e-6, max=1 - 1e-6)
        ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        self.last_entropy = ent.mean()
        return gate


class CausalChannelAttention(nn.Module):
    """因果通道交叉注意力：门控矩阵控制通道间信息流"""
    def __init__(self, d_model, n_heads, n_vars, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_vars = n_vars
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.scale = self.d_k ** -0.5

    def forward(self, x, gate_matrix):
        """
        x: [bs, nvars, d_model]
        gate_matrix: [bs, nvars, nvars]
        returns: [bs, nvars, d_model]
        """
        bs, nvars, d_model = x.shape
        residual = x
        Q = self.W_Q(x).view(bs, nvars, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(bs, nvars, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(bs, nvars, self.n_heads, self.d_k).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        gate_mask = gate_matrix.unsqueeze(1)
        # 软门控惩罚：log域加性偏置，而非硬mask的(1-g)*(-1e9)。
        # 后者对任何 g<0.9999 都会把logit压到-1e8量级，softmax后权重≈0，
        # 等价于把soft gate强行二值化。改为log(g)后，惩罚幅度与g平滑对应，
        # g=0.5时惩罚≈-0.69，g=0.1时≈-2.3，量级与attn logits (~O(1~5)) 匹配。
        attn = attn + torch.log(gate_mask.clamp(min=1e-4))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(bs, nvars, d_model)
        out = self.W_O(out)
        out = self.dropout(out)
        out = self.norm(residual + out)
        return out


class CausalChannelInteraction(nn.Module):
    """完整的因果通道交互模块
    组合: 因果稳定性门控 + 通道交叉注意力 + 信息融合
    """
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.patch_num = patch_num
        self.fusion_alpha = fusion_alpha
        self.stability_gate = CausalStabilityGate(
            n_vars=n_vars, d_model=d_model, n_envs=n_envs, rff_dim=rff_dim
        )
        self.channel_attn = CausalChannelAttention(
            d_model=d_model, n_heads=n_heads, n_vars=n_vars, dropout=dropout
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.alpha = nn.Parameter(torch.tensor(fusion_alpha))

    def forward(self, x):
        """
        x: [bs, nvars, d_model, patch_num]
        returns: (out, gate_matrix)
        """
        bs, nvars, d_model, patch_num = x.shape
        x_for_gate = x.permute(0, 1, 3, 2)   # [bs, nvars, patch_num, d_model]
        gate_matrix = self.stability_gate(x_for_gate)
        x_pooled = x.mean(dim=-1)             # [bs, nvars, d_model]
        x_channel = self.channel_attn(x_pooled, gate_matrix)
        x_channel_proj = self.fusion_proj(x_channel)
        x_channel_expand = x_channel_proj.unsqueeze(-1).expand_as(x)
        alpha = torch.sigmoid(self.alpha)
        out = (1 - alpha) * x + alpha * x_channel_expand
        return out, gate_matrix

    def get_last_entropy(self):
        """返回最近一次forward的门控熵（标量tensor或None），供上层作为正则项使用"""
        return self.stability_gate.last_entropy
