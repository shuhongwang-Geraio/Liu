"""
CausalCIT: 因果通道交互Transformer (Causal Channel Interaction Transformer)

在PatchTST基础上增加因果稳定性驱动的通道交互机制。
数据流:
    输入 [bs, nvars, seq_len]
    -> RevIN norm
    -> Patching -> [bs, nvars, patch_num, patch_len]
    -> Channel-Independent Encoder -> [bs, nvars, d_model, patch_num]
    -> ★ 因果通道交互 -> [bs, nvars, d_model, patch_num]
    -> Flatten Head -> [bs, nvars, target_window]
    -> RevIN denorm
"""

import torch
import torch.nn as nn
from typing import Optional

from models.layers import RevIN, series_decomp
from models.patchtst import TSTiEncoder, Flatten_Head
from models.causal_channel import CausalChannelInteraction


class CausalCIT_backbone(nn.Module):
    """CausalCIT骨干网络：PatchTST + 因果通道交互"""
    def __init__(self, c_in, context_window, target_window, patch_len, stride,
                 max_seq_len=1024, n_layers=3, d_model=128, n_heads=16,
                 d_k=None, d_v=None, d_ff=256, norm='BatchNorm',
                 attn_dropout=0., dropout=0., act="gelu",
                 key_padding_mask='auto', padding_var=None, attn_mask=None,
                 res_attention=True, pre_norm=False, store_attn=False,
                 pe='zeros', learn_pe=True, fc_dropout=0., head_dropout=0,
                 padding_patch=None, pretrain_head=False, head_type='flatten',
                 individual=False, revin=True, affine=True, subtract_last=False,
                 verbose=False,
                 # CausalCIT 专用参数
                 n_channel_heads=4, n_envs=4, rff_dim=32,
                 channel_dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.3,
                 temporal_mix: bool = False, temperature: float = 1.0,
                 stability_v2: bool = False, per_channel_alpha: bool = False,
                 alpha_init: float = None, **kwargs):
        super().__init__()

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Channel-Independent Encoder (复用PatchTST)
        self.backbone = TSTiEncoder(
            c_in, patch_num=patch_num, patch_len=patch_len,
            max_seq_len=max_seq_len, n_layers=n_layers,
            d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
            d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
            act=act, key_padding_mask=key_padding_mask,
            padding_var=padding_var, attn_mask=attn_mask,
            res_attention=res_attention, pre_norm=pre_norm,
            store_attn=store_attn, pe=pe, learn_pe=learn_pe,
            verbose=verbose, **kwargs
        )

        # ★ 因果通道交互模块 (核心创新)
        self.causal_channel = CausalChannelInteraction(
            n_vars=c_in, d_model=d_model, patch_num=patch_num,
            n_heads=n_channel_heads, n_envs=n_envs, rff_dim=rff_dim,
            dropout=channel_dropout, fusion_alpha=fusion_alpha,
            prior_weight=prior_weight,
            temporal_mix=temporal_mix, temperature=temperature,
            stability_v2=stability_v2,
            per_channel_alpha=per_channel_alpha, alpha_init=alpha_init,
        )

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual
        if head_type == 'flatten':
            self.head = Flatten_Head(individual, c_in, self.head_nf,
                                    target_window, head_dropout=head_dropout)

        self.last_gate_matrix = None

    def forward(self, z):
        # z: [bs, nvars, seq_len]
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        z = self.backbone(z)                           # [bs, nvars, d_model, patch_num]
        z, gate_matrix = self.causal_channel(z)        # ★ 因果通道交互
        self.last_gate_matrix = gate_matrix.detach()
        z = self.head(z)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    def get_gate_matrix(self):
        return self.last_gate_matrix

    def get_gate_entropy(self):
        """P1优化: 门控熵，供Trainer作为辅助正则项，鼓励gate做出果断的0/1选择"""
        return self.causal_channel.get_last_entropy()

    def get_diagnostics(self):
        """返回门控相关可学习参数诊断信息，供消融可观测性插桩使用"""
        return self.causal_channel.get_diagnostics()


class CausalCIT(nn.Module):
    """CausalCIT完整模型"""
    def __init__(self, enc_in, seq_len, pred_len, e_layers=3, n_heads=4,
                 d_model=16, d_ff=128, dropout=0.3, fc_dropout=0.3,
                 head_dropout=0.0, patch_len=16, stride=8,
                 padding_patch='end', individual=False,
                 revin=True, affine=True, subtract_last=False,
                 decomposition=False, kernel_size=25,
                 # CausalCIT 专用
                 n_channel_heads=4, n_envs=4, rff_dim=32,
                 channel_dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.3,
                 temporal_mix: bool = False, temperature: float = 1.0,
                 stability_v2: bool = False, per_channel_alpha: bool = False,
                 alpha_init: float = None, **kwargs):
        super().__init__()
        self.decomposition = decomposition
        backbone_kwargs = dict(
            c_in=enc_in, context_window=seq_len, target_window=pred_len,
            patch_len=patch_len, stride=stride, n_layers=e_layers,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout,
            padding_patch=padding_patch, individual=individual,
            revin=revin, affine=affine, subtract_last=subtract_last,
            n_channel_heads=n_channel_heads, n_envs=n_envs, rff_dim=rff_dim,
            channel_dropout=channel_dropout, fusion_alpha=fusion_alpha,
            prior_weight=prior_weight,
            temporal_mix=temporal_mix, temperature=temperature,
            stability_v2=stability_v2,
            per_channel_alpha=per_channel_alpha, alpha_init=alpha_init,
        )
        if decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = CausalCIT_backbone(**backbone_kwargs)
            self.model_res = CausalCIT_backbone(**backbone_kwargs)
        else:
            self.model = CausalCIT_backbone(**backbone_kwargs)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init = res_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = (res + trend).permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.model(x)
            x = x.permute(0, 2, 1)
        return x

    def get_gate_matrix(self):
        if self.decomposition:
            return self.model_res.get_gate_matrix()
        return self.model.get_gate_matrix()

    def get_gate_entropy(self):
        """P1优化: 供Trainer读取门控熵，用于熵正则化loss"""
        if self.decomposition:
            return self.model_res.get_gate_entropy()
        return self.model.get_gate_entropy()

    def get_diagnostics(self):
        """返回门控相关可学习参数诊断信息，供消融可观测性插桩使用"""
        if self.decomposition:
            return self.model_res.get_diagnostics()
        return self.model.get_diagnostics()
