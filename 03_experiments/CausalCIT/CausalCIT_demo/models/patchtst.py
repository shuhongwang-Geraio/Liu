"""
PatchTST Baseline 模型（自包含实现）
Channel-Independent Patch Time Series Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch import Tensor

from models.layers import (
    Transpose, get_activation_fn, positional_encoding,
    series_decomp, RevIN
)


# ======================== Attention ========================

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False,
                 attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads,
                                                   attn_dropout=attn_dropout,
                                                   res_attention=res_attention, lsa=lsa)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s,
                                                 key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


# ======================== Encoder ========================

class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256,
                 store_attn=False, norm='BatchNorm', attn_dropout=0, dropout=0.,
                 bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v,
                                             attn_dropout=attn_dropout,
                                             proj_dropout=dropout,
                                             res_attention=res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias), get_activation_fn(activation),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model, bias=bias))
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm

    def forward(self, src, prev=None, key_padding_mask=None, attn_mask=None):
        if self.pre_norm: src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev,
                                                key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        src = src + self.dropout_attn(src2)
        if not self.pre_norm: src = self.norm_attn(src)
        if self.pre_norm: src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm: src = self.norm_ffn(src)
        if self.res_attention: return src, scores
        else: return src


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                            d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                            dropout=dropout, activation=activation,
                            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn)
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTiEncoder(nn.Module):
    """Channel-Independent Encoder：所有通道共享同一Transformer"""
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0.,
                 act="gelu", store_attn=False, key_padding_mask='auto',
                 padding_var=None, attn_mask=None, res_attention=True,
                 pre_norm=False, pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                  norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention,
                                  n_layers=n_layers, store_attn=store_attn)

    def forward(self, x):
        # x: [bs, nvars, patch_len, patch_num]
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)          # [bs, nvars, patch_num, patch_len]
        x = self.W_P(x)                     # [bs, nvars, patch_num, d_model]
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)          # [bs, nvars, d_model, patch_num]
        return z


# ======================== Head ========================

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [bs, nvars, d_model, patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


# ======================== PatchTST Backbone ========================

class PatchTST_backbone(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, stride,
                 max_seq_len=1024, n_layers=3, d_model=128, n_heads=16,
                 d_k=None, d_v=None, d_ff=256, norm='BatchNorm',
                 attn_dropout=0., dropout=0., act="gelu",
                 key_padding_mask='auto', padding_var=None, attn_mask=None,
                 res_attention=True, pre_norm=False, store_attn=False,
                 pe='zeros', learn_pe=True, fc_dropout=0., head_dropout=0,
                 padding_patch=None, pretrain_head=False, head_type='flatten',
                 individual=False, revin=True, affine=True, subtract_last=False,
                 verbose=False, **kwargs):
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
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len,
                                    max_seq_len=max_seq_len, n_layers=n_layers,
                                    d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                                    d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                                    act=act, key_padding_mask=key_padding_mask,
                                    padding_var=padding_var, attn_mask=attn_mask,
                                    res_attention=res_attention, pre_norm=pre_norm,
                                    store_attn=store_attn, pe=pe, learn_pe=learn_pe,
                                    verbose=verbose, **kwargs)
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual
        if head_type == 'flatten':
            self.head = Flatten_Head(individual, c_in, self.head_nf,
                                    target_window, head_dropout=head_dropout)

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
        z = self.backbone(z)
        z = self.head(z)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z


# ======================== PatchTST Model ========================

class PatchTST(nn.Module):
    """PatchTST完整模型（Baseline）"""
    def __init__(self, enc_in, seq_len, pred_len, e_layers=3, n_heads=4,
                 d_model=16, d_ff=128, dropout=0.3, fc_dropout=0.3,
                 head_dropout=0.0, patch_len=16, stride=8,
                 padding_patch='end', individual=False,
                 revin=True, affine=True, subtract_last=False,
                 decomposition=False, kernel_size=25, **kwargs):
        super().__init__()
        self.decomposition = decomposition
        backbone_kwargs = dict(
            c_in=enc_in, context_window=seq_len, target_window=pred_len,
            patch_len=patch_len, stride=stride, n_layers=e_layers,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout,
            padding_patch=padding_patch, individual=individual,
            revin=revin, affine=affine, subtract_last=subtract_last,
        )
        if decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(**backbone_kwargs)
            self.model_res = PatchTST_backbone(**backbone_kwargs)
        else:
            self.model = PatchTST_backbone(**backbone_kwargs)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init = res_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)    # [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)    # [Batch, Pred length, Channel]
        return x
