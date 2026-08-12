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

    修复 A (2026-08-11, 门1静态诊断确认): 原 sigma=1.0 硬编码导致
    proj = x@W 的 std ≈ sqrt(d_model) ∈ [4, 8] (d_model=16~64), cos(proj) 剧烈震荡,
    RFF 特征退化为伪随机向量, HSIC 估计失去通道区分度。
    新增 sigma_mode='median': 首次 forward 从数据采样估计 median heuristic 带宽
    (σ = 成对距离中位数), 使 proj 差分 O(1), 核恢复区分度。
    默认 'fixed' 保持旧行为, 不影响已有结果复现。
    """
    def __init__(self, input_dim, rff_dim=64, sigma=1.0, sigma_mode='fixed'):
        super().__init__()
        self.rff_dim = rff_dim
        self.sigma_mode = sigma_mode
        self._median_sigma = None
        if sigma_mode == 'fixed':
            self.register_buffer('W', torch.randn(input_dim, rff_dim) / sigma)
            self.register_buffer('b', torch.rand(rff_dim) * 2 * math.pi)
        else:
            # median 模式: 先注册占位, 首次 forward 时用数据估计 σ 后 copy_ 填充
            self.register_buffer('W', torch.empty(input_dim, rff_dim))
            self.register_buffer('b', torch.empty(rff_dim))

    def _init_median_sigma(self, x):
        """median heuristic: σ = 成对欧氏距离的中位数 (采样估计, 零成本, 一次性)。"""
        with torch.no_grad():
            n = x.shape[0]
            k = min(n, 4096)
            if k < 8:
                # 样本过少, 退回默认
                self._median_sigma = 1.0
                self.W.normal_().div_(1.0)
                self.b.uniform_(0, 2 * math.pi)
                return
            idx_a = torch.randint(0, n, (k,), device=x.device)
            idx_b = torch.randint(0, n, (k,), device=x.device)
            dist = (x[idx_a] - x[idx_b]).norm(dim=-1)      # [k]
            med = dist.median().clamp(min=1e-3)
            self._median_sigma = med.item()
            self.W.normal_().div_(med)
            self.b.uniform_(0, 2 * math.pi)

    def forward(self, x):
        # x: [batch, features] -> [batch, rff_dim]
        if self.sigma_mode == 'median' and self._median_sigma is None:
            self._init_median_sigma(x)
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
                 stability_threshold=0.1, temperature=1.0, learn_temperature=True,
                 prior_weight: float = 0.3, stability_v2: bool = False,
                 prior_only: bool = False,
                 running_stats: bool = False, running_momentum: float = 0.1,
                 rff_sigma_mode: str = 'fixed', cka_normalize: bool = False,
                 env_mode: str = 'uniform'):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_envs = n_envs
        self.rff_dim = rff_dim
        self.prior_weight = prior_weight
        self.stability_v2 = stability_v2
        self.prior_only = prior_only
        # 修复 A+B (2026-08-11): rff_sigma_mode='median' 用 median heuristic 带宽;
        # cka_normalize=True 对 HSIC 做 CKA 归一化 (HSIC/√(HSIC_xx·HSIC_yy))。
        # 均默认关闭, 不改变旧行为; 修复版需显式开启 (见 run_large FULL_V2_KWARGS)。
        self.rff_sigma_mode = rff_sigma_mode
        self.cka_normalize = cka_normalize
        # 修 C (2026-08-12): env_mode='uniform' 保持旧行为 (batch 内 patch 均分);
        # env_mode='semantic' 时按样本的语义环境标签 (时间戳: 季节/昼夜/时段) 分组估 HSIC,
        # 由 forward 接收的 env_labels 驱动 (可行性见 docs/diagnostics/2026-08-12_env_split_feasibility.md)。
        self.env_mode = env_mode
        # 修复(回应评审 re2 §2.2): stability_v2 把 batch 维一起池化估 HSIC，
        # 若测试时直接用当前 batch 的 hsic_mean/std，同一个测试样本换一批"同伴"
        # 门控矩阵就会变 —— 预测结果依赖 batch 组成，这在部署/复现上是硬伤。
        # running_stats=True 时采用 BatchNorm 式做法：训练阶段用当前 batch 统计量
        # 更新一份 EMA 全局统计量(population estimate)，推理(eval)阶段只用这份
        # 与 batch 组成无关的全局统计量计算门控，从根本上解耦"测试预测 vs 测试batch组成"。
        # 默认 False，保证不改变已有 full_v2 结果的可复现性；新实验建议开启并与旧版对照。
        self.running_stats = running_stats
        self.running_momentum = running_momentum
        if running_stats:
            self.register_buffer('running_hsic_mean', torch.zeros(n_vars, n_vars))
            self.register_buffer('running_hsic_std', torch.zeros(n_vars, n_vars))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.rff_kernel = RFFKernel(d_model, rff_dim, sigma=1.0, sigma_mode=rff_sigma_mode)
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

    def compute_stability_score_v2(self, x):
        """改进版稳定性分数 (SOTA关键修复)。

        修复旧版两大缺陷:
        1. 旧版稳定性只用跨环境变异系数 CV (1/(1+cv))，完全忽略依赖强度，
           导致【独立通道】(HSIC≈0 但跨环境都稳定) 反而得到高门控。
           新版 = 依赖强度(hsic_mean) × 跨环境一致性(1/(1+cv))，
           只有【强依赖且稳定】的通道对才获得高分。
        2. 旧版把单个样本的 patch_num 切成 n_envs 份 (每份仅~3 patch) 估 HSIC，纯噪声。
           新版把 batch 维一起池化 (m = bs*env_size 个成对样本)，HSIC 估计稳健得多。

        x: [bs, nvars, patch_num, d_model]
        returns: [bs, nvars, nvars] (batch内共享的稳定性分数)
        """
        bs, nvars, patch_num, d_model = x.shape
        n_envs = self.n_envs
        env_size = patch_num // n_envs
        if env_size < 1:
            return torch.ones(bs, nvars, nvars, device=x.device)
        # 优化(2026-08-08): 显式转 fp32。AMP 训练下 backbone 输出可能是 fp16,
        # 而 HSIC 数值很小 (hsic_mean 会被 clamp 到 1e-8 量级), fp16 会丢精度;
        # 这里强制 fp32 保证稳定性分数/门控始终精确, 同时不影响 AMP 对注意力的加速。
        x = x.float()
        z = self.rff_kernel(x.reshape(-1, d_model)).reshape(bs, nvars, patch_num, self.rff_dim)
        z = z[:, :, :n_envs * env_size, :].reshape(bs, nvars, n_envs, env_size, self.rff_dim)
        # [n_envs, nvars, m, rff]，其中 m = bs*env_size 为每环境的成对样本数
        z = z.permute(2, 1, 0, 3, 4).reshape(n_envs, nvars, bs * env_size, self.rff_dim)
        z = z - z.mean(dim=2, keepdim=True)   # 逐环境中心化
        m = z.shape[2]
        # 线性核 HSIC (RFF): K[e,c] = Z_{e,c} Z_{e,c}^T ∈ [m,m]
        # HSIC[e,i,j] = <K[e,i], K[e,j]> / m^2 (gram 矩阵内积)。
        # 用 batched 矩阵乘实现，避免 einsum 'eip,ejp->eij' 物化 [E,C,C,m*m]
        # (C=321 时约 12GB) 导致高维 OOM。
        K = torch.einsum('ecma,ecna->ecmn', z, z)              # [E, C, m, m]
        P = m * m
        Kf = K.reshape(n_envs, nvars, P)                       # [E, C, P]
        # 优化(2026-08-08): 原 for e 循环做 n_envs 次 [C,P]@[P,C] 矩阵乘,
        # 每次单独 kernel 启动。合并为一次 batched bmm, 峰值显存与循环相同
        # (都是 C×P×C 的中间张量), 但减少调度开销, 高维(如 traffic 862 通道)更快。
        hsic = torch.bmm(Kf, Kf.transpose(1, 2)) / P           # [E, C, C]
        hsic_mean = hsic.mean(dim=0)                        # [C,C] 依赖强度
        if self.cka_normalize:
            # 修复 B (2026-08-11): CKA 归一化, 使不同通道对的可比性摆脱 HSIC 尺度
            # 差异 (未归一化 HSIC 跨通道对可差 1-2 个数量级, 淹没稳定性信号)。
            diag = torch.diagonal(hsic_mean, dim1=-2, dim2=-1)  # [C]
            denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2) + 1e-8)
            hsic_mean = hsic_mean / denom.clamp(min=1e-8)
        hsic_std = hsic.std(dim=0)
        if self.running_stats:
            if self.training:
                with torch.no_grad():
                    if self.num_batches_tracked == 0:
                        # 首个batch直接初始化，避免EMA从全零慢慢爬升
                        self.running_hsic_mean.copy_(hsic_mean.detach())
                        self.running_hsic_std.copy_(hsic_std.detach())
                    else:
                        mom = self.running_momentum
                        self.running_hsic_mean.mul_(1 - mom).add_(mom * hsic_mean.detach())
                        self.running_hsic_std.mul_(1 - mom).add_(mom * hsic_std.detach())
                    self.num_batches_tracked += 1
                # 训练阶段仍用当前batch统计量计算门控（梯度来源不变），
                # 只是"顺带"把这一batch的统计量汇入全局EMA供eval使用。
            elif self.num_batches_tracked > 0:
                # 推理阶段：只用训练期累积的全局统计量，与当前测试batch组成无关。
                hsic_mean = self.running_hsic_mean
                hsic_std = self.running_hsic_std
        cv = hsic_std / (hsic_mean + 1e-6)
        stability = hsic_mean / (1.0 + cv + self.stability_bias.abs())
        return stability.unsqueeze(0).expand(bs, nvars, nvars)

    def compute_stability_score_semantic(self, x, env_labels):
        """修 C: 语义环境切分版稳定性分数。

        与 compute_stability_score_v2 的区别: 环境不是"batch 内 patch 均分",
        而是按每个样本的**语义环境标签** (由时间戳解析, 季节/昼夜/时段) 分组。
        每个语义环境内的样本×patch 一起池化估 HSIC, 再跨语义环境看依赖强度
        是否稳定 —— 语义环境经 assess_env_split.py 验证信息量是随机均分的 4-14x。

        x: [bs, nvars, patch_num, d_model]
        env_labels: [bs] 每样本一个语义环境标签 (int)
        returns: [bs, nvars, nvars]
        """
        bs, nvars, patch_num, d_model = x.shape
        x = x.float()
        z = self.rff_kernel(x.reshape(-1, d_model)).reshape(bs, nvars, patch_num, self.rff_dim)
        z = z - z.mean(dim=2, keepdim=True)          # 逐样本中心化
        env_ids = torch.unique(env_labels)
        hsic_envs = []
        for e in env_ids:
            ze = z[env_labels == e]                  # [n_e, nvars, patch_num, rff]
            n_e = ze.shape[0]
            if n_e < 1:
                continue
            ze = ze.permute(1, 0, 2, 3).reshape(nvars, n_e * patch_num, self.rff_dim)
            ze = ze - ze.mean(dim=1, keepdim=True)   # 逐环境中心化
            m = ze.shape[1]
            P = m * m
            K = torch.einsum('cma,cna->cmn', ze, ze)             # [nvars, m, m]
            Kf = K.reshape(nvars, P)                             # [nvars, P]
            hsic_envs.append(torch.mm(Kf, Kf.t()) / P)           # [nvars, nvars]
        if len(hsic_envs) < 2:
            # 单环境无跨环境稳定性可言, 退化为 v2 (patch 均分) 保持行为
            return self.compute_stability_score_v2(x)
        hsic = torch.stack(hsic_envs)                # [E, nvars, nvars]
        hsic_mean = hsic.mean(dim=0).clamp(min=1e-8)
        if self.cka_normalize:
            diag = torch.diagonal(hsic_mean, dim1=-2, dim2=-1)
            denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2) + 1e-8)
            hsic_mean = hsic_mean / denom.clamp(min=1e-8)
        hsic_std = hsic.std(dim=0)
        cv = hsic_std / (hsic_mean + 1e-6)
        stability = hsic_mean / (1.0 + cv + self.stability_bias.abs())
        return stability.unsqueeze(0).expand(bs, nvars, nvars)

    def forward(self, x, env_labels=None):
        """
        x: [bs, nvars, patch_num, d_model]
        env_labels: [bs] 可选。env_mode='semantic' 时的语义环境标签。
        returns: gate_matrix [bs, nvars, nvars] ∈ [0,1]
        """
        temp = self.temperature_param.clamp(min=0.1, max=10.0)
        if self.prior_only:
            # 诊断对照 (gate_prior_only): 完全剥离稳定性/HSIC信号,
            # 门控退化为纯 channel_prior 的(input-independent)函数,
            # 用以验证 full_v2 的提升是否真来自因果稳定性信号。
            bs, nvars, _, _ = x.shape
            stability_std = torch.zeros(bs, nvars, nvars, device=x.device)
            logit = self.gate_mlp(stability_std.unsqueeze(-1)).squeeze(-1)
            logit = logit + self.prior_weight * self.channel_prior.unsqueeze(0)
            gate = torch.sigmoid(logit / temp)
        elif self.stability_v2:
            if self.env_mode == 'semantic' and env_labels is not None:
                stability = self.compute_stability_score_semantic(x, env_labels)
            else:
                stability = self.compute_stability_score_v2(x)
            # 逐batch标准化非对角依赖分数，保证进入MLP前尺度良好、可分化
            mean = stability.mean(dim=(1, 2), keepdim=True)
            std = stability.std(dim=(1, 2), keepdim=True) + 1e-6
            stability_std = (stability - mean) / std
            logit = self.gate_mlp(stability_std.unsqueeze(-1)).squeeze(-1)
            # 先验作为 logit 空间的可学习加性偏置 (prior_weight 控制强度)
            logit = logit + self.prior_weight * self.channel_prior.unsqueeze(0)
            gate = torch.sigmoid(logit / temp)
        else:
            stability = self.compute_stability_score(x)
            prior = torch.sigmoid(self.channel_prior)
            stability = stability * (1 - self.prior_weight) + prior.unsqueeze(0) * self.prior_weight
            logit = self.gate_mlp(stability.unsqueeze(-1)).squeeze(-1)
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

    def get_diagnostics(self):
        """返回门控相关可学习参数的诊断信息（标量字典），供消融可观测性插桩使用。"""
        prior_sig = torch.sigmoid(self.channel_prior).detach()
        diag = {
            'gate_type': 'CausalStabilityGate',
            'channel_prior_sig_mean': float(prior_sig.mean()),
            'channel_prior_sig_min': float(prior_sig.min()),
            'channel_prior_sig_max': float(prior_sig.max()),
            'stability_bias': float(self.stability_bias.detach()),
            'temperature': float(self.temperature_param.detach()),
            'prior_weight': float(self.prior_weight),
        }
        if self.last_entropy is not None:
            diag['last_entropy'] = float(self.last_entropy.detach())
        return diag


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


class CausalChannelAttentionTemporal(nn.Module):
    """时间分辨率保留的门控通道注意力。

    与 CausalChannelAttention 的关键区别：不对 patch_num 维度做池化，
    而是在【每个 patch 位置】上独立地跨通道做门控注意力。
    这样滞后因果依赖（如 Ch_i[t] ← Ch_j[t-1]）所携带的时间性信息
    不会像"先池化再广播"那样被抹平。

    门控矩阵在所有 patch 位置共享（通道对之间的因果关系被视为时间上稳定的），
    这既符合"稳定因果"的建模假设，也大幅减少参数与计算量。
    """
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
        x: [bs, nvars, d_model, patch_num]
        gate_matrix: [bs, nvars, nvars]
        returns: [bs, nvars, d_model, patch_num]
        """
        bs, nvars, d_model, patch_num = x.shape
        # 把每个 patch 位置当作独立样本： [bs*patch_num, nvars, d_model]
        xt = x.permute(0, 3, 1, 2).contiguous().view(bs * patch_num, nvars, d_model)
        residual = xt
        B = bs * patch_num
        Q = self.W_Q(xt).view(B, nvars, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(xt).view(B, nvars, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(xt).view(B, nvars, self.n_heads, self.d_k).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, heads, nvars, nvars]
        # 门控在每个 patch 位置共享。
        # 优化(2026-08-08): 保持 5D 广播, 避免把 expanded 视图 reshape 成
        # [B, nv, nv] (对 [bs*patch_num, nv, nv] 的一次显式拷贝, 高维下 ~284MB/前向),
        # 直接与 [bs, patch_num, heads, nv, nv] 的 attn 广播相加, 结果一致且更快。
        log_gate = torch.log(gate_matrix.unsqueeze(1).unsqueeze(1).clamp(min=1e-4))  # [bs,1,1,nv,nv]
        attn5 = attn.view(bs, patch_num, self.n_heads, nvars, nvars) + log_gate
        # 软门控：log域加性偏置（与 CausalChannelAttention 一致）
        attn = attn5.reshape(B, self.n_heads, nvars, nvars)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, nvars, d_model)
        out = self.W_O(out)
        out = self.dropout(out)
        out = self.norm(residual + out)
        # 还原到 [bs, nvars, d_model, patch_num]
        out = out.view(bs, patch_num, nvars, d_model).permute(0, 2, 3, 1).contiguous()
        return out


class CausalChannelInteraction(nn.Module):
    """完整的因果通道交互模块
    组合: 因果稳定性门控 + 通道交叉注意力 + 信息融合

    temporal_mix:
        False (默认, 向后兼容) — 对 patch_num 池化后做通道注意力, 再广播回所有patch。
            局限: 通道交互只能给每个时刻加一个时间上恒定的偏移。
        True  — 使用 CausalChannelAttentionTemporal, 在每个patch位置逐点做门控通道注意力,
            保留时间分辨率, 对滞后因果依赖至关重要 (SOTA 关键改进)。
    """
    def __init__(self, n_vars, d_model, patch_num, n_heads=4, n_envs=4,
                 rff_dim=32, dropout=0.1, fusion_alpha=0.3, prior_weight: float = 0.3,
                 temporal_mix: bool = False, temperature: float = 1.0,
                 stability_v2: bool = False, per_channel_alpha: bool = False,
                 alpha_init: float = None, running_stats: bool = False,
                 rff_sigma_mode: str = 'fixed', cka_normalize: bool = False,
                 env_mode: str = 'uniform'):
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
            stability_v2=stability_v2, running_stats=running_stats,
            rff_sigma_mode=rff_sigma_mode, cka_normalize=cka_normalize,
            env_mode=env_mode,
        )
        if temporal_mix:
            self.channel_attn = CausalChannelAttentionTemporal(
                d_model=d_model, n_heads=n_heads, n_vars=n_vars, dropout=dropout
            )
        else:
            self.channel_attn = CausalChannelAttention(
                d_model=d_model, n_heads=n_heads, n_vars=n_vars, dropout=dropout
            )
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        # 融合系数 alpha。per_channel_alpha=True 时为逐通道可学习向量，
        # 且 alpha_init 默认取负值 (sigmoid后接近0)，使模型【默认接近通道独立】，
        # 仅当通道混合能降低loss时才逐通道地开启混合 —— 实现"优雅回退"，
        # 保证在混合无益的场景 (低维/长horizon) 不劣于通道独立基线 (SOTA关键)。
        if per_channel_alpha:
            init = alpha_init if alpha_init is not None else -2.0
            self.alpha = nn.Parameter(torch.full((n_vars,), float(init)))
        else:
            init = alpha_init if alpha_init is not None else fusion_alpha
            self.alpha = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x, env_labels=None):
        """
        x: [bs, nvars, d_model, patch_num]
        env_labels: [bs] 可选。env_mode='semantic' 时透传给稳定性门控。
        returns: (out, gate_matrix)
        """
        bs, nvars, d_model, patch_num = x.shape
        x_for_gate = x.permute(0, 1, 3, 2)   # [bs, nvars, patch_num, d_model]
        gate_matrix = self.stability_gate(x_for_gate, env_labels)
        if self.per_channel_alpha:
            alpha_vec = torch.sigmoid(self.alpha).view(1, nvars, 1, 1)  # [1,nvars,1,1]
        else:
            alpha_vec = torch.sigmoid(self.alpha)

        if self.temporal_mix:
            # 逐patch门控通道注意力，保留时间分辨率
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

    def get_last_entropy(self):
        """返回最近一次forward的门控熵（标量tensor或None），供上层作为正则项使用"""
        return self.stability_gate.last_entropy

    def get_diagnostics(self):
        """返回门控相关可学习参数诊断信息，供消融可观测性插桩使用"""
        return self.stability_gate.get_diagnostics()
