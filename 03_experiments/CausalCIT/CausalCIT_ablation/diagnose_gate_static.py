"""
门 1 静态诊断 (零训练成本) —— 验证 05_major_improvement.md 的根因 1 & 2。

做法: 同一批 syn_ood 数据, 三个 d_model (16/32/64, 对应 traffic/electricity/weather 配置),
随机初始化模型 + 单 batch 前向, 打印:
  (a) proj = x@W 的 std          —— 根因 1: ≫1 则 RFF 核已失效 (cos 剧烈震荡)
  (b) hsic_mean 动态范围 (max/min)
  (c) log(hsic_mean) vs log(1/(1+cv)) 的方差贡献占比 —— 根因 2: 前者 >90% 则稳定性信号是装饰品
  (d) cv 分布 (mean/p99/max)      —— 环境划分是否有信息 (cv≈0 则环境间无差异)

用法: python diagnose_gate_static.py
"""
import os
import sys
import io
import torch
import torch.nn.functional as F
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CausalCIT_demo'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data import SyntheticOODDataset, get_dataloader
from models.patchtst import TSTiEncoder
from models.causal_channel import CausalStabilityGate

SEQ_LEN, PATCH_LEN, STRIDE = 96, 16, 8
PATCH_NUM = 12  # (96-16)/8 + 1 + 1(padding end)
N_VARS = 7


def get_batch(bs=32):
    ds = SyntheticOODDataset(seq_len=SEQ_LEN, pred_len=96, flag='train', regime='train',
                             seed=0, spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                             test_spurious_strengths=(0.05, -0.2, 0.1, -0.05),
                             train_noise=0.05, test_noise=0.05)
    loader = get_dataloader(ds, batch_size=bs, shuffle=False)
    xb, _ = next(iter(loader))  # [bs, seq_len, nvars]
    return xb


def patch_to_backbone(xb, d_model):
    """xb [bs, seq_len, nvars] -> patch 表示 [bs, nvars, patch_num, d_model]"""
    x = xb.permute(0, 2, 1)               # [bs, nvars, seq_len]
    x = F.pad(x, (0, STRIDE))             # padding_patch='end'
    xp = x.unfold(-1, PATCH_LEN, STRIDE)  # [bs, nvars, patch_num, patch_len]
    xp = xp.permute(0, 1, 3, 2)           # [bs, nvars, patch_len, patch_num]
    bb = TSTiEncoder(N_VARS, patch_num=PATCH_NUM, patch_len=PATCH_LEN,
                     n_layers=3, d_model=d_model, n_heads=4, d_ff=128,
                     dropout=0.2, act='gelu', res_attention=True,
                     pe='zeros', learn_pe=True)
    with torch.no_grad():
        z = bb(xp)                        # [bs, nvars, d_model, patch_num]
    return z.permute(0, 1, 3, 2)          # [bs, nvars, patch_num, d_model]


def diagnose(d_model, xb):
    x = patch_to_backbone(xb, d_model)    # [bs, nvars, patch_num, d_model]
    gate = CausalStabilityGate(n_vars=N_VARS, d_model=d_model, n_envs=4, rff_dim=32,
                               prior_weight=0.05, temperature=0.5, stability_v2=True)
    rff = gate.rff_kernel

    # (a) proj std: x @ W
    with torch.no_grad():
        W = rff.W  # [d_model, rff_dim]
        proj = x.reshape(-1, d_model) @ W
        proj_std = proj.std().item()

    # 重现 compute_stability_score_v2 的中间量
    xf = x.float()
    zf = rff(xf.reshape(-1, d_model)).reshape(xf.shape[0], N_VARS, PATCH_NUM, rff.rff_dim)
    n_envs, env_size = 4, PATCH_NUM // 4
    zf = zf[:, :, :n_envs * env_size, :].reshape(xf.shape[0], N_VARS, n_envs, env_size, rff.rff_dim)
    zf = zf.permute(2, 1, 0, 3, 4).reshape(n_envs, N_VARS, xf.shape[0] * env_size, rff.rff_dim)
    zf = zf - zf.mean(dim=2, keepdim=True)
    m = zf.shape[2]
    K = torch.einsum('ecma,ecna->ecmn', zf, zf)
    Kf = K.reshape(n_envs, N_VARS, m * m)
    hsic = torch.bmm(Kf, Kf.transpose(1, 2)) / (m * m)  # [envs, nvars, nvars]
    hsic_mean = hsic.mean(dim=0).clamp(min=1e-8)        # [nvars, nvars]
    hsic_std = hsic.std(dim=0)
    cv = hsic_std / (hsic_mean + 1e-6)

    # (b) hsic_mean 动态范围
    off = hsic_mean[~torch.eye(N_VARS, dtype=bool)]
    hsic_min, hsic_max = off.min().item(), off.max().item()
    # (c) log 分解方差贡献
    log_hsic = torch.log(off)
    log_cvterm = torch.log(1.0 / (1.0 + cv[~torch.eye(N_VARS, dtype=bool)] + 1e-6))
    v_h, v_c = log_hsic.var().item(), log_cvterm.var().item()
    frac_hsic = v_h / (v_h + v_c + 1e-12)
    # (d) cv 分布
    cv_off = cv[~torch.eye(N_VARS, dtype=bool)]
    cv_mean, cv_p99, cv_max = (cv_off.mean().item(), torch.quantile(cv_off, 0.99).item(),
                               cv_off.max().item())

    print(f"\n===== d_model={d_model} =====")
    print(f"(a) proj = x@W 的 std       : {proj_std:.3f}   {'<-- 核已失效(≫1)' if proj_std > 3 else ''}")
    print(f"(b) hsic_mean 动态范围      : min={hsic_min:.3e} max={hsic_max:.3e} ratio={hsic_max/hsic_min:.1e}")
    print(f"(c) log(hsic) 方差占比      : {frac_hsic*100:.1f}%  (log(1/(1+cv)) 占 {(1-frac_hsic)*100:.1f}%)"
          + ('   <-- 稳定性信号是装饰品' if frac_hsic > 0.9 else ''))
    print(f"(d) cv 分布 (非对角)        : mean={cv_mean:.4f} p99={cv_p99:.4f} max={cv_max:.4f}"
          + ('   <-- 环境间几乎无差异' if cv_mean < 0.05 else ''))
    return dict(d_model=d_model, proj_std=proj_std, hsic_min=hsic_min, hsic_max=hsic_max,
                frac_hsic=frac_hsic, cv_mean=cv_mean, cv_p99=cv_p99, cv_max=cv_max)


def main():
    print("门 1 静态诊断 (根因 1 & 2), syn_ood 同批数据, d_model ∈ {16, 32, 64}")
    print(f"配置: n_envs=4, patch_num={PATCH_NUM} -> env_size={PATCH_NUM//4}, rff_dim=32, sigma=1.0")
    xb = get_batch()
    print(f"batch: {tuple(xb.shape)} (来自 syn_ood train)")
    results = {}
    for dm in (16, 32, 64):
        try:
            results[dm] = diagnose(dm, xb)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  d_model={dm} 失败: {e}")
    # 汇总行
    print("\n===== 汇总 =====")
    for dm, r in results.items():
        print(f"d_model={dm}: proj.std={r['proj_std']:.2f}  hsic_ratio={r['hsic_max']/r['hsic_min']:.1e}  "
              f"log(hsic)占比={r['frac_hsic']*100:.0f}%  cv.mean={r['cv_mean']:.4f}")


if __name__ == '__main__':
    main()
