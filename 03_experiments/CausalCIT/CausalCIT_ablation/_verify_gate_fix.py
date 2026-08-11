"""验证修 A+B 效果 (门 2 分支, 0 训练成本): 对比 fixed / median+cka 的门控输入质量。"""
import os, sys, io, math
import torch
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CausalCIT_demo'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data import SyntheticOODDataset, get_dataloader
from models.patchtst import TSTiEncoder
from models.causal_channel import CausalStabilityGate

SEQ_LEN, PATCH_LEN, STRIDE, PATCH_NUM, N_VARS = 96, 16, 8, 12, 7


def get_xb():
    ds = SyntheticOODDataset(seq_len=SEQ_LEN, pred_len=96, flag='train', regime='train',
                             seed=0, spurious_strengths=(0.8, 0.5, 0.3, 0.6),
                             test_spurious_strengths=(0.05, -0.2, 0.1, -0.05),
                             train_noise=0.05, test_noise=0.05)
    loader = get_dataloader(ds, batch_size=32, shuffle=False)
    xb, _ = next(iter(loader))
    return xb


def patch_rep(xb, d_model):
    x = xb.permute(0, 2, 1)
    x = F.pad(x, (0, STRIDE))
    xp = x.unfold(-1, PATCH_LEN, STRIDE).permute(0, 1, 3, 2)
    bb = TSTiEncoder(N_VARS, patch_num=PATCH_NUM, patch_len=PATCH_LEN,
                     n_layers=3, d_model=d_model, n_heads=4, d_ff=128,
                     dropout=0.2, act='gelu', res_attention=True, pe='zeros', learn_pe=True)
    with torch.no_grad():
        return bb(xp).permute(0, 1, 3, 2)  # [bs, nvars, patch_num, d_model]


def v2_metrics(x, gate):
    """重现 v2 中间量; 返回 (proj_std, hsic_ratio, frac_hsic, cv_mean)"""
    bs, nvars, pnum, dm = x.shape
    # 触发 median 初始化 (若有)
    _ = gate.rff_kernel(x.reshape(-1, dm))
    with torch.no_grad():
        proj = x.reshape(-1, dm) @ gate.rff_kernel.W
        proj_std = proj.std().item()
    xf = x.float()
    zf = gate.rff_kernel(xf.reshape(-1, dm)).reshape(bs, nvars, pnum, gate.rff_dim)
    n_envs, env_size = gate.n_envs, pnum // gate.n_envs
    zf = zf[:, :, :n_envs * env_size, :].reshape(bs, nvars, n_envs, env_size, gate.rff_dim)
    zf = zf.permute(2, 1, 0, 3, 4).reshape(n_envs, nvars, bs * env_size, gate.rff_dim)
    zf = zf - zf.mean(dim=2, keepdim=True)
    m = zf.shape[2]
    K = torch.einsum('ecma,ecna->ecmn', zf, zf)
    P = m * m
    Kf = K.reshape(n_envs, nvars, P)
    hsic = torch.bmm(Kf, Kf.transpose(1, 2)) / P
    hsic_mean = hsic.mean(dim=0).clamp(min=1e-8)
    if gate.cka_normalize:
        diag = torch.diagonal(hsic_mean, dim1=-2, dim2=-1)
        denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2) + 1e-8)
        hsic_mean = hsic_mean / denom.clamp(min=1e-8)
    hsic_std = hsic.std(dim=0)
    cv = hsic_std / (hsic_mean + 1e-6)
    off_mask = ~torch.eye(nvars, dtype=bool)
    off = hsic_mean[off_mask]
    hsic_ratio = (off.max() / (off.min().clamp(min=1e-12))).item()
    log_hsic = torch.log(off.clamp(min=1e-12))
    log_cvterm = torch.log(1.0 / (1.0 + cv[off_mask] + 1e-6))
    v_h, v_c = log_hsic.var().item(), log_cvterm.var().item()
    frac_hsic = v_h / (v_h + v_c + 1e-12)
    cv_off = cv[off_mask]
    sigma = getattr(gate.rff_kernel, '_median_sigma', None) or 1.0
    return dict(proj_std=proj_std, hsic_ratio=hsic_ratio, frac_hsic=frac_hsic,
                cv_mean=cv_off.mean().item(), sigma=sigma)


def main():
    xb = get_xb()
    print(f"{'d_model':>6} {'mode':<12} {'sigma':>6} {'proj.std':>8} {'hsic.ratio':>10} {'log(hsic)':>9} {'cv.mean':>8}")
    for dm in (16, 32, 64):
        x = patch_rep(xb, dm)
        for mode, kw in [('fixed', dict(rff_sigma_mode='fixed', cka_normalize=False)),
                         ('median+cka', dict(rff_sigma_mode='median', cka_normalize=True))]:
            gate = CausalStabilityGate(n_vars=N_VARS, d_model=dm, n_envs=4, rff_dim=32,
                                       prior_weight=0.05, temperature=0.5,
                                       stability_v2=True, **kw)
            r = v2_metrics(x, gate)
            print(f"{dm:>6} {mode:<12} {r['sigma']:>6.3f} {r['proj_std']:>8.2f} "
                  f"{r['hsic_ratio']:>10.2f} {r['frac_hsic']*100:>8.0f}% {r['cv_mean']:>8.4f}")


if __name__ == '__main__':
    main()
