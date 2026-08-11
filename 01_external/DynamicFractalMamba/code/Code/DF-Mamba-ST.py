import os
import math
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 0. Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 1. Minimal Selective SSM
# =========================================================
class MinimalSelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            repeats=self.d_inner,
            dim=0,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.x_proj = nn.Linear(
            self.d_inner,
            (math.ceil(d_model / 16) + self.d_state * 2),
        )
        self.dt_proj = nn.Linear(math.ceil(d_model / 16), self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        self._init_weights()

    def _init_weights(self):
        dt_init_std = 2 ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        nn.init.constant_(self.dt_proj.bias, 1.0)

    def parallel_scan(self, u, delta, A, B, C):
        bsz, L, din = u.shape
        d_state = self.d_state

        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

        x = torch.zeros(bsz, din, d_state, device=u.device)
        ys = []
        for t in range(L):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            y = torch.einsum("bdn,bn->bd", x, C[:, t])
            ys.append(y)
        return torch.stack(ys, dim=1)

    def forward(self, x, time_scale_factor=1.0):
        bsz, L, _ = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :L]
        x_inner = self.act(x_inner).transpose(1, 2)
        x_inner = self.dropout(x_inner)

        dt_rank = math.ceil(self.d_model / 16)
        x_dbl = self.x_proj(x_inner)
        dt_raw, Bp, Cp = torch.split(
            x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(self.dt_proj(dt_raw))
        dt = dt * time_scale_factor

        A = -torch.exp(self.A_log)
        y = self.parallel_scan(x_inner, dt, A, Bp, Cp)

        y = y + x_inner * self.D
        y = y * F.silu(z)
        out = self.out_proj(y)
        return self.dropout(out)


# =========================================================
# 2. Backbone
# =========================================================
class InformationPreservingDownsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.decoder = nn.Linear(dim, dim * 2)
        self.last_recon_loss = 0.0

    def forward(self, x):
        B, L, D = x.shape
        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            L += 1
        x2 = x.view(B, L // 2, D * 2)
        z = self.encoder(x2)

        if self.training:
            x_recon = self.decoder(z)
            self.last_recon_loss = F.mse_loss(x_recon, x2.detach())
        else:
            self.last_recon_loss = 0.0

        return z


class RenormalizationGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, local, global_up):
        return torch.sigmoid(self.net(torch.cat([local, global_up], dim=-1)))


class DynamicFractalMamba(nn.Module):
    def __init__(self, d_model, max_depth=4, min_len=8, dropout=0.1):
        super().__init__()
        self.max_depth = max_depth
        self.min_len = min_len

        self.universal_ssm = MinimalSelectiveSSM(d_model, dropout=dropout)
        self.downsampler = InformationPreservingDownsample(d_model)
        self.gate_net = RenormalizationGate(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.aux_losses = {}

    def compute_flow_loss(self, gates_tensor):
        if gates_tensor.shape[0] < 2:
            return torch.tensor(0.0, device=gates_tensor.device)
        loss = 0.0
        for i in range(gates_tensor.shape[0] - 1):
            loss = loss + F.mse_loss(gates_tensor[i], gates_tensor[i + 1].detach())
        return loss

    def forward_recursive(self, x, depth=0):
        _, L, _ = x.shape
        time_scale = 2.0 ** depth

        local = x + self.universal_ssm(self.norm(x), time_scale_factor=time_scale)

        if depth >= self.max_depth or L < self.min_len:
            return local

        x_coarse = self.downsampler(local)
        if self.training:
            self.aux_losses["recon"].append(self.downsampler.last_recon_loss)

        global_ctx = self.forward_recursive(x_coarse, depth + 1)
        global_up = F.interpolate(
            global_ctx.transpose(1, 2),
            size=L,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        if self.training:
            self.aux_losses["consistency"].append(F.mse_loss(global_up, local.detach()))

        gate = self.gate_net(local, global_up)
        if self.training:
            self.aux_losses["gates"].append(gate.mean())

        out = local * (1.0 - gate) + global_up * gate
        return self.dropout(out)

    def forward(self, x, return_aux_loss=False):
        self.aux_losses = {"recon": [], "consistency": [], "gates": []}
        out = self.forward_recursive(x)

        if not (self.training and return_aux_loss):
            return out

        total_aux = torch.tensor(0.0, device=x.device)
        if self.aux_losses["recon"]:
            total_aux = total_aux + sum(self.aux_losses["recon"])
        if self.aux_losses["consistency"]:
            total_aux = total_aux + sum(self.aux_losses["consistency"])
        if self.aux_losses["gates"]:
            gates_tensor = torch.stack(self.aux_losses["gates"])
            total_aux = total_aux + 0.1 * self.compute_flow_loss(gates_tensor)

        return out, total_aux


# =========================================================
# 3. Forecast Head
# =========================================================
class FractalMambaForecaster(nn.Module):
    def __init__(
        self,
        in_channels,
        d_model,
        seq_len,
        pred_len,
        max_depth=4,
        min_len=8,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.in_proj = nn.Linear(1, d_model)

        self.backbone = DynamicFractalMamba(
            d_model=d_model,
            max_depth=max_depth,
            min_len=min_len,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Linear(seq_len * d_model, pred_len)
        self.linear_shortcut = nn.Linear(seq_len, pred_len)

        self.affine_weight = nn.Parameter(torch.ones(1, 1, in_channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, in_channels))

    def forward(self, x, return_aux_loss=False):
        # x: [B, L, C]
        B, L, C = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev
        x = x * self.affine_weight + self.affine_bias

        x_ci = x.permute(0, 2, 1).reshape(B * C, L, 1)

        h = self.in_proj(x_ci)
        if self.training and return_aux_loss:
            enc, aux = self.backbone(h, return_aux_loss=True)
        else:
            enc = self.backbone(h, return_aux_loss=False)
            aux = None

        enc_flat = self.flatten(enc)
        y_mamba = self.head(self.dropout(enc_flat))
        y_linear = self.linear_shortcut(x_ci.squeeze(-1))
        y_pred = y_mamba + y_linear

        y_pred = y_pred.reshape(B, C, self.pred_len).permute(0, 2, 1)

        y_pred = y_pred - self.affine_bias
        y_pred = y_pred / (self.affine_weight + 1e-10)
        y_pred = y_pred * stdev[:, :, :C] + means[:, :, :C]

        return (y_pred, aux) if (self.training and return_aux_loss) else y_pred


# =========================================================
# 4. Data utils
# =========================================================
def load_weather_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    candidates = ["date", "Date", "OT"]
    dt_col = next((c for c in candidates if c in df.columns), None)

    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.sort_values(dt_col).reset_index(drop=True)
        df = df.drop(columns=[dt_col])

    df = df.select_dtypes(include=[np.number]).copy()
    if df.shape[1] == 0:
        raise ValueError("Dataset has no numeric columns.")
    return df


def split_weather_array(
    data_np: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = data_np.shape[0]
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)

    train_raw = data_np[:n_train]
    val_raw = data_np[n_train : n_train + n_val]
    test_raw = data_np[n_train + n_val :]

    return train_raw, val_raw, test_raw


def temporal_resample_array(
    data_np: np.ndarray,
    factor: int = 2,
    mode: str = "avg",
) -> np.ndarray:
    """
    Resample along time dimension only.
    Keeps feature dimension unchanged.

    factor=2 means 2 original timestamps -> 1 coarse timestamp.
    """
    if factor <= 1:
        return data_np.copy()

    T, C = data_np.shape
    T2 = (T // factor) * factor
    if T2 < factor:
        raise ValueError(f"Sequence too short for factor={factor}")

    x = data_np[:T2]

    if mode == "avg":
        x = x.reshape(T2 // factor, factor, C).mean(axis=1)
    elif mode == "subsample":
        x = x[::factor]
    else:
        raise ValueError(f"Unknown resample mode: {mode}")

    return x.astype(np.float32)


class SlidingWindowForecastDataset(Dataset):
    def __init__(
        self,
        raw_np: np.ndarray,
        train_mu: np.ndarray,
        train_std: np.ndarray,
        seq_len: int,
        pred_len: int,
    ):
        super().__init__()
        self.raw = raw_np.astype(np.float32)
        self.mu = train_mu.astype(np.float32)
        self.std = train_std.astype(np.float32)

        self.data = (self.raw - self.mu) / self.std
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.max_len = len(self.data) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.max_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def make_loader(ds, batch_size=32, shuffle=False, drop_last=False, num_workers=4):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse, mae, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, return_aux_loss=False)
        mse += F.mse_loss(pred, y, reduction="sum").item()
        mae += F.l1_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return mse / n, mae / n


# =========================================================
# 5. Main training + zero-shot temporal cross-scale
# =========================================================
def train_weather_task_with_cross_scale(
    csv_path: str,
    save_dir: str,
    seq_len: int = 336,
    pred_len: int = 96,
    d_model: int = 64,
    max_depth: int = 4,
    min_len: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 20,
    dropout: float = 0.2,
    aux_weight: float = 0.05,
    weight_decay: float = 1e-2,
    resume: bool = False,
    cross_scale_factors: List[int] = [2, 4],
    resample_mode: str = "avg",
):
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"[Info] Device: {device}")
    print(f"[Info] Detected {num_gpus} GPUs.")

    print(f"[Info] Loading dataset: {csv_path}")
    df = load_weather_csv(csv_path)
    data_np = df.values.astype(np.float32)
    T, C = data_np.shape
    print(f"[Info] Full data shape: T={T}, C={C}")

    # -----------------------------
    # source-scale split (native)
    # -----------------------------
    train_raw, val_raw, test_raw = split_weather_array(
        data_np,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    print(
        f"[Split] train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}"
    )

    # train statistics computed ONLY from source-train
    train_mu = train_raw.mean(axis=0, keepdims=True)
    train_std = train_raw.std(axis=0, keepdims=True) + 1e-6

    train_ds = SlidingWindowForecastDataset(train_raw, train_mu, train_std, seq_len, pred_len)
    val_ds = SlidingWindowForecastDataset(val_raw, train_mu, train_std, seq_len, pred_len)
    test_ds = SlidingWindowForecastDataset(test_raw, train_mu, train_std, seq_len, pred_len)

    if len(train_ds) <= 0 or len(val_ds) <= 0 or len(test_ds) <= 0:
        raise ValueError("Dataset too short for current seq_len/pred_len after splitting.")

    train_dl = make_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = make_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl = make_loader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # -----------------------------
    # model
    # -----------------------------
    model = FractalMambaForecaster(
        in_channels=C,
        d_model=d_model,
        seq_len=seq_len,
        pred_len=pred_len,
        max_depth=max_depth,
        min_len=min_len,
        dropout=dropout,
    ).to(device)

    best_path = os.path.join(save_dir, "best_model.pth")
    best_val = float("inf")

    if resume and os.path.exists(best_path):
        print(f"[Info] Resuming from checkpoint: {best_path}")
        state_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(state_dict)
        best_val, best_mae = evaluate(model, val_dl, device)
        print(f"[Resume] current val MSE={best_val:.6f}, val MAE={best_mae:.6f}")
    elif resume:
        print(f"[Warning] resume=True but checkpoint not found: {best_path}")

    if num_gpus > 1:
        print(f"[Info] Using DataParallel on {num_gpus} GPUs.")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    patience = 10
    bad = 0

    # -----------------------------
    # train on native scale only
    # val on native scale only
    # -----------------------------
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            pred, aux = model(x, return_aux_loss=True)
            if num_gpus > 1:
                aux = aux.mean()

            loss = F.mse_loss(pred, y) + aux_weight * aux

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item()

        val_mse, val_mae = evaluate(model, val_dl, device)
        print(
            f"[Epoch {ep:03d}] "
            f"TrainLoss={loss_sum / len(train_dl):.6f} | "
            f"Val MSE={val_mse:.6f} | Val MAE={val_mae:.6f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save(state_dict, best_path)
            bad = 0
            print(f"  -> new best saved to {best_path}")
        else:
            bad += 1
            if bad >= patience:
                print("[Info] Early stopping triggered.")
                break

    # -----------------------------
    # reload best model
    # -----------------------------
    if num_gpus > 1:
        model.module.load_state_dict(torch.load(best_path, map_location=device))
    else:
        model.load_state_dict(torch.load(best_path, map_location=device))

    # -----------------------------
    # standard test: native -> native
    # -----------------------------
    native_test_mse, native_test_mae = evaluate(model, test_dl, device)
    results = {
        "standard_native_to_native": {
            "mse": float(native_test_mse),
            "mae": float(native_test_mae),
            "test_len": int(len(test_raw)),
        }
    }

    print("\n[Standard Test] native -> native")
    print(f"MSE={native_test_mse:.6f}, MAE={native_test_mae:.6f}")

    # -----------------------------
    # zero-shot temporal cross-scale
    # train/val remain native
    # ONLY test is resampled to coarser resolution
    # -----------------------------
    for factor in cross_scale_factors:
        coarse_test_raw = temporal_resample_array(
            test_raw,
            factor=factor,
            mode=resample_mode,
        )

        coarse_test_ds = SlidingWindowForecastDataset(
            coarse_test_raw,
            train_mu,
            train_std,
            seq_len,
            pred_len,
        )

        if len(coarse_test_ds) <= 0:
            print(f"[Skip] factor={factor}: coarse test set too short.")
            continue

        coarse_test_dl = make_loader(
            coarse_test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        coarse_mse, coarse_mae = evaluate(model, coarse_test_dl, device)
        results[f"zero_shot_native_to_x{factor}"] = {
            "mse": float(coarse_mse),
            "mae": float(coarse_mae),
            "coarse_test_len": int(len(coarse_test_raw)),
            "resample_mode": resample_mode,
        }

        print(f"\n[Zero-shot Temporal Cross-scale] native -> x{factor}")
        print(f"MSE={coarse_mse:.6f}, MAE={coarse_mae:.6f}")

    # -----------------------------
    # save results
    # -----------------------------
    result_path = os.path.join(save_dir, "results_cross_scale.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] {result_path}")
    return results


# =========================================================
# 6. Entry
# =========================================================
if __name__ == "__main__":
    WEATHER_CSV = "Data/ETTm2.csv"
    SAVE_DIR = ""

    results = train_weather_task_with_cross_scale(
        csv_path=WEATHER_CSV,
        save_dir=SAVE_DIR,
        seq_len=96,
        pred_len=96,
        d_model=64,
        max_depth=4,
        min_len=8,
        train_ratio=0.7,
        val_ratio=0.1,
        batch_size=128,
        lr=1e-3,
        epochs=5,
        dropout=0.2,
        aux_weight=0.05,
        weight_decay=1e-2,
        resume=False,
        cross_scale_factors=[2, 4],
        resample_mode="avg",  
    )

    print("\n================ RESULTS SUMMARY ================")
    for k, v in results.items():
        print(f"{k}: {v}")