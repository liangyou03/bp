#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练 (Train) 与 评估 (Evaluate) 逻辑
Support:
- single-target (B,) or (B,1)
- multi-target  (B,K)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset   # ← 加
from pathlib import Path                # ← 加
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Union

from utils import apply_constraint, mae_np, rmse_np, pearson_r_safe_np, r2_np
from losses import MAE_PearsonLoss

# ============== (可选) 分布先验对齐 ==============
try:
    from dist_loss import DistributionAlignmentLoss
    _HAS_DIST = True
except Exception:
    DistributionAlignmentLoss = None
    _HAS_DIST = False
# ===============================================



class BPDataset(Dataset):
    """
    For BP prediction.
    Returns:
        ecg:     (1, L)
        ppg:     (1, L)
        targets: (num_targets,)
        ssoid:   str
    """
    def __init__(self, df, npz_dir, target_cols):
        self.df = df.reset_index(drop=True)
        self.npz_dir = Path(npz_dir)
        self.target_cols = list(target_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ssoid = str(r["ssoid"])

        targets = torch.tensor(
            [float(r[c]) for c in self.target_cols],
            dtype=torch.float32
        )

        with np.load(self.npz_dir / f"{ssoid}.npz") as d:
            x = d["x"].astype(np.float32)  # (L,2), L=3630

        ecg = torch.from_numpy(x[:, 0:1].T.copy())  # (1,L)
        ppg = torch.from_numpy(x[:, 1:2].T.copy())  # (1,L)

        return ecg, ppg, targets, ssoid


def _as_2d_tensor(x: Union[float, List[float], np.ndarray, torch.Tensor],
                  device: torch.device,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert x to tensor on device and reshape to (1, K) for broadcasting.
    - scalar -> (1,1)
    - (K,)   -> (1,K)
    """
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=dtype)
    else:
        t = torch.tensor(x, device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.view(1, 1)
    elif t.ndim == 1:
        t = t.view(1, -1)
    elif t.ndim == 2 and t.shape[0] == 1:
        pass
    else:
        # keep conservative: (K,) or scalar expected
        t = t.reshape(1, -1)
    return t


def _as_2d_numpy(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y[:, None]
    return y


def _compute_metrics_dict(y: np.ndarray,
                          yhat: np.ndarray,
                          target_names: List[str]) -> Dict[str, Dict[str, float]]:
    y = _as_2d_numpy(y)
    yhat = _as_2d_numpy(yhat)
    if len(target_names) != y.shape[1]:
        raise ValueError(f"target_names length ({len(target_names)}) != target dimension ({y.shape[1]})")

    metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(target_names):
        yi = y[:, idx]
        yhi = yhat[:, idx]
        metrics[name] = {
            "mae": mae_np(yi, yhi),
            "rmse": rmse_np(yi, yhi),
            "r": pearson_r_safe_np(yi, yhi),
            "r2": r2_np(yi, yhi),
        }
    return metrics


def _ensure_2d_y(y: torch.Tensor) -> torch.Tensor:
    """Ensure y is (B,K). Accept (B,) -> (B,1)."""
    if y.ndim == 1:
        return y.view(-1, 1)
    return y


@torch.no_grad()
def evaluate_with_ids(model, loader, device, modality,
                      mu, sigma, y_min, y_max, constrain: str,
                      target_names: List[str]):
    """评估模型，并返回所有 ssoid"""
    model.eval()
    y_all = []
    yhat_all = []
    ssoid_all = []
    total = 0
    total_loss = 0.0

    mu_t = _as_2d_tensor(mu, device)
    sigma_t = _as_2d_tensor(sigma, device)
    # y_min/y_max may be scalar or per-target list/array
    y_min_t = _as_2d_tensor(y_min, device)
    y_max_t = _as_2d_tensor(y_max, device)

    for batch in loader:
        if len(batch) == 4:
            ecg, ppg, y, ssoids = batch
        else:
            ecg, ppg, y = batch
            ssoids = [""] * y.size(0)

        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        y = _ensure_2d_y(y.to(device, non_blocking=True))

        if modality == "ecg":
            ppg = torch.zeros_like(ppg)
        if modality == "ppg":
            ecg = torch.zeros_like(ecg)

        with torch.cuda.amp.autocast(enabled=True):
            raw = model(ecg, ppg)
            if raw.ndim == 1:
                raw = raw.view(-1, 1)

            if torch.isnan(raw).any():
                raise RuntimeError("NaN in model output during evaluation.")

            # apply_constraint currently expects scalar y_min/y_max.
            # If you pass vectors, apply_constraint in utils must be updated.
            # Here we support scalar bounds and per-target bounds via broadcasting,
            # but only for modes that do element-wise transforms.
            if isinstance(y_min, (list, tuple, np.ndarray, torch.Tensor)) or isinstance(y_max, (list, tuple, np.ndarray, torch.Tensor)):
                # element-wise constrain implemented here to avoid changing utils.py
                if constrain == "none":
                    yhat = raw
                elif constrain == "tanh":
                    rng = (y_max_t - y_min_t)
                    yhat = y_min_t + 0.5 * (torch.tanh(raw) + 1.0) * rng
                elif constrain == "sigmoid":
                    rng = (y_max_t - y_min_t)
                    yhat = y_min_t + torch.sigmoid(raw) * rng
                elif constrain == "clip":
                    yhat = torch.max(torch.min(raw, y_max_t), y_min_t)
                else:
                    yhat = raw
            else:
                yhat = apply_constraint(raw, constrain, float(y_min), float(y_max))

            # loss on z-score scale
            y_z = (y - mu_t) / sigma_t
            yhat_z = (yhat - mu_t) / sigma_t
            loss = F.smooth_l1_loss(yhat_z, y_z, beta=1.0)

        bs = y.size(0)
        total += bs
        total_loss += float(loss.item()) * bs

        y_all.append(y.detach().cpu().numpy())
        yhat_all.append(yhat.detach().cpu().numpy())

        if isinstance(ssoids, (list, tuple)):
            ssoid_all.extend(list(ssoids))
        else:
            ssoid_all.extend([str(ssoids)] * bs)

    if total == 0:
        raise RuntimeError("No samples were evaluated (total==0). Check your dataloader.")

    y_np = np.concatenate(y_all, axis=0).astype(np.float64)
    yhat_np = np.concatenate(yhat_all, axis=0).astype(np.float64)

    # final clip on numpy side (scalar or per-target)
    if np.ndim(y_min) == 0 and np.ndim(y_max) == 0:
        yhat_np = np.clip(yhat_np, float(y_min), float(y_max))
    else:
        y_min_np = np.asarray(y_min, dtype=np.float64).reshape(1, -1)
        y_max_np = np.asarray(y_max, dtype=np.float64).reshape(1, -1)
        yhat_np = np.maximum(np.minimum(yhat_np, y_max_np), y_min_np)

    metrics = _compute_metrics_dict(y_np, yhat_np, target_names)
    return total_loss / total, metrics, y_np, yhat_np, np.array(ssoid_all, dtype=object)


@torch.no_grad()
def evaluate(model, loader, device, modality, mu, sigma, y_min, y_max, constrain: str,
             target_names: List[str]):
    """评估模型，不返回 ssoid"""
    loss, metrics, y, yhat, _ = evaluate_with_ids(
        model, loader, device, modality, mu, sigma, y_min, y_max, constrain, target_names
    )
    return loss, metrics, y, yhat


def train_one_epoch(model, loader, optimizer, scaler, device, modality, loss_type,
                    mu, sigma, y_min, y_max, constrain: str, alpha_corr: float,
                    maepearson_criterion: Optional[MAE_PearsonLoss] = None,
                    dist_criterion: "Optional[DistributionAlignmentLoss]" = None,
                    lambda_dist: float = 1.0):
    """训练一个 Epoch"""
    model.train()
    total = 0
    total_loss = 0.0

    mu_t = _as_2d_tensor(mu, device)
    sigma_t = _as_2d_tensor(sigma, device)
    y_min_t = _as_2d_tensor(y_min, device)
    y_max_t = _as_2d_tensor(y_max, device)

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        if len(batch) == 4:
            ecg, ppg, y, _ = batch
        else:
            ecg, ppg, y = batch

        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        y = _ensure_2d_y(y.to(device, non_blocking=True))

        if modality == "ecg":
            ppg = torch.zeros_like(ppg)
        if modality == "ppg":
            ecg = torch.zeros_like(ecg)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=True):
            raw = model(ecg, ppg)
            if raw.ndim == 1:
                raw = raw.view(-1, 1)

            # constraint (support scalar or per-target bounds)
            if isinstance(y_min, (list, tuple, np.ndarray, torch.Tensor)) or isinstance(y_max, (list, tuple, np.ndarray, torch.Tensor)):
                if constrain == "none":
                    yhat = raw
                elif constrain == "tanh":
                    rng = (y_max_t - y_min_t)
                    yhat = y_min_t + 0.5 * (torch.tanh(raw) + 1.0) * rng
                elif constrain == "sigmoid":
                    rng = (y_max_t - y_min_t)
                    yhat = y_min_t + torch.sigmoid(raw) * rng
                elif constrain == "clip":
                    yhat = torch.max(torch.min(raw, y_max_t), y_min_t)
                else:
                    yhat = raw
            else:
                yhat = apply_constraint(raw, constrain, float(y_min), float(y_max))

            y_z = (y - mu_t) / sigma_t
            yhat_z = (yhat - mu_t) / sigma_t

            if loss_type == "mse":
                reg = F.mse_loss(yhat_z, y_z)
            elif loss_type == "huber":
                reg = F.smooth_l1_loss(yhat_z, y_z, beta=1.0)
            elif loss_type == "mae_pearson":
                if maepearson_criterion is None:
                    raise ValueError("maepearson_criterion is None for loss_type='mae_pearson'")
                reg = maepearson_criterion(yhat_z, y_z)
            elif loss_type == "mse+dist":
                reg = F.mse_loss(yhat_z, y_z)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss = reg

            # (可选) 分布损失（用原始尺度）
            if loss_type == "mse+dist" and (dist_criterion is not None) and (lambda_dist > 0.0):
                if not _HAS_DIST:
                    raise RuntimeError("loss_type=mse+dist, but dist_loss.py is not available.")
                dist_loss_val = dist_criterion(yhat)
                loss = loss + float(lambda_dist) * dist_loss_val

            # (可选) 相关性辅助损失：逐 target 计算后取均值
            if alpha_corr > 0.0 and loss_type != "mae_pearson":
                corr_losses = []
                for c in range(y_z.shape[1]):
                    y0 = y_z[:, c] - y_z[:, c].mean()
                    yhat0 = yhat_z[:, c] - yhat_z[:, c].mean()
                    num = (y0 * yhat0).sum()
                    den = torch.sqrt((y0 ** 2).sum()) * torch.sqrt((yhat0 ** 2).sum()) + 1e-8
                    corr = num / den
                    corr_losses.append(1.0 - corr)
                if corr_losses:
                    loss = loss + alpha_corr * torch.stack(corr_losses).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}")

    pbar.close()
    return total_loss / total
