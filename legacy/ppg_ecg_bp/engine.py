#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training and evaluation utilities (BP regression).
Key fixes:
1) Enforce (B, K) shapes everywhere to avoid broadcasting bugs.
2) Compute metrics (MAE/RMSE/r/R2) strictly on the physical scale (mmHg).
3) For mae_pearson, compute the loss on mmHg (otherwise absolute bias in mmHg is under-penalized).
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Dict, List

from utils import apply_constraint, mae_np, rmse_np, pearson_r_safe_np, r2_np
from losses import MAE_PearsonLoss

# Optional distribution alignment loss
try:
    from dist_loss import DistributionAlignmentLoss
    _HAS_DIST = True
except Exception:
    DistributionAlignmentLoss = None
    _HAS_DIST = False


def _compute_metrics_dict(
    y: np.ndarray,
    yhat: np.ndarray,
    target_names: List[str]
) -> Dict[str, Dict[str, float]]:
    # Ensure 2D arrays (N, K)
    if y.ndim == 1:
        y = y[:, None]
    if yhat.ndim == 1:
        yhat = yhat[:, None]

    if len(target_names) != y.shape[1]:
        raise ValueError(f"target_names length ({len(target_names)}) != target dimension ({y.shape[1]})")
    if yhat.shape[1] != y.shape[1]:
        raise ValueError(f"yhat dimension ({yhat.shape[1]}) != y dimension ({y.shape[1]})")

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


@torch.no_grad()
def evaluate_with_ids(
    model,
    loader,
    device,
    modality,
    mu,
    sigma,
    y_min,
    y_max,
    constrain: str,
    target_names: List[str],
):
    """
    Evaluate model and return:
      avg_loss, metrics_dict, y (N,K), yhat (N,K), ssoid (N,)
    All metrics are computed on mmHg scale.
    """
    model.eval()

    y_all: List[np.ndarray] = []
    yhat_all: List[np.ndarray] = []
    ssoid_all: List[str] = []

    total = 0
    total_loss = 0.0

    # Scalar bounds for constraint/clipping
    y_min_f = float(y_min)
    y_max_f = float(y_max)

    # Make mu/sigma tensors once (broadcastable to (B,K))
    # Note: used only if you later want z-score for auxiliary losses; metrics remain mmHg.
    mu_t = torch.as_tensor(mu, device=device, dtype=torch.float32)
    sg_t = torch.as_tensor(sigma, device=device, dtype=torch.float32)
    if mu_t.ndim == 0:
        mu_t = mu_t.view(1, 1)
    elif mu_t.ndim == 1:
        mu_t = mu_t.view(1, -1)
    if sg_t.ndim == 0:
        sg_t = sg_t.view(1, 1)
    elif sg_t.ndim == 1:
        sg_t = sg_t.view(1, -1)

    use_amp = (device.type == "cuda")

    for batch in loader:
        if len(batch) == 4:
            ecg, ppg, y, ssoids = batch
        else:
            ecg, ppg, y = batch
            ssoids = [""] * y.size(0)

        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Enforce y shape (B,K)
        if y.ndim == 1:
            y = y.view(-1, 1)

        if modality == "ecg":
            ppg = torch.zeros_like(ppg)
        if modality == "ppg":
            ecg = torch.zeros_like(ecg)

        with torch.cuda.amp.autocast(enabled=use_amp):
            raw = model(ecg, ppg)

            # Enforce raw shape (B,K)
            if raw.ndim == 1:
                raw = raw.view(-1, 1)

            if torch.isnan(raw).any():
                raise RuntimeError("NaN in model output during evaluation.")

            yhat = apply_constraint(raw, constrain, y_min_f, y_max_f)

            # Report loss on mmHg scale (consistent with MAE/RMSE)
            loss = F.smooth_l1_loss(yhat, y, beta=1.0)

        bs = y.size(0)
        total += bs
        total_loss += float(loss.item()) * bs

        # Store on CPU as numpy (mmHg)
        y_all.append(y.detach().cpu().numpy())
        yhat_all.append(yhat.detach().cpu().numpy())

        if isinstance(ssoids, (list, tuple)):
            ssoid_all.extend([str(x) for x in ssoids])
        else:
            ssoid_all.extend([str(ssoids)] * bs)

    if total == 0:
        raise RuntimeError("No samples were evaluated (total==0). Check your dataloader.")

    y_np = np.concatenate(y_all, axis=0).astype(np.float64)
    yhat_np = np.concatenate(yhat_all, axis=0).astype(np.float64)

    # Final clip on mmHg scale (safety)
    yhat_np = np.clip(yhat_np, y_min_f, y_max_f)

    metrics = _compute_metrics_dict(y_np, yhat_np, target_names)
    return total_loss / total, metrics, y_np, yhat_np, np.array(ssoid_all, dtype=object)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    modality,
    mu,
    sigma,
    y_min,
    y_max,
    constrain: str,
    target_names: List[str],
):
    """Evaluate model without returning ssoid."""
    loss, metrics, y, yhat, _ = evaluate_with_ids(
        model, loader, device, modality, mu, sigma, y_min, y_max, constrain, target_names
    )
    return loss, metrics, y, yhat


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    modality,
    loss_type,
    mu,
    sigma,
    y_min,
    y_max,
    constrain: str,
    alpha_corr: float,
    maepearson_criterion: Optional[MAE_PearsonLoss] = None,
    dist_criterion: "Optional[DistributionAlignmentLoss]" = None,
    lambda_dist: float = 1.0,
):
    """
    Train one epoch.
    - Shapes are enforced to (B,K).
    - For mae_pearson, compute on mmHg to penalize absolute bias correctly.
    """
    model.train()

    total = 0
    total_loss = 0.0

    y_min_f = float(y_min)
    y_max_f = float(y_max)

    # Make mu/sigma tensors once (broadcastable to (B,K))
    mu_t = torch.as_tensor(mu, device=device, dtype=torch.float32)
    sg_t = torch.as_tensor(sigma, device=device, dtype=torch.float32)
    if mu_t.ndim == 0:
        mu_t = mu_t.view(1, 1)
    elif mu_t.ndim == 1:
        mu_t = mu_t.view(1, -1)
    if sg_t.ndim == 0:
        sg_t = sg_t.view(1, 1)
    elif sg_t.ndim == 1:
        sg_t = sg_t.view(1, -1)

    use_amp = (device.type == "cuda")

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        if len(batch) == 4:
            ecg, ppg, y, _ = batch
        else:
            ecg, ppg, y = batch

        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Enforce y shape (B,K)
        if y.ndim == 1:
            y = y.view(-1, 1)

        if modality == "ecg":
            ppg = torch.zeros_like(ppg)
        if modality == "ppg":
            ecg = torch.zeros_like(ecg)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            raw = model(ecg, ppg)

            # Enforce raw shape (B,K)
            if raw.ndim == 1:
                raw = raw.view(-1, 1)

            yhat = apply_constraint(raw, constrain, y_min_f, y_max_f)

            # z-score versions (only for correlation helper if needed)
            y_z = (y - mu_t) / sg_t
            yhat_z = (yhat - mu_t) / sg_t

            if loss_type == "mse":
                reg = F.mse_loss(yhat, y)  # mmHg
            elif loss_type == "huber":
                reg = F.smooth_l1_loss(yhat, y, beta=1.0)  # mmHg
            elif loss_type == "mae_pearson":
                if maepearson_criterion is None:
                    raise ValueError("maepearson_criterion is None for loss_type='mae_pearson'")
                # IMPORTANT: compute in mmHg to penalize absolute bias (e.g., +20mmHg)
                reg = maepearson_criterion(yhat, y)
            elif loss_type == "mse+dist":
                reg = F.mse_loss(yhat, y)  # mmHg
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss = reg

            # Optional distribution alignment (expects mmHg)
            if loss_type == "mse+dist" and (dist_criterion is not None) and (lambda_dist > 0.0):
                if not _HAS_DIST:
                    raise RuntimeError("loss_type=mse+dist, but dist_loss.py is not available.")
                dist_loss_val = dist_criterion(yhat)
                loss = loss + float(lambda_dist) * dist_loss_val

            # Optional correlation helper (if not using mae_pearson)
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
        pbar.set_postfix(loss=f"{total_loss / total:.4f}")

    pbar.close()
    return total_loss / total