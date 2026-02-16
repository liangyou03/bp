#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _to_device_batch(batch, device):
    # dataset may return (ecg,ppg,y) or (ecg,ppg,y,ssoid)
    if len(batch) == 4:
        ecg, ppg, y, ssoid = batch
    else:
        ecg, ppg, y = batch
        ssoid = None
    ecg = ecg.to(device, non_blocking=True)
    ppg = ppg.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return ecg, ppg, y, ssoid


def _as_tensors(mu: np.ndarray, sigma: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device).view(1, -1)
    sg_t = torch.tensor(sigma, dtype=torch.float32, device=device).view(1, -1)
    return mu_t, sg_t


def _fit_affine_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-target least squares fit: y_true ≈ a * y_pred + b
    Returns a,b with shape (K,).
    """
    assert y_true.shape == y_pred.shape and y_true.ndim == 2
    K = y_true.shape[1]
    a = np.ones(K, dtype=np.float64)
    b = np.zeros(K, dtype=np.float64)

    for k in range(K):
        x = y_pred[:, k].astype(np.float64)
        y = y_true[:, k].astype(np.float64)

        vx = np.var(x)
        if vx < 1e-12:
            a[k] = 1.0
            b[k] = float(np.mean(y) - np.mean(x))
            continue

        cov = np.mean((x - x.mean()) * (y - y.mean()))
        a[k] = cov / vx
        b[k] = float(y.mean() - a[k] * x.mean())

    return a, b


def _apply_affine_calibration(y_pred: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return y_pred * a.reshape(1, -1) + b.reshape(1, -1)


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err * err, axis=0))
    return mae, rmse


def train_one_epoch_bp(
    model,
    loader,
    optimizer,
    scaler,
    device,
    targets: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    loss_type: str = "huber",
) -> float:
    model.train()
    mu_t, sg_t = _as_tensors(mu, sigma, device)

    total = 0
    total_loss = 0.0

    # ENGLISH comments: autocast should follow scaler state (so AMP can be disabled cleanly)
    use_amp = bool(getattr(scaler, "is_enabled", lambda: False)())

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        ecg, ppg, y_raw, _ = _to_device_batch(batch, device)

        # y in standardized space
        y_z = (y_raw - mu_t) / sg_t

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            yhat_z = model(ecg, ppg)  # (B,K) in z space

            if loss_type == "mse":
                loss = F.mse_loss(yhat_z, y_z)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(yhat_z, y_z, beta=1.0)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        # ENGLISH comments: if loss is non-finite, skip stepping to avoid corrupting weights
        if not torch.isfinite(loss):
            pbar.set_postfix(loss="NONFINITE_SKIP")
            continue

        if use_amp:
            scaler.scale(loss).backward()
            # ENGLISH comments: unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = y_raw.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        pbar.set_postfix(loss=f"{total_loss / max(1,total):.4f}")

    pbar.close()
    return total_loss / max(1, total)


@torch.no_grad()
def evaluate_bp(
    model,
    loader,
    device,
    targets: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    loss_type: str = "huber",
    fit_calibration: bool = False,
    apply_model_calibration: bool = True,
) -> Dict[str, Any]:
    """
    Returns dict with:
      loss (in z space), mae_macro/rmse_macro (raw BP units),
      r_macro/R2_macro (raw BP units),
      per-target mae_*/rmse_*/r_*/R2_* (raw BP units),
      right/left arm means,
      (optional) calibration params + calibrated metrics.

    If fit_calibration=True:
      - fit a,b on this loader (raw units)
      - store into model buffers via model.set_calibration(...)
      - report calibrated metrics (mae_macro_cal, r_macro_cal, etc.)
    """
    model.eval()
    mu_t, sg_t = _as_tensors(mu, sigma, device)

    tot = 0
    tot_loss = 0.0
    y_all = []
    yhat_all = []

    for batch in loader:
        ecg, ppg, y_raw, _ = _to_device_batch(batch, device)
        y_z = (y_raw - mu_t) / sg_t

        # ENGLISH comments: evaluation should be stable; use fp32 to avoid fp16 overflows masking issues
        with torch.cuda.amp.autocast(enabled=False):
            yhat_z = model(ecg, ppg)

            # ENGLISH comments: fail fast if model outputs NaN/Inf
            if not torch.isfinite(yhat_z).all():
                raise RuntimeError("Non-finite yhat_z detected in evaluate_bp")

            if loss_type == "mse":
                loss = F.mse_loss(yhat_z, y_z)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(yhat_z, y_z, beta=1.0)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        bs = y_raw.size(0)
        tot += bs
        tot_loss += float(loss.item()) * bs

        # back to raw units for metrics
        yhat_raw = yhat_z * sg_t + mu_t
        y_all.append(y_raw.detach().cpu().numpy())
        yhat_all.append(yhat_raw.detach().cpu().numpy())

    y = np.concatenate(y_all, axis=0).astype(np.float64)     # (N,K)
    yhat = np.concatenate(yhat_all, axis=0).astype(np.float64)

    out: Dict[str, Any] = {}
    out["loss"] = tot_loss / max(1, tot)
    out["N_records"] = int(y.shape[0])

    # --- MAE/RMSE (raw) ---
    mae, rmse = _mae_rmse(y, yhat)
    out["mae_macro"] = float(np.mean(mae))
    out["rmse_macro"] = float(np.mean(rmse))
    for t, m, r in zip(targets, mae, rmse):
        out[f"mae_{t}"] = float(m)
        out[f"rmse_{t}"] = float(r)

    # --- Pearson r and R2 (raw) ---
    K = y.shape[1]
    r_vec = np.full(K, np.nan, dtype=np.float64)
    r2_vec = np.full(K, np.nan, dtype=np.float64)

    for k in range(K):
        yt = y[:, k]
        yp = yhat[:, k]

        yt_m = float(np.mean(yt))
        yp_m = float(np.mean(yp))
        dy = yt - yt_m
        dp = yp - yp_m

        vy = float(np.mean(dy * dy))
        vp = float(np.mean(dp * dp))
        denom = np.sqrt(vy * vp)
        if denom > 1e-12:
            r_vec[k] = float(np.mean(dy * dp) / denom)

        sst = float(np.sum(dy * dy))
        if sst > 1e-12:
            sse = float(np.sum((yt - yp) ** 2))
            r2_vec[k] = float(1.0 - sse / sst)

    out["r_macro"] = float(np.nanmean(r_vec)) if np.isfinite(r_vec).any() else float("nan")
    out["R2_macro"] = float(np.nanmean(r2_vec)) if np.isfinite(r2_vec).any() else float("nan")
    for t, rv, r2v in zip(targets, r_vec, r2_vec):
        out[f"r_{t}"] = float(rv)
        out[f"R2_{t}"] = float(r2v)

    # arm means (if present)
    right_keys = ["right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp"]
    left_keys = ["left_arm_sbp", "left_arm_mbp", "left_arm_dbp", "left_arm_pp"]

    right_maes = [out.get(f"mae_{k}") for k in right_keys if f"mae_{k}" in out]
    left_maes = [out.get(f"mae_{k}") for k in left_keys if f"mae_{k}" in out]
    if right_maes:
        out["mae_right_arm_mean"] = float(np.mean(right_maes))
    if left_maes:
        out["mae_left_arm_mean"] = float(np.mean(left_maes))

    # --- calibration ---
    if fit_calibration:
        a_fit, b_fit = _fit_affine_calibration(y_true=y, y_pred=yhat)
        # store in model buffers
        model.set_calibration(
            scale=torch.from_numpy(a_fit.astype(np.float32)),
            bias=torch.from_numpy(b_fit.astype(np.float32)),
        )
        out["calib_fit"] = 1.0  # flag

    if apply_model_calibration:
        if not (hasattr(model, "calib_scale") and hasattr(model, "calib_bias")):
            raise RuntimeError("Model has no calib_scale/calib_bias buffers. Did you implement set_calibration()?")

        a = model.calib_scale.detach().cpu().numpy().astype(np.float64)
        b = model.calib_bias.detach().cpu().numpy().astype(np.float64)

        # store a/b as JSON-safe lists
        out["a"] = [float(x) for x in a.tolist()]
        out["b"] = [float(x) for x in b.tolist()]

        yhat_cal = _apply_affine_calibration(yhat, a, b)

        # MAE/RMSE (cal)
        mae_c, rmse_c = _mae_rmse(y, yhat_cal)
        out["mae_macro_cal"] = float(np.mean(mae_c))
        out["rmse_macro_cal"] = float(np.mean(rmse_c))
        for t, m, r in zip(targets, mae_c, rmse_c):
            out[f"mae_cal_{t}"] = float(m)
            out[f"rmse_cal_{t}"] = float(r)

        # Pearson r and R2 (cal)
        r_vec_c = np.full(K, np.nan, dtype=np.float64)
        r2_vec_c = np.full(K, np.nan, dtype=np.float64)
        for k in range(K):
            yt = y[:, k]
            yp = yhat_cal[:, k]

            yt_m = float(np.mean(yt))
            yp_m = float(np.mean(yp))
            dy = yt - yt_m
            dp = yp - yp_m

            vy = float(np.mean(dy * dy))
            vp = float(np.mean(dp * dp))
            denom = np.sqrt(vy * vp)
            if denom > 1e-12:
                r_vec_c[k] = float(np.mean(dy * dp) / denom)

            sst = float(np.sum(dy * dy))
            if sst > 1e-12:
                sse = float(np.sum((yt - yp) ** 2))
                r2_vec_c[k] = float(1.0 - sse / sst)

        out["r_macro_cal"] = float(np.nanmean(r_vec_c)) if np.isfinite(r_vec_c).any() else float("nan")
        out["R2_macro_cal"] = float(np.nanmean(r2_vec_c)) if np.isfinite(r2_vec_c).any() else float("nan")
        for t, rv, r2v in zip(targets, r_vec_c, r2_vec_c):
            out[f"r_cal_{t}"] = float(rv)
            out[f"R2_cal_{t}"] = float(r2v)

        right_maes_c = [out.get(f"mae_cal_{k}") for k in right_keys if f"mae_cal_{k}" in out]
        left_maes_c = [out.get(f"mae_cal_{k}") for k in left_keys if f"mae_cal_{k}" in out]
        if right_maes_c:
            out["mae_right_arm_mean_cal"] = float(np.mean(right_maes_c))
        if left_maes_c:
            out["mae_left_arm_mean_cal"] = float(np.mean(left_maes_c))

    return out
