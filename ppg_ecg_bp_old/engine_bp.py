#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练 (Train) 与 评估 (Evaluate) 逻辑 - BP 多目标版本
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Dict, List

from utils import mae_np, rmse_np, pearson_r_safe_np, r2_np
from losses import MAE_PearsonLoss


def _compute_metrics_dict(y: np.ndarray,
                          yhat: np.ndarray,
                          target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """计算每个目标的指标"""
    if y.ndim == 1:
        y = y[:, None]
        yhat = yhat[:, None]
    metrics = {}
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
def evaluate_with_ids(model, loader, device, modality,
                      mu, sigma, target_names: List[str]):
    """评估模型，返回所有指标和 ssoid"""
    model.eval()
    y_all, yhat_all, ssoid_all = [], [], []
    total, total_loss = 0, 0.0
    
    for batch in loader:
        ecg, ppg, y, ssoids = batch
        ecg, ppg, y = ecg.to(device), ppg.to(device), y.to(device)
        
        # 跳过 NaN 输入
        if torch.isnan(ecg).any() or torch.isnan(ppg).any() or torch.isnan(y).any():
            continue
        
        if modality == "ecg": ppg = torch.zeros_like(ppg)
        if modality == "ppg": ecg = torch.zeros_like(ecg)
        
        with torch.cuda.amp.autocast(enabled=True):
            raw = model(ecg, ppg)
            if torch.isnan(raw).any():
                continue
            # z-score 尺度的损失
            loss = F.smooth_l1_loss((raw - mu) / sigma, (y - mu) / sigma)
        
        bs = y.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        y_all.append(y.cpu().numpy())
        yhat_all.append(raw.cpu().numpy())
        ssoid_all.extend(list(ssoids))

    y = np.concatenate(y_all, axis=0)
    yhat = np.concatenate(yhat_all, axis=0)
    
    metrics = _compute_metrics_dict(y, yhat, target_names)
    avg_mae = float(np.mean([m["mae"] for m in metrics.values()]))
    
    return total_loss / total, metrics, avg_mae, y, yhat, np.array(ssoid_all, dtype=object)


@torch.no_grad()
def evaluate(model, loader, device, modality, mu, sigma, target_names: List[str]):
    """评估模型，不返回 ssoid"""
    loss, metrics, avg_mae, y, yhat, _ = evaluate_with_ids(
        model, loader, device, modality, mu, sigma, target_names
    )
    return loss, metrics, avg_mae


def train_one_epoch(model, loader, optimizer, scaler, device, modality, loss_type,
                    mu, sigma, alpha_corr: float,
                    maepearson_criterion: Optional[MAE_PearsonLoss] = None):
    """训练一个 Epoch"""
    model.train()
    total, total_loss = 0, 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        ecg, ppg, y, _ = batch
        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # 跳过 NaN 输入
        if torch.isnan(ecg).any() or torch.isnan(ppg).any() or torch.isnan(y).any():
            continue
        
        if modality == "ecg": ppg = torch.zeros_like(ppg)
        if modality == "ppg": ecg = torch.zeros_like(ecg)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=True):
            raw = model(ecg, ppg)
            
            if torch.isnan(raw).any():
                continue
            
            # z-score 标准化
            y_z = (y - mu) / sigma
            yhat_z = (raw - mu) / sigma

            if loss_type == "mse":
                reg = F.mse_loss(yhat_z, y_z)
            elif loss_type == "huber":
                reg = F.smooth_l1_loss(yhat_z, y_z)
            elif loss_type == "mae_pearson":
                if maepearson_criterion is None:
                    raise ValueError("maepearson_criterion is None")
                reg = maepearson_criterion(yhat_z, y_z)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss = reg

            # (可选) 相关性辅助损失
            if alpha_corr > 0.0 and loss_type != "mae_pearson":
                corr_losses = []
                for c in range(y_z.shape[1]):
                    y0 = y_z[:, c] - y_z[:, c].mean()
                    yhat0 = yhat_z[:, c] - yhat_z[:, c].mean()
                    std_y = torch.sqrt((y0**2).sum())
                    std_yhat = torch.sqrt((yhat0**2).sum())
                    if std_y > 1e-8 and std_yhat > 1e-8:
                        corr = (y0 * yhat0).sum() / (std_y * std_yhat + 1e-8)
                        corr_losses.append(1.0 - corr)
                if corr_losses:
                    loss = loss + alpha_corr * torch.stack(corr_losses).mean()

        if torch.isnan(loss):
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        bs = y.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}")
    
    pbar.close()
    return total_loss / max(total, 1)