#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练 (Train) 与 评估 (Evaluate) 逻辑
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional

# 本地模块导入
from utils import apply_constraint, mae_np, rmse_np, pearson_r_safe_np, r2_np
from losses import MAE_PearsonLoss

# ============== (可选) 分布先验对齐 ==============
try:
    from dist_loss import DistributionAlignmentLoss
    _HAS_DIST = True
except Exception:
    DistributionAlignmentLoss = None  # 定义占位符
    _HAS_DIST = False
# ===============================================

@torch.no_grad()
def evaluate_with_ids(model, loader, device, modality,
                      mu, sigma, y_min, y_max, constrain: str):
    """评估模型，并返回所有 ssoid"""
    model.eval()
    y_all=[]; yhat_all=[]; ssoid_all=[]
    total=0; total_loss=0.0
    for batch in loader:
        if len(batch)==4:
            ecg, ppg, y, ssoids = batch
        else:
            ecg, ppg, y = batch
            ssoids = [""]*y.size(0)
        ecg, ppg, y = ecg.to(device), ppg.to(device), y.to(device)
        if modality=="ecg": ppg = torch.zeros_like(ppg)
        if modality=="ppg": ecg = torch.zeros_like(ecg)
        
        with torch.cuda.amp.autocast(enabled=True):
            raw = model(ecg, ppg)
            yhat = apply_constraint(raw, constrain, y_min, y_max)
            # 损失使用 z-score 尺度
            loss = F.smooth_l1_loss((yhat - mu)/sigma, (y - mu)/sigma, beta=1.0)
            
        bs = y.size(0)
        total += bs; total_loss += float(loss.item())*bs
        y_all.append(y.detach().cpu().numpy())
        yhat_all.append(yhat.detach().cpu().numpy())
        if isinstance(ssoids, (list, tuple)):
            ssoid_all.extend(list(ssoids))
        else:
            ssoid_all.extend([str(ssoids)]*bs)

    y = np.concatenate(y_all).astype(np.float64)
    yhat = np.concatenate(yhat_all).astype(np.float64)
    yhat = np.clip(yhat, y_min, y_max) # 最终评估再 clip 一次
    
    mae = mae_np(y,yhat); rmse = rmse_np(y,yhat); r = pearson_r_safe_np(y,yhat); r2 = r2_np(y,yhat)
    return total_loss/total, mae, rmse, r, r2, y, yhat, np.array(ssoid_all, dtype=object)

@torch.no_grad()
def evaluate(model, loader, device, modality, mu, sigma, y_min, y_max, constrain: str):
    """评估模型，不返回 ssoid"""
    loss, mae, rmse, r, r2, y, yhat, _ = evaluate_with_ids(
        model, loader, device, modality, mu, sigma, y_min, y_max, constrain
    )
    return loss, mae, rmse, r, r2, y, yhat

def train_one_epoch(model, loader, optimizer, scaler, device, modality, loss_type,
                    mu, sigma, y_min, y_max, constrain: str, alpha_corr: float,
                    maepearson_criterion: Optional[MAE_PearsonLoss] = None,
                    dist_criterion: "Optional[DistributionAlignmentLoss]" = None,
                    lambda_dist: float = 1.0):
    """训练一个 Epoch"""
    model.train()
    total=0; total_loss=0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        if len(batch)==4:
            ecg, ppg, y, _ = batch
        else:
            ecg, ppg, y = batch
            
        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)
        
        if modality=="ecg":   ppg = torch.zeros_like(ppg)
        if modality=="ppg":   ecg = torch.zeros_like(ecg)
            
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=True):
            raw  = model(ecg, ppg)
            yhat = apply_constraint(raw, constrain, y_min, y_max)
            y_z    = (y - mu)/sigma
            yhat_z = (yhat - mu)/sigma # 损失函数在 z-score 尺度上计算

            if loss_type=="mse":
                reg = F.mse_loss(yhat_z, y_z)
            elif loss_type=="huber":
                reg = F.smooth_l1_loss(yhat_z, y_z, beta=1.0)
            elif loss_type=="mae_pearson":
                if maepearson_criterion is None:
                    raise ValueError("maepearson_criterion is None for loss_type='mae_pearson'")
                reg = maepearson_criterion(yhat_z, y_z)
            elif loss_type=="mse+dist":
                reg = F.mse_loss(yhat_z, y_z)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss = reg

            # (可选) 分布损失
            if loss_type=="mse+dist" and (dist_criterion is not None) and (lambda_dist > 0.0):
                if not _HAS_DIST:
                    raise RuntimeError("loss_type=mse+dist, but dist_loss.py is not available.")
                dist_loss_val = dist_criterion(yhat)  # 分布损失用原始尺度
                loss = loss + float(lambda_dist) * dist_loss_val

            # (可选) 相关性辅助损失
            if alpha_corr > 0.0 and loss_type != "mae_pearson":
                y0   = y_z - y_z.mean()
                yhat0= yhat_z - yhat_z.mean()
                num  = (y0 * yhat0).sum()
                den  = torch.sqrt((y0**2).sum()) * torch.sqrt((yhat0**2).sum()) + 1e-8
                corr = num / den
                loss = loss + alpha_corr * (1.0 - corr)

        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        
        bs = y.size(0); total += bs; total_loss += float(loss.item())*bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}")
        
    pbar.close()
    return total_loss/total