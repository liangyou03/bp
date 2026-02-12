#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAE_PearsonLoss(nn.Module):
    """MAE + Pearson 相关性损失
    
    L = alpha * (1 - r) + beta * MAE / y_std
    """
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    def pearson(self, y_pred, y_true):
        y_pred = y_pred - y_pred.mean()
        y_true = y_true - y_true.mean()
        denom = torch.sqrt(y_pred.pow(2).sum() * y_true.pow(2).sum()) + self.eps
        return (y_pred * y_true).sum() / denom
    
    def forward(self, y_pred, y_true):
        r = self.pearson(y_pred, y_true)
        mae = torch.abs(y_pred - y_true).mean()
        y_std = y_true.std() + self.eps
        loss = self.alpha * (1 - r) + self.beta * (mae / y_std)
        return loss


MAEPearsonLoss = MAE_PearsonLoss  # backward compatibility


class HuberLoss(nn.Module):
    """Huber Loss - 对异常值更鲁棒"""
    def __init__(self, delta=5.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        residual = torch.abs(y_pred - y_true)
        quadratic = torch.clamp(residual, max=self.delta)
        linear = residual - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        return loss.mean()


class MAELoss(nn.Module):
    """简单MAE Loss"""
    def forward(self, y_pred, y_true):
        return torch.abs(y_pred - y_true).mean()


if __name__ == "__main__":
    y_pred = torch.randn(100)
    y_true = torch.randn(100)
    loss1 = MAE_PearsonLoss()(y_pred, y_true)
    loss2 = HuberLoss()(y_pred, y_true)
    loss3 = MAELoss()(y_pred, y_true)
    print(f"MAE_Pearson: {loss1.item():.4f}")
    print(f"Huber: {loss2.item():.4f}")
    print(f"MAE: {loss3.item():.4f}")
