#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自定义损失函数 (Custom Loss Functions)
"""

import torch
import torch.nn as nn

class MAE_PearsonLoss(nn.Module):
    """
    计算 (alpha * (1 - PearsonR)) + (beta * MAE)
    """
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-8):
        super().__init__()
        self.alpha = float(alpha); self.beta = float(beta); self.eps = float(eps)
        
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat.reshape(-1); y = y.reshape(-1)
        
        # Pearson Loss (1 - r)
        vx = y_hat - y_hat.mean(); vy = y - y.mean()
        corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()) + self.eps)
        pearson_loss = 1.0 - corr
        
        # MAE Loss
        mae_loss = torch.mean(torch.abs(y_hat - y))
        
        return self.alpha * pearson_loss + self.beta * mae_loss