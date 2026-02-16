#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class MAE_PearsonLoss(nn.Module):
    """
    Multi-target version:
      loss = alpha * mean_k(1 - corr_k) + beta * mean(|y_hat - y|)
    y_hat, y: (B,K) or (B,)
    """
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y_hat.dim() == 1:
            y_hat = y_hat[:, None]
        if y.dim() == 1:
            y = y[:, None]

        # MAE over all entries
        mae_loss = torch.mean(torch.abs(y_hat - y))

        # Pearson per target (over batch dimension)
        vx = y_hat - y_hat.mean(dim=0, keepdim=True)
        vy = y - y.mean(dim=0, keepdim=True)
        num = torch.sum(vx * vy, dim=0)
        den = torch.sqrt(torch.sum(vx * vx, dim=0)) * torch.sqrt(torch.sum(vy * vy, dim=0)) + self.eps
        corr = num / den  # (K,)
        pearson_loss = torch.mean(1.0 - corr)

        return self.alpha * pearson_loss + self.beta * mae_loss
