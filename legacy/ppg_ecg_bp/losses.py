#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom Loss Functions

MAE_PearsonLoss:
  alpha * (1 - PearsonR) + beta * MAE

Fixes vs your version:
1) Do NOT flatten all outputs together. Compute per-target then average.
2) Keep shapes consistent for K=1 and K>1.
3) Avoid creating CPU float tensor in AMP path; keep dtype/device consistent.
"""

import torch
import torch.nn as nn


class MAE_PearsonLoss(nn.Module):
    """
    alpha * (1 - PearsonR) + beta * MAE
    Pearson and MAE are computed per-target (column) and then averaged.
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure (B, K)
        if y_hat.ndim == 1:
            y_hat = y_hat.view(-1, 1)
        if y.ndim == 1:
            y = y.view(-1, 1)

        if y_hat.shape != y.shape:
            raise ValueError(f"Shape mismatch: y_hat {tuple(y_hat.shape)} vs y {tuple(y.shape)}")

        losses = []
        K = y.shape[1]

        for c in range(K):
            yh = y_hat[:, c]
            yt = y[:, c]

            # Pearson loss: 1 - r
            vx = yh - yh.mean()
            vy = yt - yt.mean()

            std_x = torch.sqrt((vx * vx).sum())
            std_y = torch.sqrt((vy * vy).sum())

            # Use same dtype/device scalar
            zero = yh.new_zeros(())
            one = yh.new_ones(())

            if (std_x < self.eps) or (std_y < self.eps):
                pearson_loss = zero
            else:
                corr = (vx * vy).sum() / (std_x * std_y + self.eps)
                pearson_loss = one - corr

            # MAE loss
            mae_loss = torch.mean(torch.abs(yh - yt))

            losses.append(self.alpha * pearson_loss + self.beta * mae_loss)

        return torch.stack(losses).mean()