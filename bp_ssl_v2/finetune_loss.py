# finetune_loss.py
# Loss utilities for regression fine-tuning (BP or any continuous target).
# Includes:
# 1) DistributionAlignmentLoss (KDE prior + batch soft histogram alignment)
# 2) Inverse-frequency weighted MSE (imbalance handling)
# 3) MSE * PearsonLoss (product form)

from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# 1) Dist Loss: KDE Prior Builder
# -----------------------------
def build_kde_prior(
    train_targets: np.ndarray,
    y_min: float,
    y_max: float,
    bin_width: float = 1.0,
    sigma: float = 2.0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    train_targets = np.asarray(train_targets, dtype=np.float64)
    bin_centers = np.arange(y_min, y_max + 1e-6, bin_width, dtype=np.float64)

    diffs = (bin_centers[None, :] - train_targets[:, None]) / (sigma + eps)
    kernel_vals = np.exp(-0.5 * diffs ** 2)
    prior = kernel_vals.sum(axis=0)
    prior = prior + eps
    prior = prior / prior.sum()
    return bin_centers.astype(np.float32), prior.astype(np.float32)


class DistributionAlignmentLoss(nn.Module):
    def __init__(
        self,
        bin_centers: np.ndarray,
        prior_probs: np.ndarray,
        sigma: float = 2.0,
        inv_weight: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("bin_centers", torch.as_tensor(bin_centers).float())
        self.register_buffer("prior_probs", torch.as_tensor(prior_probs).float())
        self.sigma2 = float(sigma) ** 2
        self.inv_weight = bool(inv_weight)
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        y = y_pred.reshape(-1, 1)
        centers = self.bin_centers.reshape(1, -1)
        d2 = (y - centers) ** 2

        weights = torch.exp(-0.5 * d2 / (self.sigma2 + self.eps))
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)

        q = weights.sum(dim=0)
        q = q / (q.sum() + self.eps)

        p = torch.clamp(self.prior_probs, min=self.eps)

        if self.inv_weight:
            w = 1.0 / p
            w = w / (w.mean() + self.eps)
        else:
            w = torch.ones_like(p)

        loss = torch.mean(w * (q - p) ** 2)
        return loss


# -----------------------------
# 2) Inverse-frequency weighted MSE
# -----------------------------
def build_invfreq_weights(
    train_targets: np.ndarray,
    y_min: float,
    y_max: float,
    bin_width: float = 1.0,
    smoothing: float = 1.0,
    normalize_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    train_targets = np.asarray(train_targets, dtype=np.float64)
    bin_centers = np.arange(y_min, y_max + 1e-6, bin_width, dtype=np.float64)

    idx = np.clip(np.rint(train_targets).astype(np.int64), int(y_min), int(y_max)) - int(y_min)
    counts = np.bincount(idx, minlength=len(bin_centers)).astype(np.float64)

    counts = counts + float(smoothing)
    inv = 1.0 / counts
    if normalize_mean:
        inv = inv / (inv.mean() + 1e-12)

    return bin_centers.astype(np.float32), inv.astype(np.float32)


class InverseFreqWeightedMSELoss(nn.Module):
    def __init__(
        self,
        bin_centers: np.ndarray,
        bin_weights: np.ndarray,
        y_min: float,
        y_max: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("bin_centers", torch.as_tensor(bin_centers).float())
        self.register_buffer("bin_weights", torch.as_tensor(bin_weights).float())
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true_clamped = torch.clamp(y_true, self.y_min, self.y_max)
        idx = torch.round(y_true_clamped).long() - int(self.y_min)
        idx = torch.clamp(idx, 0, self.bin_weights.numel() - 1)
        w = self.bin_weights[idx]

        mse = (y_pred - y_true) ** 2
        return torch.mean(w * mse)


# -----------------------------
# 3) Pearson Loss & MSE * Pearson
# -----------------------------
class PearsonLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred.view(-1)
        y = y_true.view(-1)

        x = x - x.mean()
        y = y - y.mean()

        vx = torch.sqrt(torch.sum(x * x) + self.eps)
        vy = torch.sqrt(torch.sum(y * y) + self.eps)

        corr = torch.sum(x * y) / (vx * vy + self.eps)
        return 1.0 - corr


class MSEPearsonProductLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.mse = nn.MSELoss()
        self.ploss = PearsonLoss(eps=eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        mse = self.mse(y_pred, y_true)
        pl = self.ploss(y_pred, y_true)
        loss = mse * (1.0 + self.alpha * pl)
        stats = {
            "mse": float(mse.detach().cpu().item()),
            "pearson_loss": float(pl.detach().cpu().item()),
        }
        return loss, stats


# -----------------------------
# Helper to prepare loss objects
# -----------------------------
def prepare_losses(
    loss_mode: str,
    train_targets: np.ndarray,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    dist_bin_width: float = 1.0,
    dist_kde_sigma: float = 2.0,
    dist_assign_sigma: float = 2.0,
    dist_inv_weight: bool = True,
    wmse_bin_width: float = 1.0,
    wmse_smoothing: float = 1.0,
    pearson_alpha: float = 1.0,
):
    assert loss_mode in ["mse", "mse+dist", "wmse", "mse*pearson"]

    train_targets = np.asarray(train_targets, dtype=np.float32)
    if y_min is None:
        y_min = float(np.floor(train_targets.min()))
    if y_max is None:
        y_max = float(np.ceil(train_targets.max()))

    pack = {"loss_mode": loss_mode, "y_min": float(y_min), "y_max": float(y_max)}

    if loss_mode == "mse":
        pack["mse"] = nn.MSELoss()

    elif loss_mode == "mse+dist":
        centers, prior = build_kde_prior(
            train_targets=train_targets,
            y_min=y_min,
            y_max=y_max,
            bin_width=dist_bin_width,
            sigma=dist_kde_sigma,
        )
        pack["mse"] = nn.MSELoss()
        pack["dist"] = DistributionAlignmentLoss(
            bin_centers=centers,
            prior_probs=prior,
            sigma=dist_assign_sigma,
            inv_weight=dist_inv_weight,
        )

    elif loss_mode == "wmse":
        centers, weights = build_invfreq_weights(
            train_targets=train_targets,
            y_min=y_min,
            y_max=y_max,
            bin_width=wmse_bin_width,
            smoothing=wmse_smoothing,
            normalize_mean=True,
        )
        pack["wmse"] = InverseFreqWeightedMSELoss(
            bin_centers=centers,
            bin_weights=weights,
            y_min=y_min,
            y_max=y_max,
        )

    else:  # "mse*pearson"
        pack["mse_pearson"] = MSEPearsonProductLoss(alpha=pearson_alpha)

    return pack
