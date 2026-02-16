#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数
"""

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def mae_np(y, yhat):
    """MAE"""
    return float(np.mean(np.abs(y - yhat)))


def rmse_np(y, yhat):
    """RMSE"""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def pearson_r_np(y, yhat, eps=1e-12):
    """Pearson相关系数"""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    y0 = y - y.mean()
    h0 = yhat - yhat.mean()
    denom = np.sqrt((y0**2).sum()) * np.sqrt((h0**2).sum()) + eps
    return float(np.clip((y0 * h0).sum() / denom, -1, 1))


def pearson_r_safe_np(y, yhat, eps: float = 1e-12):
    """数值稳定的Pearson相关"""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return 0.0
    y = y[mask]
    yhat = yhat[mask]
    if len(y) < 2:
        return 0.0
    y0 = y - y.mean()
    h0 = yhat - yhat.mean()
    denom_y = np.sqrt((y0**2).sum())
    denom_h = np.sqrt((h0**2).sum())
    denom = (denom_y * denom_h) + eps
    if denom <= eps or denom_y <= eps or denom_h <= eps:
        return 0.0
    val = float((y0 * h0).sum() / denom)
    return max(min(val, 1.0), -1.0)


def r2_np(y, yhat, eps=1e-12):
    """R²分数"""
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + eps
    return float(1.0 - ss_res / ss_tot)


def subject_id_from_ssoid(ssoid: str) -> str:
    """从ssoid提取subject ID (默认取下划线前缀)"""
    if not isinstance(ssoid, str):
        ssoid = str(ssoid)
    return ssoid.split("_", 1)[0]


def apply_constraint(raw: torch.Tensor, mode: str, y_min: float, y_max: float) -> torch.Tensor:
    """对模型原始输出应用范围约束"""
    if mode == "none":
        return raw
    rng = (y_max - y_min)
    if mode == "tanh":
        return y_min + 0.5 * (torch.tanh(raw) + 1.0) * rng
    if mode == "sigmoid":
        return y_min + torch.sigmoid(raw) * rng
    if mode == "clip":
        return torch.clamp(raw, y_min, y_max)
    return raw


def aggregate_by_subject_prefix(y: np.ndarray,
                                yhat: np.ndarray,
                                ssoids: np.ndarray,
                                agg: str = "mean") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据ssoid前缀聚合到subject级别"""
    assert len(y) == len(yhat) == len(ssoids)
    df = pd.DataFrame({
        "ssoid": ssoids.astype(str),
        "subject": [subject_id_from_ssoid(s) for s in ssoids],
        "y": y.astype(np.float64),
        "yhat": yhat.astype(np.float64),
    })
    agg_map = {"mean": "mean", "median": "median", "max": "max"}
    agg_fn = agg_map.get(agg, "mean")
    grouped = df.groupby("subject").agg(
        y=("y", agg_fn),
        yhat=("yhat", agg_fn),
        n_records=("ssoid", "size")
    ).reset_index()
    subjects = grouped["subject"].to_numpy(dtype=object)
    y_subj = grouped["y"].to_numpy(dtype=np.float64)
    yhat_subj = grouped["yhat"].to_numpy(dtype=np.float64)
    n_records = grouped["n_records"].to_numpy(dtype=np.int32)
    return subjects, y_subj, yhat_subj, n_records


class LinearCalibrator:
    """线性校准 y_cal = a + b * y_pred"""
    def __init__(self):
        self.a = 0.0  # intercept
        self.b = 1.0  # slope

    def fit(self, y_pred, y_true):
        """最小二乘拟合 (返回 self 以便链式调用)"""
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        if y_pred.ndim != 1:
            y_pred = y_pred.ravel()
        if y_true.ndim != 1:
            y_true = y_true.ravel()
        A = np.vstack([y_pred, np.ones_like(y_pred)]).T
        try:
            self.b, self.a = np.linalg.lstsq(A, y_true, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.a = 0.0
            self.b = 1.0
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        return self.a + self.b * y_pred

    def predict(self, y_pred):
        return self.transform(y_pred)

    def fit_transform(self, y_pred, y_true):
        return self.fit(y_pred, y_true).transform(y_pred)

    def to_dict(self):
        return {"a": float(self.a), "b": float(self.b)}

    def __repr__(self):
        return f"LinearCalibrator(a={self.a:.4f}, b={self.b:.4f})"


def compute_metrics(y_true, y_pred):
    """计算所有指标"""
    return {
        "MAE": mae_np(y_true, y_pred),
        "RMSE": rmse_np(y_true, y_pred),
        "r": pearson_r_safe_np(y_true, y_pred),
        "R2": r2_np(y_true, y_pred),
    }


if __name__ == "__main__":
    y_true = np.array([100, 120, 140, 160, 180])
    y_pred = np.array([105, 115, 145, 155, 175])
    metrics = compute_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    cal = LinearCalibrator().fit(y_pred, y_true)
    print(f"\nCalibration: {cal}")
