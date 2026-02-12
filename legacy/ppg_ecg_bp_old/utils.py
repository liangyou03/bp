#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
辅助工具函数 (Utility Functions)
包括：设置随机种子、Numpy版评估指标、ID处理、输出约束、个体聚合
"""

import os
import random
import numpy as np
import pandas as pd
import torch

def set_seed(seed: int = 666):
    """设置全局随机种子"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def subject_id_from_ssoid(ssoid: str) -> str:
    """从 ssoid (e.g., 'Subject123_Rec01') 提取个体ID (e.g., 'Subject123')"""
    return str(ssoid).split("_", 1)[0]

# ===================== Numpy 评估指标 =====================
def mae_np(y, yhat): 
    return float(np.mean(np.abs(y - yhat)))

def rmse_np(y, yhat): 
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def pearson_r_safe_np(y, yhat, eps=1e-12):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask): return 0.0
    y = y[mask]; yhat = yhat[mask]
    if len(y) < 2: return 0.0
    y0 = y - y.mean(); h0 = yhat - yhat.mean()
    denom_y = np.sqrt((y0**2).sum()); denom_h = np.sqrt((h0**2).sum())
    denom = (denom_y * denom_h) + eps
    if denom <= eps: return 0.0
    # 检查标准差是否为0
    if denom_y <= eps or denom_h <= eps: return 0.0
    r = float((y0 * h0).sum() / denom)
    return max(min(r, 1.0), -1.0)

def r2_np(y, yhat, eps=1e-12):
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2) + eps)
    return 1.0 - ss_res/ss_tot

# ===================== 约束与聚合 =====================
def apply_constraint(raw: torch.Tensor, mode: str, y_min, y_max) -> torch.Tensor:
    """对模型原始输出应用范围约束，支持逐维上下界"""
    if mode == "none":
        return raw
    y_min_t = torch.as_tensor(y_min, dtype=raw.dtype, device=raw.device)
    y_max_t = torch.as_tensor(y_max, dtype=raw.dtype, device=raw.device)
    rng = (y_max_t - y_min_t)
    if mode == "tanh":
        # Tanh 映射到 (y_min, y_max)
        return y_min_t + 0.5 * (torch.tanh(raw) + 1.0) * rng
    if mode == "sigmoid":
        # Sigmoid 映射到 (y_min, y_max)
        return y_min_t + torch.sigmoid(raw) * rng
    if mode == "clip":
        # 硬裁剪
        return torch.clamp(raw, y_min_t, y_max_t)
    return raw

def aggregate_by_subject_prefix(y: np.ndarray,
                                yhat: np.ndarray,
                                ssoids: np.ndarray,
                                agg: str = "mean"):
    """
    根据 ssoid 的下划线前缀聚合为 subject。
    返回：
     subjects(np.str_), y_subj(np.float64), yhat_subj(np.float64), n_records(np.int32)
    """
    assert len(y)==len(yhat)==len(ssoids)
    df = pd.DataFrame({
        "ssoid": ssoids.astype(str),
        "subject": [subject_id_from_ssoid(s) for s in ssoids],
        "y": y.astype(np.float64),
        "yhat": yhat.astype(np.float64),
    })
    agg_map = {"mean":"mean","median":"median","max":"max"}
    agg_fn = agg_map.get(agg, "mean")
    g = df.groupby("subject").agg(
        y=("y", agg_fn),
        yhat=("yhat", agg_fn),
        n_records=("ssoid", "size")
    ).reset_index()
    subjects = g["subject"].to_numpy(dtype=object)
    y_subj = g["y"].to_numpy(dtype=np.float64)
    yhat_subj = g["yhat"].to_numpy(dtype=np.float64)
    n_records = g["n_records"].to_numpy(dtype=np.int32)
    return subjects, y_subj, yhat_subj, n_records

# ===================== NEW: 线性校准器 =====================
class LinearCalibrator:
    """
    线性校准器 (Linear Calibrator)
    
    用于拟合 y_true ≈ a + b * y_pred
    在 验证集(validation set) 上 `fit`，
    在 测试集(test set) 上 `transform`。
    """
    def __init__(self):
        self.a = 0.0  # 截距 (Intercept)
        self.b = 1.0  # 斜率 (Slope)

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        在验证集上学习校准参数 a 和 b
        y_true ≈ a + b * y_pred
        """
        # 构造 A 矩阵 (y_pred, 1)
        A = np.vstack([y_pred, np.ones_like(y_pred)]).T
        # 求解 (b, a)
        try:
            self.b, self.a = np.linalg.lstsq(A, y_true, rcond=None)[0]
        except np.linalg.LinAlgError:
            print("[Warning] 线性校准拟合失败，使用默认值 a=0, b=1")
            self.a = 0.0
            self.b = 1.0
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """应用校准: a + b * y_pred"""
        return self.a + self.b * y_pred

    def fit_transform(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """拟合并立即应用（通常在验证集上自用）"""
        self.fit(y_pred, y_true)
        return self.transform(y_pred)

    def __repr__(self):
        return f"LinearCalibrator(a={self.a:.4f}, b={self.b:.4f})"
