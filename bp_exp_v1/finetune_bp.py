#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BP预测微调 - 使用CLIP预训练权重

工作流程:
1. 先运行 pretrain_clip.py 预训练 ECG/PPG encoder
2. 运行此脚本，加载预训练权重微调BP预测
"""

import os
import json
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from backbones import ECGEncoderCLIP, PPGEncoderCLIP
from utils import set_seed


# ==================== Dataset ====================
class BPDataset(Dataset):
    """BP预测数据集 - 新格式"""
    def __init__(self, df, npz_dir, target_col):
        self.df = df.reset_index(drop=True)
        self.npz_dir = Path(npz_dir)
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ssoid = str(row["ssoid"])
        
        with np.load(self.npz_dir / f"{ssoid}.npz") as d:
            ecg = torch.from_numpy(d["ecg"].astype(np.float32)).unsqueeze(0)
            ppg = torch.from_numpy(d["ppg"].astype(np.float32)).unsqueeze(0)
        
        target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return ecg, ppg, target


def get_loaders(data_dir, target_col, batch_size=64, num_workers=4):
    """获取train/val/test数据加载器"""
    data_dir = Path(data_dir)
    labels_df = pd.read_csv(data_dir / "labels.csv")
    
    train_df = labels_df[labels_df['split'] == 'train']
    val_df = labels_df[labels_df['split'] == 'val']
    test_df = labels_df[labels_df['split'] == 'test']
    
    train_df = train_df[train_df[target_col].notna()]
    val_df = val_df[val_df[target_col].notna()]
    test_df = test_df[test_df[target_col].notna()]
    
    print(f"[{target_col}] train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    
    train_ds = BPDataset(train_df, data_dir / "npz", target_col)
    val_ds = BPDataset(val_df, data_dir / "npz", target_col)
    test_ds = BPDataset(test_df, data_dir / "npz", target_col)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


# ==================== Model ====================
class BPRegressionHead(nn.Module):
    """BP回归头"""
    def __init__(self, ecg_dim=1024, ppg_dim=512, hidden_dim=256):
        super().__init__()
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(ecg_dim + ppg_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, ecg_feat, ppg_feat):
        """
        ecg_feat: (B, ecg_dim)
        ppg_feat: (B, ppg_dim)
        """
        fused = torch.cat([ecg_feat, ppg_feat], dim=1)
        return self.fusion(fused).squeeze(-1)


class SingleModalBPHead(nn.Module):
    """单模态BP回归头"""
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, feat):
        return self.mlp(feat).squeeze(-1)


class BPModel(nn.Module):
    """BP预测模型 - 封装encoder + head"""
    def __init__(self, ecg_encoder, ppg_encoder, head):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder
        self.head = head
    
    def forward(self, ecg, ppg):
        # ecg, ppg: (B, 1, T)
        ecg_feat = self.ecg_encoder(ecg)  # (B, D)
        ppg_feat = self.ppg_encoder(ppg)  # (B, D)
        return self.head(ecg_feat, ppg_feat)


class SingleModalBPModel(nn.Module):
    """单模态BP预测模型"""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


# ==================== Loss ====================
class MAEPearsonLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, y_pred, y_true):
        r = torch.corrcoef(torch.stack([y_pred, y_true]))[0, 1]
        mae = torch.abs(y_pred - y_true).mean()
        y_std = y_true.std() + 1e-8
        loss = self.alpha * (1 - r) + self.beta * (mae / y_std)
        return loss


# ==================== Training ====================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    
    for ecg, ppg, target in tqdm(dataloader, desc="Train", leave=False):
        ecg = ecg.to(device)
        ppg = ppg.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # 判断模型类型
        try:
            pred = model(ecg, ppg)
        except:
            # 单模态模型
            if hasattr(model, 'encoder'):
                # SingleModalBPModel
                if 'ECG' in model.encoder.__class__.__name__:
                    pred = model(ecg)
                else:
                    pred = model(ppg)
            else:
                # BPModel with single modality
                if model.ecg_encoder.__class__.__name__ == 'Identity':
                    pred = model.head(torch.zeros_like(target), model.ppg_encoder(ppg))
                else:
                    pred = model.head(model.ecg_encoder(ecg), torch.zeros_like(target))
        
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * target.size(0)
        n += target.size(0)
    
    return total_loss / n if n > 0 else 0


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    n = 0
    
    for ecg, ppg, target in tqdm(dataloader, desc="Eval", leave=False):
        ecg = ecg.to(device)
        ppg = ppg.to(device)
        target = target.to(device)
        
        try:
            pred = model(ecg, ppg)
        except:
            if hasattr(model, 'encoder'):
                if 'ECG' in model.encoder.__class__.__name__:
                    pred = model(ecg)
                else:
                    pred = model(ppg)
            else:
                if model.ecg_encoder.__class__.__name__ == 'Identity':
                    pred = model.head(torch.zeros_like(target), model.ppg_encoder(ppg))
                else:
                    pred = model.head(model.ecg_encoder(ecg), torch.zeros_like(target))
        
        loss = criterion(pred, target)
        total_loss += loss.item() * target.size(0)
        n += target.size(0)
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    y0 = y_true - y_true.mean()
    h0 = y_pred - y_pred.mean()
    r = float(np.clip((y0 * h0).sum() / (np.sqrt((y0**2).sum()) * np.sqrt((h0**2).sum()) + 1e-12), -1, 1))
    
    r2 = float(1.0 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-12))
    
    metrics = {
        "loss": total_loss / n if n > 0 else 0,
        "MAE": mae,
        "RMSE": rmse,
        "r": r,
        "R2": r2,
    }
    
    return metrics, y_pred, y_true


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    # CLIP预训练权重
    parser.add_argument("--clip_ckpt", type=str, required=True)
    # Model
    parser.add_argument("--modality", type=str, default="both", choices=["ecg", "ppg", "both"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--freeze_backbone", action="store_true")
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="mae_pearson", choices=["mse", "mae", "mae_pearson"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # System
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 数据加载器
    train_loader, val_loader, test_loader = get_loaders(
        args.data_dir, args.target_col, args.batch_size
    )
    
    # 加载CLIP预训练权重
    print(f"Loading CLIP pretrained: {args.clip_ckpt}")
    clip_ckpt = torch.load(args.clip_ckpt, map_location="cpu")
    
    # 创建encoder
    ecg_encoder = ECGEncoderCLIP(with_proj=False)  # 不用projection head
    ppg_encoder = PPGEncoderCLIP(with_proj=False)
    
    # 加载预训练权重
    if "ecg_encoder" in clip_ckpt:
        ecg_encoder.load_state_dict(clip_ckpt["ecg_encoder"])
    elif "model" in clip_ckpt:
        # 从完整模型中提取
        state = clip_ckpt["model"]
        ecg_state = {k.replace("ecg_enc.", ""): v for k, v in state.items() if k.startswith("ecg_enc.")}
        ecg_encoder.load_state_dict(ecg_state, strict=False)
    
    if "ppg_encoder" in clip_ckpt:
        ppg_encoder.load_state_dict(clip_ckpt["ppg_encoder"])
    elif "model" in clip_ckpt:
        state = clip_ckpt["model"]
        ppg_state = {k.replace("ppg_enc.", ""): v for k, v in state.items() if k.startswith("ppg_enc.")}
        ppg_encoder.load_state_dict(ppg_state, strict=False)
    
    ecg_encoder = ecg_encoder.to(device)
    ppg_encoder = ppg_encoder.to(device)
    
    # 根据模态创建模型
    if args.modality == "both":
        head = BPRegressionHead(ecg_dim=1024, ppg_dim=512, hidden_dim=args.hidden_dim)
        model = BPModel(ecg_encoder, ppg_encoder, head)
    elif args.modality == "ecg":
        head = SingleModalBPHead(feat_dim=1024, hidden_dim=args.hidden_dim // 2)
        model = SingleModalBPModel(ecg_encoder, head)
    else:  # ppg
        head = SingleModalBPHead(feat_dim=512, hidden_dim=args.hidden_dim // 2)
        model = SingleModalBPModel(ppg_encoder, head)
    
    model = model.to(device)
    
    # 冻结backbone
    if args.freeze_backbone:
        print("Freezing backbone...")
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
    
    # 损失函数
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:  # mae_pearson
        criterion = MAEPearsonLoss()
    
    # 优化器
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # 训练循环
    best_val_loss = float("inf")
    best_metrics = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics["loss"])
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['MAE']:.2f} | r: {val_metrics['r']:.3f}")
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = deepcopy(val_metrics)
            patience_counter = 0
            
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 测试集评估
    print("\n" + "="*50)
    print("Evaluating on TEST set...")
    model.load_state_dict(torch.load(out_dir / "best_model.pth"))
    test_metrics, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    
    print(f"TEST (Raw):")
    for k, v in test_metrics.items():
        if k != "loss":
            print(f"  {k}: {v:.4f}")
    
    # 线性校准
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(y_pred, y_true)
    y_pred_cal = slope * y_pred + intercept
    
    mae_cal = float(np.mean(np.abs(y_true - y_pred_cal)))
    rmse_cal = float(np.sqrt(np.mean((y_true - y_pred_cal) ** 2)))
    
    y0 = y_true - y_true.mean()
    h0 = y_pred_cal - y_pred_cal.mean()
    r_cal = float(np.clip((y0 * h0).sum() / (np.sqrt((y0**2).sum()) * np.sqrt((h0**2).sum()) + 1e-12), -1, 1))
    r2_cal = float(1.0 - np.sum((y_true - y_pred_cal)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-12))
    
    print(f"\nTEST (Calibrated):")
    print(f"  MAE: {mae_cal:.2f}")
    print(f"  RMSE: {rmse_cal:.2f}")
    print(f"  r: {r_cal:.3f}")
    print(f"  R2: {r2_cal:.3f}")
    
    # 保存结果
    results = {
        "test_raw": test_metrics,
        "test_calibrated": {
            "MAE": mae_cal,
            "RMSE": rmse_cal,
            "r": r_cal,
            "R2": r2_cal,
        },
        "calibration": {"slope": slope, "intercept": intercept},
        "best_val": best_metrics,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
