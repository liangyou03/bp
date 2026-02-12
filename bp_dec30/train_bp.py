#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BP Multi-target Regression Training Script
用 ECG + PPG 信号预测多个血压指标

Usage:
python train_bp.py \
    --npz_dir /path/to/npz \
    --labels_csv /path/to/labels.csv \
    --ecg_ckpt /path/to/1_lead_ECGFounder.pth \
    --ppg_ckpt /path/to/best_checkpoint.pth \
    --out_dir ./bp_run1 \
    --target_cols right_arm_dbp left_arm_mbp right_arm_pp right_arm_sbp left_arm_sbp

"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==================== 工具函数 ====================
def set_seed(seed: int = 666):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def subject_id_from_ssoid(ssoid: str) -> str:
    return str(ssoid).split("_", 1)[0]

def mae_np(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def rmse_np(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def pearson_r_np(y, yhat, eps=1e-12):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    y0 = y - y.mean()
    h0 = yhat - yhat.mean()
    denom = np.sqrt((y0**2).sum()) * np.sqrt((h0**2).sum()) + eps
    return float(np.clip((y0 * h0).sum() / denom, -1, 1))

def r2_np(y, yhat, eps=1e-12):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + eps
    return float(1.0 - ss_res / ss_tot)


# ==================== Dataset ====================
class BPDataset(Dataset):
    """返回 (ecg, ppg, targets, ssoid)"""
    def __init__(self, df: pd.DataFrame, npz_dir: Path, target_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.npz_dir = Path(npz_dir)
        self.target_cols = target_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ssoid = str(row["ssoid"])
        targets = torch.tensor([float(row[c]) for c in self.target_cols], dtype=torch.float32)
        
        with np.load(self.npz_dir / f"{ssoid}.npz") as d:
            x = d["x"].astype(np.float32)  # (7500, 2)
        
        ecg = torch.from_numpy(x[:, 0:1].T.copy())  # (1, 7500)
        ppg = torch.from_numpy(x[:, 1:2].T.copy())  # (1, 7500)
        return ecg, ppg, targets, ssoid


# ==================== 模型组件 ====================
class MyConv1dPadSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, groups=groups)

    def forward(self, x):
        L = x.shape[-1]
        out_len = (L + self.stride - 1) // self.stride
        p = max(0, (out_len - 1) * self.stride + self.kernel_size - L)
        left, right = p // 2, p - p // 2
        return self.conv(F.pad(x, (left, right)))


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x):
        p = max(0, self.kernel_size - 1)
        left, right = p // 2, p - p // 2
        return self.pool(F.pad(x, (left, right)))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ==================== PPG Backbone ====================
class BasicBlockPPG(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, downsample):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = Swish()
        self.do1 = nn.Dropout(0.5)
        self.conv1 = MyConv1dPadSame(in_ch, out_ch, 1, 1)
        
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = Swish()
        self.do2 = nn.Dropout(0.5)
        self.conv2 = MyConv1dPadSame(out_ch, out_ch, kernel_size, stride if downsample else 1, groups=out_ch // 16)
        
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.act3 = Swish()
        self.do3 = nn.Dropout(0.5)
        self.conv3 = MyConv1dPadSame(out_ch, out_ch, 1, 1)
        
        self.se_fc1 = nn.Linear(out_ch, out_ch // 2)
        self.se_fc2 = nn.Linear(out_ch // 2, out_ch)
        
        self.downsample = downsample
        if downsample:
            self.pool = MyMaxPool1dPadSame(stride)
        self.in_ch, self.out_ch = in_ch, out_ch

    def forward(self, x):
        idt = x
        out = self.do1(self.act1(self.bn1(x)))
        out = self.conv1(out)
        out = self.do2(self.act2(self.bn2(out)))
        out = self.conv2(out)
        out = self.do3(self.act3(self.bn3(out)))
        out = self.conv3(out)
        
        se = torch.sigmoid(self.se_fc2(Swish()(self.se_fc1(out.mean(-1)))))
        out = out * se.unsqueeze(-1)
        
        if self.downsample:
            idt = self.pool(idt)
        if self.out_ch != self.in_ch:
            idt = F.pad(idt.transpose(-1, -2), ((self.out_ch - self.in_ch) // 2,) * 2).transpose(-1, -2)
        return out + idt


class PPGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = MyConv1dPadSame(1, 32, 3, 1)
        self.first_bn = nn.BatchNorm1d(32)
        self.first_act = Swish()
        
        # stage_list: 5 stages
        filters = [32, 64, 128, 256, 512]
        n_blocks = [3, 4, 4, 4, 2]
        self.stage_list = nn.ModuleList()
        in_ch = 32
        for out_ch, nb in zip(filters, n_blocks):
            blocks = nn.ModuleList()
            for i in range(nb):
                blocks.append(BasicBlockPPG(in_ch if i == 0 else out_ch, out_ch, 3, 2, i == 0))
            self.stage_list.append(blocks)
            in_ch = out_ch

    def forward(self, x):
        x = self.first_act(self.first_bn(self.first_conv(x)))
        for stage in self.stage_list:
            for block in stage:
                x = block(x)
        return x.mean(dim=-1)  # (B, 512)


# ==================== ECG Backbone ====================
class BasicBlockECG(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, downsample):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = Swish()
        self.do1 = nn.Dropout(0.5)
        self.conv1 = MyConv1dPadSame(in_ch, out_ch, 1, 1)
        
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = Swish()
        self.do2 = nn.Dropout(0.5)
        self.conv2 = MyConv1dPadSame(out_ch, out_ch, kernel_size, stride if downsample else 1, groups=out_ch // 16)
        
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.act3 = Swish()
        self.do3 = nn.Dropout(0.5)
        self.conv3 = MyConv1dPadSame(out_ch, out_ch, 1, 1)
        
        self.se1 = nn.Linear(out_ch, out_ch // 2)
        self.se2 = nn.Linear(out_ch // 2, out_ch)
        
        self.downsample = downsample
        if downsample:
            self.pool = MyMaxPool1dPadSame(stride)
        self.in_ch, self.out_ch = in_ch, out_ch

    def forward(self, x):
        idt = x
        out = self.do1(self.act1(self.bn1(x)))
        out = self.conv1(out)
        out = self.do2(self.act2(self.bn2(out)))
        out = self.conv2(out)
        out = self.do3(self.act3(self.bn3(out)))
        out = self.conv3(out)
        
        se = torch.sigmoid(self.se2(self.act3(self.se1(out.mean(-1)))))
        out = out * se.unsqueeze(-1)
        
        if self.downsample:
            idt = self.pool(idt)
        if self.out_ch != self.in_ch:
            idt = F.pad(idt.transpose(-1, -2), ((self.out_ch - self.in_ch) // 2,) * 2).transpose(-1, -2)
        return out + idt


class ECGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = MyConv1dPadSame(1, 64, 16, 2)
        self.first_bn = nn.BatchNorm1d(64)
        self.first_act = Swish()
        
        # stage_list: 7 stages
        filters = [64, 160, 160, 400, 400, 1024, 1024]
        n_blocks = [2, 2, 2, 3, 3, 4, 4]
        self.stage_list = nn.ModuleList()
        in_ch = 64
        for out_ch, nb in zip(filters, n_blocks):
            blocks = nn.ModuleList()
            for i in range(nb):
                blocks.append(BasicBlockECG(in_ch if i == 0 else out_ch, out_ch, 16, 2, i == 0))
            self.stage_list.append(blocks)
            in_ch = out_ch

    def forward(self, x):
        x = self.first_act(self.first_bn(self.first_conv(x)))
        for stage in self.stage_list:
            for block in stage:
                x = block(x)
        return x.mean(dim=-1)  # (B, 1024)


# ==================== BP Model ====================
class BPModel(nn.Module):
    def __init__(self, target_dim: int, modality: str = "both"):
        super().__init__()
        self.modality = modality
        self.ppg_backbone = PPGBackbone()
        self.ecg_backbone = ECGBackbone()
        self.ecg_proj = nn.Linear(1024, 512)
        
        in_dim = 512 if modality != "both" else 1024
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, target_dim)
        )

    def load_pretrain(self, ecg_ckpt: str, ppg_ckpt: str, device):
        """加载预训练权重，处理 key 映射"""
        # ECG
        ecg_state = torch.load(ecg_ckpt, map_location="cpu")["state_dict"]
        ecg_mapped = {}
        for k, v in ecg_state.items():
            if k.startswith("dense."):
                continue
            new_k = k.replace(".block_list.", ".")
            new_k = new_k.replace("se_fc1", "se1").replace("se_fc2", "se2")  # 新增
            ecg_mapped[new_k] = v
        
        missing, unexpected = self.ecg_backbone.load_state_dict(ecg_mapped, strict=False)
        print(f"[ECG] loaded {ecg_ckpt}")
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        
        # PPG
        ppg_state = torch.load(ppg_ckpt, map_location="cpu")["model_state_dict"]
        ppg_mapped = {}
        for k, v in ppg_state.items():
            if k.startswith("ppg_encoder."):
                new_k = k.replace("ppg_encoder.", "")
                new_k = new_k.replace(".block_list.", ".")
                new_k = new_k.replace("se_fc1", "se_fc1").replace("se_fc2", "se_fc2")  # PPG 用 se_fc
                ppg_mapped[new_k] = v
        
        missing, unexpected = self.ppg_backbone.load_state_dict(ppg_mapped, strict=False)
        print(f"[PPG] loaded {ppg_ckpt}")
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        
        self.to(device)

    def forward(self, ecg, ppg):
        feats = []
        if self.modality in ["ecg", "both"]:
            z_ecg = self.ecg_proj(self.ecg_backbone(ecg))  # (B, 512)
            feats.append(z_ecg)
        if self.modality in ["ppg", "both"]:
            z_ppg = self.ppg_backbone(ppg)  # (B, 512)
            feats.append(z_ppg)
        
        z = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        return self.head(z)  # (B, target_dim)


# ==================== 训练与评估 ====================
def train_one_epoch(model, loader, optimizer, scaler, device, mu, sigma):
    model.train()
    total_loss, total_n = 0.0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    
    for batch_idx, (ecg, ppg, y, _) in enumerate(pbar):
        ecg = ecg.to(device)
        ppg = ppg.to(device)
        y = y.to(device)
        
        # 检查输入
        if torch.isnan(ecg).any() or torch.isnan(ppg).any() or torch.isnan(y).any():
            print(f"[batch {batch_idx}] NaN in input!")
            continue
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(ecg, ppg)
            
            # 检查输出
            if torch.isnan(pred).any():
                print(f"[batch {batch_idx}] NaN in pred!")
                print(f"  ecg: min={ecg.min():.2f}, max={ecg.max():.2f}")
                print(f"  ppg: min={ppg.min():.2f}, max={ppg.max():.2f}")
                # 分别测试 ECG 和 PPG
                with torch.no_grad():
                    z_ecg = model.ecg_proj(model.ecg_backbone(ecg))
                    z_ppg = model.ppg_backbone(ppg)
                    print(f"  z_ecg nan: {torch.isnan(z_ecg).any()}, z_ppg nan: {torch.isnan(z_ppg).any()}")
                continue
            
            loss = F.smooth_l1_loss((pred - mu) / sigma, (y - mu) / sigma)
        
        if torch.isnan(loss):
            print(f"[batch {batch_idx}] NaN loss!")
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        pbar.set_postfix(loss=f"{total_loss/total_n:.4f}")
    
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device, mu, sigma, target_cols):
    model.eval()
    all_y, all_pred, all_ids = [], [], []
    total_loss, total_n = 0.0, 0
    
    for ecg, ppg, y, ssoids in loader:
        ecg = ecg.to(device)
        ppg = ppg.to(device)
        y = y.to(device)
        
        with torch.cuda.amp.autocast():
            pred = model(ecg, ppg)
            loss = F.smooth_l1_loss((pred - mu) / sigma, (y - mu) / sigma)
        
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        
        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_ids.extend(ssoids)
    
    y_np = np.concatenate(all_y, axis=0)
    pred_np = np.concatenate(all_pred, axis=0)
    
    # 计算每个目标的指标
    metrics = {}
    for i, col in enumerate(target_cols):
        metrics[col] = {
            "mae": mae_np(y_np[:, i], pred_np[:, i]),
            "rmse": rmse_np(y_np[:, i], pred_np[:, i]),
            "r": pearson_r_np(y_np[:, i], pred_np[:, i]),
            "r2": r2_np(y_np[:, i], pred_np[:, i]),
        }
    
    avg_mae = np.mean([m["mae"] for m in metrics.values()])
    return total_loss / total_n, metrics, avg_mae, y_np, pred_np, all_ids


# ==================== 主函数 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--ecg_ckpt", required=True, help="Path to 1_lead_ECGFounder.pth")
    ap.add_argument("--ppg_ckpt", required=True, help="Path to best_checkpoint.pth")
    ap.add_argument("--out_dir", default="./bp_run")
    ap.add_argument("--target_cols", nargs="+", default=[
        "right_arm_dbp", "left_arm_mbp", "right_arm_pp", "right_arm_sbp", "left_arm_sbp"
    ])
    ap.add_argument("--modality", choices=["ecg", "ppg", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 加载数据
    df = pd.read_csv(args.labels_csv)
    df["ssoid"] = df["ssoid"].astype(str)
    df = df[["ssoid"] + args.target_cols].dropna().reset_index(drop=True)
    
    npz_dir = Path(args.npz_dir)
    have = {p.stem for p in npz_dir.glob("*.npz")}
    df = df[df["ssoid"].isin(have)].reset_index(drop=True)
    print(f"[data] {len(df)} samples")

    # Subject-level split
    df["subject"] = df["ssoid"].apply(subject_id_from_ssoid)
    subjects = df["subject"].unique().tolist()
    np.random.shuffle(subjects)
    n = len(subjects)
    n_tr, n_va = int(0.7 * n), int(0.1 * n)
    
    s_tr = set(subjects[:n_tr])
    s_va = set(subjects[n_tr:n_tr + n_va])
    s_te = set(subjects[n_tr + n_va:])
    
    df_tr = df[df["subject"].isin(s_tr)].copy()
    df_va = df[df["subject"].isin(s_va)].copy()
    df_te = df[df["subject"].isin(s_te)].copy()
    print(f"[split] train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}")

    # 目标统计
    tr_targets = df_tr[args.target_cols].to_numpy(dtype=np.float32)
    mu = torch.tensor(tr_targets.mean(axis=0), device=device)
    sigma = torch.tensor(np.maximum(tr_targets.std(axis=0), 1e-6), device=device)
    print(f"[stats] mu={mu.cpu().numpy()}, sigma={sigma.cpu().numpy()}")

    # DataLoader
    ds_tr = BPDataset(df_tr, npz_dir, args.target_cols)
    ds_va = BPDataset(df_va, npz_dir, args.target_cols)
    ds_te = BPDataset(df_te, npz_dir, args.target_cols)
    
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 模型
    model = BPModel(target_dim=len(args.target_cols), modality=args.modality).to(device)
    model.load_pretrain(args.ecg_ckpt, args.ppg_ckpt, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # 训练
    best_mae, best_epoch, patience_cnt = float("inf"), -1, 0
    ckpt_path = out_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, dl_tr, optimizer, scaler, device, mu, sigma)
        val_loss, val_metrics, val_mae, _, _, _ = evaluate(model, dl_va, device, mu, sigma, args.target_cols)
        
        metric_str = " ".join([f"{k}:{v['mae']:.2f}" for k, v in val_metrics.items()])
        print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} avg_MAE={val_mae:.2f} | {metric_str}")

        if val_mae < best_mae - 1e-4:
            best_mae, best_epoch, patience_cnt = val_mae, epoch, 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "mu": mu.cpu().tolist(),
                "sigma": sigma.cpu().tolist(),
                "target_cols": args.target_cols,
            }, ckpt_path)
            print(f"  -> saved best (MAE={best_mae:.2f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"[early stop] best @ epoch {best_epoch}")
                break

    # 最终测试
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        _, test_metrics, test_mae, y_te, pred_te, ids_te = evaluate(model, dl_te, device, mu, sigma, args.target_cols)
        
        print("\n[TEST] Final metrics:")
        for col, m in test_metrics.items():
            print(f"  {col}: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, r={m['r']:.3f}, R2={m['r2']:.3f}")
        print(f"  Average MAE: {test_mae:.2f}")

        # 保存预测结果
        df_pred = pd.DataFrame({"ssoid": ids_te})
        for i, col in enumerate(args.target_cols):
            df_pred[f"{col}_true"] = y_te[:, i]
            df_pred[f"{col}_pred"] = pred_te[:, i]
        df_pred.to_csv(out_dir / "test_predictions.csv", index=False)
        
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump({"avg_mae": test_mae, "metrics": test_metrics}, f, indent=2)
        
        print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()