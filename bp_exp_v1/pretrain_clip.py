#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG-PPG CLIP预训练 - 适配新数据格式

数据格式:
- NPZ文件包含 ecg, ppg 两个数组
- PPG: 50Hz (从600Hz降采样)
- ECG: 500Hz 或 600Hz
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from backbones import ECGEncoderCLIP, PPGEncoderCLIP


class CLIPDataset(Dataset):
    """
    新数据格式: NPZ包含独立的 ecg 和 ppg 数组
    """
    def __init__(self, data_dir: str, split: str = "train"):
        data_dir = Path(data_dir)
        
        # 读取labels.csv获取split信息
        import pandas as pd
        labels_df = pd.read_csv(data_dir / "labels.csv")
        self.df = labels_df[labels_df['split'] == split].reset_index(drop=True)
        self.npz_dir = data_dir / "npz"
        
        if len(self.df) == 0:
            raise RuntimeError(f"No {split} data found")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        ssoid = str(row["ssoid"])
        
        with np.load(self.npz_dir / f"{ssoid}.npz") as d:
            ecg = d["ecg"].astype(np.float32)  # (seq_len,)
            ppg = d["ppg"].astype(np.float32)  # (seq_len,)
        
        return (
            torch.from_numpy(ecg).unsqueeze(0),  # (1, T)
            torch.from_numpy(ppg).unsqueeze(0),  # (1, T)
        )


class ECGPPG_CLIP(nn.Module):
    def __init__(self, proj_hidden: int = 0):
        super().__init__()
        self.ecg_enc = ECGEncoderCLIP(with_proj=True, proj_hidden=proj_hidden)
        self.ppg_enc = PPGEncoderCLIP(with_proj=True, proj_hidden=proj_hidden)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor):
        z_ecg = self.ecg_enc(ecg)
        z_ppg = self.ppg_enc(ppg)
        return z_ecg, z_ppg


def load_founders_into_clip(model, ecg_ckpt, ppg_ckpt, device):
    """加载预训练的Founder权重"""
    own = model.state_dict()
    load_dict = {}

    def _load(path, prefix):
        if not path or not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location="cpu")
        
        # 处理不同的checkpoint格式
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                st = ckpt["model"]
            elif "state_dict" in ckpt:
                st = ckpt["state_dict"]
            elif "ecg_encoder" in ckpt:
                # CLIP预训练格式
                if prefix == "ecg_enc.":
                    st = ckpt["ecg_encoder"]
                else:
                    st = ckpt["ppg_encoder"]
            else:
                st = ckpt
        else:
            st = ckpt
        
        for k, v in st.items():
            k2 = k.replace("module.", "")
            if k2.startswith(prefix) and k2 in own and own[k2].shape == v.shape:
                load_dict[k2] = v

    _load(ecg_ckpt, "ecg_enc.")
    _load(ppg_ckpt, "ppg_enc.")

    model.load_state_dict({**own, **load_dict}, strict=False)
    model.to(device)


def train_one_epoch(model, loader, optimizer, scaler, device, temperature):
    model.train()
    total, loss_sum = 0, 0.0

    for ecg, ppg in tqdm(loader, desc="Pretrain", leave=False):
        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            z_ecg, z_ppg = model(ecg, ppg)
            z_ecg = F.normalize(z_ecg, dim=1)
            z_ppg = F.normalize(z_ppg, dim=1)

            logits = (z_ecg @ z_ppg.t()) / temperature
            labels = torch.arange(logits.size(0), device=device)
            loss = 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.t(), labels)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = ecg.size(0)
        total += bs
        loss_sum += loss.item() * bs

    return loss_sum / total


@torch.no_grad()
def evaluate(model, loader, device, temperature):
    model.eval()
    total, loss_sum = 0, 0.0

    for ecg, ppg in tqdm(loader, desc="Eval", leave=False):
        ecg = ecg.to(device)
        ppg = ppg.to(device)

        z_ecg, z_ppg = model(ecg, ppg)
        z_ecg = F.normalize(z_ecg, dim=1)
        z_ppg = F.normalize(z_ppg, dim=1)

        logits = (z_ecg @ z_ppg.t()) / temperature
        labels = torch.arange(logits.size(0), device=device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)
        )

        bs = ecg.size(0)
        total += bs
        loss_sum += loss.item() * bs

    return loss_sum / total


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--data_dir", required=True, help="包含data_500hz或data_600hz的目录")
    ap.add_argument("--out_dir", required=True)
    # Pretrained weights (可选)
    ap.add_argument("--ecg_ckpt", default="")
    ap.add_argument("--ppg_ckpt", default="")
    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # Freeze options
    ap.add_argument("--freeze_ecg", action="store_true", help="冻结ECG backbone")
    ap.add_argument("--freeze_ppg", action="store_true", help="冻结PPG backbone")
    
    args = ap.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Data dir: {args.data_dir}")
    
    # 数据加载器
    train_ds = CLIPDataset(args.data_dir, split="train")
    val_ds = CLIPDataset(args.data_dir, split="val")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 模型
    model = ECGPPG_CLIP().to(device)
    
    # 加载预训练权重
    load_founders_into_clip(model, args.ecg_ckpt, args.ppg_ckpt, device)

    # 冻结策略
    if args.freeze_ecg:
        print("Freezing ECG backbone...")
        for p in model.ecg_enc.backbone.parameters():
            p.requires_grad = False
    
    if args.freeze_ppg:
        print("Freezing PPG backbone...")
        for p in model.ppg_enc.backbone.parameters():
            p.requires_grad = False

    # 优化器 - 只优化非冻结参数
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0
    
    for ep in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args.temperature
        )
        val_loss = evaluate(model, val_loader, device, args.temperature)
        scheduler.step()
        
        print(f"[E{ep:03d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # 保存最新
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "ecg_encoder": model.ecg_enc.state_dict(),
            "ppg_encoder": model.ppg_enc.state_dict(),
        }, out_dir / "clip_last.pth")

        # 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "ecg_encoder": model.ecg_enc.state_dict(),
                "ppg_encoder": model.ppg_enc.state_dict(),
                "val_loss": val_loss,
            }, out_dir / "clip_best.pth")
            print(f"  -> Saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {ep}")
                break

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {out_dir / 'clip_best.pth'}")


if __name__ == "__main__":
    main()
