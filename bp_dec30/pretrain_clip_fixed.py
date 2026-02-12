#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECG–PPG CLIP-style pretraining (BP version)

重采样策略：
- ECG: 50 Hz -> 500 Hz (3630 -> 36300)，保持完整 72.6 秒时长
- PPG: 保持 50 Hz 不变 (3630)
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.signal import resample_poly

from backbones import ECGEncoderCLIP, PPGEncoderCLIP


# ---------------------------
# Dataset
# ---------------------------
class UnlabeledECGPPGNPZ(Dataset):
    """
    Returns (ecg, ppg):
      ecg: (1, 36300) @ 500 Hz  -- 上采样保持完整时长
      ppg: (1, 3630)  @ 50 Hz   -- 不变
    """
    def __init__(self, npz_dir: str, key_x: str = "x", expected_len: int = 3630):
        self.npz_dir = Path(npz_dir)
        self.key_x = key_x
        self.expected_len = int(expected_len)
        self.files = sorted(self.npz_dir.glob("*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in: {self.npz_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        with np.load(fp) as d:
            x = d[self.key_x].astype(np.float32)  # (3630, 2) @ 50 Hz

        if x.ndim != 2 or x.shape[1] != 2:
            raise RuntimeError(f"Bad x shape in {fp}: {x.shape}")
        if x.shape[0] != self.expected_len:
            raise RuntimeError(
                f"Bad length in {fp}: {x.shape[0]}, expected {self.expected_len}"
            )

        ecg_raw = x[:, 0]  # 3630 @ 50 Hz
        ppg_raw = x[:, 1]  # 3630 @ 50 Hz

        # ECG: 50 Hz -> 500 Hz (上采样 10x，保持完整 72.6 秒时长)
        ecg_up = resample_poly(ecg_raw, 10, 1)  # 3630 -> 36300

        ecg = torch.from_numpy(ecg_up[None, :].astype(np.float32))
        ppg = torch.from_numpy(ppg_raw[None, :].astype(np.float32))

        return ecg, ppg  # (1, 36300), (1, 3630)


# ---------------------------
# CLIP model
# ---------------------------
class ECGPPG_CLIP(nn.Module):
    def __init__(self, proj_hidden: int = 0):
        super().__init__()
        self.ecg_enc = ECGEncoderCLIP(with_proj=True, proj_hidden=proj_hidden)
        self.ppg_enc = PPGEncoderCLIP(with_proj=True, proj_hidden=proj_hidden)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor):
        z_ecg = self.ecg_enc(ecg)  # (B, 512)
        z_ppg = self.ppg_enc(ppg)  # (B, 512)
        return z_ecg, z_ppg


# ---------------------------
# Checkpoint helpers
# ---------------------------
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise RuntimeError("Unrecognized checkpoint format")


def load_founders_into_clip(model, ecg_ckpt, ppg_ckpt, device):
    own = model.state_dict()
    load_dict = {}

    def _load(path, prefix):
        if not path:
            return
        ckpt = torch.load(path, map_location="cpu")
        st = _extract_state_dict(ckpt)
        for k, v in st.items():
            k2 = k.replace("module.", "")
            if k2.startswith(prefix) and k2 in own and own[k2].shape == v.shape:
                load_dict[k2] = v

    _load(ecg_ckpt, "ecg_enc.")
    _load(ppg_ckpt, "ppg_enc.")

    model.load_state_dict({**own, **load_dict}, strict=False)
    model.to(device)


# ---------------------------
# Training utils
# ---------------------------
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


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ecg_ckpt", default="")
    ap.add_argument("--ppg_ckpt", default="")
    ap.add_argument("--expected_len", type=int, default=3630)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = UnlabeledECGPPGNPZ(args.npz_dir, expected_len=args.expected_len)
    print(f"[data] {len(ds)} samples")
    
    # 测试一个样本，确认形状
    ecg_test, ppg_test = ds[0]
    print(f"[shape] ECG: {ecg_test.shape}, PPG: {ppg_test.shape}")
    assert ecg_test.shape == (1, 36300), f"ECG shape mismatch: {ecg_test.shape}"
    assert ppg_test.shape == (1, 3630), f"PPG shape mismatch: {ppg_test.shape}"
    
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = ECGPPG_CLIP().to(device)
    load_founders_into_clip(
        model,
        args.ecg_ckpt or None,
        args.ppg_ckpt or None,
        device,
    )

    # Freeze ECG backbone
    for p in model.ecg_enc.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float("inf")
    epochs_no_improve = 0
    min_delta = 0.002

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, dl, optimizer, scaler, device, args.temperature
        )
        print(f"[E{ep:03d}] loss={loss:.4f}")

        if best_loss - loss > min_delta:
            best_loss = loss
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "loss": loss,
                    "expected_len": args.expected_len,
                    "ecg_resampled": True,  # 标记 ECG 已重采样
                    "ecg_len": 36300,
                    "ppg_len": 3630,
                },
                out_dir / "clip_foundation_best.pth",
            )
            print(f"  -> saved best (loss={loss:.4f})")
        else:
            epochs_no_improve += 1

        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "loss": loss,
                "expected_len": args.expected_len,
                "ecg_resampled": True,
                "ecg_len": 36300,
                "ppg_len": 3630,
            },
            out_dir / "clip_foundation_last.pth",
        )

        if epochs_no_improve >= args.patience:
            print(
                f"[early stop] no improvement > {min_delta} "
                f"for {args.patience} epochs. stopping at epoch {ep}."
            )
            break

    print("[done]")


if __name__ == "__main__":
    main()