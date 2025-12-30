#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECG–PPG CLIP-style pretraining (BP version, fixed length = 3630)

Goal:
- Train a joint ECG+PPG foundation model by contrastive learning (InfoNCE/CLIP).
- Input: aligned ECG/PPG pairs from NPZ files (each contains key "x" with shape (3630, 2)).
- Output: a single checkpoint containing BOTH ecg_enc.* and ppg_enc.* weights:
    {"model": model.state_dict(), ...}

Usage example:
python pretrain_clip_temp.py \
  --npz_dir /home/youliang/youliang_data2/bp/bp_npz_truncate/npz \
  --ecg_ckpt /home/youliang/youliang_data2/bp/ppg_ecg_age/1_lead_ECGFounder.pth \
  --ppg_ckpt /home/youliang/youliang_data2/bp/ppg_ecg_age/best_checkpoint.pth \
  --out_dir /home/youliang/youliang_data2/bp/ppg_ecg_clip_bp/run1 \
  --epochs 20 \
  --batch_size 128 \
  --gpu 1

Notes:
- This script is designed to run inside your repo folder that contains backbones.py.
- It does NOT require labels.csv.
- It expects all NPZ files to have fixed length 3630. It will assert and fail otherwise.
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from backbones import ECGEncoderCLIP, PPGEncoderCLIP


# ---------------------------
# Dataset
# ---------------------------
class UnlabeledECGPPGNPZ(Dataset):
    """
    Returns (ecg, ppg) where:
      ecg: (1, 3630)
      ppg: (1, 3630)
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
            x = d[self.key_x].astype(np.float32)  # (L,2)
        if x.ndim != 2 or x.shape[1] != 2:
            raise RuntimeError(f"Bad x shape in {fp}: {x.shape}, expected (L,2)")
        if x.shape[0] != self.expected_len:
            raise RuntimeError(f"Bad length in {fp}: {x.shape[0]}, expected {self.expected_len}")

        ecg = torch.from_numpy(x[:, 0:1].T.copy())  # (1,L)
        ppg = torch.from_numpy(x[:, 1:2].T.copy())  # (1,L)
        return ecg, ppg


# ---------------------------
# CLIP model wrapper
# ---------------------------
class ECGPPG_CLIP(nn.Module):
    def __init__(self, proj_hidden: int = 0):
        super().__init__()
        # Keep projection ON (matches your backbones.py definition)
        self.ecg_enc = ECGEncoderCLIP(with_proj=True, proj_hidden=proj_hidden)
        self.ppg_enc = PPGEncoderCLIP(with_proj=True, proj_hidden=proj_hidden)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_ecg = self.ecg_enc(ecg)  # (B,512)
        z_ppg = self.ppg_enc(ppg)  # (B,512)
        return z_ecg, z_ppg


# ---------------------------
# Checkpoint loading helpers
# ---------------------------
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Accepts common checkpoint formats:
    - raw state_dict (dict of param_name -> tensor)
    - {"model": state_dict}
    - {"state_dict": state_dict}
    - {"model_state_dict": state_dict}
    """
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # Could be raw state dict already
        tensor_like = [v for v in ckpt.values() if torch.is_tensor(v)]
        if len(tensor_like) > 0:
            return ckpt
    raise RuntimeError("Unrecognized checkpoint format; cannot extract state_dict.")


def _map_ppg_keys_to_clip(ppg_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map older PPG keys (e.g., ppg_encoder.stage_list.0.block_list.0.se_fc1.weight)
    to current backbones.py keys under ppg_enc.backbone.*.

    This is intentionally conservative:
    - Apply a small set of string replacements
    - The final load step will take intersection with actual model keys
    """
    mapped = {}
    for k, v in ppg_state.items():
        nk = k

        # Strip common wrappers
        nk = nk.replace("module.", "")

        # Old naming -> new naming
        # Old: ppg_encoder.*  ->  ppg_enc.backbone.*
        if nk.startswith("ppg_encoder."):
            nk = nk.replace("ppg_encoder.", "ppg_enc.backbone.", 1)

        # Some older code uses first_conv instead of first
        nk = nk.replace("first_conv.", "first.")
        nk = nk.replace("stage_list.", "stages.")
        nk = nk.replace(".block_list.", ".blocks.")

        # SE layers in your backbones.py are named se1/se2
        nk = nk.replace("se_fc1", "se1")
        nk = nk.replace("se_fc2", "se2")

        mapped[nk] = v
    return mapped


def _map_ecg_keys_to_clip(ecg_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map older ECG keys to current backbones.py keys under ecg_enc.backbone.*.

    If your ECGFounder ckpt already matches ecg_enc.backbone.*, it will pass through.
    If it uses ecg_encoder.* naming, we map similarly to PPG.
    """
    mapped = {}
    for k, v in ecg_state.items():
        nk = k
        nk = nk.replace("module.", "")

        # Old: ecg_encoder.* -> ecg_enc.backbone.*
        if nk.startswith("ecg_encoder."):
            nk = nk.replace("ecg_encoder.", "ecg_enc.backbone.", 1)

        # Some ECG codebases use first_conv naming
        nk = nk.replace("first_conv.", "first.")
        nk = nk.replace("stage_list.", "stages.")
        nk = nk.replace(".block_list.", ".blocks.")
        nk = nk.replace("se_fc1", "se1")
        nk = nk.replace("se_fc2", "se2")

        mapped[nk] = v
    return mapped


def load_founders_into_clip(
    model: ECGPPG_CLIP,
    ecg_ckpt_path: Optional[str],
    ppg_ckpt_path: Optional[str],
    device: torch.device,
) -> None:
    """
    Load ECGFounder and PPGFounder (or PPG-only checkpoint) into the CLIP model encoders.
    This function is robust:
      - extracts state dict from many formats
      - applies conservative key mapping
      - only loads keys that match by name AND shape
    """
    own = model.state_dict()
    load_dict = {}

    def _add_matching(mapped: Dict[str, torch.Tensor], tag: str):
        added = 0
        for k, v in mapped.items():
            if k in own and own[k].shape == v.shape:
                load_dict[k] = v
                added += 1
        print(f"[init] matched params from {tag}: {added}")

    if ecg_ckpt_path:
        ckpt = torch.load(ecg_ckpt_path, map_location="cpu")
        st = _extract_state_dict(ckpt)
        st_m = _map_ecg_keys_to_clip(st)
        _add_matching(st_m, f"ECG ckpt {ecg_ckpt_path}")

    if ppg_ckpt_path:
        ckpt = torch.load(ppg_ckpt_path, map_location="cpu")
        st = _extract_state_dict(ckpt)
        st_m = _map_ppg_keys_to_clip(st)
        _add_matching(st_m, f"PPG ckpt {ppg_ckpt_path}")

    # Merge and load
    merged = {**own, **load_dict}
    msg = model.load_state_dict(merged, strict=False)
    print(f"[init] load_state_dict strict=False")
    print(f"  missing_keys: {len(msg.missing_keys)}")
    print(f"  unexpected_keys: {len(msg.unexpected_keys)}")
    model.to(device)


# ---------------------------
# Metrics
# ---------------------------
@torch.no_grad()
def retrieval_top1(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """
    Given normalized embeddings:
      z_a: (B,D), z_b: (B,D)
    Compute top-1 retrieval accuracy for a->b using cosine similarity.
    """
    logits = z_a @ z_b.t()
    pred = logits.argmax(dim=1)
    gt = torch.arange(z_a.shape[0], device=z_a.device)
    acc = (pred == gt).float().mean().item()
    return float(acc)


# ---------------------------
# Train
# ---------------------------
def train_one_epoch(
    model: ECGPPG_CLIP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    temperature: float,
) -> Tuple[float, float, float]:
    model.train()
    total = 0
    total_loss = 0.0
    total_acc_e2p = 0.0
    total_acc_p2e = 0.0

    pbar = tqdm(loader, desc="Pretrain", leave=False)
    for ecg, ppg in pbar:
        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=True):
            z_ecg, z_ppg = model(ecg, ppg)  # (B,512)
            z_ecg = F.normalize(z_ecg, dim=1)
            z_ppg = F.normalize(z_ppg, dim=1)

            logits = (z_ecg @ z_ppg.t()) / float(temperature)
            labels = torch.arange(logits.shape[0], device=device)

            loss_e2p = F.cross_entropy(logits, labels)
            loss_p2e = F.cross_entropy(logits.t(), labels)
            loss = 0.5 * (loss_e2p + loss_p2e)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = ecg.size(0)
        total += bs
        total_loss += float(loss.item()) * bs

        with torch.no_grad():
            acc_e2p = retrieval_top1(z_ecg, z_ppg)
            acc_p2e = retrieval_top1(z_ppg, z_ecg)
        total_acc_e2p += acc_e2p * bs
        total_acc_p2e += acc_p2e * bs

        pbar.set_postfix(
            loss=f"{total_loss/total:.4f}",
            acc_e2p=f"{total_acc_e2p/total:.3f}",
            acc_p2e=f"{total_acc_p2e/total:.3f}",
        )

    pbar.close()
    return total_loss / total, total_acc_e2p / total, total_acc_p2e / total


@torch.no_grad()
def eval_one_epoch(
    model: ECGPPG_CLIP,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> Tuple[float, float, float]:
    model.eval()
    total = 0
    total_loss = 0.0
    total_acc_e2p = 0.0
    total_acc_p2e = 0.0

    for ecg, ppg in loader:
        ecg = ecg.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            z_ecg, z_ppg = model(ecg, ppg)
            z_ecg = F.normalize(z_ecg, dim=1)
            z_ppg = F.normalize(z_ppg, dim=1)
            logits = (z_ecg @ z_ppg.t()) / float(temperature)
            labels = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        bs = ecg.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        total_acc_e2p += retrieval_top1(z_ecg, z_ppg) * bs
        total_acc_p2e += retrieval_top1(z_ppg, z_ecg) * bs

    return total_loss / total, total_acc_e2p / total, total_acc_p2e / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Directory with *.npz files containing x=(3630,2)")
    ap.add_argument("--out_dir", required=True, help="Output directory for logs/checkpoints")

    ap.add_argument("--ecg_ckpt", default="", help="ECGFounder checkpoint path (optional)")
    ap.add_argument("--ppg_ckpt", default="", help="PPGFounder or PPG-only checkpoint path (optional)")

    ap.add_argument("--expected_len", type=int, default=3630)
    ap.add_argument("--npz_key_x", type=str, default="x")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--proj_hidden", type=int, default=0, help="Optional hidden dim for projection MLP; 0 uses Linear")
    ap.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio based on file list")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(f"cuda:{args.gpu}")
    print(f"device={device} | {torch.cuda.get_device_name(args.gpu)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args_pretrain_clip_bp.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Dataset + split
    ds_all = UnlabeledECGPPGNPZ(args.npz_dir, key_x=args.npz_key_x, expected_len=args.expected_len)
    n = len(ds_all)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * float(args.val_ratio))))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    ds_tr = torch.utils.data.Subset(ds_all, tr_idx.tolist())
    ds_va = torch.utils.data.Subset(ds_all, val_idx.tolist())

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print(f"[data] total={n} train={len(ds_tr)} val={len(ds_va)} expected_len={args.expected_len}")

    # Model
    model = ECGPPG_CLIP(proj_hidden=args.proj_hidden).to(device)

    # Initialize from founders (optional)
    ecg_ckpt = args.ecg_ckpt.strip() or None
    ppg_ckpt = args.ppg_ckpt.strip() or None
    if ecg_ckpt or ppg_ckpt:
        load_founders_into_clip(model, ecg_ckpt, ppg_ckpt, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_val_loss = float("inf")
    best_ep = -1

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_a1, tr_a2 = train_one_epoch(
            model, dl_tr, optimizer, scaler, device, args.temperature
        )
        va_loss, va_a1, va_a2 = eval_one_epoch(
            model, dl_va, device, args.temperature
        )

        print(
            f"[E{ep:03d}] "
            f"train_loss={tr_loss:.4f} acc_e2p={tr_a1:.3f} acc_p2e={tr_a2:.3f} | "
            f"val_loss={va_loss:.4f} acc_e2p={va_a1:.3f} acc_p2e={va_a2:.3f}"
        )

        # Save "foundation" checkpoint (clean format)
        last_path = out_dir / "clip_foundation_last.pth"
        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "temperature": float(args.temperature),
                "expected_len": int(args.expected_len),
            },
            str(last_path),
        )

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_ep = ep
            best_path = out_dir / "clip_foundation_best.pth"
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "temperature": float(args.temperature),
                    "expected_len": int(args.expected_len),
                },
                str(best_path),
            )
            print(f"[best] val_loss={best_val_loss:.4f} @E{best_ep} saved={best_path}")

    print(f"[done] best_epoch={best_ep} best_val_loss={best_val_loss:.4f}")
    print(f"[output] {out_dir}/clip_foundation_best.pth")

    # Quick self-check: ensure both encoders exist in saved state_dict
    ckpt = torch.load(str(out_dir / "clip_foundation_best.pth"), map_location="cpu")
    keys = ckpt["model"].keys()
    has_ecg = any(k.startswith("ecg_enc.") for k in keys)
    has_ppg = any(k.startswith("ppg_enc.") for k in keys)
    print(f"[check] has_ecg_enc={has_ecg} has_ppg_enc={has_ppg}")
    if not (has_ecg and has_ppg):
        print("[warn] Saved checkpoint does not appear to contain both encoders. Check your backbones.py names.")


if __name__ == "__main__":
    main()
