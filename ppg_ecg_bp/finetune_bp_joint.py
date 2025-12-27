#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint ECG–PPG BP finetuning using a single pretrained checkpoint

This script mirrors the Age finetune style:
- one joint checkpoint
- jointly aligned ECG + PPG encoders
- multi-target BP regression head
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from backbones2 import AgeModel
from dataset import BPDataset
from losses import MAE_PearsonLoss
from utils import set_seed, subject_id_from_ssoid
from engine_bp import train_one_epoch, evaluate, evaluate_with_ids


DEFAULT_NPZ_DIR   = "/home/youliang/youliang_data2/bp/bp_npz_run1/npz"
DEFAULT_LABELS    = "/home/youliang/youliang_data2/bp/bp_npz_run1/labels.csv"
DEFAULT_PRETRAIN  = "/home/youliang/youliang_data2/bp/ppg_ecg_age/best_checkpoint.pth"
DEFAULT_OUT_DIR   = "/home/youliang/youliang_data2/bp/bp_joint_run1"

DEFAULT_TARGET_COLS = [
    "right_arm_dbp", "left_arm_mbp",
    "right_arm_pp", "right_arm_sbp", "left_arm_sbp"
]


def main():
    ap = argparse.ArgumentParser("Joint BP finetune (ECG + PPG)")
    ap.add_argument("--npz_dir", default=DEFAULT_NPZ_DIR)
    ap.add_argument("--labels_csv", default=DEFAULT_LABELS)
    ap.add_argument("--pretrain", default=DEFAULT_PRETRAIN)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--target_cols", nargs="+", default=DEFAULT_TARGET_COLS)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--maepearson_alpha", type=float, default=0.5)
    ap.add_argument("--maepearson_beta", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=5)

    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- load labels ----
    df = pd.read_csv(args.labels_csv)
    df = df[["ssoid"] + args.target_cols].dropna().copy()
    df["ssoid"] = df["ssoid"].astype(str)

    have = {p.stem for p in Path(args.npz_dir).glob("*.npz")}
    df = df[df["ssoid"].isin(have)].reset_index(drop=True)

    # ---- subject split ----
    df["subject"] = df["ssoid"].apply(subject_id_from_ssoid)
    subjects = df["subject"].unique().tolist()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    s_tr = set(subjects[:int(0.7 * n)])
    s_va = set(subjects[int(0.7 * n):int(0.8 * n)])
    s_te = set(subjects[int(0.8 * n):])

    df_tr = df[df["subject"].isin(s_tr)]
    df_va = df[df["subject"].isin(s_va)]
    df_te = df[df["subject"].isin(s_te)]

    # ---- target normalization ----
    y_tr = df_tr[args.target_cols].to_numpy(np.float32)
    mu = torch.tensor(y_tr.mean(axis=0), device=device)
    sigma = torch.tensor(y_tr.std(axis=0), device=device)
    sigma = torch.where(sigma < 1e-6, torch.ones_like(sigma), sigma)

    # ---- loaders ----
    dl_tr = DataLoader(BPDataset(df_tr, args.npz_dir, args.target_cols),
                       batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(BPDataset(df_va, args.npz_dir, args.target_cols),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(BPDataset(df_te, args.npz_dir, args.target_cols),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # ---- model ----
    model = AgeModel(
        modality="both",
        proj_hidden=0,
        target_dim=len(args.target_cols)
    ).to(device)

    model.load_from_pretrain(args.pretrain, device=device)

    # ---- optimizer ----
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            p.requires_grad = ("head" in n)
        params = [{"params": filter(lambda p: p.requires_grad, model.parameters()),
                   "lr": args.lr_head}]
    else:
        enc, head = [], []
        for n, p in model.named_parameters():
            (head if "head" in n else enc).append(p)
        params = [
            {"params": enc, "lr": args.lr_backbone},
            {"params": head, "lr": args.lr_head},
        ]

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    criterion = MAE_PearsonLoss(
        alpha=args.maepearson_alpha,
        beta=args.maepearson_beta
    ).to(device)

    # ---- training ----
    best_mae, wait = float("inf"), 0
    for ep in range(1, args.epochs + 1):
        train_one_epoch(
            model, dl_tr, optimizer, scaler, device,
            modality="both", reg_loss="mae_pearson",
            mu=mu, sigma=sigma,
            maepearson_criterion=criterion
        )

        _, _, val_mae = evaluate(
            model, dl_va, device, "both", mu, sigma, args.target_cols
        )

        print(f"[E{ep}] val_avg_MAE = {val_mae:.2f}")

        if val_mae < best_mae:
            best_mae = val_mae
            wait = 0
            torch.save(model.state_dict(), out_dir / "best.pth")
        else:
            wait += 1
            if wait >= args.patience:
                break

    # ---- test ----
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device))
    evaluate_with_ids(
        model, dl_te, device, "both", mu, sigma, args.target_cols
    )


if __name__ == "__main__":
    main()
