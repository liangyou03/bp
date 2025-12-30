#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Script: BP Finetuning (Multi-target Regression)

Minimal changes from finetune_age.py:
- Use BPDataset
- Multi-target labels via --target_cols
- AgeModel(..., target_dim=K)
- mu/sigma are vectors (K,)
- Evaluate per-target metrics and early stop on average MAE

python finetune_bp.py \
  --npz_dir /home/youliang/youliang_data2/bp/bp_npz_truncate/npz \
  --labels_csv /home/youliang/youliang_data2/bp/bp_npz_truncate/labels.csv \
  --pretrain /home/youliang/youliang_data2/bp/ppg_ecg_clip_bp/run1/clip_foundation_best.pth \
  --out_dir /home/youliang/youliang_data2/bp/ppg_ecg_bp/runs/bp_run1 \
  --modality both \
  --target_cols \
    right_arm_sbp \
  --epochs 40 \
  --batch_size 32 \
  --reg_loss mae_pearson \
  --gpu 1


"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from backbones import AgeModel
from dataset import BPDataset
from utils import set_seed, subject_id_from_ssoid
from engine import train_one_epoch, evaluate, evaluate_with_ids


class MultiTarget_MAE_PearsonLoss(torch.nn.Module):
    """
    Multi-target version of MAE+Pearson:
    average over targets: mean_k [ alpha*(1-r_k) + beta*MAE_k ]
    Input shapes: (B,K)
    """
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y_hat, y: (B,K)
        if y_hat.ndim == 1:
            y_hat = y_hat.view(-1, 1)
        if y.ndim == 1:
            y = y.view(-1, 1)

        K = y.shape[1]
        losses = []
        for k in range(K):
            yh = y_hat[:, k]
            yt = y[:, k]

            vx = yh - yh.mean()
            vy = yt - yt.mean()
            corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + self.eps)
            pearson_loss = 1.0 - corr
            mae_loss = torch.mean(torch.abs(yh - yt))
            losses.append(self.alpha * pearson_loss + self.beta * mae_loss)

        return torch.stack(losses).mean()


# ===================== Default paths (your setup) =====================
DEFAULT_NPZ_DIR   = "/home/notebook/data/personal/S9061270/bp_ready/labeled_zscore"
DEFAULT_LABELS    = "/home/notebook/data/personal/S9061270/bp_ready/labeled_labels.csv"
DEFAULT_PRETRAIN  = "/home/notebook/data/personal/S9061270/model/clip_ppg_ecg_founder/pretrain_temp/best.pth"
DEFAULT_OUT_DIR   = "/home/notebook/data/personal/S9061270/model/clip_ppg_ecg_founder/finetune_bp_run1"


def main():
    ap = argparse.ArgumentParser(description="BP finetune (multi-target) with subject-wise split.")
    ap.add_argument("--npz_dir",    default=DEFAULT_NPZ_DIR)
    ap.add_argument("--labels_csv", default=DEFAULT_LABELS)
    ap.add_argument("--pretrain",   default=DEFAULT_PRETRAIN)
    ap.add_argument("--out_dir",    default=DEFAULT_OUT_DIR)
    ap.add_argument("--modality",   choices=["ecg", "ppg", "both"], default="both")

    # Multi-target columns
    ap.add_argument("--target_cols", nargs="+", required=True,
                    help="BP target columns in labels_csv, e.g. sbp dbp mbp pp ...")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)

    # Optim & freeze
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze encoders+projectors (linear probe).")
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head",     type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # Loss
    ap.add_argument("--reg_loss", choices=["mse", "huber", "mae_pearson"], default="huber")
    ap.add_argument("--alpha_corr", type=float, default=0.0, help="Extra (1-corr) term, ignored if reg_loss=mae_pearson")

    ap.add_argument("--maepearson_alpha", type=float, default=0.5)
    ap.add_argument("--maepearson_beta",  type=float, default=0.5)

    # Output constraint (use scalar bounds for all targets to keep minimal)
    ap.add_argument("--constrain", choices=["none", "tanh", "sigmoid", "clip"], default="none")
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=300.0)

    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    # device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(f"cuda:{args.gpu}")
    print(f"device = {device} | name = {torch.cuda.get_device_name(args.gpu)}")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args_bp.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ----- load labels & align npz -----
    df = pd.read_csv(args.labels_csv)
    need_cols = ["ssoid"] + list(args.target_cols)
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"labels_csv missing column: {c}")

    df = df[need_cols].copy()
    df["ssoid"] = df["ssoid"].astype(str)

    # numeric conversion + drop NA
    for c in args.target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=args.target_cols).reset_index(drop=True)

    have = set(p.stem for p in Path(args.npz_dir).glob("*.npz"))
    df = df[df["ssoid"].isin(have)].reset_index(drop=True)

    # subject-wise split (7:1:2)
    df["subject"] = df["ssoid"].apply(subject_id_from_ssoid)
    subjects = df["subject"].unique().tolist()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_tr = int(0.7 * n)
    n_va = int(0.1 * n)
    s_tr = set(subjects[:n_tr])
    s_va = set(subjects[n_tr:n_tr + n_va])
    s_te = set(subjects[n_tr + n_va:])

    df_tr = df[df["subject"].isin(s_tr)][["ssoid"] + list(args.target_cols)].copy()
    df_va = df[df["subject"].isin(s_va)][["ssoid"] + list(args.target_cols)].copy()
    df_te = df[df["subject"].isin(s_te)][["ssoid"] + list(args.target_cols)].copy()

    print(f"[split] train={len(df_tr)}  val={len(df_va)}  test={len(df_te)}  (subjects: {len(s_tr)}/{len(s_va)}/{len(s_te)})")
    print(f"[targets] K={len(args.target_cols)} -> {args.target_cols}")

    # target stats from train: mu/sigma are vectors (K,)
    tr_targets = df_tr[args.target_cols].to_numpy(dtype=np.float32)
    mu = tr_targets.mean(axis=0).astype(np.float32)
    sigma = tr_targets.std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, 1e-6).astype(np.float32)
    y_min = float(args.y_min)
    y_max = float(args.y_max)
    print(f"[target stats] mu={mu}  sigma={sigma}  y_min={y_min}  y_max={y_max}")

    # datasets / loaders
    ds_tr = BPDataset(df_tr, args.npz_dir, target_cols=args.target_cols)
    ds_va = BPDataset(df_va, args.npz_dir, target_cols=args.target_cols)
    ds_te = BPDataset(df_te, args.npz_dir, target_cols=args.target_cols)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # model (reuse AgeModel, only change head dim)
    model = AgeModel(modality=args.modality, proj_hidden=0, target_dim=len(args.target_cols)).to(device)
    model.load_from_pretrain(args.pretrain, device=device)

    # freeze / params groups
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            p.requires_grad = ("head" in n)
        params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_head}]
    else:
        enc_params, head_params = [], []
        for n, p in model.named_parameters():
            if "head" in n:
                head_params.append(p)
            else:
                enc_params.append(p)
        params = [
            {"params": enc_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ]

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    maepearson_criterion = None
    if args.reg_loss == "mae_pearson":
        maepearson_criterion = MultiTarget_MAE_PearsonLoss(
            alpha=args.maepearson_alpha,
            beta=args.maepearson_beta
        ).to(device)

    # training loop with early stopping on average MAE
    best_avg_mae = float("inf")
    best_ep = -1
    patience_cnt = 0
    ckpt_name = f"bp_{args.modality}_{args.reg_loss}_{args.constrain}_best.pth"

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, dl_tr, optimizer, scaler, device, args.modality,
            args.reg_loss, mu, sigma, y_min, y_max, args.constrain, args.alpha_corr,
            maepearson_criterion=maepearson_criterion,
            dist_criterion=None,
            lambda_dist=0.0
        )

        # engine.evaluate should return (loss, metrics, y, yhat) for multi-target
        val_loss, val_metrics, _, _ = evaluate(
            model, dl_va, device, args.modality, mu, sigma, y_min, y_max, args.constrain,
            target_names=list(args.target_cols)
        )

        avg_mae = float(np.mean([val_metrics[k]["mae"] for k in args.target_cols]))
        metric_str = " ".join([f"{k}:{val_metrics[k]['mae']:.2f}" for k in args.target_cols])
        print(f"[E{ep}] train_loss={tr_loss:.6f} | val_loss={val_loss:.6f} avg_MAE={avg_mae:.3f} | {metric_str}")

        if avg_mae < best_avg_mae - 1e-6:
            best_avg_mae = avg_mae
            best_ep = ep
            patience_cnt = 0
            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "val_avg_mae": float(avg_mae),
                "modality": args.modality,
                "target_cols": list(args.target_cols),
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "y_min": y_min,
                "y_max": y_max,
                "constrain": args.constrain,
            }
            ckpt_path = out_dir / ckpt_name
            torch.save(ckpt, str(ckpt_path))
            print(f"[best] avg_MAE={best_avg_mae:.3f} @epoch{ep} | saved: {ckpt_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs (best @ {best_ep})")
                break

    # ======= Load best and test eval =======
    ckpt_path = out_dir / ckpt_name
    if not ckpt_path.exists():
        raise RuntimeError("No best checkpoint was saved. Check training logs.")

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state["model"])
    mu = np.array(state["mu"], dtype=np.float32)
    sigma = np.array(state["sigma"], dtype=np.float32)
    y_min = float(state["y_min"])
    y_max = float(state["y_max"])
    constrain = state["constrain"]
    target_cols = state["target_cols"]

    print(f"Loaded best ckpt: epoch={state['epoch']} | val_avg_MAE={state['val_avg_mae']:.3f} | modality={state['modality']} | constrain={constrain}")

    test_loss, test_metrics, y_te, yhat_te, sids_te = evaluate_with_ids(
        model, dl_te, device, args.modality, mu, sigma, y_min, y_max, constrain,
        target_names=list(target_cols)
    )

    avg_mae_te = float(np.mean([test_metrics[k]["mae"] for k in target_cols]))
    print("\n[TEST] Final metrics:")
    for k in target_cols:
        m = test_metrics[k]
        print(f"  {k}: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, r={m['r']:.3f}, R2={m['r2']:.3f}")
    print(f"  Average MAE: {avg_mae_te:.2f}")

    # Save predictions
    df_pred = pd.DataFrame({"ssoid": sids_te})
    y_te = np.asarray(y_te)
    yhat_te = np.asarray(yhat_te)
    if y_te.ndim == 1:
        y_te = y_te[:, None]
    if yhat_te.ndim == 1:
        yhat_te = yhat_te[:, None]

    for i, k in enumerate(target_cols):
        df_pred[f"{k}_true"] = y_te[:, i]
        df_pred[f"{k}_pred"] = yhat_te[:, i]

    df_pred.to_csv(out_dir / "test_predictions.csv", index=False)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({
            "avg_mae": avg_mae_te,
            "metrics": test_metrics,
            "target_cols": target_cols,
            "test_loss": float(test_loss),
        }, f, indent=2)

    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
