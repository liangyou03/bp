#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Script: BP Finetuning (Multi-target Regression)

Key changes in this version:
1) Report train/val MAE/RMSE/r/R2/bias each epoch (train metrics via non-shuffled loader).
2) Exclude bias/Norm params from weight decay (AdamW).
3) Initialize the last head bias with train-set mu (helps remove large global offset early).

Assumes your engine.py provides:
- train_one_epoch(...)
- evaluate(model, loader, device, modality, mu, sigma, y_min, y_max, constrain, target_names)
  -> returns (loss, metrics_dict, y, yhat)
- evaluate_with_ids(...)
  -> returns (loss, metrics_dict, y, yhat, ssoids)
"""

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
    Multi-target MAE+Pearson:
      mean_k [ alpha*(1-r_k) + beta*MAE_k ]
    Input shapes: (B,K)
    """
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y_hat.ndim == 1:
            y_hat = y_hat.view(-1, 1)
        if y.ndim == 1:
            y = y.view(-1, 1)
        if y_hat.shape != y.shape:
            raise ValueError(f"Shape mismatch: y_hat {tuple(y_hat.shape)} vs y {tuple(y.shape)}")

        K = y.shape[1]
        losses = []
        for k in range(K):
            yh = y_hat[:, k]
            yt = y[:, k]

            vx = yh - yh.mean()
            vy = yt - yt.mean()
            den = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + self.eps
            corr = (vx * vy).sum() / den
            pearson_loss = 1.0 - corr

            mae_loss = torch.mean(torch.abs(yh - yt))
            losses.append(self.alpha * pearson_loss + self.beta * mae_loss)

        return torch.stack(losses).mean()


DEFAULT_NPZ_DIR = "/home/notebook/data/personal/S9061270/bp_ready/labeled_zscore"
DEFAULT_LABELS = "/home/notebook/data/personal/S9061270/bp_ready/labeled_labels.csv"
DEFAULT_PRETRAIN = "/home/notebook/data/personal/S9061270/model/clip_ppg_ecg_founder/pretrain_temp/best.pth"
DEFAULT_OUT_DIR = "/home/notebook/data/personal/S9061270/model/clip_ppg_ecg_founder/finetune_bp_run1"


def main():
    ap = argparse.ArgumentParser(description="BP finetune (multi-target) with subject-wise split.")
    ap.add_argument("--npz_dir", default=DEFAULT_NPZ_DIR)
    ap.add_argument("--labels_csv", default=DEFAULT_LABELS)
    ap.add_argument("--pretrain", default=DEFAULT_PRETRAIN)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--modality", choices=["ecg", "ppg", "both"], default="both")

    ap.add_argument("--target_cols", nargs="+", required=True)

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--reg_loss", choices=["mse", "huber", "mae_pearson"], default="huber")
    ap.add_argument("--alpha_corr", type=float, default=0.0)

    ap.add_argument("--maepearson_alpha", type=float, default=0.5)
    ap.add_argument("--maepearson_beta", type=float, default=0.5)

    ap.add_argument("--constrain", choices=["none", "tanh", "sigmoid", "clip"], default="none")
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=300.0)

    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(f"cuda:{args.gpu}")
    print(f"device = {device} | name = {torch.cuda.get_device_name(args.gpu)}")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args_bp.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    df = pd.read_csv(args.labels_csv)
    need_cols = ["ssoid"] + list(args.target_cols)
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"labels_csv missing column: {c}")

    df = df[need_cols].copy()
    df["ssoid"] = df["ssoid"].astype(str)
    for c in args.target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=args.target_cols).reset_index(drop=True)

    have = set(p.stem for p in Path(args.npz_dir).glob("*.npz"))
    df = df[df["ssoid"].isin(have)].reset_index(drop=True)

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

    tr_targets = df_tr[args.target_cols].to_numpy(dtype=np.float32)
    mu = tr_targets.mean(axis=0).astype(np.float32)
    sigma = tr_targets.std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, 1e-6).astype(np.float32)
    y_min = float(args.y_min)
    y_max = float(args.y_max)
    print(f"[target stats] mu={mu}  sigma={sigma}  y_min={y_min}  y_max={y_max}")

    ds_tr = BPDataset(df_tr, args.npz_dir, target_cols=args.target_cols)
    ds_va = BPDataset(df_va, args.npz_dir, target_cols=args.target_cols)
    ds_te = BPDataset(df_te, args.npz_dir, target_cols=args.target_cols)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_tr_eval = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = AgeModel(modality=args.modality, proj_hidden=0, target_dim=len(args.target_cols)).to(device)
    model.load_from_pretrain(args.pretrain, device=device)

    # Initialize last head bias with train mu to reduce global offset at start
    if isinstance(model.head, torch.nn.Sequential) and isinstance(model.head[-1], torch.nn.Linear):
        with torch.no_grad():
            b = model.head[-1].bias
            if b is not None and b.numel() == len(mu):
                b.copy_(torch.as_tensor(mu, device=device, dtype=b.dtype))

    # Freeze backbone if requested
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            p.requires_grad = ("head" in n)

    # Build AdamW param groups with no weight decay for bias/norm
    decay_bb, no_decay_bb = [], []
    decay_hd, no_decay_hd = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_no_decay = (
            n.endswith(".bias")
            or ("bn" in n.lower())
            or ("norm" in n.lower())
            or ("ln" in n.lower())
        )

        is_head = ("head" in n)

        if args.freeze_backbone:
            # only head is trainable anyway
            if is_no_decay:
                no_decay_hd.append(p)
            else:
                decay_hd.append(p)
        else:
            if is_head:
                if is_no_decay:
                    no_decay_hd.append(p)
                else:
                    decay_hd.append(p)
            else:
                if is_no_decay:
                    no_decay_bb.append(p)
                else:
                    decay_bb.append(p)

    param_groups = []
    if not args.freeze_backbone:
        if decay_bb:
            param_groups.append({"params": decay_bb, "lr": args.lr_backbone, "weight_decay": args.weight_decay})
        if no_decay_bb:
            param_groups.append({"params": no_decay_bb, "lr": args.lr_backbone, "weight_decay": 0.0})

    if decay_hd:
        param_groups.append({"params": decay_hd, "lr": args.lr_head, "weight_decay": args.weight_decay})
    if no_decay_hd:
        param_groups.append({"params": no_decay_hd, "lr": args.lr_head, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    maepearson_criterion = None
    if args.reg_loss == "mae_pearson":
        maepearson_criterion = MultiTarget_MAE_PearsonLoss(
            alpha=args.maepearson_alpha,
            beta=args.maepearson_beta
        ).to(device)

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

        tr_eval_loss, tr_metrics, y_tr, yhat_tr = evaluate(
            model, dl_tr_eval, device, args.modality, mu, sigma, y_min, y_max, args.constrain,
            target_names=list(args.target_cols)
        )
        val_loss, val_metrics, y_va, yhat_va = evaluate(
            model, dl_va, device, args.modality, mu, sigma, y_min, y_max, args.constrain,
            target_names=list(args.target_cols)
        )

        tr_avg_mae = float(np.mean([tr_metrics[k]["mae"] for k in args.target_cols]))
        tr_avg_rmse = float(np.mean([tr_metrics[k]["rmse"] for k in args.target_cols]))
        tr_avg_r = float(np.mean([tr_metrics[k]["r"] for k in args.target_cols]))
        tr_avg_r2 = float(np.mean([tr_metrics[k]["r2"] for k in args.target_cols]))
        tr_bias = (np.asarray(yhat_tr) - np.asarray(y_tr)).mean(axis=0)

        va_avg_mae = float(np.mean([val_metrics[k]["mae"] for k in args.target_cols]))
        va_avg_rmse = float(np.mean([val_metrics[k]["rmse"] for k in args.target_cols]))
        va_avg_r = float(np.mean([val_metrics[k]["r"] for k in args.target_cols]))
        va_avg_r2 = float(np.mean([val_metrics[k]["r2"] for k in args.target_cols]))
        va_bias = (np.asarray(yhat_va) - np.asarray(y_va)).mean(axis=0)

        metric_str = " ".join([
            f"{k}:MAE={val_metrics[k]['mae']:.2f},RMSE={val_metrics[k]['rmse']:.2f},r={val_metrics[k]['r']:.3f},R2={val_metrics[k]['r2']:.3f}"
            for k in args.target_cols
        ])

        print(
            f"[E{ep}] "
            f"train_loss={tr_loss:.6f} | "
            f"train_eval_loss={tr_eval_loss:.6f} MAE={tr_avg_mae:.2f} RMSE={tr_avg_rmse:.2f} r={tr_avg_r:.3f} R2={tr_avg_r2:.3f} bias={tr_bias} | "
            f"val_loss={val_loss:.6f} MAE={va_avg_mae:.2f} RMSE={va_avg_rmse:.2f} r={va_avg_r:.3f} R2={va_avg_r2:.3f} bias={va_bias} | "
            f"{metric_str}"
        )

        avg_mae = va_avg_mae
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

    print("test_bias:", (yhat_te - y_te).mean(axis=0))
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