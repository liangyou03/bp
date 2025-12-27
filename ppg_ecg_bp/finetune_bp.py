#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主脚本：血压预测微调 (Main Script: BP Finetuning)

基于 finetune_age.py 修改，支持多目标 BP 回归

python finetune_bp.py \
  --npz_dir /home/youliang/youliang_data2/bp/bp_npz_run1/npz \
  --labels_csv /home/youliang/youliang_data2/bp/bp_npz_run1/labels.csv \
  --ecg_ckpt /home/youliang/youliang_data2/bp/ppg_ecg_age/1_lead_ECGFounder.pth \
  --ppg_ckpt /home/youliang/youliang_data2/bp/ppg_ecg_age/best_checkpoint.pth \
  --out_dir /home/youliang/youliang_data2/bp/bp_run2 \
  --target_cols right_arm_dbp left_arm_mbp right_arm_pp right_arm_sbp left_arm_sbp \
  --batch_size 64 \
  --epochs 40 \
  --gpu 1 \
  --reg_loss mae_pearson
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

# ============== 本地模块导入 ==============
from backbones import AgeModel
from dataset import BPDataset  # 改用 BP Dataset
from losses import MAE_PearsonLoss
from utils import (set_seed, subject_id_from_ssoid,
                   mae_np, rmse_np, pearson_r_safe_np, r2_np)
from engine_bp import train_one_epoch, evaluate, evaluate_with_ids  # 改用 BP engine
# =========================================

# ===================== Default paths =====================
DEFAULT_NPZ_DIR   = "/home/youliang/youliang_data2/bp/bp_npz_run1/npz"
DEFAULT_LABELS    = "/home/youliang/youliang_data2/bp/bp_npz_run1/labels.csv"
DEFAULT_ECG_CKPT  = "/home/youliang/youliang_data2/bp/ppg_ecg_age/1_lead_ECGFounder.pth"
DEFAULT_PPG_CKPT  = "/home/youliang/youliang_data2/bp/ppg_ecg_age/best_checkpoint.pth"
DEFAULT_OUT_DIR   = "/home/youliang/youliang_data2/bp/bp_run1"

DEFAULT_TARGET_COLS = [
    "right_arm_dbp", "left_arm_mbp", "right_arm_pp", 
    "right_arm_sbp", "left_arm_sbp"
]

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="BP finetune with multi-target regression.")
    ap.add_argument("--npz_dir",   default=DEFAULT_NPZ_DIR)
    ap.add_argument("--labels_csv",default=DEFAULT_LABELS)
    ap.add_argument("--ecg_ckpt",  default=DEFAULT_ECG_CKPT)
    ap.add_argument("--ppg_ckpt",  default=DEFAULT_PPG_CKPT)
    ap.add_argument("--out_dir",   default=DEFAULT_OUT_DIR)
    ap.add_argument("--target_cols", nargs="+", default=DEFAULT_TARGET_COLS)
    ap.add_argument("--modality",  choices=["ecg","ppg","both"], default="both")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)

    # 优化 & 冻结策略
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze encoders (linear probe).")
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head",     type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # 损失设置
    ap.add_argument("--reg_loss", choices=["mse","huber","mae_pearson"], default="mae_pearson")
    ap.add_argument("--alpha_corr", type=float, default=0.0, help="相关性辅助项权重(1-corr)，'mae_pearson'时忽略")

    # MAE+Pearson 参数
    ap.add_argument("--maepearson_alpha", type=float, default=0.5, help="权重: (1-r) 项")
    ap.add_argument("--maepearson_beta",  type=float, default=0.5, help="权重: MAE 项")

    # 早停
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    # device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(f"cuda:{args.gpu}")
    print(f"device = {device} | name = {torch.cuda.get_device_name(args.gpu)}")

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"args_bp.json","w") as f: json.dump(vars(args), f, indent=2)

    # ----- load labels & align npz -----
    df = pd.read_csv(args.labels_csv)
    if "ssoid" not in df.columns:
        raise RuntimeError("labels_csv must contain column: ssoid")
    missing_cols = [c for c in args.target_cols if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"labels_csv missing target columns: {missing_cols}")
    
    df = df[["ssoid"] + args.target_cols].copy()
    df["ssoid"] = df["ssoid"].astype(str)
    df = df.dropna().reset_index(drop=True)
    
    have = set(p.stem for p in Path(args.npz_dir).glob("*.npz"))
    df = df[df["ssoid"].isin(have)].reset_index(drop=True)
    print(f"[data] {len(df)} labeled records after alignment with npz files.")

    # subject-wise split (7:1:2)
    df["subject"] = df["ssoid"].apply(subject_id_from_ssoid)
    subjects = df["subject"].unique().tolist()
    rng = np.random.default_rng(args.seed); rng.shuffle(subjects)
    n=len(subjects); n_tr=int(0.7*n); n_va=int(0.1*n)
    s_tr=set(subjects[:n_tr]); s_va=set(subjects[n_tr:n_tr+n_va]); s_te=set(subjects[n_tr+n_va:])
    df_tr=df[df["subject"].isin(s_tr)].copy()
    df_va=df[df["subject"].isin(s_va)].copy()
    df_te=df[df["subject"].isin(s_te)].copy()
    print(f"[split] train={len(df_tr)}  val={len(df_va)}  test={len(df_te)}  (subjects: {len(s_tr)}/{len(s_va)}/{len(s_te)})")

    # 目标标准化参数（基于训练集，多目标）
    tr_targets = df_tr[args.target_cols].to_numpy(dtype=np.float32)
    mu_np = tr_targets.mean(axis=0)
    sigma_np = tr_targets.std(axis=0)
    sigma_np = np.where(sigma_np < 1e-6, 1.0, sigma_np)
    
    mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    print(f"[target stats] mu={mu_np}, sigma={sigma_np}")

    # datasets / loaders
    ds_tr = BPDataset(df_tr, args.npz_dir, args.target_cols)
    ds_va = BPDataset(df_va, args.npz_dir, args.target_cols)
    ds_te = BPDataset(df_te, args.npz_dir, args.target_cols)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # model (使用 AgeModel，但 target_dim 改为多目标)
    model = AgeModel(modality=args.modality, proj_hidden=0, target_dim=len(args.target_cols)).to(device)
    model.load_from_pretrain(args.ecg_ckpt, args.ppg_ckpt, device=device)

    # freeze if linear probe
    params=[]
    if args.freeze_backbone:
        for n,p in model.named_parameters():
            if ("head" in n): p.requires_grad=True
            else: p.requires_grad=False
        params = [{"params":[p for p in model.parameters() if p.requires_grad], "lr": args.lr_head}]
    else:
        enc_params = []; head_params = []
        for n,p in model.named_parameters():
            if "head" in n: head_params.append(p)
            else: enc_params.append(p)
        params = [{"params": enc_params, "lr": args.lr_backbone},
                  {"params": head_params, "lr": args.lr_head}]

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # MAE_Pearson
    maepearson_criterion = None
    if args.reg_loss == "mae_pearson":
        maepearson_criterion = MAE_PearsonLoss(
            alpha=args.maepearson_alpha,
            beta=args.maepearson_beta
        ).to(device)
        print(f"[loss] MAE+Pearson (alpha={args.maepearson_alpha}, beta={args.maepearson_beta})")
    else:
        print(f"[loss] {args.reg_loss}")

    # training loop with early stopping on avg val MAE
    best_mae = float("inf"); best_ep = -1; patience_cnt=0
    ckpt_name = f"bp_{args.modality}_{args.reg_loss}_best.pth"

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(
            model, dl_tr, optimizer, scaler, device, args.modality,
            args.reg_loss, mu, sigma, args.alpha_corr,
            maepearson_criterion=maepearson_criterion
        )
        val_loss, val_metrics, avg_val_mae = evaluate(
            model, dl_va, device, args.modality, mu, sigma, args.target_cols
        )
        metric_str = " ".join([f"{name}:{vals['mae']:.2f}" for name, vals in val_metrics.items()])
        print(f"[E{ep}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} avg_MAE={avg_val_mae:.2f} | {metric_str}")

        if avg_val_mae < best_mae - 1e-4:
            best_mae = avg_val_mae; best_ep = ep; patience_cnt=0
            ckpt = {"epoch": ep, "model": model.state_dict(),
                    "avg_val_mae": float(avg_val_mae), "modality": args.modality,
                    "mu": mu.cpu().tolist(), "sigma": sigma.cpu().tolist(),
                    "target_cols": args.target_cols}
            ckpt_path = Path(args.out_dir)/ckpt_name
            torch.save(ckpt, str(ckpt_path))
            print(f"  -> saved best (avg_MAE={best_mae:.2f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs (best @ {best_ep})")
                break

    # ======= Load best and full eval =======
    ckpt_path = Path(args.out_dir)/ckpt_name
    if not ckpt_path.exists():
        print("[Error] No best checkpoint was saved. Skipping final evaluation.")
        return
        
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state["model"])
    mu = torch.tensor(state["mu"], dtype=torch.float32, device=device)
    sigma = torch.tensor(state["sigma"], dtype=torch.float32, device=device)
    target_cols = state["target_cols"]
    print(f"Loaded best ckpt: epoch={state['epoch']} | avg_MAE={state['avg_val_mae']:.2f} | modality={state['modality']}")

    # ---- Final evaluation on test set ----
    te_loss, te_metrics, avg_te_mae, y_te, yhat_te, sids_te = evaluate_with_ids(
        model, dl_te, device, args.modality, mu, sigma, target_cols
    )
    
    print("\n[TEST] Final metrics:")
    for name, vals in te_metrics.items():
        print(f"  {name}: MAE={vals['mae']:.2f} RMSE={vals['rmse']:.2f} r={vals['r']:.3f} R2={vals['r2']:.3f}")
    print(f"  Average MAE: {avg_te_mae:.2f}")

    # ---- Save predictions ----
    df_pred = pd.DataFrame({"ssoid": sids_te})
    for idx, col in enumerate(target_cols):
        df_pred[f"{col}_true"] = y_te[:, idx]
        df_pred[f"{col}_pred"] = yhat_te[:, idx]
    pred_path = out_dir / "test_predictions.csv"
    df_pred.to_csv(pred_path, index=False)
    print(f"[saved] predictions -> {pred_path}")

    # ---- Save metrics ----
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "test_loss": te_loss,
            "avg_mae": avg_te_mae,
            "metrics": te_metrics,
            "best_epoch": state["epoch"],
            "modality": args.modality
        }, f, indent=2)
    print(f"[saved] metrics -> {metrics_path}")


if __name__ == "__main__":
    main()