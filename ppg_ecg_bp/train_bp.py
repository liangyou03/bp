#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版血压预测微调脚本 (Improved BP Finetuning Script)

改进点:
1. 支持 6 个血压目标: right_arm_sbp, right_arm_mbp, right_arm_dbp, 
                      left_arm_sbp, left_arm_mbp, left_arm_dbp
2. 每个 epoch 汇报训练集和验证集的详细指标 (MAE, RMSE, r, R2)
3. 保存完整的训练历史日志 (training_history.json)
4. 更清晰的日志输出格式

Usage:
python finetune_bp_v2.py \
  --npz_dir /path/to/npz \
  --labels_csv /path/to/labels.csv \
  --ecg_ckpt /path/to/1_lead_ECGFounder.pth \
  --ppg_ckpt /path/to/best_checkpoint.pth \
  --out_dir /path/to/output \
  --target_cols right_arm_sbp right_arm_mbp right_arm_dbp left_arm_sbp left_arm_mbp left_arm_dbp \
  --batch_size 64 \
  --epochs 40 \
  --gpu 0

python finetune_bp.py \
  --npz_dir /home/youliang/youliang_data2/bp/bp_npz_truncate/npz \
  --labels_csv /home/youliang/youliang_data2/bp/bp_npz_truncate/labels.csv \
  --ecg_ckpt /home/youliang/youliang_data2/bp/ppg_ecg_age/1_lead_ECGFounder.pth \
  --ppg_ckpt /home/youliang/youliang_data2/bp/ppg_ecg_age/best_checkpoint.pth \
  --out_dir /home/youliang/youliang_data2/bp/bp_ppgobly \
  --target_cols right_arm_sbp right_arm_mbp right_arm_dbp right_arm_pp left_arm_sbp left_arm_mbp left_arm_dbp left_arm_pp \
  --modality both \
  --batch_size 64 \
  --epochs 40 \
  --gpu 0
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

# ============== 本地模块导入 ==============
from backbones import AgeModel
from dataset import BPDataset
from losses import MAE_PearsonLoss
from utils import (set_seed, subject_id_from_ssoid,
                   mae_np, rmse_np, pearson_r_safe_np, r2_np)
from engine_bp import train_one_epoch, evaluate, evaluate_with_ids
# =========================================

# ===================== Default paths =====================
# 数据目录（truncate 或 resample 版本）
DEFAULT_NPZ_DIR   = "/home/youliang/youliang_data2/bp/bp_npz_truncate/npz"
DEFAULT_LABELS    = "/home/youliang/youliang_data2/bp/bp_npz_truncate/labels.csv"

# 预训练权重
DEFAULT_ECG_CKPT  = "/home/youliang/youliang_data2/bp/ppg_ecg_age/1_lead_ECGFounder.pth"
DEFAULT_PPG_CKPT  = "/home/youliang/youliang_data2/bp/ppg_ecg_age/best_checkpoint.pth"

# 输出目录
DEFAULT_OUT_DIR   = "/home/youliang/youliang_data2/bp/bp_run_v2"

# 8 个血压目标 (SBP, MBP, DBP, PP × 左右臂)
DEFAULT_TARGET_COLS = [
    "right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp",
    "left_arm_sbp", "left_arm_mbp", "left_arm_dbp", "left_arm_pp"
]


def compute_metrics_per_target(y_true, y_pred, target_cols):
    """计算每个目标的详细指标"""
    metrics = {}
    for idx, col in enumerate(target_cols):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        metrics[col] = {
            "mae": float(mae_np(yt, yp)),
            "rmse": float(rmse_np(yt, yp)),
            "r": float(pearson_r_safe_np(yt, yp)),
            "r2": float(r2_np(yt, yp))
        }
    # 计算平均 MAE
    avg_mae = np.mean([m["mae"] for m in metrics.values()])
    return metrics, avg_mae


def format_metrics_table(metrics, target_cols, prefix=""):
    """格式化指标输出为表格形式"""
    lines = []
    header = f"{prefix:>8} | {'Target':<16} | {'MAE':>7} | {'RMSE':>7} | {'r':>7} | {'R2':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for col in target_cols:
        m = metrics[col]
        lines.append(f"{prefix:>8} | {col:<16} | {m['mae']:>7.2f} | {m['rmse']:>7.2f} | {m['r']:>7.3f} | {m['r2']:>7.3f}")
    avg_mae = np.mean([metrics[col]["mae"] for col in target_cols])
    avg_rmse = np.mean([metrics[col]["rmse"] for col in target_cols])
    avg_r = np.mean([metrics[col]["r"] for col in target_cols])
    avg_r2 = np.mean([metrics[col]["r2"] for col in target_cols])
    lines.append("-" * len(header))
    lines.append(f"{prefix:>8} | {'AVERAGE':<16} | {avg_mae:>7.2f} | {avg_rmse:>7.2f} | {avg_r:>7.3f} | {avg_r2:>7.3f}")
    return "\n".join(lines)


def evaluate_full(model, dataloader, device, modality, mu, sigma, target_cols):
    """
    完整评估函数，返回 loss、详细指标、预测结果
    """
    model.eval()
    all_y = []
    all_yhat = []
    all_sids = []
    total_loss = 0.0
    n_batches = 0
    
    mse_criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in dataloader:
            ecg = batch["ecg"].to(device) if "ecg" in batch else None
            ppg = batch["ppg"].to(device) if "ppg" in batch else None
            y = batch["target"].to(device)  # (B, num_targets)
            sids = batch["ssoid"]
            
            # forward
            if modality == "ecg":
                pred_z = model(ecg, None)
            elif modality == "ppg":
                pred_z = model(None, ppg)
            else:
                pred_z = model(ecg, ppg)
            
            # 标准化后的 target
            y_z = (y - mu) / sigma
            
            # 计算 loss (在标准化空间)
            loss = mse_criterion(pred_z, y_z)
            total_loss += loss.item()
            n_batches += 1
            
            # 反标准化预测值
            pred = pred_z * sigma + mu
            
            all_y.append(y.cpu().numpy())
            all_yhat.append(pred.cpu().numpy())
            all_sids.extend(sids)
    
    y_np = np.concatenate(all_y, axis=0)
    yhat_np = np.concatenate(all_yhat, axis=0)
    avg_loss = total_loss / max(n_batches, 1)
    
    metrics, avg_mae = compute_metrics_per_target(y_np, yhat_np, target_cols)
    
    return avg_loss, metrics, avg_mae, y_np, yhat_np, all_sids


def main():
    ap = argparse.ArgumentParser(description="Improved BP finetune with detailed logging.")
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
    ap.add_argument("--alpha_corr", type=float, default=0.0, help="相关性辅助项权重")

    # MAE+Pearson 参数
    ap.add_argument("--maepearson_alpha", type=float, default=0.5, help="权重: (1-r) 项")
    ap.add_argument("--maepearson_beta",  type=float, default=0.5, help="权重: MAE 项")

    # 早停
    ap.add_argument("--patience", type=int, default=5)
    
    # 新增：是否在每个 epoch 评估训练集（会增加时间）
    ap.add_argument("--eval_train", action="store_true", default=True,
                    help="Evaluate on training set each epoch (slower but more informative)")
    
    args = ap.parse_args()

    # ==================== Setup ====================
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(f"cuda:{args.gpu}")
    print("=" * 70)
    print(f"BP Finetuning Script (Improved Version)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(args.gpu)}")
    print("=" * 70)

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = vars(args).copy()
    config["start_time"] = datetime.now().isoformat()
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[Config] Saved to {out_dir / 'config.json'}")

    # ==================== Data Loading ====================
    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    
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
    print(f"Total samples with labels and npz files: {len(df)}")

    # Subject-wise split (7:1:2)
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
    
    df_tr = df[df["subject"].isin(s_tr)].copy()
    df_va = df[df["subject"].isin(s_va)].copy()
    df_te = df[df["subject"].isin(s_te)].copy()
    
    print(f"\nData Split (by subject):")
    print(f"  Train:      {len(df_tr):>6} samples ({len(s_tr):>4} subjects)")
    print(f"  Validation: {len(df_va):>6} samples ({len(s_va):>4} subjects)")
    print(f"  Test:       {len(df_te):>6} samples ({len(s_te):>4} subjects)")

    # 目标统计
    print(f"\nTarget Statistics (Training Set):")
    print(f"  {'Target':<16} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print(f"  {'-'*60}")
    for col in args.target_cols:
        vals = df_tr[col].values
        print(f"  {col:<16} | {vals.mean():>8.2f} | {vals.std():>8.2f} | {vals.min():>8.2f} | {vals.max():>8.2f}")

    # 标准化参数
    tr_targets = df_tr[args.target_cols].to_numpy(dtype=np.float32)
    mu_np = tr_targets.mean(axis=0)
    sigma_np = tr_targets.std(axis=0)
    sigma_np = np.where(sigma_np < 1e-6, 1.0, sigma_np)
    
    mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)

    # Datasets & DataLoaders
    ds_tr = BPDataset(df_tr, args.npz_dir, args.target_cols)
    ds_va = BPDataset(df_va, args.npz_dir, args.target_cols)
    ds_te = BPDataset(df_te, args.npz_dir, args.target_cols)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # ==================== Model Setup ====================
    print("\n" + "=" * 70)
    print("MODEL SETUP")
    print("=" * 70)
    
    model = AgeModel(
        modality=args.modality,
        proj_hidden=0,
        target_dim=len(args.target_cols)
    ).to(device)
    model.load_from_pretrain(args.ecg_ckpt, args.ppg_ckpt, device=device)
    
    print(f"Modality: {args.modality}")
    print(f"Target dimensions: {len(args.target_cols)}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    # 参数设置
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if "head" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_head}]
        print(f"Mode: Linear Probe (head lr={args.lr_head})")
    else:
        enc_params = []
        head_params = []
        for n, p in model.named_parameters():
            if "head" in n:
                head_params.append(p)
            else:
                enc_params.append(p)
        params = [
            {"params": enc_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head}
        ]
        print(f"Mode: End-to-end (backbone lr={args.lr_backbone}, head lr={args.lr_head})")

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Loss function
    maepearson_criterion = None
    if args.reg_loss == "mae_pearson":
        maepearson_criterion = MAE_PearsonLoss(
            alpha=args.maepearson_alpha,
            beta=args.maepearson_beta
        ).to(device)
        print(f"Loss: MAE+Pearson (alpha={args.maepearson_alpha}, beta={args.maepearson_beta})")
    else:
        print(f"Loss: {args.reg_loss}")

    # ==================== Training ====================
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_mae = float("inf")
    best_ep = -1
    patience_cnt = 0
    ckpt_name = f"bp_{args.modality}_{args.reg_loss}_best.pth"
    
    # 训练历史记录
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_avg_mae": [],
        "val_avg_mae": [],
        "train_metrics": [],
        "val_metrics": []
    }

    for ep in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {ep}/{args.epochs}")
        print(f"{'='*70}")
        
        # ---- Training ----
        tr_loss = train_one_epoch(
            model, dl_tr, optimizer, scaler, device, args.modality,
            args.reg_loss, mu, sigma, args.alpha_corr,
            maepearson_criterion=maepearson_criterion
        )
        
        # ---- Evaluate on Training Set (optional but informative) ----
        if args.eval_train:
            tr_eval_loss, tr_metrics, tr_avg_mae, _, _, _ = evaluate_full(
                model, dl_tr, device, args.modality, mu, sigma, args.target_cols
            )
            print(f"\n[Train] Loss={tr_loss:.4f} | Eval Loss={tr_eval_loss:.4f} | Avg MAE={tr_avg_mae:.2f}")
            print(format_metrics_table(tr_metrics, args.target_cols, prefix="Train"))
        else:
            tr_metrics = {}
            tr_avg_mae = 0.0
            tr_eval_loss = tr_loss
        
        # ---- Evaluate on Validation Set ----
        val_loss, val_metrics, val_avg_mae, _, _, _ = evaluate_full(
            model, dl_va, device, args.modality, mu, sigma, args.target_cols
        )
        print(f"\n[Val] Loss={val_loss:.4f} | Avg MAE={val_avg_mae:.2f}")
        print(format_metrics_table(val_metrics, args.target_cols, prefix="Val"))
        
        # ---- Record history ----
        history["epochs"].append(ep)
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(val_loss))
        history["train_avg_mae"].append(float(tr_avg_mae) if args.eval_train else None)
        history["val_avg_mae"].append(float(val_avg_mae))
        history["train_metrics"].append(tr_metrics if args.eval_train else {})
        history["val_metrics"].append(val_metrics)
        
        # ---- Early stopping check ----
        if val_avg_mae < best_mae - 1e-4:
            best_mae = val_avg_mae
            best_ep = ep
            patience_cnt = 0
            
            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_val_mae": float(val_avg_mae),
                "val_metrics": val_metrics,
                "modality": args.modality,
                "mu": mu.cpu().tolist(),
                "sigma": sigma.cpu().tolist(),
                "target_cols": args.target_cols
            }
            ckpt_path = out_dir / ckpt_name
            torch.save(ckpt, str(ckpt_path))
            print(f"\n*** NEW BEST: Saved checkpoint (Avg MAE={best_mae:.2f}) ***")
        else:
            patience_cnt += 1
            print(f"\nNo improvement ({patience_cnt}/{args.patience}). Best: epoch {best_ep} (MAE={best_mae:.2f})")
            
            if patience_cnt >= args.patience:
                print(f"\n[EARLY STOP] No improvement for {args.patience} epochs.")
                break
        
        # 保存训练历史（每个 epoch 更新）
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ==================== Final Evaluation ====================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    ckpt_path = out_dir / ckpt_name
    if not ckpt_path.exists():
        print("[Error] No best checkpoint was saved. Skipping final evaluation.")
        return
    
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state["model"])
    mu = torch.tensor(state["mu"], dtype=torch.float32, device=device)
    sigma = torch.tensor(state["sigma"], dtype=torch.float32, device=device)
    target_cols = state["target_cols"]
    
    print(f"\nLoaded best checkpoint: Epoch {state['epoch']} | Val Avg MAE={state['avg_val_mae']:.2f}")

    # Test evaluation
    te_loss, te_metrics, te_avg_mae, y_te, yhat_te, sids_te = evaluate_full(
        model, dl_te, device, args.modality, mu, sigma, target_cols
    )
    
    print(f"\n[Test] Loss={te_loss:.4f} | Avg MAE={te_avg_mae:.2f}")
    print(format_metrics_table(te_metrics, target_cols, prefix="Test"))

    # ==================== Save Results ====================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # 保存预测结果
    df_pred = pd.DataFrame({"ssoid": sids_te})
    for idx, col in enumerate(target_cols):
        df_pred[f"{col}_true"] = y_te[:, idx]
        df_pred[f"{col}_pred"] = yhat_te[:, idx]
    pred_path = out_dir / "test_predictions.csv"
    df_pred.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")

    # 保存最终指标
    final_results = {
        "best_epoch": state["epoch"],
        "modality": args.modality,
        "train_samples": len(df_tr),
        "val_samples": len(df_va),
        "test_samples": len(df_te),
        "test_loss": float(te_loss),
        "test_avg_mae": float(te_avg_mae),
        "test_metrics": te_metrics,
        "val_metrics_at_best": state["val_metrics"],
        "target_cols": target_cols,
        "normalization": {
            "mu": state["mu"],
            "sigma": state["sigma"]
        }
    }
    
    metrics_path = out_dir / "final_results.json"
    with open(metrics_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Final results saved to: {metrics_path}")

    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best Epoch: {state['epoch']}")
    print(f"Modality: {args.modality}")
    print(f"\nTest Set Performance:")
    print(f"  Average MAE: {te_avg_mae:.2f} mmHg")
    print(f"\n  Per-target MAE:")
    for col in target_cols:
        print(f"    {col}: {te_metrics[col]['mae']:.2f} mmHg")
    
    print(f"\nOutput directory: {out_dir}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()