#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主脚本：血压预测微调 (Main Script: BP Finetuning)

从 finetune_age.py 迁移，主要改动：
1. target_col 参数化 (right_arm_sbp, left_arm_dbp 等)
2. y_min/y_max 默认值调整为 BP 范围
3. 移除 DistLoss (年龄专用)
4. 输出文件名改为 bp 前缀
5. 支持 ECG 重采样 (--resample_ecg)
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
from backbones import AgeModel  # 结构不变，仍可复用
from dataset import LabeledECGPPGDataset
from losses import MAE_PearsonLoss
from utils import (set_seed, subject_id_from_ssoid, aggregate_by_subject_prefix, 
                   mae_np, rmse_np, pearson_r_safe_np, r2_np, 
                   LinearCalibrator)
from engine import train_one_epoch, evaluate, evaluate_with_ids
# =========================================

# ===================== Default paths =====================
DEFAULT_NPZ_DIR   = "/home/youliang/youliang_data2/bp/bp_npz_truncate/npz"
DEFAULT_LABELS    = "/home/youliang/youliang_data2/bp/bp_npz_truncate/labels.csv"
DEFAULT_PRETRAIN  = "/home/youliang/youliang_data2/bp/ppg_ecg_clip_bp/run_resample_only_long/clip_foundation_best.pth"
DEFAULT_OUT_DIR   = "/home/youliang/youliang_data2/bp/ppg_ecg_bp/runs/bp_run1"

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="BP finetune with target-standardization & bounded outputs.")
    ap.add_argument("--npz_dir",    default=DEFAULT_NPZ_DIR)
    ap.add_argument("--labels_csv", default=DEFAULT_LABELS)
    ap.add_argument("--pretrain",   default=DEFAULT_PRETRAIN)
    ap.add_argument("--out_dir",    default=DEFAULT_OUT_DIR)
    ap.add_argument("--modality",   choices=["ecg","ppg","both"], default="both")
    
    # BP 特有：目标列
    ap.add_argument("--target_col", type=str, default="right_arm_sbp",
                    help="BP target column in labels_csv (e.g., right_arm_sbp, left_arm_dbp)")
    
    # ECG 重采样（与 pretrain 保持一致）
    ap.add_argument("--resample_ecg", action="store_true", default=False,
                    help="Resample ECG from 50 Hz to 500 Hz (must match pretrain setting)")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)

    # 优化 & 冻结策略
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze encoders+projectors (linear probe).")
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head",     type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # 损失设置 (移除 mse+dist)
    ap.add_argument("--reg_loss", choices=["mse","huber","mae_pearson"], default="huber")
    ap.add_argument("--alpha_corr", type=float, default=0.0, help="相关性辅助项权重(1-corr)")

    # MAE+Pearson 参数
    ap.add_argument("--maepearson_alpha", type=float, default=0.5, help="权重: (1-r) 项")
    ap.add_argument("--maepearson_beta",  type=float, default=0.5, help="权重: MAE 项")

    # 约束设置 (BP 范围)
    ap.add_argument("--constrain", choices=["none","tanh","sigmoid","clip"], default="clip")
    ap.add_argument("--y_min", type=float, default=60.0,  help="BP lower bound (mmHg)")
    ap.add_argument("--y_max", type=float, default=220.0, help="BP upper bound (mmHg)")

    # 个体聚合设置
    ap.add_argument("--subj_agg", choices=["mean","max","median"], default="mean",
                    help="个体级别聚合方式")

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
    target_col = args.target_col
    
    if "ssoid" not in df.columns:
        raise RuntimeError("labels_csv must contain column: ssoid")
    if target_col not in df.columns:
        raise RuntimeError(f"labels_csv must contain column: {target_col}")
    
    df = df[["ssoid", target_col]].copy()
    df["ssoid"] = df["ssoid"].astype(str)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()]
    
    have = set(p.stem for p in Path(args.npz_dir).glob("*.npz"))
    df = df[df["ssoid"].isin(have)].reset_index(drop=True)
    print(f"[data] {len(df)} samples with valid {target_col}")

    # subject-wise split (7:1:2)
    df["subject"] = df["ssoid"].apply(subject_id_from_ssoid)
    subjects = df["subject"].unique().tolist()
    rng = np.random.default_rng(args.seed); rng.shuffle(subjects)
    n = len(subjects); n_tr = int(0.7*n); n_va = int(0.1*n)
    s_tr = set(subjects[:n_tr])
    s_va = set(subjects[n_tr:n_tr+n_va])
    s_te = set(subjects[n_tr+n_va:])
    
    df_tr = df[df["subject"].isin(s_tr)][["ssoid", target_col]].copy()
    df_va = df[df["subject"].isin(s_va)][["ssoid", target_col]].copy()
    df_te = df[df["subject"].isin(s_te)][["ssoid", target_col]].copy()
    print(f"[split] train={len(df_tr)}  val={len(df_va)}  test={len(df_te)}  (subjects: {len(s_tr)}/{len(s_va)}/{len(s_te)})")

    # 目标标准化参数（基于训练集）
    mu    = float(df_tr[target_col].mean())
    sigma = float(df_tr[target_col].std(ddof=0))
    if sigma < 1e-6: sigma = 1.0
    y_min = float(args.y_min); y_max = float(args.y_max)
    print(f"[target stats] mu={mu:.3f}  sigma={sigma:.3f}  y_min={y_min}  y_max={y_max}")

    # datasets / loaders
    ds_tr = LabeledECGPPGDataset(df_tr, args.npz_dir, target_col=target_col, resample_ecg=args.resample_ecg)
    ds_va = LabeledECGPPGDataset(df_va, args.npz_dir, target_col=target_col, resample_ecg=args.resample_ecg)
    ds_te = LabeledECGPPGDataset(df_te, args.npz_dir, target_col=target_col, resample_ecg=args.resample_ecg)

    # 打印数据形状确认
    ecg_test, ppg_test, _, _ = ds_tr[0]
    print(f"[shape] ECG: {ecg_test.shape}, PPG: {ppg_test.shape} (resample_ecg={args.resample_ecg})")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # model (复用 AgeModel，结构完全相同)
    model = AgeModel(modality=args.modality, proj_hidden=0).to(device)
    model.load_from_pretrain(args.pretrain, device=device)

    # freeze if linear probe
    params = []
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if "head" in n: p.requires_grad = True
            else: p.requires_grad = False
        params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_head}]
    else:
        enc_params = []; head_params = []
        for n, p in model.named_parameters():
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

    # training loop with early stopping on val MAE
    best_mae = float("inf"); best_ep = -1; patience_cnt = 0
    ckpt_name = f"bp_{target_col}_{args.modality}_{args.reg_loss}_{args.constrain}_best.pth"

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_mae = train_one_epoch(
            model, dl_tr, optimizer, scaler, device, args.modality,
            args.reg_loss, mu, sigma, y_min, y_max, args.constrain, args.alpha_corr,
            maepearson_criterion=maepearson_criterion,
            dist_criterion=None,
            lambda_dist=0.0
        )
        val_loss, val_mae, val_rmse, r, r2, y_val, yhat_val = evaluate(
            model, dl_va, device, args.modality, mu, sigma, y_min, y_max, args.constrain
        )
        # 在验证集上 fit calibrator 并计算校准后 MAE
        calib = LinearCalibrator().fit(yhat_val, y_val)
        yhat_val_cal = calib.transform(yhat_val)
        val_mae_cal = mae_np(y_val, yhat_val_cal)
        print(f"[E{ep}] tr_MAE={tr_mae:.2f} | val_MAE={val_mae:.2f} (cal={val_mae_cal:.2f}) r={r:.3f}")

        if val_mae < best_mae - 1e-6:
            best_mae = val_mae; best_ep = ep; patience_cnt = 0
            ckpt = {"epoch": ep, "model": model.state_dict(),
                    "val_mae": float(val_mae), "modality": args.modality,
                    "mu": mu, "sigma": sigma, "y_min": y_min, "y_max": y_max,
                    "constrain": args.constrain, "target_col": target_col,
                    "resample_ecg": args.resample_ecg}
            ckpt_path = Path(args.out_dir) / ckpt_name
            torch.save(ckpt, str(ckpt_path))
            print(f"[best] val_MAE={best_mae:.3f} @epoch{ep} | saved: {ckpt_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs (best @ {best_ep})")
                break

    # ======= Load best and full eval =======
    ckpt_path = Path(args.out_dir) / ckpt_name
    if not ckpt_path.exists():
        print("[Error] No best checkpoint was saved. Skipping final evaluation.")
        return
        
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state["model"])
    mu    = float(state.get("mu", mu))
    sigma = float(state.get("sigma", sigma))
    y_min = float(state.get("y_min", y_min))
    y_max = float(state.get("y_max", y_max))
    constrain = state.get("constrain", args.constrain)
    print(f"Loaded best ckpt: epoch={state['epoch']} | val_MAE={state['val_mae']:.3f} | modality={state['modality']} | constrain={constrain}")

    # ---- 1. 获取验证集(VAL)和测试集(TEST)的原始预测 ----
    _, _, _, _, _, y_val_rec, yhat_val_rec, sids_val_rec = evaluate_with_ids(
        model, dl_va, device, args.modality, mu, sigma, y_min, y_max, constrain
    )
    loss_te, _, _, _, _, y_te_rec, yhat_te_rec, sids_te_rec = evaluate_with_ids(
        model, dl_te, device, args.modality, mu, sigma, y_min, y_max, constrain
    )
    
    # ---- 2. 记录级 (Record-level) 校准 ----
    calibrator_rec = LinearCalibrator()
    calibrator_rec.fit(yhat_val_rec, y_val_rec)
    print(f"[Calibrator] (record) {calibrator_rec}")
    
    yhat_te_rec_cal = calibrator_rec.transform(yhat_te_rec)
    a_rec, b_rec = calibrator_rec.a, calibrator_rec.b

    print(f"[TEST/raw] (record) ({args.modality})  MAE={mae_np(y_te_rec,yhat_te_rec):.3f}  RMSE={rmse_np(y_te_rec,yhat_te_rec):.3f}  r={pearson_r_safe_np(y_te_rec,yhat_te_rec):.3f}  R2={r2_np(y_te_rec,yhat_te_rec):.3f}")
    print(f"[TEST/cal] (record) ({args.modality})  MAE={mae_np(y_te_rec,yhat_te_rec_cal):.3f}  RMSE={rmse_np(y_te_rec,yhat_te_rec_cal):.3f}  r={pearson_r_safe_np(y_te_rec,yhat_te_rec_cal):.3f}  R2={r2_np(y_te_rec,yhat_te_rec_cal):.3f}  (a={a_rec:.4f}, b={b_rec:.4f})")

    # ---- 3. 个体级 (Subject-level) 聚合与校准 ----
    subj_val, y_val_subj, yhat_val_subj, n_val = aggregate_by_subject_prefix(y_val_rec, yhat_val_rec, sids_val_rec, agg=args.subj_agg)
    subj_te, y_te_subj, yhat_te_subj, n_te = aggregate_by_subject_prefix(y_te_rec, yhat_te_rec, sids_te_rec, agg=args.subj_agg)
    
    print(f"[VAL/raw] (subject-{args.subj_agg}) Nsubj={len(subj_val)}  MAE={mae_np(y_val_subj,yhat_val_subj):.3f}  RMSE={rmse_np(y_val_subj,yhat_val_subj):.3f}  r={pearson_r_safe_np(y_val_subj,yhat_val_subj):.3f}  R2={r2_np(y_val_subj,yhat_val_subj):.3f}")
    print(f"[TEST/raw] (subject-{args.subj_agg}) Nsubj={len(subj_te)}  MAE={mae_np(y_te_subj,yhat_te_subj):.3f}  RMSE={rmse_np(y_te_subj,yhat_te_subj):.3f}  r={pearson_r_safe_np(y_te_subj,yhat_te_subj):.3f}  R2={r2_np(y_te_subj,yhat_te_subj):.3f}")

    calibrator_subj = LinearCalibrator()
    calibrator_subj.fit(yhat_val_subj, y_val_subj)
    print(f"[Calibrator] (subject) {calibrator_subj}")

    yhat_te_subj_cal = calibrator_subj.transform(yhat_te_subj)
    a_subj, b_subj = calibrator_subj.a, calibrator_subj.b
    
    print(f"[TEST/cal] (subject-{args.subj_agg}) Nsubj={len(subj_te)}  MAE={mae_np(y_te_subj,yhat_te_subj_cal):.3f}  RMSE={rmse_np(y_te_subj,yhat_te_subj_cal):.3f}  r={pearson_r_safe_np(y_te_subj,yhat_te_subj_cal):.3f}  R2={r2_np(y_te_subj,yhat_te_subj_cal):.3f}  (a={a_subj:.4f}, b={b_subj:.4f})")

    # ---- 4. 保存所有结果 ----
    
    # ---- Save record-level outputs ----
    pred_rec = Path(args.out_dir) / f"pred_test_bp_{target_col}_{args.modality}_{args.reg_loss}_{args.constrain}_records.npz"
    np.savez(pred_rec,
             ssoid=sids_te_rec,
             y_true=y_te_rec.astype(np.float32),
             y_pred=yhat_te_rec.astype(np.float32),
             y_pred_cal=yhat_te_rec_cal.astype(np.float32),
             target_col=np.array(target_col),
             modality=np.array(args.modality),
             a=float(a_rec), b=float(b_rec))
    
    metrics_rec = Path(args.out_dir) / f"metrics_test_bp_{target_col}_{args.modality}_{args.reg_loss}_{args.constrain}_records.json"
    with open(metrics_rec, "w") as f:
        json.dump({
            "target_col": target_col,
            "MAE_raw": mae_np(y_te_rec, yhat_te_rec),
            "RMSE_raw": rmse_np(y_te_rec, yhat_te_rec),
            "r_raw": pearson_r_safe_np(y_te_rec, yhat_te_rec),
            "R2_raw": r2_np(y_te_rec, yhat_te_rec),
            "MAE_cal": mae_np(y_te_rec, yhat_te_rec_cal),
            "RMSE_cal": rmse_np(y_te_rec, yhat_te_rec_cal),
            "r_cal": pearson_r_safe_np(y_te_rec, yhat_te_rec_cal),
            "R2_cal": r2_np(y_te_rec, yhat_te_rec_cal),
            "N_records": int(len(y_te_rec)),
            "a": float(a_rec), "b": float(b_rec),
            "modality": args.modality,
            "resample_ecg": args.resample_ecg
        }, f, indent=2)
    print(f"[saved] record-level predictions -> {pred_rec}")
    print(f"[saved] record-level metrics     -> {metrics_rec}")

    # ---- Save subject-level outputs ----
    pred_subj = Path(args.out_dir) / f"pred_test_bp_{target_col}_{args.modality}_{args.reg_loss}_{args.constrain}_subjects.npz"
    np.savez(pred_subj,
             subject=subj_te.astype(object),
             n_records=n_te.astype(np.int32),
             y_true=y_te_subj.astype(np.float32),
             y_pred=yhat_te_subj.astype(np.float32),
             y_pred_cal=yhat_te_subj_cal.astype(np.float32),
             target_col=np.array(target_col),
             modality=np.array(args.modality),
             a=float(a_subj), b=float(b_subj),
             agg=np.array(args.subj_agg))
    
    metrics_subj = Path(args.out_dir) / f"metrics_test_bp_{target_col}_{args.modality}_{args.reg_loss}_{args.constrain}_subjects.json"
    with open(metrics_subj, "w") as f:
        json.dump({
            "target_col": target_col,
            "MAE_raw": mae_np(y_te_subj, yhat_te_subj),
            "RMSE_raw": rmse_np(y_te_subj, yhat_te_subj),
            "r_raw": pearson_r_safe_np(y_te_subj, yhat_te_subj),
            "R2_raw": r2_np(y_te_subj, yhat_te_subj),
            "MAE_cal": mae_np(y_te_subj, yhat_te_subj_cal),
            "RMSE_cal": rmse_np(y_te_subj, yhat_te_subj_cal),
            "r_cal": pearson_r_safe_np(y_te_subj, yhat_te_subj_cal),
            "R2_cal": r2_np(y_te_subj, yhat_te_subj_cal),
            "N_subjects": int(len(subj_te)),
            "a": float(a_subj), "b": float(b_subj),
            "agg": args.subj_agg,
            "modality": args.modality,
            "resample_ecg": args.resample_ecg
        }, f, indent=2)
    print(f"[saved] subject-level predictions -> {pred_subj}")
    print(f"[saved] subject-level metrics     -> {metrics_subj}")

    # ---- 生成测试集个体小表 CSV ----
    df_all = pd.read_csv(args.labels_csv)
    df_all["ssoid"] = df_all["ssoid"].astype(str)
    df_all["subject"] = df_all["ssoid"].apply(subject_id_from_ssoid)

    df_all_te = df_all[df_all["subject"].isin(set(subj_te))].copy()

    num_agg = "median" if args.subj_agg == "median" else "mean"

    agg_dict = {}
    for col in df_all_te.columns:
        if col in ("ssoid", "subject"): 
            continue
        if pd.api.types.is_numeric_dtype(df_all_te[col]):
            agg_dict[col] = num_agg
        else:
            agg_dict[col] = "first"

    df_labels_by_subject = df_all_te.groupby("subject", as_index=False).agg(agg_dict)

    df_pred_subj = pd.DataFrame({
        "subject": subj_te,
        "n_records": n_te,
        f"predicted_{target_col}_raw": yhat_te_subj,
        f"predicted_{target_col}_cal": yhat_te_subj_cal,
        f"{target_col}_agg": y_te_subj,
    })
    df_pred_subj[f"{target_col}_gap"] = df_pred_subj[f"predicted_{target_col}_raw"] - df_pred_subj[f"{target_col}_agg"]
    df_pred_subj[f"{target_col}_gap_cal"] = df_pred_subj[f"predicted_{target_col}_cal"] - df_pred_subj[f"{target_col}_agg"]

    df_subject_summary = pd.merge(df_labels_by_subject, df_pred_subj, on="subject", how="inner")

    table_path = Path(args.out_dir) / f"subject_summary_test_bp_{target_col}_{args.modality}_{args.reg_loss}_{args.constrain}_agg-{args.subj_agg}.csv"
    df_subject_summary.to_csv(table_path, index=False)
    print(f"[saved] subject summary table -> {table_path}  (rows={len(df_subject_summary)})")

if __name__ == "__main__":
    main()