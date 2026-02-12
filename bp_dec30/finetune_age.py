#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主脚本：年龄预测微调 (Main Script: Age Finetuning)

负责：
1. 参数解析
2. 数据准备与划分
3. 模型、优化器、损失函数设置
4. 调用 engine.py 执行训练与评估
5. 保存最终结果 (模型, 指标, 预测)
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
from dataset import LabeledECGPPGDataset
from losses import MAE_PearsonLoss
from utils import (set_seed, subject_id_from_ssoid, aggregate_by_subject_prefix, 
                   mae_np, rmse_np, pearson_r_safe_np, r2_np, 
                   LinearCalibrator)
from engine import train_one_epoch, evaluate, evaluate_with_ids
# =========================================

# ============== (可选) 分布先验对齐 ==============
try:
    from dist_loss import DistributionAlignmentLoss, build_kde_prior
    _HAS_DIST = True
except Exception:
    DistributionAlignmentLoss = None
    build_kde_prior = None
    _HAS_DIST = False
# ===============================================

# ===================== Default paths (your setup) =====================
DEFAULT_NPZ_DIR   = "/home/notebook/data/personal/S9061270/age_ready/labeled_zscore"
DEFAULT_LABELS    = "/home/notebook/data/personal/S9061270/age_ready/labeled_labels.csv"
DEFAULT_PRETRAIN  = "/home/notebook/data/personal/S9061270/model/clip_ppg_ecg_founder/pretrain_temp/best.pth"
DEFAULT_OUT_DIR   = "/home/notebook/data/personal/S9061270/model/clip_ppg_ecg_founder/finetune_age_temp_251116"

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="Age finetune with target-standardization & bounded outputs + subject-level metrics & summary table.")
    ap.add_argument("--npz_dir",   default=DEFAULT_NPZ_DIR)
    ap.add_argument("--labels_csv",default=DEFAULT_LABELS)
    ap.add_argument("--pretrain",  default=DEFAULT_PRETRAIN)
    ap.add_argument("--out_dir",   default=DEFAULT_OUT_DIR)
    ap.add_argument("--modality",  choices=["ecg","ppg","both"], default="both")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)

    # 优化 & 冻结策略
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze encoders+projectors (linear probe).")
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head",     type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # 损失设置
    ap.add_argument("--reg_loss", choices=["mse","huber","mae_pearson","mse+dist"], default="mae_pearson")
    ap.add_argument("--alpha_corr", type=float, default=0.3, help="相关性辅助项权重(1-corr)，'mae_pearson'时忽略")

    # MAE+Pearson 参数
    ap.add_argument("--maepearson_alpha", type=float, default=0.5, help="权重: (1-r) 项")
    ap.add_argument("--maepearson_beta",  type=float, default=0.5, help="权重: MAE 项")

    # DistLoss 参数
    ap.add_argument("--lambda_dist", type=float, default=1.0, help="DistLoss 权重（仅 'mse+dist' 有效）")
    ap.add_argument("--dist_binsz", type=float, default=1.0, help="年龄 bin 宽度（岁）")
    ap.add_argument("--dist_sigma", type=float, default=2.0, help="KDE/软分配的高斯核宽（岁）")

    # 约束设置
    ap.add_argument("--constrain", choices=["none","tanh","sigmoid","clip"], default="tanh")
    ap.add_argument("--y_min", type=float, default=20.0)
    ap.add_argument("--y_max", type=float, default=80.0)

    # 个体聚合设置
    ap.add_argument("--subj_agg", choices=["mean","max","median"], default="mean",
                      help="个体级别聚合方式（预测与真实都用这个：mean / median / max）")

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
    with open(out_dir/"args_age_v6.json","w") as f: json.dump(vars(args), f, indent=2)

    # ----- load labels & align npz -----
    df = pd.read_csv(args.labels_csv)
    if "ssoid" not in df or "age" not in df:
        raise RuntimeError("labels_csv must contain columns: ssoid, age")
    df = df[["ssoid","age"]].copy()
    df["ssoid"]=df["ssoid"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[df["age"].notna()]
    have = set(p.stem for p in Path(args.npz_dir).glob("*.npz"))
    df = df[df["ssoid"].isin(have)].reset_index(drop=True)

    # subject-wise split (7:1:2)
    df["subject"] = df["ssoid"].apply(subject_id_from_ssoid)
    subjects = df["subject"].unique().tolist()
    rng = np.random.default_rng(args.seed); rng.shuffle(subjects)
    n=len(subjects); n_tr=int(0.7*n); n_va=int(0.1*n)
    s_tr=set(subjects[:n_tr]); s_va=set(subjects[n_tr:n_tr+n_va]); s_te=set(subjects[n_tr+n_va:])
    df_tr=df[df["subject"].isin(s_tr)][["ssoid","age"]].copy()
    df_va=df[df["subject"].isin(s_va)][["ssoid","age"]].copy()
    df_te=df[df["subject"].isin(s_te)][["ssoid","age"]].copy()
    print(f"[split] train={len(df_tr)}  val={len(df_va)}  test={len(df_te)}  (subjects: {len(s_tr)}/{len(s_va)}/{len(s_te)})")

    # 目标标准化参数（基于训练集）
    mu   = float(df_tr["age"].mean())
    sigma= float(df_tr["age"].std(ddof=0))
    if sigma < 1e-6: sigma = 1.0
    y_min = float(args.y_min); y_max = float(args.y_max)
    print(f"[target stats] mu={mu:.3f}  sigma={sigma:.3f}  y_min={y_min}  y_max={y_max}")

    # datasets / loaders
    ds_tr = LabeledECGPPGDataset(df_tr, args.npz_dir)
    ds_va = LabeledECGPPGDataset(df_va, args.npz_dir)
    ds_te = LabeledECGPPGDataset(df_te, args.npz_dir)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # model
    model = AgeModel(modality=args.modality, proj_hidden=0).to(device)
    model.load_from_pretrain(args.pretrain, device=device)

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

    # DistLoss（仅在 'mse+dist'）
    dist_criterion = None
    if args.reg_loss == "mse+dist":
        if not _HAS_DIST:
            raise RuntimeError("reg_loss=mse+dist 但未找到 dist_loss.py 或 build_kde_prior")
        train_ages_np = df_tr["age"].to_numpy(dtype=np.float64)
        bin_centers, prior_probs = build_kde_prior(
            train_ages=train_ages_np,
            y_min=y_min, y_max=y_max,
            bin_width=args.dist_binsz,
            sigma=args.dist_sigma
        )
        dist_criterion = DistributionAlignmentLoss(
            bin_centers=bin_centers,
            prior_probs=prior_probs,
            sigma=args.dist_sigma,
            inv_weight=True
        ).to(device)
        print(f"[DistLoss] bins={len(bin_centers)}  sigma={args.dist_sigma}  lambda={args.lambda_dist}")

    # MAE_Pearson
    maepearson_criterion = None
    if args.reg_loss == "mae_pearson":
        maepearson_criterion = MAE_PearsonLoss(
            alpha=args.maepearson_alpha,
            beta=args.maepearson_beta
        ).to(device)

    # training loop with early stopping on val MAE
    best_mae = float("inf"); best_ep = -1; patience_cnt=0
    ckpt_name = f"age_{args.modality}_{args.reg_loss}_{args.constrain}_best.pth"

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(
            model, dl_tr, optimizer, scaler, device, args.modality,
            args.reg_loss, mu, sigma, y_min, y_max, args.constrain, args.alpha_corr,
            maepearson_criterion=maepearson_criterion,
            dist_criterion=dist_criterion,
            lambda_dist=args.lambda_dist
        )
        val_loss, val_mae, val_rmse, r, r2, _, _ = evaluate(
            model, dl_va, device, args.modality, mu, sigma, y_min, y_max, args.constrain
        )
        print(f"[E{ep}] train_loss={tr_loss:.6f} | val_loss={val_loss:.6f} val_MAE={val_mae:.3f} val_RMSE={val_rmse:.3f} r={r:.3f} R2={r2:.3f}")

        if val_mae < best_mae - 1e-6:
            best_mae = val_mae; best_ep = ep; patience_cnt=0
            ckpt = {"epoch": ep, "model": model.state_dict(),
                    "val_mae": float(val_mae), "modality": args.modality,
                    "mu": mu, "sigma": sigma, "y_min": y_min, "y_max": y_max,
                    "constrain": args.constrain}
            ckpt_path = Path(args.out_dir)/ckpt_name
            torch.save(ckpt, str(ckpt_path))
            print(f"[best] val_MAE={best_mae:.3f} @epoch{ep} | saved: {ckpt_path}")
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
    # 在 验证集(VAL) 上 拟合(fit) 校准器
    calibrator_rec = LinearCalibrator()
    calibrator_rec.fit(yhat_val_rec, y_val_rec)
    print(f"[Calibrator] (record) {calibrator_rec}")
    
    # 在 测试集(TEST) 上 应用(transform) 校准
    yhat_te_rec_cal = calibrator_rec.transform(yhat_te_rec)
    a_rec, b_rec = calibrator_rec.a, calibrator_rec.b

    print(f"[TEST/raw] (record) ({args.modality})  MAE={mae_np(y_te_rec,yhat_te_rec):.3f}  RMSE={rmse_np(y_te_rec,yhat_te_rec):.3f}  r={pearson_r_safe_np(y_te_rec,yhat_te_rec):.3f}  R2={r2_np(y_te_rec,yhat_te_rec):.3f}")
    print(f"[TEST/cal] (record) ({args.modality})  MAE={mae_np(y_te_rec,yhat_te_rec_cal):.3f}  RMSE={rmse_np(y_te_rec,yhat_te_rec_cal):.3f}  r={pearson_r_safe_np(y_te_rec,yhat_te_rec_cal):.3f}  R2={r2_np(y_te_rec,yhat_te_rec_cal):.3f}  (a={a_rec:.4f}, b={b_rec:.4f})")

    # ---- 3. 个体级 (Subject-level) 聚合与校准 ----
    # 3.1. 聚合 VAL 和 TEST
    subj_val, y_val_subj, yhat_val_subj, n_val = aggregate_by_subject_prefix(y_val_rec, yhat_val_rec, sids_val_rec, agg=args.subj_agg)
    subj_te, y_te_subj, yhat_te_subj, n_te = aggregate_by_subject_prefix(y_te_rec, yhat_te_rec, sids_te_rec, agg=args.subj_agg)
    
    print(f"[VAL/raw] (subject-{args.subj_agg}) Nsubj={len(subj_val)}  MAE={mae_np(y_val_subj,yhat_val_subj):.3f}  RMSE={rmse_np(y_val_subj,yhat_val_subj):.3f}  r={pearson_r_safe_np(y_val_subj,yhat_val_subj):.3f}  R2={r2_np(y_val_subj,yhat_val_subj):.3f}")
    print(f"[TEST/raw] (subject-{args.subj_agg}) Nsubj={len(subj_te)}  MAE={mae_np(y_te_subj,yhat_te_subj):.3f}  RMSE={rmse_np(y_te_subj,yhat_te_subj):.3f}  r={pearson_r_safe_np(y_te_subj,yhat_te_subj):.3f}  R2={r2_np(y_te_subj,yhat_te_subj):.3f}")

    # 3.2. 在 聚合后的验证集(VAL-subject) 上 拟合(fit) 校准器
    calibrator_subj = LinearCalibrator()
    calibrator_subj.fit(yhat_val_subj, y_val_subj)
    print(f"[Calibrator] (subject) {calibrator_subj}")

    # 3.3. 在 聚合后的测试集(TEST-subject) 上 应用(transform) 校准
    yhat_te_subj_cal = calibrator_subj.transform(yhat_te_subj)
    a_subj, b_subj = calibrator_subj.a, calibrator_subj.b
    
    print(f"[TEST/cal] (subject-{args.subj_agg}) Nsubj={len(subj_te)}  MAE={mae_np(y_te_subj,yhat_te_subj_cal):.3f}  RMSE={rmse_np(y_te_subj,yhat_te_subj_cal):.3f}  r={pearson_r_safe_np(y_te_subj,yhat_te_subj_cal):.3f}  R2={r2_np(y_te_subj,yhat_te_subj_cal):.3f}  (a={a_subj:.4f}, b={b_subj:.4f})")

    # ---- 4. 保存所有结果 ----
    
    # ---- Save record-level outputs ----
    pred_rec = Path(args.out_dir)/f"pred_test_age_{args.modality}_{args.reg_loss}_{args.constrain}_records.npz"
    np.savez(pred_rec,
             ssoid=sids_te_rec,
             ChronologicalAge=y_te_rec.astype(np.float32),
             VascularAge=yhat_te_rec.astype(np.float32),
             VascularAge_cal=yhat_te_rec_cal.astype(np.float32), # <-- 使用校准后
             modality=np.array(args.modality),
             a=float(a_rec), b=float(b_rec)) # <-- 保存校准参数
    metrics_rec = Path(args.out_dir)/f"metrics_test_age_{args.modality}_{args.reg_loss}_{args.constrain}_records.json"
    with open(metrics_rec,"w") as f:
        json.dump({
            "MAE_raw": mae_np(y_te_rec,yhat_te_rec),
            "RMSE_raw": rmse_np(y_te_rec,yhat_te_rec),
            "r_raw": pearson_r_safe_np(y_te_rec,yhat_te_rec),
            "R2_raw": r2_np(y_te_rec,yhat_te_rec),
            "MAE_cal": mae_np(y_te_rec,yhat_te_rec_cal),
            "RMSE_cal": rmse_np(y_te_rec,yhat_te_rec_cal),
            "r_cal": pearson_r_safe_np(y_te_rec,yhat_te_rec_cal),
            "R2_cal": r2_np(y_te_rec,yhat_te_rec_cal),
            "N_records": int(len(y_te_rec)),
            "a": float(a_rec), "b": float(b_rec),
            "modality": args.modality
        }, f, indent=2)
    print(f"[saved] record-level predictions -> {pred_rec}")
    print(f"[saved] record-level metrics     -> {metrics_rec}")

    # ---- Save subject-level outputs ----
    pred_subj = Path(args.out_dir)/f"pred_test_age_{args.modality}_{args.reg_loss}_{args.constrain}_subjects.npz"
    np.savez(pred_subj,
             subject=subj_te.astype(object),
             n_records=n_te.astype(np.int32),
             ChronologicalAge=y_te_subj.astype(np.float32),
             VascularAge=yhat_te_subj.astype(np.float32),
             VascularAge_cal=yhat_te_subj_cal.astype(np.float32), # <-- 使用校准后
             modality=np.array(args.modality),
             a=float(a_subj), b=float(b_subj), # <-- 保存校准参数
             agg=np.array(args.subj_agg))
    metrics_subj = Path(args.out_dir)/f"metrics_test_age_{args.modality}_{args.reg_loss}_{args.constrain}_subjects.json"
    with open(metrics_subj,"w") as f:
        json.dump({
            "MAE_raw": mae_np(y_te_subj,yhat_te_subj),
            "RMSE_raw": rmse_np(y_te_subj,yhat_te_subj),
            "r_raw": pearson_r_safe_np(y_te_subj,yhat_te_subj),
            "R2_raw": r2_np(y_te_subj,yhat_te_subj),
            "MAE_cal": mae_np(y_te_subj,yhat_te_subj_cal),
            "RMSE_cal": rmse_np(y_te_subj,yhat_te_subj_cal),
            "r_cal": pearson_r_safe_np(y_te_subj,yhat_te_subj_cal),
            "R2_cal": r2_np(y_te_subj,yhat_te_subj_cal),
            "N_subjects": int(len(subj_te)),
            "a": float(a_subj), "b": float(b_subj),
            "agg": args.subj_agg,
            "modality": args.modality
        }, f, indent=2)
    print(f"[saved] subject-level predictions -> {pred_subj}")
    print(f"[saved] subject-level metrics     -> {metrics_subj}")

    # ---- NEW: 生成“测试集个体小表” CSV（合并 labels 的原有列 + 预测 + gap）----
    # 读全量 labels
    df_all = pd.read_csv(args.labels_csv)
    df_all["ssoid"] = df_all["ssoid"].astype(str)
    df_all["subject"] = df_all["ssoid"].apply(subject_id_from_ssoid)

    # 仅保留测试集 subject
    df_all_te = df_all[df_all["subject"].isin(set(subj_te))].copy()

    # 数值列聚合策略：与 subj_agg 对齐（median => median，否则 mean）
    num_agg = "median" if args.subj_agg == "median" else "mean"

    # 为了让“非数值列”也能聚合，统一构造 agg 字典
    agg_dict = {}
    for col in df_all_te.columns:
        if col in ("ssoid","subject"): 
            continue
        if pd.api.types.is_numeric_dtype(df_all_te[col]):
            agg_dict[col] = num_agg
        else:
            # 对于非数值列（如 'gender', 'group'），取第一个
            agg_dict[col] = "first"

    df_labels_by_subject = df_all_te.groupby("subject", as_index=False).agg(agg_dict)

    # 预测 side（按 subject 汇总后的 y/yhat 已有） -> 做成 DataFrame
    df_pred_subj = pd.DataFrame({
        "subject": subj_te,
        "n_records": n_te,
        "predicted_age_raw": yhat_te_subj,
        "predicted_age_cal": yhat_te_subj_cal, # <-- 使用校准后
        "age_agg": y_te_subj,  # 用同一聚合得到的真实年龄
    })
    # age_gap 用 raw
    df_pred_subj["age_gap"] = df_pred_subj["predicted_age_raw"] - df_pred_subj["age_agg"]
    # 也可以增加一个校准后的 gap
    df_pred_subj["age_gap_cal"] = df_pred_subj["predicted_age_cal"] - df_pred_subj["age_agg"]


    # 合并 labels 聚合表 + 预测表（保留 labels 的所有原始列的聚合版本）
    df_subject_summary = pd.merge(df_labels_by_subject, df_pred_subj, on="subject", how="inner")

    # 输出
    table_path = Path(args.out_dir) / f"subject_summary_test_{args.modality}_{args.reg_loss}_{args.constrain}_agg-{args.subj_agg}.csv"
    df_subject_summary.to_csv(table_path, index=False)
    print(f"[saved] subject summary table -> {table_path}  (rows={len(df_subject_summary)})")

if __name__ == "__main__":
    main()