#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BP finetune + evaluation (supports BOTH multi-target and single-target).

Key features:
- Single-target experiment via --target_col (overrides --targets)
- Eval-only mode via --eval_only to (re)generate metrics JSON from best.pth
- Detailed metrics: MAE/RMSE/Pearson r/R2 for raw + calibrated
- Saves: metrics_test_bp_<tag>_records.json (and also val metrics)

Important fixes for "prediction collapsed to mean / range compressed":
1) Do NOT write calibration into model state during training/eval (no model.set_calibration).
   Calibration is computed externally only, so "raw" stays raw.
2) Eval-only now rebuilds datasets/loaders AFTER reading ckpt overrides (targets/use_raw/modality).
3) Adds strict shape checks to prevent silent broadcasting / squeezing issues.
4) Default early stop is on raw MAE (you can still choose cal, but raw is safer).
python /home/youliang/youliang_data2/bp/bp_recode_v1/finetune_bp.py \
  --target_col right_arm_sbp --modality both \
  --early_on raw \
  --out_dir /home/youliang/youliang_data2/bp/bp_recode_v1/test_run

New loss:
- mae_pearson: combines MAE + (1 - Pearson correlation) using bp_losses.MAE_PearsonLoss
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import set_seed
from backbones import PPGEncoderCLIP, ECGEncoderCLIP
from bp_dataset import BPLabeledDataset, BP_COLS_DEFAULT
from bp_backbones import BPModel
from bp_engine import train_one_epoch_bp  # keep your existing training loop
from bp_losses import MAE_PearsonLoss


DEF_NPZ_DIR  = "/home/youliang/youliang_data2/bp/bp_recode_v1/output/npz"
DEF_LABELS   = "/home/youliang/youliang_data2/bp/bp_recode_v1/output/labels.csv"
DEF_CLIP     = "/home/youliang/youliang_data2/bp/bp_recode_v1/clip_finetune_out"
DEF_OUT_DIR  = "/home/youliang/youliang_data2/bp/bp_recode_v1/bp_finetune_out"


ARM_KEYS = [
    "right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp",
    "left_arm_sbp",  "left_arm_mbp",  "left_arm_dbp",  "left_arm_pp",
]


def _to_device_batch(batch, device):
    # dataset may return (ecg,ppg,y) or (ecg,ppg,y,ssoid)
    if len(batch) == 4:
        ecg, ppg, y, ssoid = batch
    else:
        ecg, ppg, y = batch
        ssoid = None
    ecg = ecg.to(device, non_blocking=True)
    ppg = ppg.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return ecg, ppg, y, ssoid


def _as_tensors(mu: np.ndarray, sigma: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device).view(1, -1)
    sg_t = torch.tensor(sigma, dtype=torch.float32, device=device).view(1, -1)
    return mu_t, sg_t


def _fit_affine_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-target least squares fit: y_true ≈ a * y_pred + b
    Returns a,b with shape (K,).
    """
    assert y_true.shape == y_pred.shape and y_true.ndim == 2
    K = y_true.shape[1]
    a = np.ones(K, dtype=np.float64)
    b = np.zeros(K, dtype=np.float64)

    for k in range(K):
        x = y_pred[:, k].astype(np.float64)
        y = y_true[:, k].astype(np.float64)

        vx = np.var(x)
        if vx < 1e-12:
            a[k] = 1.0
            b[k] = float(np.mean(y) - np.mean(x))
            continue

        cov = np.mean((x - x.mean()) * (y - y.mean()))
        a[k] = cov / vx
        b[k] = float(y.mean() - a[k] * x.mean())

    return a, b


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err * err, axis=0))
    return mae, rmse


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # returns r per target (K,)
    assert y_true.shape == y_pred.shape and y_true.ndim == 2
    K = y_true.shape[1]
    out = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        yt = y_true[:, k].astype(np.float64)
        yp = y_pred[:, k].astype(np.float64)
        yt_std = np.std(yt)
        yp_std = np.std(yp)
        if yt_std < 1e-12 or yp_std < 1e-12:
            out[k] = np.nan
        else:
            out[k] = float(np.corrcoef(yt, yp)[0, 1])
    return out


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # returns R2 per target (K,)
    assert y_true.shape == y_pred.shape and y_true.ndim == 2
    K = y_true.shape[1]
    out = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        yt = y_true[:, k].astype(np.float64)
        yp = y_pred[:, k].astype(np.float64)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        if ss_tot < 1e-12:
            out[k] = np.nan
        else:
            out[k] = float(1.0 - ss_res / ss_tot)
    return out


@torch.no_grad()
def _infer_on_loader(
    model,
    loader,
    device,
    mu: np.ndarray,
    sigma: np.ndarray,
    loss_type: str,
    loss_fn: Optional[torch.nn.Module] = None,
) -> Tuple[float, np.ndarray, np.ndarray, List[Optional[str]]]:
    """
    Run inference on loader.
    Returns:
      loss_z (float),
      y_true_raw (N,K),
      y_pred_raw (N,K),
      ssoid_list (len N; may contain None)
    """
    model.eval()
    mu_t, sg_t = _as_tensors(mu, sigma, device)

    tot = 0
    tot_loss = 0.0
    y_all = []
    yhat_all = []
    ssoids: List[Optional[str]] = []

    for batch in loader:
        ecg, ppg, y_raw, ssoid = _to_device_batch(batch, device)
        y_z = (y_raw - mu_t) / sg_t

        with torch.cuda.amp.autocast(enabled=False):
            yhat_z = model(ecg, ppg)

            # ENGLISH comments: strict shape checks to prevent silent broadcasting / squeeze bugs
            if yhat_z.ndim != 2 or y_z.ndim != 2:
                raise RuntimeError(f"Bad dims: yhat_z={tuple(yhat_z.shape)} y_z={tuple(y_z.shape)}")
            if yhat_z.shape != y_z.shape:
                raise RuntimeError(
                    f"Shape mismatch: yhat_z={tuple(yhat_z.shape)} vs y_z={tuple(y_z.shape)}. "
                    "This can cause silent broadcasting and collapsed predictions."
                )

            if loss_type == "mse":
                loss = F.mse_loss(yhat_z, y_z)
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(yhat_z, y_z, beta=1.0)
            elif loss_type == "mae_pearson":
                if loss_fn is None:
                    raise RuntimeError("loss_fn must be provided for mae_pearson")
                loss = loss_fn(yhat_z, y_z)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        bs = y_raw.size(0)
        tot += bs
        tot_loss += float(loss.item()) * bs

        yhat_raw = (yhat_z * sg_t + mu_t).detach().cpu().numpy()
        y_all.append(y_raw.detach().cpu().numpy())
        yhat_all.append(yhat_raw)

        if ssoid is None:
            ssoids.extend([None] * bs)
        else:
            if isinstance(ssoid, (list, tuple)):
                ssoids.extend([str(x) for x in ssoid])
            else:
                ssoids.extend([str(ssoid)] * bs)

    y_true = np.concatenate(y_all, axis=0).astype(np.float64)
    y_pred = np.concatenate(yhat_all, axis=0).astype(np.float64)
    loss_z = tot_loss / max(1, tot)
    return loss_z, y_true, y_pred, ssoids


def _build_metrics_dict(
    loss_z: float,
    targets: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    mae, rmse = _mae_rmse(y_true, y_pred)
    rr = _pearson_r(y_true, y_pred)
    r2 = _r2_score(y_true, y_pred)

    out: Dict[str, object] = {}
    out["loss"] = float(loss_z)
    out["N_records"] = int(y_true.shape[0])
    out["targets"] = list(targets)

    out["mae_per_target"] = {t: float(v) for t, v in zip(targets, mae)}
    out["rmse_per_target"] = {t: float(v) for t, v in zip(targets, rmse)}
    out["r_per_target"] = {t: (None if np.isnan(v) else float(v)) for t, v in zip(targets, rr)}
    out["R2_per_target"] = {t: (None if np.isnan(v) else float(v)) for t, v in zip(targets, r2)}

    out["MAE"] = float(np.nanmean(mae))
    out["RMSE"] = float(np.nanmean(rmse))
    out["r"] = None if np.isnan(np.nanmean(rr)) else float(np.nanmean(rr))
    out["R2"] = None if np.isnan(np.nanmean(r2)) else float(np.nanmean(r2))

    right_maes = [out["mae_per_target"][k] for k in ARM_KEYS[:4] if k in out["mae_per_target"]]
    left_maes  = [out["mae_per_target"][k] for k in ARM_KEYS[4:] if k in out["mae_per_target"]]
    if right_maes:
        out["right_arm_mae_mean"] = float(np.mean(right_maes))
    if left_maes:
        out["left_arm_mae_mean"] = float(np.mean(left_maes))

    return out


def _print_metrics(prefix: str, m: Dict[str, object], targets: List[str], detailed: bool = True):
    line = (
        f"{prefix} loss={m.get('loss', float('nan')):.6f} "
        f"| MAE={m.get('MAE', float('nan')):.3f} RMSE={m.get('RMSE', float('nan')):.3f} "
        f"| r={m.get('r', None)} R2={m.get('R2', None)}"
    )
    if "right_arm_mae_mean" in m or "left_arm_mae_mean" in m:
        line += f" | right_mae={m.get('right_arm_mae_mean', None)} left_mae={m.get('left_arm_mae_mean', None)}"
    print(line)

    if not detailed:
        return

    mae_pt = m.get("mae_per_target", {})
    rmse_pt = m.get("rmse_per_target", {})
    r_pt = m.get("r_per_target", {})
    r2_pt = m.get("R2_per_target", {})
    for t in targets:
        if t in mae_pt:
            print(
                f"  {t:>14s}  "
                f"MAE={mae_pt[t]:.3f}  RMSE={rmse_pt.get(t, float('nan')):.3f}  "
                f"r={r_pt.get(t, None)}  R2={r2_pt.get(t, None)}"
            )


def _save_metrics_json(out_path: Path, payload: Dict[str, object]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] {out_path}")


def _make_records(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_cal: Optional[np.ndarray],
    ssoids: List[Optional[str]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    N = y_true.shape[0]
    for i in range(N):
        rec = {
            "ssoid": None if ssoids[i] is None else str(ssoids[i]),
            "y_true": y_true[i].astype(float).tolist(),
            "y_pred_raw": y_pred_raw[i].astype(float).tolist(),
        }
        if y_pred_cal is not None:
            rec["y_pred_cal"] = y_pred_cal[i].astype(float).tolist()
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser("BP finetune (supports single-target + detailed eval json)")

    ap.add_argument("--npz_dir", default=DEF_NPZ_DIR)
    ap.add_argument("--labels_csv", default=DEF_LABELS)
    ap.add_argument("--clip_ckpt", default=DEF_CLIP, help="Path to CLIP finetune dir or best.pth (only used in training).")
    ap.add_argument("--out_dir", default=DEF_OUT_DIR)

    ap.add_argument("--targets", nargs="*", default=BP_COLS_DEFAULT, help="BP target columns (multi-output).")
    ap.add_argument("--target_col", type=str, default=None, help="Single target column (overrides --targets).")

    ap.add_argument("--use_raw", action="store_true", help="Use ecg_raw/ppg_raw instead of ecg/ppg (zscore).")
    ap.add_argument("--modality", choices=["ecg", "ppg", "both"], default="both")

    ap.add_argument("--eval_only", action="store_true", help="Only run eval from best.pth; no training.")
    ap.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path for eval_only (default: out_dir/best.pth).")
    ap.add_argument("--save_records", action="store_true", help="Save per-record predictions in metrics json (default: True).")
    ap.add_argument("--no_save_records", action="store_true", help="Disable per-record saving.")

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--gpu", type=int, default=0)

    # New loss choice added
    ap.add_argument("--loss", choices=["mse", "huber", "mae_pearson"], default="huber")
    ap.add_argument("--loss_alpha", type=float, default=0.5, help="alpha for mae_pearson loss")
    ap.add_argument("--loss_beta", type=float, default=0.5, help="beta for mae_pearson loss")

    ap.add_argument("--freeze_backbone", action="store_true", help="Linear probe only (freeze encoders).")
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--head_hidden", type=int, default=256)

    ap.add_argument("--patience", type=int, default=5)
    # Default raw is safer to avoid selecting collapsed-range ckpt via per-epoch val calibration
    ap.add_argument("--early_on", choices=["raw", "cal"], default="raw",
                    help="Early stop on val MAE (raw) or val MAE (calibrated).")

    args = ap.parse_args()

    # Default behavior: save records unless explicitly disabled
    if args.no_save_records:
        save_records = False
    else:
        save_records = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(f"cuda:{args.gpu}")
    print(f"device={device} | name={torch.cuda.get_device_name(args.gpu)}")

    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve targets (single-target overrides)
    if args.target_col is not None and str(args.target_col).strip() != "":
        targets: List[str] = [str(args.target_col).strip()]
    else:
        targets = list(args.targets)
    print(f"[targets] K={len(targets)} -> {targets}")
    is_single = (len(targets) == 1)

    # Load labels & split
    df = pd.read_csv(args.labels_csv)
    if "ssoid" not in df.columns:
        raise RuntimeError("labels_csv must have column: ssoid")
    if "split" not in df.columns:
        raise RuntimeError("labels_csv has no 'split' column")

    df_tr = df[df["split"] == "train"].copy()
    df_va = df[df["split"] == "val"].copy()
    df_te = df[df["split"] == "test"].copy()

    # Compute train mu/sigma for current targets (used in training; eval_only may override from ckpt)
    y_tr = df_tr[targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(y_tr).any():
        raise RuntimeError("NaN in training targets (check labels.csv)")
    mu = y_tr.mean(axis=0)
    sigma = y_tr.std(axis=0, ddof=0)
    sigma[sigma < 1e-6] = 1.0

    print("[target stats] per-dim mu/sigma:")
    for t, m, s in zip(targets, mu, sigma):
        print(f"  {t:>14s}  mu={m:.3f}  sigma={s:.3f}")

    # Save args for traceability
    with open(out_dir / "args_bp.json", "w") as f:
        json.dump({
            **vars(args),
            "resolved_targets": targets,
            "resolved_is_single": is_single,
            "resolved_save_records": save_records,
        }, f, indent=2)

    # -------------------------
    # Eval-only mode
    # -------------------------
    if args.eval_only:
        ckpt_path = Path(args.ckpt_path) if args.ckpt_path is not None else (out_dir / "best.pth")
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

        st = torch.load(str(ckpt_path), map_location="cpu")

        # Prefer ckpt config (targets/modality/use_raw/loss/head_hidden)
        ckpt_targets = st.get("targets", None)
        if ckpt_targets is not None and isinstance(ckpt_targets, list) and len(ckpt_targets) > 0:
            targets = [str(x) for x in ckpt_targets]
            is_single = (len(targets) == 1)
            print(f"[eval_only] targets from ckpt: K={len(targets)} -> {targets}")

        if "modality" in st and isinstance(st["modality"], str):
            args.modality = st["modality"]
            print(f"[eval_only] modality from ckpt: {args.modality}")

        if "use_raw" in st:
            args.use_raw = bool(st["use_raw"])
            print(f"[eval_only] use_raw from ckpt: {args.use_raw}")

        if "loss" in st and isinstance(st["loss"], str):
            args.loss = st["loss"]
            print(f"[eval_only] loss from ckpt: {args.loss}")

        if "head_hidden" in st:
            try:
                args.head_hidden = int(st["head_hidden"])
                print(f"[eval_only] head_hidden from ckpt: {args.head_hidden}")
            except Exception:
                pass

        if "loss_alpha" in st:
            try:
                args.loss_alpha = float(st["loss_alpha"])
            except Exception:
                pass
        if "loss_beta" in st:
            try:
                args.loss_beta = float(st["loss_beta"])
            except Exception:
                pass

        # Rebuild datasets/loaders AFTER overrides (critical for correctness)
        ds_va = BPLabeledDataset(df_va[["ssoid"] + targets], args.npz_dir, targets, use_raw=args.use_raw)
        ds_te = BPLabeledDataset(df_te[["ssoid"] + targets], args.npz_dir, targets, use_raw=args.use_raw)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False)

        # Build model with correct target count
        model = BPModel(
            ppg_enc=PPGEncoderCLIP(with_proj=True),
            ecg_enc=ECGEncoderCLIP(with_proj=True),
            modality=args.modality,
            n_targets=len(targets),
            head_hidden=args.head_hidden,
            normalize_emb=False,
        ).to(device)

        model.load_state_dict(st["model"], strict=True)

        mu_eval = st.get("mu", mu)
        sigma_eval = st.get("sigma", sigma)
        mu_eval = np.asarray(mu_eval, dtype=np.float64).reshape(-1)
        sigma_eval = np.asarray(sigma_eval, dtype=np.float64).reshape(-1)
        if mu_eval.shape[0] != len(targets) or sigma_eval.shape[0] != len(targets):
            raise RuntimeError(
                f"mu/sigma shape mismatch: mu={mu_eval.shape} sigma={sigma_eval.shape} targets={len(targets)}"
            )

        loss_fn = None
        if args.loss == "mae_pearson":
            loss_fn = MAE_PearsonLoss(alpha=args.loss_alpha, beta=args.loss_beta)

        # VAL raw
        val_loss, yv, yvhat, _ = _infer_on_loader(model, dl_va, device, mu_eval, sigma_eval, args.loss, loss_fn)

        # Fit calibration on VAL, apply externally only (do NOT set into model)
        a, b = _fit_affine_calibration(y_true=yv, y_pred=yvhat)

        val_raw = _build_metrics_dict(val_loss, targets, yv, yvhat)
        yvhat_cal = yvhat * a.reshape(1, -1) + b.reshape(1, -1)
        val_cal = _build_metrics_dict(val_loss, targets, yv, yvhat_cal)

        print("\n[VAL raw]")
        _print_metrics("[VAL raw]", val_raw, targets, detailed=True)
        print("[VAL cal]")
        _print_metrics("[VAL cal]", val_cal, targets, detailed=True)

        # TEST raw + cal
        test_loss, yt, ythat, ssoids = _infer_on_loader(model, dl_te, device, mu_eval, sigma_eval, args.loss, loss_fn)
        test_raw = _build_metrics_dict(test_loss, targets, yt, ythat)
        ythat_cal = ythat * a.reshape(1, -1) + b.reshape(1, -1)
        test_cal = _build_metrics_dict(test_loss, targets, yt, ythat_cal)

        print("\n[TEST raw]")
        _print_metrics("[TEST raw]", test_raw, targets, detailed=True)
        print("[TEST cal]")
        _print_metrics("[TEST cal]", test_cal, targets, detailed=True)

        tag = targets[0] if is_single else "all"
        payload = {
            "target_col": (targets[0] if is_single else "all"),
            "modality": args.modality,
            "loss": args.loss,
            "loss_alpha": float(args.loss_alpha),
            "loss_beta": float(args.loss_beta),
            "constrain": "",
            "use_raw": bool(args.use_raw),
            "ckpt_path": str(ckpt_path),
            "N_records": int(yt.shape[0]),
            "MAE_raw": float(test_raw["MAE"]),
            "RMSE_raw": float(test_raw["RMSE"]),
            "r_raw": test_raw["r"],
            "R2_raw": test_raw["R2"],
            "MAE_cal": float(test_cal["MAE"]),
            "RMSE_cal": float(test_cal["RMSE"]),
            "r_cal": test_cal["r"],
            "R2_cal": test_cal["R2"],
            "a": float(a[0]) if is_single else a.astype(float).tolist(),
            "b": float(b[0]) if is_single else b.astype(float).tolist(),
            "per_target_raw": {
                "MAE": test_raw["mae_per_target"],
                "RMSE": test_raw["rmse_per_target"],
                "r": test_raw["r_per_target"],
                "R2": test_raw["R2_per_target"],
            },
            "per_target_cal": {
                "MAE": test_cal["mae_per_target"],
                "RMSE": test_cal["rmse_per_target"],
                "r": test_cal["r_per_target"],
                "R2": test_cal["R2_per_target"],
            },
        }
        if save_records:
            payload["records"] = _make_records(yt, ythat, ythat_cal, ssoids)
        _save_metrics_json(out_dir / f"metrics_test_bp_{tag}_records.json", payload)

        val_payload = {
            "target_col": (targets[0] if is_single else "all"),
            "modality": args.modality,
            "loss": args.loss,
            "loss_alpha": float(args.loss_alpha),
            "loss_beta": float(args.loss_beta),
            "use_raw": bool(args.use_raw),
            "N_records": int(yv.shape[0]),
            "MAE_raw": float(val_raw["MAE"]),
            "RMSE_raw": float(val_raw["RMSE"]),
            "r_raw": val_raw["r"],
            "R2_raw": val_raw["R2"],
            "MAE_cal": float(val_cal["MAE"]),
            "RMSE_cal": float(val_cal["RMSE"]),
            "r_cal": val_cal["r"],
            "R2_cal": val_cal["R2"],
            "a": float(a[0]) if is_single else a.astype(float).tolist(),
            "b": float(b[0]) if is_single else b.astype(float).tolist(),
        }
        _save_metrics_json(out_dir / f"metrics_val_bp_{tag}_records.json", val_payload)
        return

    # -------------------------
    # Training mode
    # -------------------------
    # Datasets/loaders (training path uses args.use_raw as given)
    ds_tr = BPLabeledDataset(df_tr[["ssoid"] + targets], args.npz_dir, targets, use_raw=args.use_raw)
    ds_va = BPLabeledDataset(df_va[["ssoid"] + targets], args.npz_dir, targets, use_raw=args.use_raw)
    ds_te = BPLabeledDataset(df_te[["ssoid"] + targets], args.npz_dir, targets, use_raw=args.use_raw)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Build model
    model = BPModel(
        ppg_enc=PPGEncoderCLIP(with_proj=True),
        ecg_enc=ECGEncoderCLIP(with_proj=True),
        modality=args.modality,
        n_targets=len(targets),
        head_hidden=args.head_hidden,
        normalize_emb=False,
    ).to(device)

    # Load CLIP weights into encoders (training only)
    model.load_from_clip(args.clip_ckpt, device=device)

    # Prepare loss_fn if needed
    loss_fn = None
    if args.loss == "mae_pearson":
        loss_fn = MAE_PearsonLoss(alpha=args.loss_alpha, beta=args.loss_beta)

    # Freeze strategy (more robust than name-prefix only)
    if args.freeze_backbone:
        # ENGLISH comments: freeze everything except the regression head if present
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
            params = [{"params": list(model.head.parameters()), "lr": args.lr_head}]
        else:
            # fallback: keep old behavior by name
            for n, p in model.named_parameters():
                p.requires_grad = n.startswith("head.")
            params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_head}]
    else:
        if hasattr(model, "head"):
            head_param_ids = set(id(p) for p in model.head.parameters())
            enc_params = [p for p in model.parameters() if id(p) not in head_param_ids]
            head_params = list(model.head.parameters())
        else:
            enc_params = []
            head_params = []
            for n, p in model.named_parameters():
                if n.startswith("head."):
                    head_params.append(p)
                else:
                    enc_params.append(p)
        params = [
            {"params": enc_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ]

    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best = float("inf")
    best_ep = -1
    wait = 0

    mu_t, sg_t = _as_tensors(mu, sigma, device)

    for ep in range(1, args.epochs + 1):
        # Train one epoch
        if args.loss in ("mse", "huber"):
            tr_loss = train_one_epoch_bp(
                model, dl_tr, optimizer, scaler, device,
                targets=targets, mu=mu, sigma=sigma, loss_type=args.loss
            )
        elif args.loss == "mae_pearson":
            # Inline training loop to avoid changing bp_engine.py
            model.train()
            tot = 0
            tot_loss = 0.0
            for batch in dl_tr:
                ecg, ppg, y_raw, _ = _to_device_batch(batch, device)
                y_z = (y_raw - mu_t) / sg_t

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=True):
                    yhat_z = model(ecg, ppg)
                    if not torch.isfinite(yhat_z).all():
                        raise RuntimeError(f"Non-finite yhat_z in eval: min={float(torch.nan_to_num(yhat_z).min())} "
                                        f"max={float(torch.nan_to_num(yhat_z).max())}")
                    print("debug yhat_z min/max:", float(yhat_z.min().item()), float(yhat_z.max().item()))


                    # ENGLISH comments: strict shape checks
                    if yhat_z.ndim != 2 or y_z.ndim != 2:
                        raise RuntimeError(f"Bad dims: yhat_z={tuple(yhat_z.shape)} y_z={tuple(y_z.shape)}")
                    if yhat_z.shape != y_z.shape:
                        raise RuntimeError(
                            f"Shape mismatch: yhat_z={tuple(yhat_z.shape)} vs y_z={tuple(y_z.shape)}"
                        )

                    loss = loss_fn(yhat_z, y_z)

                bs = y_raw.size(0)
                tot += bs
                tot_loss += float(loss.item()) * bs

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            tr_loss = tot_loss / max(1, tot)
        else:
            raise RuntimeError(f"Unknown loss: {args.loss}")

        # VAL raw
        val_loss, yv, yvhat, _ = _infer_on_loader(model, dl_va, device, mu, sigma, args.loss, loss_fn)
        val_raw = _build_metrics_dict(val_loss, targets, yv, yvhat)

        # VAL cal: fit on VAL, apply externally only (do NOT set into model)
        a, b = _fit_affine_calibration(y_true=yv, y_pred=yvhat)
        yvhat_cal = yvhat * a.reshape(1, -1) + b.reshape(1, -1)
        val_cal = _build_metrics_dict(val_loss, targets, yv, yvhat_cal)

        print(f"\n[E{ep}] train_loss={tr_loss:.6f}")
        print("[VAL raw]")
        _print_metrics("[VAL raw]", val_raw, targets, detailed=True)
        print("[VAL cal]")
        _print_metrics("[VAL cal]", val_cal, targets, detailed=True)

        # Early stop metric
        score = float(val_raw["MAE"]) if args.early_on == "raw" else float(val_cal["MAE"])
        improved = score < best - 1e-6
        if improved:
            best = score
            best_ep = ep
            wait = 0

            ckpt = {
                "epoch": ep,
                "targets": targets,
                "mu": np.asarray(mu, dtype=np.float32),
                "sigma": np.asarray(sigma, dtype=np.float32),
                "model": model.state_dict(),
                "val_score": float(best),
                "val_metric": f"MAE_{args.early_on}",
                "modality": args.modality,
                "use_raw": bool(args.use_raw),
                "loss": args.loss,
                "loss_alpha": float(args.loss_alpha),
                "loss_beta": float(args.loss_beta),
                "head_hidden": int(args.head_hidden),
            }
            save_path = out_dir / "best.pth"
            torch.save(ckpt, str(save_path))
            print(f"[best] {args.early_on}={best:.6f} @epoch{ep} | saved: {save_path}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[early stop] no improve for {args.patience} epochs (best @ {best_ep}, score={best:.6f})")
                break

    # -------------------------
    # Final TEST using best.pth
    # -------------------------
    best_path = out_dir / "best.pth"
    if not best_path.exists():
        raise RuntimeError(f"best checkpoint not found: {best_path}")

    st = torch.load(str(best_path), map_location="cpu")
    model.load_state_dict(st["model"], strict=True)
    mu_best = np.asarray(st.get("mu", mu), dtype=np.float64).reshape(-1)
    sigma_best = np.asarray(st.get("sigma", sigma), dtype=np.float64).reshape(-1)

    # Fit calibration on VAL with best model (external only)
    val_loss, yv, yvhat, _ = _infer_on_loader(model, dl_va, device, mu_best, sigma_best, args.loss, loss_fn)
    a, b = _fit_affine_calibration(y_true=yv, y_pred=yvhat)

    # TEST
    test_loss, yt, ythat, ssoids = _infer_on_loader(model, dl_te, device, mu_best, sigma_best, args.loss, loss_fn)
    test_raw = _build_metrics_dict(test_loss, targets, yt, ythat)
    ythat_cal = ythat * a.reshape(1, -1) + b.reshape(1, -1)
    test_cal = _build_metrics_dict(test_loss, targets, yt, ythat_cal)

    print("\n[TEST raw]")
    _print_metrics("[TEST raw]", test_raw, targets, detailed=True)
    print("[TEST cal]")
    _print_metrics("[TEST cal]", test_cal, targets, detailed=True)

    # Save metrics JSON
    tag = targets[0] if is_single else "all"
    payload = {
        "target_col": (targets[0] if is_single else "all"),
        "modality": args.modality,
        "loss": args.loss,
        "loss_alpha": float(args.loss_alpha),
        "loss_beta": float(args.loss_beta),
        "constrain": "",
        "use_raw": bool(args.use_raw),
        "ckpt_path": str(best_path),
        "epoch": int(st.get("epoch", -1)),
        "N_records": int(yt.shape[0]),
        "MAE_raw": float(test_raw["MAE"]),
        "RMSE_raw": float(test_raw["RMSE"]),
        "r_raw": test_raw["r"],
        "R2_raw": test_raw["R2"],
        "MAE_cal": float(test_cal["MAE"]),
        "RMSE_cal": float(test_cal["RMSE"]),
        "r_cal": test_cal["r"],
        "R2_cal": test_cal["R2"],
        "a": float(a[0]) if is_single else a.astype(float).tolist(),
        "b": float(b[0]) if is_single else b.astype(float).tolist(),
        "per_target_raw": {
            "MAE": test_raw["mae_per_target"],
            "RMSE": test_raw["rmse_per_target"],
            "r": test_raw["r_per_target"],
            "R2": test_raw["R2_per_target"],
        },
        "per_target_cal": {
            "MAE": test_cal["mae_per_target"],
            "RMSE": test_cal["rmse_per_target"],
            "r": test_cal["r_per_target"],
            "R2": test_cal["R2_per_target"],
        },
    }
    if save_records:
        payload["records"] = _make_records(yt, ythat, ythat_cal, ssoids)

    _save_metrics_json(out_dir / f"metrics_test_bp_{tag}_records.json", payload)

    # Also save val metrics
    yvhat_cal = yvhat * a.reshape(1, -1) + b.reshape(1, -1)
    val_raw = _build_metrics_dict(val_loss, targets, yv, yvhat)
    val_cal = _build_metrics_dict(val_loss, targets, yv, yvhat_cal)
    val_payload = {
        "target_col": (targets[0] if is_single else "all"),
        "modality": args.modality,
        "loss": args.loss,
        "loss_alpha": float(args.loss_alpha),
        "loss_beta": float(args.loss_beta),
        "use_raw": bool(args.use_raw),
        "N_records": int(yv.shape[0]),
        "MAE_raw": float(val_raw["MAE"]),
        "RMSE_raw": float(val_raw["RMSE"]),
        "r_raw": val_raw["r"],
        "R2_raw": val_raw["R2"],
        "MAE_cal": float(val_cal["MAE"]),
        "RMSE_cal": float(val_cal["RMSE"]),
        "r_cal": val_cal["r"],
        "R2_cal": val_cal["R2"],
        "a": float(a[0]) if is_single else a.astype(float).tolist(),
        "b": float(b[0]) if is_single else b.astype(float).tolist(),
    }
    _save_metrics_json(out_dir / f"metrics_val_bp_{tag}_records.json", val_payload)


if __name__ == "__main__":
    main()
