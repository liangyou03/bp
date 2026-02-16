#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from age_tune_v2 import (
    Config,
    FusionAgeRegressor,
    LabeledECGPPGDataset,
    metrics_from_arrays,
    predict_records,
    subject_level_aggregate,
    _fit_affine_calibration,
    fit_metadata_calibrator,
    apply_metadata_calibrator,
)


def _apply_global_calibrator(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return (a * y_pred + b).astype(np.float32)


def evaluate_one_run(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    best_model_path = run_dir / "best_model.pth"
    if not metrics_path.exists() or not best_model_path.exists():
        raise FileNotFoundError(f"Missing metrics/best_model in {run_dir}")

    m = json.load(open(metrics_path, "r"))
    cfg = m.get("config", {})

    # keep same data/split source as training pipeline
    labels_csv = Path(Config.LABEL_CSV)
    data_dir = Path(Config.DATA_DIR)
    target = cfg["TARGET_COL"]
    mode = cfg["MODE"]
    use_zscore = bool(cfg.get("USE_ZSCORE", False))
    batch_size = int(cfg.get("BATCH_SIZE", 256))

    df_all = pd.read_csv(labels_csv)
    df_all["ssoid"] = df_all["ssoid"].astype(str)
    if "subject_uid" in df_all.columns:
        df_all["subject_id"] = df_all["subject_uid"].astype(str)
    else:
        df_all["subject_id"] = df_all["ssoid"].astype(str).str.split("_").str[0]

    df_val = df_all[df_all["split"] == "val"].copy().reset_index(drop=True)
    df_test = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)
    for part in (df_val, df_test):
        part[target] = pd.to_numeric(part[target], errors="coerce")
    df_val = df_val.dropna(subset=[target]).reset_index(drop=True)
    df_test = df_test.dropna(subset=[target]).reset_index(drop=True)

    # apply config into runtime
    Config.TARGET_COL = target
    Config.MODE = mode
    Config.USE_ZSCORE = use_zscore
    Config.BATCH_SIZE = batch_size

    val_ds = LabeledECGPPGDataset(df_val, str(data_dir), Config)
    test_ds = LabeledECGPPGDataset(df_test, str(data_dir), Config)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = FusionAgeRegressor(mode=mode, feature_dim=256)
    state = torch.load(best_model_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model = model.to(Config.DEVICE)

    # val/test subject-level predictions
    val_preds, _, val_ssoids = predict_records(model, val_loader, Config.DEVICE, mode)
    test_preds, _, test_ssoids = predict_records(model, test_loader, Config.DEVICE, mode)

    subj_val = subject_level_aggregate(df_val, val_preds, val_ssoids)
    subj_test = subject_level_aggregate(df_test, test_preds, test_ssoids)

    yv = subj_val[target].values.astype(np.float32)
    pv = subj_val["pred_value"].values.astype(np.float32)
    yt = subj_test[target].values.astype(np.float32)
    pt = subj_test["pred_value"].values.astype(np.float32)

    # 1) no calibration
    raw_mae, raw_rmse, raw_r, raw_r2 = metrics_from_arrays(yt, pt)

    # 2) global calibration (fit on val only)
    a, b = _fit_affine_calibration(y_true=yv, y_pred=pv)
    pt_global = _apply_global_calibrator(pt, a, b)
    g_mae, g_rmse, g_r, g_r2 = metrics_from_arrays(yt, pt_global)

    # 3) metadata calibration (fit on val only)
    cal = fit_metadata_calibrator(subj_val, target_col=target, pred_col="pred_value", min_group_size=25)
    pt_meta = apply_metadata_calibrator(subj_test, cal, pred_col="pred_value")
    m_mae, m_rmse, m_r, m_r2 = metrics_from_arrays(yt, pt_meta)

    return {
        "run_dir": str(run_dir),
        "target": target,
        "raw_mae": raw_mae,
        "raw_r": raw_r,
        "global_mae": g_mae,
        "global_r": g_r,
        "meta_mae": m_mae,
        "meta_r": m_r,
        "global_a": float(a),
        "global_b": float(b),
        "meta_groups": int(len(cal.get("groups", {}))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    rows = []
    for rd in args.run_dirs:
        rows.append(evaluate_one_run(Path(rd)))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()
