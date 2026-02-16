"""
Finetune (ECG / PPG / ECG+PPG) BP regression using MULTIMODAL SSL checkpoints.

Outputs (in Config.OUTPUT_DIR):
1) best_model.pth                         (best val MAE)
2) metrics.json                           (test + trainval metrics)
3) subject_level_predictions.csv          (TEST subject-level predictions)
4) subject_level_predictions_trainval.csv (TRAIN+VAL subject-level predictions)

Loss options (Config.LOSS_MODE):
- "mse"         : standard MSE (your original behavior)
- "mse+dist"    : MSE + lambda * DistributionAlignmentLoss
- "wmse"        : inverse-frequency weighted MSE
- "mse*pearson" : MSE * (1 + alpha * PearsonLoss)
"""

import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import signal, stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from models import ResNet18_1D
from finetune_loss import prepare_losses


# ================= 配置参数 (Config) =================
class Config:
    # ---- Paths ----
    DATA_DIR = "/home/youliang/youliang_data2/bp/bp_recode_v1/output_600hz/npz"
    LABEL_CSV = "/home/youliang/youliang_data2/bp/bp_recode_v1/output_600hz/labels.csv"

    PRETRAINED_CKPT = "./checkpoints/ssl_multimodal_bp_600hz_v1/checkpoint_epoch_100.pth"

    PPG_ENCODER_PATH = None
    ECG_ENCODER_PATH = None

    OUTPUT_DIR = "./results/finetune_bp_from_multimodal_ckpt_minimal"

    TARGET_COL = "right_arm_sbp"
    USE_ZSCORE = False

    # ---- Data params ----
    ECG_CROP_LEN = 3630
    PPG_CROP_LEN = 3630

    # ---- Choose downstream mode ----
    # "ppg" | "ecg" | "fusion"
    MODE = "fusion"

    # ---- Training params ----
    SEED = 42
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 7

    FREEZE_ENCODERS = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Loss choice (DEFAULT = original MSE) ----
    LOSS_MODE = "mse"  # "mse" | "mse+dist" | "wmse" | "mse*pearson"

    # ---- Loss hyperparams ----
    # mse+dist
    LAMBDA_DIST = 0.2
    DIST_KDE_SIGMA = 2.0
    DIST_ASSIGN_SIGMA = 2.0
    DIST_INV_WEIGHT = True
    DIST_BIN_WIDTH = 1.0

    # wmse
    WMSE_SMOOTHING = 1.0
    WMSE_BIN_WIDTH = 1.0

    # mse*pearson
    PEARSON_ALPHA = 1.0

    # Optional fixed range for dist/wmse binning; None => inferred from train ages
    Y_MIN = None
    Y_MAX = None


# ================= 工具函数 =================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_subject_id(ssoid: str) -> str:
    return str(ssoid).split('_')[0]


# ================= Model =================
class FusionAgeRegressor(nn.Module):
    def __init__(self, mode: str, feature_dim: int = 256):
        super().__init__()
        assert mode in ["ppg", "ecg", "fusion"]
        self.mode = mode

        self.ppg_encoder = ResNet18_1D(input_channels=1, feature_dim=feature_dim)
        self.ecg_encoder = ResNet18_1D(input_channels=1, feature_dim=feature_dim)

        if self.mode == "fusion":
            self.head = nn.Linear(feature_dim * 2, 1)
        else:
            self.head = nn.Linear(feature_dim, 1)

    def freeze_encoders(self):
        for p in self.ppg_encoder.parameters():
            p.requires_grad = False
        for p in self.ecg_encoder.parameters():
            p.requires_grad = False

    def forward(self, ppg_x=None, ecg_x=None):
        if self.mode == "ppg":
            feats = self.ppg_encoder(ppg_x)
        elif self.mode == "ecg":
            feats = self.ecg_encoder(ecg_x)
        else:
            ppg_feats = self.ppg_encoder(ppg_x)
            ecg_feats = self.ecg_encoder(ecg_x)
            feats = torch.cat([ppg_feats, ecg_feats], dim=1)
        return self.head(feats).squeeze(-1)


def _strip_encoder_state(simclr_state: dict) -> dict:
    out = {}
    for k, v in simclr_state.items():
        if "projector" in k:
            continue
        if k.startswith("encoder."):
            out[k.replace("encoder.", "")] = v
        else:
            out[k] = v
    return out


def load_pretrained_from_multimodal_ckpt(model: FusionAgeRegressor, ckpt_path: str, device: str = "cpu"):
    print(f"Loading MULTIMODAL checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "ppg_model_state_dict" not in ckpt or "ecg_model_state_dict" not in ckpt:
        raise ValueError("Checkpoint does not contain ppg_model_state_dict/ecg_model_state_dict.")

    ppg_state = _strip_encoder_state(ckpt["ppg_model_state_dict"])
    ecg_state = _strip_encoder_state(ckpt["ecg_model_state_dict"])

    msg_ppg = model.ppg_encoder.load_state_dict(ppg_state, strict=False)
    msg_ecg = model.ecg_encoder.load_state_dict(ecg_state, strict=False)

    print(f"PPG encoder loaded. Missing keys: {msg_ppg.missing_keys}")
    print(f"ECG encoder loaded. Missing keys: {msg_ecg.missing_keys}")


def load_pretrained_from_encoder_files(
    model: FusionAgeRegressor,
    ppg_encoder_path: str,
    ecg_encoder_path: str,
    device: str = "cpu",
):
    if ppg_encoder_path:
        print(f"Loading PPG encoder: {ppg_encoder_path}")
        ppg_state = torch.load(ppg_encoder_path, map_location=device)
        msg_ppg = model.ppg_encoder.load_state_dict(ppg_state, strict=False)
        print(f"PPG encoder loaded. Missing keys: {msg_ppg.missing_keys}")

    if ecg_encoder_path:
        print(f"Loading ECG encoder: {ecg_encoder_path}")
        ecg_state = torch.load(ecg_encoder_path, map_location=device)
        msg_ecg = model.ecg_encoder.load_state_dict(ecg_state, strict=False)
        print(f"ECG encoder loaded. Missing keys: {msg_ecg.missing_keys}")


# ================= Dataset =================
class LabeledECGPPGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, config: Config):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.cfg = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ssoid = str(row["ssoid"])
        target_value = np.float32(row[self.cfg.TARGET_COL])
        file_path = self.data_dir / f"{ssoid}.npz"

        try:
            loaded = np.load(file_path)
            if self.cfg.USE_ZSCORE:
                ecg = loaded["ecg"].astype(np.float32).reshape(-1, 1)
                ppg = loaded["ppg"].astype(np.float32).reshape(-1, 1)
            else:
                ecg = loaded["ecg_raw"].astype(np.float32).reshape(-1, 1)
                ppg = loaded["ppg_raw"].astype(np.float32).reshape(-1, 1)

            if ecg.shape[0] < self.cfg.ECG_CROP_LEN:
                pad = self.cfg.ECG_CROP_LEN - ecg.shape[0]
                ecg = np.pad(ecg, ((0, pad), (0, 0)), mode="constant")
            ecg = ecg[: self.cfg.ECG_CROP_LEN, :].transpose(1, 0)  # (1, L_ecg)

            if ppg.shape[0] < self.cfg.PPG_CROP_LEN:
                pad = self.cfg.PPG_CROP_LEN - ppg.shape[0]
                ppg = np.pad(ppg, ((0, pad), (0, 0)), mode="constant")
            ppg = ppg[: self.cfg.PPG_CROP_LEN, :].transpose(1, 0)  # (1, L_ppg)

            return (
                torch.FloatTensor(ppg),
                torch.FloatTensor(ecg),
                torch.tensor(target_value, dtype=torch.float32),
                ssoid,
            )
        except Exception as e:
            print(f"Error loading {ssoid}: {e}")
            return (
                torch.zeros((1, self.cfg.PPG_CROP_LEN)),
                torch.zeros((1, self.cfg.ECG_CROP_LEN)),
                torch.tensor(target_value, dtype=torch.float32),
                ssoid,
            )


# ================= Metrics / Aggregation =================
@torch.no_grad()
def predict_records(model, loader, device, mode: str):
    model.eval()
    preds, targets, ssoids = [], [], []
    for ppg, ecg, y, ssoid in tqdm(loader, desc="Predict", leave=False):
        ppg = ppg.to(device)
        ecg = ecg.to(device)
        y = y.to(device)

        if mode == "ppg":
            out = model(ppg_x=ppg, ecg_x=None)
        elif mode == "ecg":
            out = model(ppg_x=None, ecg_x=ecg)
        else:
            out = model(ppg_x=ppg, ecg_x=ecg)

        preds.extend(out.detach().cpu().numpy())
        targets.extend(y.detach().cpu().numpy())
        ssoids.extend(list(ssoid))

    return np.array(preds, dtype=np.float32), np.array(targets, dtype=np.float32), ssoids


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r = float("nan")
    if len(y_true) >= 2 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        r, _ = stats.pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return float(mae), float(rmse), float(r), float(r2)


def subject_level_aggregate(df_split: pd.DataFrame, preds: np.ndarray, ssoids: list):
    pred_col = "pred_value"
    target_col = Config.TARGET_COL
    results_df = pd.DataFrame({"ssoid": ssoids, pred_col: preds})
    merged = df_split.merge(results_df, on="ssoid", how="left")

    agg = {pred_col: "mean", target_col: "first", "ssoid": "first"}
    for col in ["sex", "age", "subject_uid"]:
        if col in merged.columns:
            agg[col] = "first"

    subj_df = merged.groupby("subject_id").agg(agg).reset_index()
    subj_df["residual"] = subj_df[pred_col] - subj_df[target_col]
    return subj_df


def _fit_affine_calibration(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2 or np.std(y_pred) < 1e-8:
        return 1.0, 0.0
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def _meta_group_key(df: pd.DataFrame) -> pd.Series:
    keys = pd.Series(["all"] * len(df), index=df.index, dtype=object)
    if "sex" in df.columns:
        sx = df["sex"].fillna("UNK").astype(str).str.strip().str.upper()
        keys = sx
    if "age" in df.columns:
        age_num = pd.to_numeric(df["age"], errors="coerce")
        age_bin = (np.floor(age_num / 10.0) * 10.0)
        age_bin = age_bin.where(np.isfinite(age_bin), -1).astype(int)
        keys = keys.astype(str) + "_A" + age_bin.astype(str)
    return keys


def fit_metadata_calibrator(
    val_subj_df: pd.DataFrame,
    target_col: str,
    pred_col: str = "pred_value",
    min_group_size: int = 25,
):
    # Leakage-safe: fit calibration strictly on validation labels.
    yv = pd.to_numeric(val_subj_df[target_col], errors="coerce").values
    pv = pd.to_numeric(val_subj_df[pred_col], errors="coerce").values
    a_g, b_g = _fit_affine_calibration(y_true=yv, y_pred=pv)

    keys = _meta_group_key(val_subj_df)
    groups = {}
    for k, part in val_subj_df.assign(_k=keys).groupby("_k"):
        if len(part) < min_group_size:
            continue
        a, b = _fit_affine_calibration(
            y_true=pd.to_numeric(part[target_col], errors="coerce").values,
            y_pred=pd.to_numeric(part[pred_col], errors="coerce").values,
        )
        groups[str(k)] = {"a": float(a), "b": float(b), "n": int(len(part))}

    return {
        "global": {"a": float(a_g), "b": float(b_g), "n": int(len(val_subj_df))},
        "groups": groups,
        "group_rule": "sex+age_decade",
        "min_group_size": int(min_group_size),
    }


def apply_metadata_calibrator(df_subj: pd.DataFrame, cal: dict, pred_col: str = "pred_value") -> np.ndarray:
    keys = _meta_group_key(df_subj)
    pred = pd.to_numeric(df_subj[pred_col], errors="coerce").values.astype(np.float64)
    out = pred.copy()
    ag = float(cal.get("global", {}).get("a", 1.0))
    bg = float(cal.get("global", {}).get("b", 0.0))
    groups = cal.get("groups", {})

    for i, k in enumerate(keys.astype(str).values):
        g = groups.get(k)
        if g is None:
            a, b = ag, bg
        else:
            a, b = float(g.get("a", ag)), float(g.get("b", bg))
        out[i] = a * pred[i] + b
    return out.astype(np.float32)


# ================= Loss builder (no CLI) =================
def build_loss_from_config(df_train: pd.DataFrame, device: str):
    """
    Build the selected loss based on Config, using TRAIN ages only.
    Returns:
      loss_pack (dict of modules)
    """
    train_targets = df_train[Config.TARGET_COL].values.astype(np.float32)

    loss_pack = prepare_losses(
        loss_mode=Config.LOSS_MODE,
        train_ages=train_targets,
        y_min=Config.Y_MIN,
        y_max=Config.Y_MAX,
        dist_bin_width=Config.DIST_BIN_WIDTH,
        dist_kde_sigma=Config.DIST_KDE_SIGMA,
        dist_assign_sigma=Config.DIST_ASSIGN_SIGMA,
        dist_inv_weight=Config.DIST_INV_WEIGHT,
        wmse_bin_width=Config.WMSE_BIN_WIDTH,
        wmse_smoothing=Config.WMSE_SMOOTHING,
        pearson_alpha=Config.PEARSON_ALPHA,
    )

    # move nn.Module to device
    for k, v in list(loss_pack.items()):
        if isinstance(v, nn.Module):
            loss_pack[k] = v.to(device)
    return loss_pack


# ================= Main =================
def main():
    set_seed(Config.SEED)

    outdir = Path(Config.OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {outdir}")

    # ---- Load labels ----
    df_all = pd.read_csv(Config.LABEL_CSV)
    df_all["ssoid"] = df_all["ssoid"].astype(str)
    if "subject_uid" in df_all.columns:
        df_all["subject_id"] = df_all["subject_uid"].astype(str)
    else:
        df_all["subject_id"] = df_all["ssoid"].apply(get_subject_id)
    print(f"Total records in CSV: {len(df_all)}")

    if "split" in df_all.columns:
        df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
        df_val = df_all[df_all["split"] == "val"].reset_index(drop=True)
        df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)
        train_subjs = df_train["subject_id"].unique()
        val_subjs = df_val["subject_id"].unique()
        test_subjs = df_test["subject_id"].unique()
    else:
        unique_subjects = df_all["subject_id"].unique()
        np.random.shuffle(unique_subjects)
        n_subjects = len(unique_subjects)
        n_train = int(n_subjects * 0.7)
        n_val = int(n_subjects * 0.1)
        train_subjs = unique_subjects[:n_train]
        val_subjs = unique_subjects[n_train:n_train + n_val]
        test_subjs = unique_subjects[n_train + n_val:]
        df_train = df_all[df_all["subject_id"].isin(train_subjs)].reset_index(drop=True)
        df_val = df_all[df_all["subject_id"].isin(val_subjs)].reset_index(drop=True)
        df_test = df_all[df_all["subject_id"].isin(test_subjs)].reset_index(drop=True)

    for part in (df_train, df_val, df_test):
        part[Config.TARGET_COL] = pd.to_numeric(part[Config.TARGET_COL], errors="coerce")
    df_train = df_train.dropna(subset=[Config.TARGET_COL]).reset_index(drop=True)
    df_val = df_val.dropna(subset=[Config.TARGET_COL]).reset_index(drop=True)
    df_test = df_test.dropna(subset=[Config.TARGET_COL]).reset_index(drop=True)
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)

    print(f"Subjects Split: Train={len(train_subjs)}, Val={len(val_subjs)}, Test={len(test_subjs)}")
    print(f"Records Split:  Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # ---- Datasets & Loaders ----
    train_ds = LabeledECGPPGDataset(df_train, Config.DATA_DIR, Config)
    val_ds = LabeledECGPPGDataset(df_val, Config.DATA_DIR, Config)
    test_ds = LabeledECGPPGDataset(df_test, Config.DATA_DIR, Config)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # ---- Model ----
    model = FusionAgeRegressor(mode=Config.MODE, feature_dim=256)

    if Config.PRETRAINED_CKPT and os.path.exists(Config.PRETRAINED_CKPT):
        load_pretrained_from_multimodal_ckpt(model, Config.PRETRAINED_CKPT, device="cpu")
    else:
        if Config.PPG_ENCODER_PATH or Config.ECG_ENCODER_PATH:
            load_pretrained_from_encoder_files(model, Config.PPG_ENCODER_PATH, Config.ECG_ENCODER_PATH, device="cpu")
        else:
            print("⚠️  No pretrained weights found. Training from scratch.")

    if Config.FREEZE_ENCODERS:
        model.freeze_encoders()
        print("Encoders frozen. Only training head.")

    model = model.to(Config.DEVICE)

    # ---- Loss pack ----
    loss_pack = build_loss_from_config(df_train, Config.DEVICE)
    y_min = float(loss_pack.get("y_min", float("nan")))
    y_max = float(loss_pack.get("y_max", float("nan")))
    print(f"LOSS_MODE={Config.LOSS_MODE} | y_range=[{y_min}, {y_max}]")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)

    # ---- Train loop ----
    best_val_mae = float("inf")
    patience_counter = 0
    best_model_path = outdir / "best_model.pth"

    print(
        f"\nStarting finetune. TARGET={Config.TARGET_COL} MODE={Config.MODE} "
        f"FREEZE_ENCODERS={Config.FREEZE_ENCODERS} LOSS={Config.LOSS_MODE} USE_ZSCORE={Config.USE_ZSCORE}"
    )

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0

        # optional logs
        log_mse, log_dist, log_pl = [], [], []

        for ppg, ecg, y, _ in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            ppg = ppg.to(Config.DEVICE)
            ecg = ecg.to(Config.DEVICE)
            y = y.to(Config.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if Config.MODE == "ppg":
                pred = model(ppg_x=ppg, ecg_x=None)
            elif Config.MODE == "ecg":
                pred = model(ppg_x=None, ecg_x=ecg)
            else:
                pred = model(ppg_x=ppg, ecg_x=ecg)

            # -------- choose loss --------
            if Config.LOSS_MODE == "mse":
                loss = loss_pack["mse"](pred, y)

            elif Config.LOSS_MODE == "mse+dist":
                mse = loss_pack["mse"](pred, y)
                pred_c = torch.clamp(pred, y_min, y_max)
                dist = loss_pack["dist"](pred_c)
                loss = mse + Config.LAMBDA_DIST * dist
                log_mse.append(float(mse.detach().cpu().item()))
                log_dist.append(float(dist.detach().cpu().item()))

            elif Config.LOSS_MODE == "wmse":
                loss = loss_pack["wmse"](pred, y)

            elif Config.LOSS_MODE == "mse*pearson":
                loss, stats_dict = loss_pack["mse_pearson"](pred, y)
                log_mse.append(stats_dict["mse"])
                log_pl.append(stats_dict["pearson_loss"])

            else:
                raise ValueError(f"Unknown LOSS_MODE: {Config.LOSS_MODE}")

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y.size(0)

        train_loss = train_loss_sum / len(train_ds)

        # ---- Validation (MAE) ----
        val_preds, val_targets, _ = predict_records(model, val_loader, Config.DEVICE, Config.MODE)
        val_mae = mean_absolute_error(val_targets, val_preds)

        msg = f"Epoch {epoch}: TrainLoss={train_loss:.4f}, Val MAE={val_mae:.4f}"
        if Config.LOSS_MODE == "mse+dist" and len(log_dist) > 0:
            msg += f", TrainMSE~{np.mean(log_mse):.4f}, TrainDist~{np.mean(log_dist):.4f}, lambda={Config.LAMBDA_DIST}"
        if Config.LOSS_MODE == "mse*pearson" and len(log_pl) > 0:
            msg += f", TrainMSE~{np.mean(log_mse):.4f}, TrainPearsonLoss~{np.mean(log_pl):.4f}, alpha={Config.PEARSON_ALPHA}"
        print(msg)

        # ---- Early stopping ----
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved (Val MAE: {best_val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # ---- Load best model ----
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
        print(f"\nLoaded best model from {best_model_path}")
    else:
        print("\n⚠️ Best model not found; using last epoch weights.")

    # ---- Fit metadata-aware calibration on VAL only (no test-label leakage) ----
    print("\nFitting metadata-aware calibration on VAL set only (leakage-safe)...")
    val_preds_eval, _, val_ssoids_eval = predict_records(model, val_loader, Config.DEVICE, Config.MODE)
    subj_val_df = subject_level_aggregate(df_val, val_preds_eval, val_ssoids_eval)
    meta_cal = fit_metadata_calibrator(subj_val_df, target_col=Config.TARGET_COL, pred_col="pred_value", min_group_size=25)

    # ---- Evaluate on TEST ----
    print("\nRunning Evaluation on TEST set...")
    test_preds, test_targets, test_ssoids = predict_records(model, test_loader, Config.DEVICE, Config.MODE)

    rec_mae, rec_rmse, rec_r, rec_r2 = metrics_from_arrays(test_targets, test_preds)
    print("\n=== TEST Record-Level Metrics ===")
    print(f"MAE: {rec_mae:.4f}")
    print(f"RMSE: {rec_rmse:.4f}")
    print(f"Pearson r: {rec_r:.4f}")
    print(f"R2: {rec_r2:.4f}")

    subj_test_df = subject_level_aggregate(df_test, test_preds, test_ssoids)
    subj_mae, subj_rmse, subj_r, subj_r2 = metrics_from_arrays(
        subj_test_df[Config.TARGET_COL].values.astype(np.float32),
        subj_test_df["pred_value"].values.astype(np.float32),
    )

    print("\n=== TEST Subject-Level Metrics ===")
    print(f"MAE: {subj_mae:.4f}")
    print(f"RMSE: {subj_rmse:.4f}")
    print(f"Pearson r: {subj_r:.4f}")
    print(f"R2: {subj_r2:.4f}")

    subj_test_df["pred_value_calibrated"] = apply_metadata_calibrator(subj_test_df, meta_cal, pred_col="pred_value")
    cal_mae, cal_rmse, cal_r, cal_r2 = metrics_from_arrays(
        subj_test_df[Config.TARGET_COL].values.astype(np.float32),
        subj_test_df["pred_value_calibrated"].values.astype(np.float32),
    )
    print("\n=== TEST Subject-Level Metrics (Metadata-Calibrated) ===")
    print(f"MAE: {cal_mae:.4f}")
    print(f"RMSE: {cal_rmse:.4f}")
    print(f"Pearson r: {cal_r:.4f}")
    print(f"R2: {cal_r2:.4f}")

    out_test_csv = outdir / "subject_level_predictions.csv"
    subj_test_df.to_csv(out_test_csv, index=False)
    print(f"Saved TEST subject-level results to: {out_test_csv}")

    # ---- TRAIN+VAL subject-level predictions ----
    print("\nGenerating TRAIN+VAL subject-level predictions...")
    trainval_ds = LabeledECGPPGDataset(df_trainval, Config.DATA_DIR, Config)
    trainval_loader = DataLoader(trainval_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    trainval_preds, trainval_targets, trainval_ssoids = predict_records(model, trainval_loader, Config.DEVICE, Config.MODE)

    trv_rec_mae, trv_rec_rmse, trv_rec_r, trv_rec_r2 = metrics_from_arrays(trainval_targets, trainval_preds)

    subj_trainval_df = subject_level_aggregate(df_trainval, trainval_preds, trainval_ssoids)
    trv_subj_mae, trv_subj_rmse, trv_subj_r, trv_subj_r2 = metrics_from_arrays(
        subj_trainval_df[Config.TARGET_COL].values.astype(np.float32),
        subj_trainval_df["pred_value"].values.astype(np.float32),
    )

    out_trainval_csv = outdir / "subject_level_predictions_trainval.csv"
    subj_trainval_df.to_csv(out_trainval_csv, index=False)
    print(f"Saved TRAIN+VAL subject-level results to: {out_trainval_csv}")

    # ---- Save metrics.json ----
    metrics = {
        "config": {
            "MODE": Config.MODE,
            "FREEZE_ENCODERS": Config.FREEZE_ENCODERS,
            "PRETRAINED_CKPT": Config.PRETRAINED_CKPT,
            "TARGET_COL": Config.TARGET_COL,
            "USE_ZSCORE": Config.USE_ZSCORE,
            "BATCH_SIZE": Config.BATCH_SIZE,
            "LR": Config.LR,
            "WEIGHT_DECAY": Config.WEIGHT_DECAY,
            "EPOCHS": Config.EPOCHS,
            "PATIENCE": Config.PATIENCE,
            "LOSS_MODE": Config.LOSS_MODE,
            "LOSS_HPARAMS": {
                "LAMBDA_DIST": Config.LAMBDA_DIST if Config.LOSS_MODE == "mse+dist" else None,
                "DIST_KDE_SIGMA": Config.DIST_KDE_SIGMA if Config.LOSS_MODE == "mse+dist" else None,
                "DIST_ASSIGN_SIGMA": Config.DIST_ASSIGN_SIGMA if Config.LOSS_MODE == "mse+dist" else None,
                "DIST_INV_WEIGHT": bool(Config.DIST_INV_WEIGHT) if Config.LOSS_MODE == "mse+dist" else None,
                "WMSE_SMOOTHING": Config.WMSE_SMOOTHING if Config.LOSS_MODE == "wmse" else None,
                "PEARSON_ALPHA": Config.PEARSON_ALPHA if Config.LOSS_MODE == "mse*pearson" else None,
                "Y_MIN": y_min,
                "Y_MAX": y_max,
            },
        },
        "val": {
            "best_mae": float(best_val_mae),
        },
        "test": {
            "record_level": {"MAE": rec_mae, "RMSE": rec_rmse, "r": rec_r, "R2": rec_r2},
            "subject_level": {"MAE": subj_mae, "RMSE": subj_rmse, "r": subj_r, "R2": subj_r2},
            "subject_level_calibrated": {"MAE": cal_mae, "RMSE": cal_rmse, "r": cal_r, "R2": cal_r2},
        },
        "calibration": meta_cal,
        "trainval": {
            "record_level": {"MAE": trv_rec_mae, "RMSE": trv_rec_rmse, "r": trv_rec_r, "R2": trv_rec_r2},
            "subject_level": {"MAE": trv_subj_mae, "RMSE": trv_subj_rmse, "r": trv_subj_r, "R2": trv_subj_r2},
        },
    }

    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
