#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BP data preprocessing pipeline (Kailuan H5 -> NPZ, keep BOTH raw+zscore)

Run directly:
    python /home/youliang/youliang_data2/bp/bp_recode_v1/prepare_bp_data.py

What it saves per NPZ:
- ecg, ppg: (default) z-scored signals (resampled)
- ecg_raw, ppg_raw: resampled but NOT normalized
- ecg_mu, ecg_sd, ppg_mu, ppg_sd: stats computed on raw resampled signals
- fs_ecg, fs_ppg, orig_fs, truncate_start, target_raw_len
"""

print("start")

import argparse
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm


# ---------------------------
# Defaults (edit here if you want)
# ---------------------------
DEF_H5_PATH = "/home/youliang/youliang_data2/bp/kailuan_dataset.h5"
DEF_OUT_DIR = "/home/youliang/youliang_data2/bp/bp_recode_v1/output"

DEF_ORIG_FREQ = 600
DEF_ECG_FREQ  = 500
DEF_PPG_FREQ  = 50

DEF_TRUNCATE_START = 600
DEF_TARGET_RAW_LEN = 3630

DEF_TRAIN_RATIO = 0.7
DEF_VAL_RATIO   = 0.15
DEF_TEST_RATIO  = 0.15
DEF_SEED        = 42


def create_subject_uid(name: str, age: float, sex: str) -> str:
    """Create subject UID from (name, age, sex)."""
    key = f"{name}_{age}_{sex}".strip().lower()
    return hashlib.md5(key.encode()).hexdigest()[:16]


def resample_signal(signal: np.ndarray, orig_freq: int, target_freq: int) -> np.ndarray:
    """Anti-aliasing resampling using resample_poly."""
    if orig_freq == target_freq:
        return signal.astype(np.float32)

    import math
    g = math.gcd(orig_freq, target_freq)
    up = target_freq // g
    down = orig_freq // g

    try:
        y = resample_poly(signal, up, down, padtype="line")
    except TypeError:
        y = resample_poly(signal, up, down)

    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="BP data preprocessing (Kailuan H5 -> NPZ, keep raw+zscore)")
    ap.add_argument("--h5_path", default=DEF_H5_PATH)
    ap.add_argument("--output_dir", default=DEF_OUT_DIR)

    ap.add_argument("--orig_freq", type=int, default=DEF_ORIG_FREQ)
    ap.add_argument("--ecg_freq", type=int, default=DEF_ECG_FREQ)
    ap.add_argument("--ppg_freq", type=int, default=DEF_PPG_FREQ)

    ap.add_argument("--truncate_start", type=int, default=DEF_TRUNCATE_START,
                    help="Drop first N samples from BOTH ECG & PPG to keep alignment.")
    ap.add_argument("--target_raw_len", type=int, default=DEF_TARGET_RAW_LEN,
                    help="After truncation, crop/pad BOTH ECG & PPG to this raw length (at orig_freq).")

    ap.add_argument("--train_ratio", type=float, default=DEF_TRAIN_RATIO)
    ap.add_argument("--val_ratio", type=float, default=DEF_VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=DEF_TEST_RATIO)
    ap.add_argument("--seed", type=int, default=DEF_SEED)

    ap.add_argument("--no_normalize", action="store_true")

    args = ap.parse_args()

    s = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {s}")

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 not found: {h5_path}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_dir = outdir / "npz"
    npz_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # Read metadata directly from H5 (do NOT trust metadata/uid/new_id)
    # ---------------------------------------------------------
    import h5py

    bp_cols = [
        "right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp",
        "left_arm_sbp", "left_arm_mbp", "left_arm_dbp", "left_arm_pp"
    ]

    with h5py.File(str(h5_path), "r") as h5f:
        meta = h5f["metadata"]

        name = meta["name"][:]
        age  = meta["age"][:]
        sex  = meta["sex"][:]

        if name.dtype.kind in ("S", "O"):
            name = np.array([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in name], dtype=object)
        if sex.dtype.kind in ("S", "O"):
            sex = np.array([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in sex], dtype=object)

        meta_df = pd.DataFrame({
            "name": name,
            "age": age,
            "sex": sex,
        })
        for c in bp_cols:
            meta_df[c] = meta[c][:]

        meta_df["h5_index"] = np.arange(len(meta_df), dtype=int)

        meta_df["subject_uid"] = meta_df.apply(
            lambda r: create_subject_uid(str(r["name"]), float(r["age"]), str(r["sex"])),
            axis=1
        )

        print(f"Metadata raw count = {len(meta_df)}")

        finite_mask = np.ones(len(meta_df), dtype=bool)
        for c in bp_cols:
            finite_mask &= np.isfinite(meta_df[c].to_numpy(dtype=np.float64))
        meta_df = meta_df[finite_mask].copy().reset_index(drop=True)

        print(f"Valid records (finite BP) = {len(meta_df)}")
        if len(meta_df) == 0:
            print("No valid data after BP finite filter.")
            return

        ecg_ds = h5f["signals"]["ecg"]
        ppg_ds = h5f["signals"]["ppg"]
        print(f"H5 shape: ECG={ecg_ds.shape}, PPG={ppg_ds.shape}")

        records = []
        skipped = 0

        start = int(args.truncate_start)
        target_raw_len = int(args.target_raw_len)

        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Processing"):
            idx_h5 = int(row["h5_index"])
            if idx_h5 >= ecg_ds.shape[0] or idx_h5 >= ppg_ds.shape[0]:
                skipped += 1
                continue

            ecg_raw0 = np.array(ecg_ds[idx_h5], dtype=np.float32)
            ppg_raw0 = np.array(ppg_ds[idx_h5], dtype=np.float32)

            if ecg_raw0.size == 0 or ppg_raw0.size == 0:
                skipped += 1
                continue
            if (not np.isfinite(ecg_raw0).all()) or (not np.isfinite(ppg_raw0).all()):
                skipped += 1
                continue

            # synchronized truncation: drop first N samples from BOTH
            if start > 0:
                if len(ecg_raw0) <= start or len(ppg_raw0) <= start:
                    skipped += 1
                    continue
                ecg_raw0 = ecg_raw0[start:]
                ppg_raw0 = ppg_raw0[start:]

            # synchronized crop/pad to fixed raw length (at orig_freq)
            if len(ecg_raw0) < target_raw_len:
                tmp = np.zeros(target_raw_len, dtype=np.float32)
                tmp[:len(ecg_raw0)] = ecg_raw0
                ecg_raw0 = tmp
            else:
                ecg_raw0 = ecg_raw0[:target_raw_len]

            if len(ppg_raw0) < target_raw_len:
                tmp = np.zeros(target_raw_len, dtype=np.float32)
                tmp[:len(ppg_raw0)] = ppg_raw0
                ppg_raw0 = tmp
            else:
                ppg_raw0 = ppg_raw0[:target_raw_len]

            # resampling (anti-alias)
            try:
                ecg_res_raw = resample_signal(ecg_raw0, args.orig_freq, args.ecg_freq)  # resampled, NOT normalized
                ppg_res_raw = resample_signal(ppg_raw0, args.orig_freq, args.ppg_freq)  # resampled, NOT normalized
            except Exception:
                skipped += 1
                continue

            # compute stats on raw resampled (before any normalization)
            eps = 1e-8
            ecg_mu = float(ecg_res_raw.mean())
            ecg_sd = float(ecg_res_raw.std())
            if ecg_sd < eps:
                ecg_sd = eps

            ppg_mu = float(ppg_res_raw.mean())
            ppg_sd = float(ppg_res_raw.std())
            if ppg_sd < eps:
                ppg_sd = eps

            # z-score versions (for CLIP fine-tune)
            if args.no_normalize:
                ecg_res = ecg_res_raw
                ppg_res = ppg_res_raw
            else:
                ecg_res = ((ecg_res_raw - ecg_mu) / ecg_sd).astype(np.float32)
                ppg_res = ((ppg_res_raw - ppg_mu) / ppg_sd).astype(np.float32)

            ssoid = str(idx_h5)
            np.savez_compressed(
                npz_dir / f"{ssoid}.npz",
                # normalized (default)
                ecg=ecg_res, ppg=ppg_res,
                # raw (resampled but not normalized)
                ecg_raw=ecg_res_raw, ppg_raw=ppg_res_raw,
                # stats
                ecg_mu=ecg_mu, ecg_sd=ecg_sd,
                ppg_mu=ppg_mu, ppg_sd=ppg_sd,
                # meta
                orig_fs=int(args.orig_freq),
                fs_ecg=int(args.ecg_freq),
                fs_ppg=int(args.ppg_freq),
                truncate_start=int(args.truncate_start),
                target_raw_len=int(target_raw_len),
            )

            rec = {
                "ssoid": ssoid,
                "h5_index": idx_h5,
                "subject_uid": row["subject_uid"],
                "age": float(row["age"]),
                "sex": str(row["sex"]),
                "ecg_len": int(len(ecg_res_raw)),
                "ppg_len": int(len(ppg_res_raw)),
                "ecg_mu": ecg_mu,
                "ecg_sd": ecg_sd,
                "ppg_mu": ppg_mu,
                "ppg_sd": ppg_sd,
            }
            for c in bp_cols:
                rec[c] = float(row[c])
            records.append(rec)

    df = pd.DataFrame(records)
    print(f"Processed = {len(df)}, skipped = {skipped}")
    if len(df) == 0:
        print("No processed samples.")
        return

    np.random.seed(args.seed)
    subjects = df["subject_uid"].unique()
    np.random.shuffle(subjects)

    n_train = int(len(subjects) * args.train_ratio)
    n_val = int(len(subjects) * args.val_ratio)

    train_sub = set(subjects[:n_train])
    val_sub = set(subjects[n_train:n_train + n_val])
    test_sub = set(subjects[n_train + n_val:])

    df["split"] = df["subject_uid"].map(
        lambda x: "train" if x in train_sub else ("val" if x in val_sub else "test")
    )

    df.to_csv(outdir / "labels.csv", index=False)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        split_df[["ssoid"]].to_csv(outdir / f"{split}.txt", index=False, header=False)

    config = {
        "h5_path": str(h5_path),
        "orig_freq": args.orig_freq,
        "ecg_freq": args.ecg_freq,
        "ppg_freq": args.ppg_freq,
        "truncate_start": args.truncate_start,
        "target_raw_len": target_raw_len,
        "normalized_signal_saved_as": "ecg/ppg (unless --no_normalize)",
        "raw_resampled_saved_as": "ecg_raw/ppg_raw",
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "n_subjects": int(len(subjects)),
        "n_records": int(len(df)),
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Saved NPZ to {npz_dir}")
    print("Each NPZ keys: ecg, ppg, ecg_raw, ppg_raw, ecg_mu, ecg_sd, ppg_mu, ppg_sd, fs_ecg, fs_ppg, orig_fs, ...")


if __name__ == "__main__":
    main()
