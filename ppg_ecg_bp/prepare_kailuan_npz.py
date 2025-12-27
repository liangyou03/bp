#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Kailuan HDF5 dataset into per-record NPZ files + labels CSV for BP training.

两种对齐模式：
  --align_mode truncate : PPG 截断前 600 点，两者都变成 3630
  --align_mode resample : ECG 重采样到 PPG 长度 (4230)

Example:
python prepare_kailuan_npz.py \
    --h5_path /home/youliang/youliang_data2/bp/kailuan_dataset.h5 \
    --id_key new_id \
    --target_cols right_arm_sbp right_arm_mbp right_arm_dbp left_arm_sbp left_arm_mbp left_arm_dbp left_arm_pp right_arm_pp\
    --output_npz /home/youliang/youliang_data2/bp/bp_npz_truncate/npz \
    --output_csv /home/youliang/youliang_data2/bp/bp_npz_truncate/labels.csv \
    --align_mode truncate


python prepare_kailuan_npz.py \
  --h5_path /home/youliang/youliang_data2/bp/kailuan_dataset.h5 \
  --id_key new_id \
  --target_cols right_arm_sbp right_arm_mbp right_arm_dbp left_arm_sbp left_arm_mbp left_arm_dbp left_arm_pp right_arm_pp\
  --output_npz /home/youliang/youliang_data2/bp/bp_npz_resample/npz \
  --output_csv /home/youliang/youliang_data2/bp/bp_npz_resample/labels.csv \
  --align_mode resample
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


def resample_signal(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Resample 1D signal to target_len using linear interpolation."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if signal.size == target_len:
        return signal
    x_old = np.linspace(0.0, 1.0, num=signal.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
    resampled = np.interp(x_new, x_old, signal).astype(np.float32)
    return resampled


def main():
    ap = argparse.ArgumentParser(description="Prepare Kailuan ECG/PPG npz files and BP labels CSV.")
    ap.add_argument("--h5_path", required=True, help="Path to kailuan_dataset.h5")
    ap.add_argument("--output_npz", required=True, help="Directory to store <ssoid>.npz files.")
    ap.add_argument("--output_csv", required=True, help="Path to write labels CSV.")
    ap.add_argument("--id_key", default="new_id", help="Metadata field used as unique ssoid.")
    ap.add_argument("--target_cols", nargs="+", required=True, help="BP metadata columns to predict.")
    ap.add_argument("--align_mode", choices=["truncate", "resample"], default="truncate",
                    help="truncate: PPG截断前600点(两者3630); resample: ECG重采样到PPG长度(4230)")
    ap.add_argument("--lowercase_ids", action="store_true", help="Convert IDs to lowercase.")
    args = ap.parse_args()

    out_dir = Path(args.output_npz)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    skipped_nan = 0

    with h5py.File(args.h5_path, "r") as h5f:
        metadata = h5f["metadata"]
        signals = h5f["signals"]
        ecg_ds = signals["ecg"]
        ppg_ds = signals["ppg"]

        if args.id_key not in metadata:
            raise RuntimeError(f"id_key '{args.id_key}' not found in metadata group.")
        for col in args.target_cols:
            if col not in metadata:
                raise RuntimeError(f"target column '{col}' not present in metadata group.")

        total = metadata[args.id_key].shape[0]
        for idx in tqdm(range(total), desc="Converting records"):
            raw_id = metadata[args.id_key][idx]
            if isinstance(raw_id, bytes):
                raw_id = raw_id.decode("utf-8")
            ssoid = str(raw_id).strip()
            if args.lowercase_ids:
                ssoid = ssoid.lower()
            if len(ssoid) == 0:
                continue

            ecg = np.array(ecg_ds[idx], dtype=np.float32)
            ppg = np.array(ppg_ds[idx], dtype=np.float32)
            if ecg.size == 0 or ppg.size == 0:
                continue

            # 对齐 ECG 和 PPG
            if args.align_mode == "truncate":
                # PPG 截断前 600 点，变成 3630；ECG 保持 3630
                ppg_aligned = ppg[600:] if ppg.size > 600 else ppg  # 4230 -> 3630
                ecg_aligned = ecg  # 3630
                # 确保长度一致（取较短的）
                min_len = min(ecg_aligned.size, ppg_aligned.size)
                ecg_aligned = ecg_aligned[:min_len]
                ppg_aligned = ppg_aligned[:min_len]
            else:  # resample
                # ECG 重采样到 PPG 长度 (4230)
                target_len = ppg.size
                ecg_aligned = resample_signal(ecg, target_len)
                ppg_aligned = ppg

            # 跳过含有 NaN 或 Inf 的记录
            if (np.isnan(ecg_aligned).any() or np.isnan(ppg_aligned).any() or 
                np.isinf(ecg_aligned).any() or np.isinf(ppg_aligned).any()):
                print(f"[skip] {ssoid} contains NaN/Inf")
                skipped_nan += 1
                continue

            stacked = np.stack([ecg_aligned, ppg_aligned], axis=1)  # (seq_len, 2)
            np.savez_compressed(out_dir / f"{ssoid}.npz", x=stacked)

            record = {"ssoid": ssoid}
            skip_record = False
            for col in args.target_cols:
                val = metadata[col][idx]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                try:
                    record[col] = float(val)
                except (TypeError, ValueError):
                    skip_record = True
                    break
            if not skip_record:
                labels.append(record)

    if not labels:
        raise RuntimeError("No valid records were exported.")

    df = pd.DataFrame(labels)
    df = df.drop_duplicates(subset="ssoid")
    df.to_csv(args.output_csv, index=False)
    
    # 打印信息
    seq_len = 3630 if args.align_mode == "truncate" else 4230
    print(f"[done] align_mode={args.align_mode}, seq_len={seq_len}")
    print(f"[done] wrote {len(df)} label rows to {args.output_csv}")
    print(f"[done] saved npz files to {out_dir}")
    print(f"[done] skipped {skipped_nan} records with NaN/Inf")


if __name__ == "__main__":
    main()