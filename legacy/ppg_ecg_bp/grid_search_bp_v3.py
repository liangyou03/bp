#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid Search (V3) for BP finetuning on CLIP foundation
====================================================

Assumptions (IMPORTANT):
- You already have a CLIP foundation checkpoint:
    clip_foundation_best.pth
- finetune_bp.py takes ONLY --pretrain for initialization
- No ecg_ckpt / ppg_ckpt are used anymore

Focus:
- modality: ppg / ecg / both
- freeze vs finetune backbone
- lr_backbone / lr_head
"""

import subprocess
from pathlib import Path
import argparse
import json
import itertools

# ===================== 基础配置 =====================
BASE_CONFIG = {
    "npz_dir": "/home/youliang/youliang_data2/bp/bp_npz_truncate/npz",
    "labels_csv": "/home/youliang/youliang_data2/bp/bp_npz_truncate/labels.csv",
    "pretrain": "/home/youliang/youliang_data2/bp/ppg_ecg_clip_bp/run1/clip_foundation_best.pth",
    "epochs": 40,
    "patience": 6,
    "batch_size": 64,
    "num_workers": 4,
    "seed": 666,
}

TARGET_COLS = [
    "right_arm_sbp",
    "right_arm_mbp",
    "right_arm_dbp",
    "right_arm_pp",
    "left_arm_sbp",
    "left_arm_mbp",
    "left_arm_dbp",
    "left_arm_pp",
]

# ===================== 搜索空间 =====================
SEARCH_SPACE = {
    "modality": ["ppg", "ecg", "both"],
    "freeze_backbone": [True, False],
    "lr_backbone": [3e-6, 1e-5],   # only used if not frozen
    "lr_head": [1e-4, 3e-4],
}

# ===================== util =====================
def build_exp_name(cfg):
    parts = [
        f"mod_{cfg['modality']}",
        "freeze" if cfg["freeze_backbone"] else "finetune",
    ]
    if not cfg["freeze_backbone"]:
        parts.append(f"lrb{cfg['lr_backbone']:.0e}")
    parts.append(f"lrh{cfg['lr_head']:.0e}")
    return "_".join(parts)


def build_command(cfg, out_dir, gpu):
    cmd = [
        "python", "finetune_bp.py",
        "--npz_dir", cfg["npz_dir"],
        "--labels_csv", cfg["labels_csv"],
        "--pretrain", cfg["pretrain"],
        "--out_dir", str(out_dir),
        "--modality", cfg["modality"],
        "--batch_size", str(cfg["batch_size"]),
        "--epochs", str(cfg["epochs"]),
        "--patience", str(cfg["patience"]),
        "--num_workers", str(cfg["num_workers"]),
        "--seed", str(cfg["seed"]),
        "--gpu", str(gpu),
    ]

    cmd += ["--target_cols"] + TARGET_COLS

    if cfg["freeze_backbone"]:
        cmd.append("--freeze_backbone")
        cmd += ["--lr_head", f"{cfg['lr_head']:.0e}"]
    else:
        cmd += ["--lr_backbone", f"{cfg['lr_backbone']:.0e}"]
        cmd += ["--lr_head", f"{cfg['lr_head']:.0e}"]

    return cmd


# ===================== main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="./grid_runs_bp_v3",
                    help="Base output directory")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]

    all_cfgs = []
    for comb in itertools.product(*values):
        cfg = dict(zip(keys, comb))
        # skip invalid: frozen backbone should not specify lr_backbone
        if cfg["freeze_backbone"] is True:
            cfg["lr_backbone"] = None
        all_cfgs.append(cfg)

    print(f"Total experiments: {len(all_cfgs)}")

    for idx, cfg_delta in enumerate(all_cfgs, 1):
        cfg = {**BASE_CONFIG, **cfg_delta}
        exp_name = build_exp_name(cfg)
        out_dir = out_base / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "test_metrics.json").exists():
            print(f"[{idx}] skip {exp_name}")
            continue

        cmd = build_command(cfg, out_dir, args.gpu)

        print(f"\n[{idx}/{len(all_cfgs)}] {exp_name}")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        with open(out_dir / "command.txt", "w") as f:
            f.write(" ".join(cmd) + "\n")

        with open(out_dir / "train.log", "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
