#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BP实验运行器 - 逐个运行实验避免OOM
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def build_cmd(exp_config, base_config, gpu, out_dir):
    """构建 finetune_bp.py 的命令行参数"""
    data_dir = exp_config.get("data_dir") or base_config.get("data_dir")
    if data_dir is None:
        raise ValueError("data_dir is not specified in experiment or config")

    cmd = [
        "python", "finetune_bp.py",
        "--data_dir", str(data_dir),
        "--target_col", exp_config["target_col"],
        "--modality", exp_config.get("modality", "both"),
        "--loss", exp_config.get("loss", base_config.get("loss", "mae_pearson")),
        "--lr_backbone", str(exp_config.get("lr_backbone", base_config.get("lr_backbone", 1e-5))),
        "--lr_head", str(exp_config.get("lr_head", base_config.get("lr_head", 3e-4))),
        "--epochs", str(exp_config.get("epochs", base_config.get("epochs", 50))),
        "--batch_size", str(exp_config.get("batch_size", base_config.get("batch_size", 64))),
        "--patience", str(exp_config.get("patience", base_config.get("patience", 10))),
        "--gpu", str(gpu),
        "--out_dir", str(out_dir),
    ]

    if exp_config.get("freeze_backbone") or base_config.get("freeze_backbone"):
        cmd.append("--freeze_backbone")

    pretrain = exp_config.get("pretrain") or base_config.get("pretrain")
    if pretrain:
        cmd.extend(["--pretrain", str(pretrain)])

    pretrain_ecg = exp_config.get("pretrain_ecg") or base_config.get("pretrain_ecg")
    pretrain_ppg = exp_config.get("pretrain_ppg") or base_config.get("pretrain_ppg")
    if pretrain_ecg and pretrain_ppg:
        cmd.extend(["--pretrain_ecg", str(pretrain_ecg), "--pretrain_ppg", str(pretrain_ppg)])

    optional_keys = [
        ("y_min", "--y_min"),
        ("y_max", "--y_max"),
        ("subj_agg", "--subj_agg"),
        ("weight_decay", "--weight_decay"),
        ("alpha_corr", "--alpha_corr"),
    ]
    for key, flag in optional_keys:
        value = exp_config.get(key, base_config.get(key))
        if value is not None:
            cmd.extend([flag, str(value)])

    return cmd


def run_single_exp(exp_config, base_config, gpu, exp_name, base_out_dir):
    out_dir = Path(base_out_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = out_dir / "results.json"
    if results_file.exists():
        print(f"[Skip] {exp_name} already has results")
        return True

    cmd = build_cmd(exp_config, base_config, gpu, out_dir)

    print("\n" + "=" * 60)
    print(f"Running: {exp_name}")
    print(" ".join(cmd))
    print("=" * 60 + "\n")

    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"[Done] {exp_name} in {elapsed/60:.1f}min")
        return True
    else:
        print(f"[Failed] {exp_name}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments_config.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    experiments = config.get("experiments", [])

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(experiments)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running experiments {start_idx} to {end_idx-1} (total {end_idx-start_idx})")
    print(f"GPU: {args.gpu}")

    results = {"success": [], "failed": []}

    for i in range(start_idx, end_idx):
        exp = experiments[i]
        exp_name = exp.get("name", f"exp_{i:03d}")

        try:
            success = run_single_exp(exp, config, args.gpu, exp_name, output_dir)
        except Exception as exc:
            print(f"[Failed] {exp_name} (config error: {exc})")
            success = False

        if success:
            results["success"].append(exp_name)
        else:
            results["failed"].append(exp_name)

        with open(output_dir / "progress.json", "w") as f:
            json.dump(results, f, indent=2)

        subprocess.run(["python", "-c", "import torch; torch.cuda.empty_cache()"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Success: {len(results[success])}")
    print(f"Failed: {len(results[failed])}")
    if results[failed]:
        print(f"Failed experiments: {results[failed]}")


if __name__ == "__main__":
    main()
