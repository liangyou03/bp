#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动化参数搜索脚本 V2 - 血压预测
Automated Grid Search for BP Prediction (更多实验组合)

功能:
1. 自动运行多组参数组合
2. 每组实验保存到独立文件夹
3. 自动生成对比报告 (Markdown + CSV)
4. 支持断点续传（跳过已完成的实验）

Usage:
    python grid_search_bp_v2.py --gpu 0 --phases phase2_ppg_lr phase2_both_lr
"""

import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse


# ===================== 实验配置 =====================
BASE_CONFIG = {
    "npz_dir": "/home/youliang/youliang_data2/bp/bp_npz_truncate/npz",
    "labels_csv": "/home/youliang/youliang_data2/bp/bp_npz_truncate/labels.csv",
    "ecg_ckpt": "/home/youliang/youliang_data2/bp/ppg_ecg_age/1_lead_ECGFounder.pth",
    "ppg_ckpt": "/home/youliang/youliang_data2/bp/ppg_ecg_age/best_checkpoint.pth",
    "batch_size": 64,
    "epochs": 40,
    "patience": 5,
    "num_workers": 4,
    "seed": 666,
}

# 搜索空间定义
SEARCH_SPACE = {
    # Phase 1: 基础对比 (已完成)
    "phase1_basic": [
        {"modality": "ppg", "freeze_backbone": True, "lr_head": 3e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4},
        {"modality": "ecg", "freeze_backbone": True, "lr_head": 3e-4},
        {"modality": "ecg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4},
        {"modality": "both", "freeze_backbone": True, "lr_head": 3e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4},
    ],
    
    # Phase 2a: PPG 学习率优化
    "phase2_ppg_lr": [
        # 调整 backbone 学习率 (固定 head=3e-4)
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 3e-6, "lr_head": 3e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 5e-6, "lr_head": 3e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 3e-5, "lr_head": 3e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 5e-5, "lr_head": 3e-4},
        
        # 调整 head 学习率 (固定 backbone=1e-5)
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 1e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 5e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 1e-3},
        
        # 同步调整
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 5e-6, "lr_head": 1e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 3e-5, "lr_head": 5e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 5e-5, "lr_head": 1e-3},
    ],
    
    # Phase 2b: Both 学习率优化
    "phase2_both_lr": [
        # 调整 backbone 学习率
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 3e-6, "lr_head": 3e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 5e-6, "lr_head": 3e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 3e-5, "lr_head": 3e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 5e-5, "lr_head": 3e-4},
        
        # 调整 head 学习率
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 1e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 5e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 1e-3},
        
        # 同步调整
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 5e-6, "lr_head": 1e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 3e-5, "lr_head": 5e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 5e-5, "lr_head": 1e-3},
    ],
    
    # Phase 3a: PPG 损失函数对比
    "phase3_ppg_loss": [
        # MAE+Pearson 变体
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.3, "maepearson_beta": 0.5},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.5, "maepearson_beta": 0.5},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.7, "maepearson_beta": 0.5},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.5, "maepearson_beta": 0.3},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.5, "maepearson_beta": 0.7},
        
        # 其他损失函数
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mse"},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "huber"},
    ],
    
    # Phase 3b: Both 损失函数对比
    "phase3_both_loss": [
        # MAE+Pearson 变体
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.3, "maepearson_beta": 0.5},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.5, "maepearson_beta": 0.5},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.7, "maepearson_beta": 0.5},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.5, "maepearson_beta": 0.3},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.5, "maepearson_beta": 0.7},
        
        # 其他损失函数
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mse"},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "huber"},
    ],
    
    # Phase 4a: PPG Weight Decay
    "phase4_ppg_wd": [
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "weight_decay": 0},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "weight_decay": 1e-5},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "weight_decay": 5e-5},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "weight_decay": 1e-3},
    ],
    
    # Phase 4b: Both Weight Decay
    "phase4_both_wd": [
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "weight_decay": 0},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "weight_decay": 1e-5},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "weight_decay": 5e-5},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "weight_decay": 1e-3},
    ],
}

# 默认的 target_cols
DEFAULT_TARGET_COLS = [
    "right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp",
    "left_arm_sbp", "left_arm_mbp", "left_arm_dbp", "left_arm_pp"
]


def generate_exp_name(config):
    """根据配置生成实验名称"""
    parts = []
    parts.append(f"mod_{config['modality']}")
    parts.append("freeze" if config.get("freeze_backbone", False) else "finetune")
    
    if not config.get("freeze_backbone", False):
        parts.append(f"lrb{config.get('lr_backbone', 1e-5):.0e}")
    parts.append(f"lrh{config.get('lr_head', 3e-4):.0e}")
    
    # 添加 weight_decay
    if "weight_decay" in config:
        wd = config["weight_decay"]
        if wd == 0:
            parts.append("wd0")
        else:
            parts.append(f"wd{wd:.0e}")
    
    # 添加损失函数
    if "reg_loss" in config and config["reg_loss"] != "mae_pearson":
        parts.append(f"loss_{config['reg_loss']}")
    elif config.get("reg_loss") == "mae_pearson":
        alpha = config.get("maepearson_alpha", 0.5)
        beta = config.get("maepearson_beta", 0.5)
        parts.append(f"maep_a{alpha:.1f}_b{beta:.1f}")
    
    return "_".join(parts)


def build_command(config, output_dir, gpu):
    """构建训练命令"""
    cmd = [
        "python", "finetune_bp.py",
        "--npz_dir", config["npz_dir"],
        "--labels_csv", config["labels_csv"],
        "--ecg_ckpt", config["ecg_ckpt"],
        "--ppg_ckpt", config["ppg_ckpt"],
        "--out_dir", str(output_dir),
        "--modality", config["modality"],
        "--batch_size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--patience", str(config["patience"]),
        "--num_workers", str(config["num_workers"]),
        "--seed", str(config["seed"]),
        "--gpu", str(gpu),
    ]
    
    # 添加 target_cols
    cmd.extend(["--target_cols"] + config.get("target_cols", DEFAULT_TARGET_COLS))
    
    # 学习率
    if config.get("freeze_backbone", False):
        cmd.extend(["--freeze_backbone"])
        cmd.extend(["--lr_head", f"{config['lr_head']:.0e}"])
    else:
        cmd.extend(["--lr_backbone", f"{config.get('lr_backbone', 1e-5):.0e}"])
        cmd.extend(["--lr_head", f"{config['lr_head']:.0e}"])
    
    # 正则化
    if "weight_decay" in config:
        cmd.extend(["--weight_decay", f"{config['weight_decay']:.0e}"])
    
    # 损失函数
    if "reg_loss" in config:
        cmd.extend(["--reg_loss", config["reg_loss"]])
    
    if config.get("reg_loss") == "mae_pearson":
        cmd.extend(["--maepearson_alpha", str(config.get("maepearson_alpha", 0.5))])
        cmd.extend(["--maepearson_beta", str(config.get("maepearson_beta", 0.5))])
    
    return cmd


def run_experiment(exp_name, config, base_output_dir, gpu):
    """运行单个实验"""
    output_dir = base_output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已完成
    if (output_dir / "test_metrics.json").exists():
        print(f"✓ Skip (already done): {exp_name}")
        return True, "skipped"
    
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}")
    
    # 构建完整配置
    full_config = {**BASE_CONFIG, **config}
    cmd = build_command(full_config, output_dir, gpu)
    
    # 保存命令
    with open(output_dir / "command.txt", "w") as f:
        f.write(" ".join(cmd) + "\n")
    
    # 运行训练
    log_file = output_dir / "train.log"
    try:
        with open(log_file, "w") as f:
            process = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True
            )
        print(f"✓ Success: {exp_name}")
        return True, "success"
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {exp_name}")
        print(f"  Check log: {log_file}")
        return False, "failed"
    except Exception as e:
        print(f"✗ Error: {exp_name}")
        print(f"  {str(e)}")
        return False, "error"


def main():
    parser = argparse.ArgumentParser(description="Grid search for BP prediction (V2 - more experiments)")
    parser.add_argument("--base_dir", default="/home/youliang/youliang_data2/bp/grid_search_results",
                        help="Base directory for all experiments")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--phases", nargs="+", 
                        default=["phase2_ppg_lr"],
                        choices=list(SEARCH_SPACE.keys()),
                        help="Which phases to run")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()
    
    base_output_dir = Path(args.base_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Blood Pressure Prediction - Grid Search V2")
    print("=" * 70)
    print(f"Base output directory: {base_output_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Phases: {args.phases}")
    print("=" * 70)
    
    # 选择要运行的 phases
    selected_phases = {k: v for k, v in SEARCH_SPACE.items() if k in args.phases}
    
    # 统计实验数量
    total_exps = sum(len(configs) for configs in selected_phases.values())
    print(f"\nTotal experiments to run: {total_exps}")
    
    # 显示每个 phase 的实验数
    for phase_name, configs in selected_phases.items():
        print(f"  {phase_name}: {len(configs)} experiments")
    
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for phase_name, configs in selected_phases.items():
            print(f"\n{phase_name}:")
            for config in configs:
                exp_name = generate_exp_name(config)
                print(f"  - {exp_name}")
        return
    
    # 运行实验
    results_summary = []
    for phase_name, configs in selected_phases.items():
        print(f"\n{'='*70}")
        print(f"Phase: {phase_name} ({len(configs)} experiments)")
        print(f"{'='*70}")
        
        for idx, config in enumerate(configs, 1):
            exp_name = generate_exp_name(config)
            print(f"\n[{idx}/{len(configs)}] {phase_name}")
            
            success, status = run_experiment(exp_name, config, base_output_dir, args.gpu)
            results_summary.append({
                "phase": phase_name,
                "exp_name": exp_name,
                "status": status
            })
    
    # 统计
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    
    df_summary = pd.DataFrame(results_summary)
    for status in ["success", "skipped", "failed", "error"]:
        count = len(df_summary[df_summary["status"] == status])
        if count > 0:
            print(f"{status.capitalize()}: {count}")
    
    print(f"\nRun collect_results.py to generate report!")
    print("="*70)


if __name__ == "__main__":
    main()