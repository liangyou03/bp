#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动化参数搜索脚本 - 血压预测
Automated Grid Search for BP Prediction

功能:
1. 自动运行多组参数组合
2. 每组实验保存到独立文件夹
3. 自动生成对比报告 (Markdown + CSV)
4. 支持断点续传（跳过已完成的实验）

Usage:
    python grid_search_bp.py --gpu 0
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
    # Phase 1: 基础对比 (信号类型 + 训练策略)
    "phase1_basic": [
        # PPG only
        {"modality": "ppg", "freeze_backbone": True, "lr_head": 3e-4},
        {"modality": "ppg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4},
        
        # ECG only
        {"modality": "ecg", "freeze_backbone": True, "lr_head": 3e-4},
        {"modality": "ecg", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4},
        
        # Both (synchronized)
        {"modality": "both", "freeze_backbone": True, "lr_head": 3e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4},
    ],
    
    # Phase 2: 学习率优化 (基于 Phase 1 最佳模态)
    "phase2_lr": [
        # 针对最佳模态，尝试不同学习率组合
        # 这里以 both 为例，实际使用时根据 Phase 1 结果调整
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 5e-6, "lr_head": 1e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 1e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 5e-5, "lr_head": 1e-4},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 1e-3},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 5e-5, "lr_head": 1e-3},
    ],
    
    # Phase 3: 损失函数对比
    "phase3_loss": [
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mse"},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "huber"},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.3, "maepearson_beta": 0.5},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.5, "maepearson_beta": 0.5},
        {"modality": "both", "freeze_backbone": False, "lr_backbone": 1e-5, "lr_head": 3e-4, 
         "reg_loss": "mae_pearson", "maepearson_alpha": 0.7, "maepearson_beta": 0.5},
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
    if (output_dir / "final_results.json").exists():
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


def collect_results(base_output_dir, phases):
    """收集所有实验结果"""
    results = []
    
    for phase_name, configs in phases.items():
        for config in configs:
            exp_name = generate_exp_name(config)
            output_dir = base_output_dir / exp_name
            result_file = output_dir / "final_results.json"
            
            if not result_file.exists():
                continue
            
            with open(result_file, "r") as f:
                data = json.load(f)
            
            # 提取关键信息
            row = {
                "exp_name": exp_name,
                "phase": phase_name,
                "modality": config["modality"],
                "freeze": config.get("freeze_backbone", False),
                "lr_backbone": config.get("lr_backbone", "-"),
                "lr_head": config["lr_head"],
                "reg_loss": config.get("reg_loss", "mae_pearson"),
                "test_avg_mae": data["test_avg_mae"],
                "best_epoch": data["best_epoch"],
            }
            
            # 添加每个目标的 MAE
            for target, metrics in data["test_metrics"].items():
                row[f"{target}_mae"] = metrics["mae"]
                row[f"{target}_r"] = metrics["r"]
            
            results.append(row)
    
    return pd.DataFrame(results)


def generate_report(df, output_path):
    """生成 Markdown 格式的对比报告"""
    lines = []
    lines.append("# Blood Pressure Prediction - Grid Search Results")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal experiments: {len(df)}")
    
    if len(df) == 0:
        lines.append("\n**No completed experiments found.**")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        return
    
    # 按 Phase 分组
    for phase in df["phase"].unique():
        df_phase = df[df["phase"] == phase].sort_values("test_avg_mae")
        
        lines.append(f"\n## {phase}")
        lines.append(f"\n### Top 3 Models (by Average MAE)")
        
        # 表格
        lines.append("\n| Rank | Exp Name | Modality | Freeze | LR Head | Loss | Avg MAE | Best Epoch |")
        lines.append("|------|----------|----------|--------|---------|------|---------|------------|")
        
        for idx, row in df_phase.head(3).iterrows():
            lines.append(
                f"| {idx+1} | `{row['exp_name']}` | {row['modality']} | "
                f"{row['freeze']} | {row['lr_head']:.0e} | {row['reg_loss']} | "
                f"**{row['test_avg_mae']:.2f}** | {row['best_epoch']} |"
            )
        
        # 详细指标（最佳模型）
        best_row = df_phase.iloc[0]
        lines.append(f"\n### Best Model: `{best_row['exp_name']}`")
        lines.append("\n**Per-target Performance:**")
        lines.append("\n| Target | MAE | Pearson r |")
        lines.append("|--------|-----|-----------|")
        
        targets = ["right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp",
                   "left_arm_sbp", "left_arm_mbp", "left_arm_dbp", "left_arm_pp"]
        for target in targets:
            if f"{target}_mae" in best_row:
                lines.append(f"| {target} | {best_row[f'{target}_mae']:.2f} | {best_row[f'{target}_r']:.3f} |")
    
    # 总体最佳
    lines.append("\n## Overall Best Model")
    best_overall = df.sort_values("test_avg_mae").iloc[0]
    lines.append(f"\n- **Experiment**: `{best_overall['exp_name']}`")
    lines.append(f"- **Modality**: {best_overall['modality']}")
    lines.append(f"- **Average MAE**: **{best_overall['test_avg_mae']:.2f} mmHg**")
    lines.append(f"- **Best Epoch**: {best_overall['best_epoch']}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n✓ Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid search for BP prediction")
    parser.add_argument("--base_dir", default="/home/youliang/youliang_data2/bp/grid_search_results",
                        help="Base directory for all experiments")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--phases", nargs="+", default=["phase1_basic"],
                        choices=["phase1_basic", "phase2_lr", "phase3_loss"],
                        help="Which phases to run")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()
    
    base_output_dir = Path(args.base_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Blood Pressure Prediction - Grid Search")
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
    
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for phase_name, configs in selected_phases.items():
            print(f"\n{phase_name}:")
            for config in configs:
                exp_name = generate_exp_name(config)
                full_config = {**BASE_CONFIG, **config}
                cmd = build_command(full_config, base_output_dir / exp_name, args.gpu)
                print(f"  {exp_name}:")
                print(f"    {' '.join(cmd)}")
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
    
    # 收集结果
    print("\n" + "="*70)
    print("Collecting results...")
    print("="*70)
    
    df_results = collect_results(base_output_dir, selected_phases)
    
    if len(df_results) > 0:
        # 保存 CSV
        csv_path = base_output_dir / "results_summary.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"✓ Results CSV saved: {csv_path}")
        
        # 生成报告
        report_path = base_output_dir / "REPORT.md"
        generate_report(df_results, report_path)
        
        # 打印简要总结
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Completed: {len(df_results)} / {total_exps}")
        if len(df_results) > 0:
            best = df_results.sort_values("test_avg_mae").iloc[0]
            print(f"\nBest Model: {best['exp_name']}")
            print(f"  Average MAE: {best['test_avg_mae']:.2f} mmHg")
            print(f"  Modality: {best['modality']}")
    else:
        print("\n[Warning] No completed experiments found.")
    
    print(f"\nAll results saved to: {base_output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()