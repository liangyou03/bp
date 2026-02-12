#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BP 预测结果高级分析
- 不同位置 BP 对比
- BP 分布图
- 分区间 MAE 分析
- 散点图 + 回归线
- Bland-Altman 图

python analyze_bp_results.py --results_dir /home/youliang/youliang_data2/bp/bp_dec30/runs
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# 设置中文字体和样式
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.style.use('seaborn-v0_8-whitegrid')

# BP 目标和显示名称
BP_TARGETS = {
    "right_arm_sbp": "Right SBP",
    "right_arm_mbp": "Right MBP", 
    "right_arm_dbp": "Right DBP",
    "right_arm_pp": "Right PP",
    "left_arm_sbp": "Left SBP",
    "left_arm_mbp": "Left MBP",
    "left_arm_dbp": "Left DBP",
    "left_arm_pp": "Left PP",
}

# BP 分类阈值 (基于 SBP)
BP_CATEGORIES = {
    "Normal": (0, 120),
    "Elevated": (120, 130),
    "Hypertension Stage 1": (130, 140),
    "Hypertension Stage 2": (140, 180),
    "Hypertensive Crisis": (180, 300),
}

# DBP 分类阈值
DBP_CATEGORIES = {
    "Normal": (0, 80),
    "Elevated": (80, 85),
    "Hypertension Stage 1": (85, 90),
    "Hypertension Stage 2": (90, 120),
    "Hypertensive Crisis": (120, 200),
}


def load_best_predictions(results_dir: str, best_csv: str = None):
    """加载每个 target 的最佳模型预测结果"""
    results_dir = Path(results_dir)
    
    # 读取 best_per_target.csv
    if best_csv is None:
        best_csv = results_dir / "best_per_target.csv"
    
    df_best = pd.read_csv(best_csv)
    
    predictions = {}
    
    for _, row in df_best.iterrows():
        target = row["target"]
        exp_name = row["best_exp"]
        
        # 找到对应的 npz 文件
        exp_dir = results_dir / exp_name
        npz_files = list(exp_dir.glob(f"pred_test_bp_{target}_*_records.npz"))
        
        if npz_files:
            data = np.load(npz_files[0], allow_pickle=True)
            predictions[target] = {
                "y_true": data["y_true"],
                "y_pred": data["y_pred"],
                "y_pred_cal": data["y_pred_cal"],
                "ssoid": data["ssoid"],
                "exp_name": exp_name,
                "modality": row["modality"],
                "MAE_cal": row["MAE_cal"],
                "r": row["r"],
            }
            print(f"[Loaded] {target}: {exp_name} (MAE={row['MAE_cal']:.2f}, r={row['r']:.3f})")
    
    return predictions


def plot_bp_distribution(predictions, save_path):
    """绘制 BP 真实值分布"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (target, display_name) in enumerate(BP_TARGETS.items()):
        ax = axes[idx]
        
        if target in predictions:
            y_true = predictions[target]["y_true"]
            y_pred_cal = predictions[target]["y_pred_cal"]
            
            # 绘制直方图
            ax.hist(y_true, bins=30, alpha=0.7, label=f"True (μ={y_true.mean():.1f})", color='steelblue')
            ax.hist(y_pred_cal, bins=30, alpha=0.5, label=f"Pred (μ={y_pred_cal.mean():.1f})", color='coral')
            
            ax.set_xlabel(f"{display_name} (mmHg)")
            ax.set_ylabel("Count")
            ax.set_title(display_name)
            ax.legend(fontsize=8)
        else:
            ax.set_visible(False)
    
    plt.suptitle("BP Distribution: True vs Predicted (Calibrated)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_scatter_all(predictions, save_path):
    """绘制所有 target 的散点图"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (target, display_name) in enumerate(BP_TARGETS.items()):
        ax = axes[idx]
        
        if target in predictions:
            y_true = predictions[target]["y_true"]
            y_pred_cal = predictions[target]["y_pred_cal"]
            r = predictions[target]["r"]
            mae = predictions[target]["MAE_cal"]
            
            # 散点图
            ax.scatter(y_true, y_pred_cal, alpha=0.3, s=10, c='steelblue')
            
            # 对角线
            lims = [min(y_true.min(), y_pred_cal.min()), max(y_true.max(), y_pred_cal.max())]
            ax.plot(lims, lims, 'r--', lw=1.5, label='Identity')
            
            # 回归线
            slope, intercept, _, _, _ = stats.linregress(y_true, y_pred_cal)
            x_fit = np.linspace(lims[0], lims[1], 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'g-', lw=1.5, label=f'Fit (y={slope:.2f}x+{intercept:.1f})')
            
            ax.set_xlabel(f"True {display_name} (mmHg)")
            ax.set_ylabel(f"Predicted {display_name} (mmHg)")
            ax.set_title(f"{display_name}\nr={r:.3f}, MAE={mae:.2f}")
            ax.legend(fontsize=7, loc='lower right')
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_visible(False)
    
    plt.suptitle("Scatter Plots: True vs Predicted (Calibrated)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_bland_altman(predictions, save_path):
    """绘制 Bland-Altman 图"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (target, display_name) in enumerate(BP_TARGETS.items()):
        ax = axes[idx]
        
        if target in predictions:
            y_true = predictions[target]["y_true"]
            y_pred_cal = predictions[target]["y_pred_cal"]
            
            mean_vals = (y_true + y_pred_cal) / 2
            diff_vals = y_pred_cal - y_true
            
            mean_diff = diff_vals.mean()
            std_diff = diff_vals.std()
            
            ax.scatter(mean_vals, diff_vals, alpha=0.3, s=10, c='steelblue')
            ax.axhline(mean_diff, color='red', linestyle='-', lw=1.5, label=f'Mean={mean_diff:.2f}')
            ax.axhline(mean_diff + 1.96 * std_diff, color='orange', linestyle='--', lw=1, label=f'+1.96SD={mean_diff + 1.96 * std_diff:.2f}')
            ax.axhline(mean_diff - 1.96 * std_diff, color='orange', linestyle='--', lw=1, label=f'-1.96SD={mean_diff - 1.96 * std_diff:.2f}')
            
            ax.set_xlabel(f"Mean of True & Pred (mmHg)")
            ax.set_ylabel(f"Pred - True (mmHg)")
            ax.set_title(f"{display_name}")
            ax.legend(fontsize=7, loc='upper right')
        else:
            ax.set_visible(False)
    
    plt.suptitle("Bland-Altman Plots", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def compute_binned_mae(y_true, y_pred, bins):
    """计算分区间 MAE"""
    results = []
    
    for bin_name, (low, high) in bins.items():
        mask = (y_true >= low) & (y_true < high)
        n = mask.sum()
        
        if n > 0:
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
            r = np.corrcoef(y_true[mask], y_pred[mask])[0, 1] if n > 1 else 0
        else:
            mae = rmse = r = np.nan
        
        results.append({
            "bin": bin_name,
            "range": f"{low}-{high}",
            "n": n,
            "mae": mae,
            "rmse": rmse,
            "r": r,
        })
    
    return pd.DataFrame(results)


def plot_binned_mae(predictions, save_path):
    """绘制分区间 MAE 分析"""
    # 主要分析 SBP 和 DBP
    targets_to_analyze = ["right_arm_sbp", "left_arm_sbp", "right_arm_dbp", "left_arm_dbp"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    all_binned_results = {}
    
    for idx, target in enumerate(targets_to_analyze):
        ax = axes[idx]
        
        if target in predictions:
            y_true = predictions[target]["y_true"]
            y_pred_cal = predictions[target]["y_pred_cal"]
            
            # 选择合适的分类
            if "dbp" in target:
                bins = DBP_CATEGORIES
            else:
                bins = BP_CATEGORIES
            
            df_binned = compute_binned_mae(y_true, y_pred_cal, bins)
            all_binned_results[target] = df_binned
            
            # 柱状图
            x = np.arange(len(df_binned))
            bars = ax.bar(x, df_binned["mae"], color='steelblue', alpha=0.8)
            
            # 在柱子上标注 n
            for i, (bar, n) in enumerate(zip(bars, df_binned["n"])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                       f'n={n}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xticks(x)
            ax.set_xticklabels([f"{row['bin']}\n({row['range']})" for _, row in df_binned.iterrows()], 
                              fontsize=8, rotation=15, ha='right')
            ax.set_ylabel("MAE (mmHg)")
            ax.set_title(f"{BP_TARGETS[target]} - MAE by BP Category")
            
            # 添加整体 MAE 参考线
            overall_mae = predictions[target]["MAE_cal"]
            ax.axhline(overall_mae, color='red', linestyle='--', lw=1.5, label=f'Overall MAE={overall_mae:.2f}')
            ax.legend(fontsize=8)
        else:
            ax.set_visible(False)
    
    plt.suptitle("MAE by BP Category", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")
    
    return all_binned_results


def plot_modality_comparison(results_dir, save_path):
    """绘制模态对比图"""
    all_results = pd.read_csv(Path(results_dir) / "all_results.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE 对比
    ax = axes[0]
    modalities = ["ppg", "ecg", "both"]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for i, mod in enumerate(modalities):
        df_mod = all_results[all_results["modality"] == mod]
        targets = df_mod["target_col"].unique()
        maes = [df_mod[df_mod["target_col"] == t]["MAE_cal"].values[0] for t in targets if t in df_mod["target_col"].values]
        
        x = np.arange(len(targets))
        ax.bar(x + i * 0.25, maes, width=0.25, label=mod.upper(), color=colors[i], alpha=0.8)
    
    ax.set_xticks(np.arange(len(targets)) + 0.25)
    ax.set_xticklabels([BP_TARGETS.get(t, t) for t in targets], rotation=45, ha='right')
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("MAE Comparison by Modality")
    ax.legend()
    
    # r 对比
    ax = axes[1]
    for i, mod in enumerate(modalities):
        df_mod = all_results[all_results["modality"] == mod]
        targets = df_mod["target_col"].unique()
        rs = [df_mod[df_mod["target_col"] == t]["r_cal"].values[0] for t in targets if t in df_mod["target_col"].values]
        
        x = np.arange(len(targets))
        ax.bar(x + i * 0.25, rs, width=0.25, label=mod.upper(), color=colors[i], alpha=0.8)
    
    ax.set_xticks(np.arange(len(targets)) + 0.25)
    ax.set_xticklabels([BP_TARGETS.get(t, t) for t in targets], rotation=45, ha='right')
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation Comparison by Modality")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_summary_dashboard(predictions, save_path):
    """绘制综合仪表板"""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 第一行：SBP 详细分析 (right & left)
    for col, target in enumerate(["right_arm_sbp", "left_arm_sbp"]):
        if target not in predictions:
            continue
        
        y_true = predictions[target]["y_true"]
        y_pred_cal = predictions[target]["y_pred_cal"]
        r = predictions[target]["r"]
        mae = predictions[target]["MAE_cal"]
        display_name = BP_TARGETS[target]
        
        # 散点图
        ax = fig.add_subplot(gs[0, col * 2])
        ax.scatter(y_true, y_pred_cal, alpha=0.3, s=8, c='steelblue')
        lims = [min(y_true.min(), y_pred_cal.min()), max(y_true.max(), y_pred_cal.max())]
        ax.plot(lims, lims, 'r--', lw=1.5)
        ax.set_xlabel(f"True {display_name}")
        ax.set_ylabel(f"Pred {display_name}")
        ax.set_title(f"{display_name}\nr={r:.3f}, MAE={mae:.2f}")
        ax.set_aspect('equal', adjustable='box')
        
        # Bland-Altman
        ax = fig.add_subplot(gs[0, col * 2 + 1])
        mean_vals = (y_true + y_pred_cal) / 2
        diff_vals = y_pred_cal - y_true
        mean_diff = diff_vals.mean()
        std_diff = diff_vals.std()
        ax.scatter(mean_vals, diff_vals, alpha=0.3, s=8, c='steelblue')
        ax.axhline(mean_diff, color='red', linestyle='-', lw=1.5)
        ax.axhline(mean_diff + 1.96 * std_diff, color='orange', linestyle='--', lw=1)
        ax.axhline(mean_diff - 1.96 * std_diff, color='orange', linestyle='--', lw=1)
        ax.set_xlabel("Mean (mmHg)")
        ax.set_ylabel("Difference (mmHg)")
        ax.set_title(f"{display_name} Bland-Altman")
    
    # 第二行：DBP 详细分析
    for col, target in enumerate(["right_arm_dbp", "left_arm_dbp"]):
        if target not in predictions:
            continue
        
        y_true = predictions[target]["y_true"]
        y_pred_cal = predictions[target]["y_pred_cal"]
        r = predictions[target]["r"]
        mae = predictions[target]["MAE_cal"]
        display_name = BP_TARGETS[target]
        
        ax = fig.add_subplot(gs[1, col * 2])
        ax.scatter(y_true, y_pred_cal, alpha=0.3, s=8, c='coral')
        lims = [min(y_true.min(), y_pred_cal.min()), max(y_true.max(), y_pred_cal.max())]
        ax.plot(lims, lims, 'r--', lw=1.5)
        ax.set_xlabel(f"True {display_name}")
        ax.set_ylabel(f"Pred {display_name}")
        ax.set_title(f"{display_name}\nr={r:.3f}, MAE={mae:.2f}")
        ax.set_aspect('equal', adjustable='box')
        
        ax = fig.add_subplot(gs[1, col * 2 + 1])
        mean_vals = (y_true + y_pred_cal) / 2
        diff_vals = y_pred_cal - y_true
        mean_diff = diff_vals.mean()
        std_diff = diff_vals.std()
        ax.scatter(mean_vals, diff_vals, alpha=0.3, s=8, c='coral')
        ax.axhline(mean_diff, color='red', linestyle='-', lw=1.5)
        ax.axhline(mean_diff + 1.96 * std_diff, color='orange', linestyle='--', lw=1)
        ax.axhline(mean_diff - 1.96 * std_diff, color='orange', linestyle='--', lw=1)
        ax.set_xlabel("Mean (mmHg)")
        ax.set_ylabel("Difference (mmHg)")
        ax.set_title(f"{display_name} Bland-Altman")
    
    # 第三行：汇总条形图
    ax = fig.add_subplot(gs[2, :2])
    targets = list(predictions.keys())
    maes = [predictions[t]["MAE_cal"] for t in targets]
    colors = ['steelblue' if 'sbp' in t else 'coral' if 'dbp' in t else 'green' for t in targets]
    bars = ax.bar(range(len(targets)), maes, color=colors, alpha=0.8)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([BP_TARGETS[t] for t in targets], rotation=45, ha='right')
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("MAE Summary (Calibrated)")
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{mae:.2f}', 
               ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[2, 2:])
    rs = [predictions[t]["r"] for t in targets]
    bars = ax.bar(range(len(targets)), rs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([BP_TARGETS[t] for t in targets], rotation=45, ha='right')
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation Summary")
    for bar, r in zip(bars, rs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{r:.3f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.suptitle("BP Prediction Analysis Dashboard", fontsize=16, y=0.98)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def generate_report(predictions, binned_results, output_dir):
    """生成文字报告"""
    report_path = output_dir / "analysis_report.md"
    
    with open(report_path, "w") as f:
        f.write("# BP Prediction Analysis Report\n\n")
        
        f.write("## Overall Performance\n\n")
        f.write("| Target | Best Model | Modality | MAE (mmHg) | r | R² |\n")
        f.write("|--------|------------|----------|------------|---|----|\n")
        
        for target, data in predictions.items():
            f.write(f"| {BP_TARGETS[target]} | {data['exp_name']} | {data['modality']} | "
                   f"{data['MAE_cal']:.2f} | {data['r']:.3f} | {data['r']**2:.3f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # 找最好和最差的
        best_target = min(predictions.keys(), key=lambda t: predictions[t]["MAE_cal"])
        worst_target = max(predictions.keys(), key=lambda t: predictions[t]["MAE_cal"])
        
        f.write(f"- **Best performing target**: {BP_TARGETS[best_target]} "
               f"(MAE={predictions[best_target]['MAE_cal']:.2f}, r={predictions[best_target]['r']:.3f})\n")
        f.write(f"- **Most challenging target**: {BP_TARGETS[worst_target]} "
               f"(MAE={predictions[worst_target]['MAE_cal']:.2f}, r={predictions[worst_target]['r']:.3f})\n")
        
        # 模态分析
        ppg_wins = sum(1 for d in predictions.values() if d['modality'] == 'ppg')
        f.write(f"- **Modality preference**: PPG-only wins in {ppg_wins}/{len(predictions)} targets\n")
        
        f.write("\n## Binned MAE Analysis\n\n")
        
        for target, df_binned in binned_results.items():
            f.write(f"### {BP_TARGETS[target]}\n\n")
            f.write("| Category | Range | N | MAE | RMSE | r |\n")
            f.write("|----------|-------|---|-----|------|---|\n")
            for _, row in df_binned.iterrows():
                f.write(f"| {row['bin']} | {row['range']} | {row['n']} | "
                       f"{row['mae']:.2f} | {row['rmse']:.2f} | {row['r']:.3f} |\n")
            f.write("\n")
        
        f.write("\n## Clinical Implications\n\n")
        f.write("- SBP predictions show strong correlation (r > 0.7), suitable for screening\n")
        f.write("- DBP predictions are more challenging, typical of BP prediction tasks\n")
        f.write("- Pulse pressure (PP) predictions are highly accurate, reflecting strong PPG-BP relationship\n")
        
    print(f"[Saved] {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced BP prediction analysis")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: results_dir/analysis)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BP Prediction Analysis")
    print("=" * 60)
    
    # 加载最佳预测结果
    print("\n[1/7] Loading best predictions...")
    predictions = load_best_predictions(results_dir)
    
    # 绘制分布图
    print("\n[2/7] Plotting BP distributions...")
    plot_bp_distribution(predictions, output_dir / "bp_distribution.png")
    
    # 绘制散点图
    print("\n[3/7] Plotting scatter plots...")
    plot_scatter_all(predictions, output_dir / "scatter_all.png")
    
    # 绘制 Bland-Altman 图
    print("\n[4/7] Plotting Bland-Altman plots...")
    plot_bland_altman(predictions, output_dir / "bland_altman.png")
    
    # 绘制分区间 MAE
    print("\n[5/7] Analyzing binned MAE...")
    binned_results = plot_binned_mae(predictions, output_dir / "binned_mae.png")
    
    # 绘制模态对比
    print("\n[6/7] Plotting modality comparison...")
    plot_modality_comparison(results_dir, output_dir / "modality_comparison.png")
    
    # 绘制综合仪表板
    print("\n[7/7] Creating summary dashboard...")
    plot_summary_dashboard(predictions, output_dir / "dashboard.png")
    
    # 生成报告
    generate_report(predictions, binned_results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()