#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速收集 Grid Search 结果（自动计算置信区间）
Quick Result Collection Script with Auto-calculated Confidence Intervals
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy import stats


def calculate_ci(errors, confidence=0.95):
    """计算置信区间（使用 t 分布）"""
    n = len(errors)
    if n < 2:
        return None, None
    
    mean = np.mean(errors)
    # 使用 t 分布计算置信区间
    sem = stats.sem(errors)
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
    
    return ci[0], ci[1]


def calculate_r_ci(r, n, confidence=0.95):
    """计算 Pearson r 的置信区间（使用 Fisher's z transformation）"""
    if n < 3 or np.isnan(r):
        return None, None
    
    # Fisher's z transformation
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    
    # z 的置信区间
    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - alpha / 2)
    z_lower = z - z_critical * se
    z_upper = z + z_critical * se
    
    # 转换回 r
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    
    return r_lower, r_upper


def calculate_metrics_with_ci(predictions_file):
    """从 test_predictions.csv 计算指标和置信区间"""
    try:
        df = pd.read_csv(predictions_file)
    except Exception as e:
        print(f"Error reading predictions: {e}")
        return None
    
    targets = ["right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp",
               "left_arm_sbp", "left_arm_mbp", "left_arm_dbp", "left_arm_pp"]
    
    metrics = {}
    all_maes = []
    all_r_values = []
    n_samples = len(df)
    
    for target in targets:
        true_col = f"{target}_true"
        pred_col = f"{target}_pred"
        
        if true_col not in df.columns or pred_col not in df.columns:
            continue
        
        y_true = df[true_col].values
        y_pred = df[pred_col].values
        
        # 计算 MAE
        errors = np.abs(y_true - y_pred)
        mae = np.mean(errors)
        all_maes.extend(errors)
        
        # 计算 MAE 置信区间
        mae_ci_lower, mae_ci_upper = calculate_ci(errors)
        
        # 计算 Pearson r
        r = np.corrcoef(y_true, y_pred)[0, 1]
        all_r_values.append(r)
        
        # 计算 r 的置信区间
        r_ci_lower, r_ci_upper = calculate_r_ci(r, n_samples)
        
        metrics[target] = {
            "mae": mae,
            "mae_ci_lower": mae_ci_lower,
            "mae_ci_upper": mae_ci_upper,
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
            "r": r,
            "r_ci_lower": r_ci_lower,
            "r_ci_upper": r_ci_upper,
            "r2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }
    
    # 计算整体平均 MAE 的置信区间
    avg_mae = np.mean(all_maes)
    avg_mae_ci_lower, avg_mae_ci_upper = calculate_ci(all_maes)
    
    # 计算平均 Pearson r 及其置信区间
    avg_r = np.mean(all_r_values)
    # 对于平均 r，使用所有目标 r 值的标准误差
    avg_r_ci_lower, avg_r_ci_upper = calculate_ci(all_r_values)
    
    return {
        "metrics": metrics,
        "avg_mae": avg_mae,
        "avg_mae_ci_lower": avg_mae_ci_lower,
        "avg_mae_ci_upper": avg_mae_ci_upper,
        "avg_r": avg_r,
        "avg_r_ci_lower": avg_r_ci_lower,
        "avg_r_ci_upper": avg_r_ci_upper,
        "num_samples": n_samples
    }


def collect_all_results(base_dir):
    """收集所有实验结果"""
    base_path = Path(base_dir)
    results = []
    
    # 遍历所有子目录
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        result_file = exp_dir / "test_metrics.json"
        predictions_file = exp_dir / "test_predictions.csv"
        
        if not result_file.exists():
            print(f"⚠ Missing results: {exp_dir.name}")
            continue
        
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            
            # 提取配置
            config_file = exp_dir / "config.json"
            config = {}
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)
            
            # 构建结果行
            row = {
                "exp_name": exp_dir.name,
                "modality": data.get("modality", config.get("modality", "unknown")),
                "test_avg_mae": data.get("avg_mae", 0),
                "test_loss": data.get("test_loss", 0),
                "best_epoch": data.get("best_epoch", 0),
                "test_samples": data.get("test_samples", data.get("num_samples", "N/A"))
            }
            
            # 如果存在 predictions 文件，计算置信区间
            if predictions_file.exists():
                ci_data = calculate_metrics_with_ci(predictions_file)
                if ci_data:
                    row["avg_mae_ci_lower"] = ci_data["avg_mae_ci_lower"]
                    row["avg_mae_ci_upper"] = ci_data["avg_mae_ci_upper"]
                    row["avg_r"] = ci_data["avg_r"]
                    row["avg_r_ci_lower"] = ci_data["avg_r_ci_lower"]
                    row["avg_r_ci_upper"] = ci_data["avg_r_ci_upper"]
                    row["test_samples"] = ci_data["num_samples"]
                    
                    # 更新每个目标的置信区间
                    for target, metrics in ci_data["metrics"].items():
                        row[f"{target}_mae_ci_lower"] = metrics["mae_ci_lower"]
                        row[f"{target}_mae_ci_upper"] = metrics["mae_ci_upper"]
                        row[f"{target}_r_ci_lower"] = metrics["r_ci_lower"]
                        row[f"{target}_r_ci_upper"] = metrics["r_ci_upper"]
            
            # 添加每个目标的详细指标
            for target, metrics in data.get("metrics", {}).items():
                row[f"{target}_mae"] = metrics["mae"]
                row[f"{target}_rmse"] = metrics["rmse"]
                row[f"{target}_r"] = metrics["r"]
                row[f"{target}_r2"] = metrics["r2"]
            
            results.append(row)
            print(f"✓ Collected: {exp_dir.name}")
            
        except Exception as e:
            print(f"✗ Error reading {exp_dir.name}: {e}")
            continue
    
    return pd.DataFrame(results)


def generate_report(df, output_path):
    """生成 Markdown 报告"""
    lines = []
    lines.append("# Blood Pressure Prediction - Results Summary\n")
    lines.append(f"Total experiments: {len(df)}\n")
    
    if len(df) == 0:
        lines.append("**No results found.**\n")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        return
    
    # 按 modality 分组
    lines.append("## Results by Modality\n")
    
    has_mae_ci = "avg_mae_ci_lower" in df.columns
    has_r = "avg_r" in df.columns
    has_r_ci = "avg_r_ci_lower" in df.columns
    
    for modality in sorted(df["modality"].unique()):
        df_mod = df[df["modality"] == modality].sort_values("test_avg_mae")
        lines.append(f"### {modality.upper()}\n")
        
        if has_mae_ci and has_r_ci:
            lines.append("| Exp Name | Avg MAE (95% CI) | Avg r (95% CI) | Best Epoch |")
            lines.append("|----------|------------------|----------------|------------|")
        elif has_mae_ci and has_r:
            lines.append("| Exp Name | Avg MAE (95% CI) | Avg r | Best Epoch |")
            lines.append("|----------|------------------|-------|------------|")
        elif has_mae_ci:
            lines.append("| Exp Name | Avg MAE (95% CI) | Best Epoch |")
            lines.append("|----------|------------------|------------|")
        else:
            lines.append("| Exp Name | Avg MAE | Best Epoch |")
            lines.append("|----------|---------|------------|")
        
        for _, row in df_mod.iterrows():
            # MAE with CI
            if has_mae_ci and pd.notna(row.get("avg_mae_ci_lower")):
                mae_str = f"**{row['test_avg_mae']:.2f}** ({row['avg_mae_ci_lower']:.2f}-{row['avg_mae_ci_upper']:.2f})"
            else:
                mae_str = f"**{row['test_avg_mae']:.2f}**"
            
            # r with CI
            if has_r_ci and pd.notna(row.get("avg_r_ci_lower")):
                r_str = f"{row['avg_r']:.3f} ({row['avg_r_ci_lower']:.3f}-{row['avg_r_ci_upper']:.3f})"
                lines.append(
                    f"| `{row['exp_name']}` | {mae_str} | {r_str} | "
                    f"{row['best_epoch']} |"
                )
            elif has_r and pd.notna(row.get("avg_r")):
                lines.append(
                    f"| `{row['exp_name']}` | {mae_str} | {row['avg_r']:.3f} | "
                    f"{row['best_epoch']} |"
                )
            else:
                lines.append(
                    f"| `{row['exp_name']}` | {mae_str} | "
                    f"{row['best_epoch']} |"
                )
        lines.append("")
    
    # 总体排名
    lines.append("## Overall Ranking (Top 10)\n")
    
    if has_mae_ci and has_r_ci:
        lines.append("| Rank | Exp Name | Modality | Avg MAE (95% CI) | Avg r (95% CI) |")
        lines.append("|------|----------|----------|------------------|----------------|")
    elif has_mae_ci and has_r:
        lines.append("| Rank | Exp Name | Modality | Avg MAE (95% CI) | Avg r |")
        lines.append("|------|----------|----------|------------------|-------|")
    elif has_mae_ci:
        lines.append("| Rank | Exp Name | Modality | Avg MAE (95% CI) |")
        lines.append("|------|----------|----------|------------------|")
    else:
        lines.append("| Rank | Exp Name | Modality | Avg MAE |")
        lines.append("|------|----------|----------|---------|")
    
    for i, (idx, row) in enumerate(df.nsmallest(10, "test_avg_mae").iterrows(), 1):
        # MAE with CI
        if has_mae_ci and pd.notna(row.get("avg_mae_ci_lower")):
            mae_str = f"**{row['test_avg_mae']:.2f}** ({row['avg_mae_ci_lower']:.2f}-{row['avg_mae_ci_upper']:.2f})"
        else:
            mae_str = f"**{row['test_avg_mae']:.2f}**"
        
        # r with CI
        if has_r_ci and pd.notna(row.get("avg_r_ci_lower")):
            r_str = f"{row['avg_r']:.3f} ({row['avg_r_ci_lower']:.3f}-{row['avg_r_ci_upper']:.3f})"
            lines.append(
                f"| {i} | `{row['exp_name']}` | {row['modality']} | {mae_str} | {r_str} |"
            )
        elif has_r and pd.notna(row.get("avg_r")):
            lines.append(
                f"| {i} | `{row['exp_name']}` | {row['modality']} | {mae_str} | {row['avg_r']:.3f} |"
            )
        else:
            lines.append(
                f"| {i} | `{row['exp_name']}` | {row['modality']} | {mae_str} |"
            )
    
    # 最佳模型详细信息
    best = df.nsmallest(1, "test_avg_mae").iloc[0]
    lines.append(f"\n## Best Model: `{best['exp_name']}`\n")
    lines.append(f"- **Modality**: {best['modality']}")
    
    if has_mae_ci and pd.notna(best.get("avg_mae_ci_lower")):
        lines.append(f"- **Average MAE**: **{best['test_avg_mae']:.2f} mmHg** (95% CI: {best['avg_mae_ci_lower']:.2f}-{best['avg_mae_ci_upper']:.2f})")
    else:
        lines.append(f"- **Average MAE**: **{best['test_avg_mae']:.2f} mmHg**")
    
    if has_r_ci and pd.notna(best.get("avg_r_ci_lower")):
        lines.append(f"- **Average Pearson r**: **{best['avg_r']:.3f}** (95% CI: {best['avg_r_ci_lower']:.3f}-{best['avg_r_ci_upper']:.3f})")
    elif has_r and pd.notna(best.get("avg_r")):
        lines.append(f"- **Average Pearson r**: **{best['avg_r']:.3f}**")
    
    lines.append(f"- **Best Epoch**: {best['best_epoch']}")
    if 'test_samples' in best and pd.notna(best['test_samples']) and best['test_samples'] != "N/A":
        lines.append(f"- **Test Samples**: {best['test_samples']}\n")
    else:
        lines.append("")
    
    # 每个目标的表现
    lines.append("### Per-target Performance:\n")
    
    targets = ["right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp",
               "left_arm_sbp", "left_arm_mbp", "left_arm_dbp", "left_arm_pp"]
    
    has_target_mae_ci = any(f"{target}_mae_ci_lower" in best for target in targets if f"{target}_mae" in best)
    has_target_r_ci = any(f"{target}_r_ci_lower" in best for target in targets if f"{target}_r" in best)
    
    if has_target_mae_ci and has_target_r_ci:
        lines.append("| Target | MAE (95% CI) | RMSE | Pearson r (95% CI) | R² |")
        lines.append("|--------|--------------|------|---------------------|-----|")
    elif has_target_mae_ci:
        lines.append("| Target | MAE (95% CI) | RMSE | Pearson r | R² |")
        lines.append("|--------|--------------|------|-----------|-----|")
    else:
        lines.append("| Target | MAE | RMSE | Pearson r | R² |")
        lines.append("|--------|-----|------|-----------|-----|")
    
    for target in targets:
        if f"{target}_mae" in best:
            # MAE with CI
            if has_target_mae_ci and f"{target}_mae_ci_lower" in best and pd.notna(best[f"{target}_mae_ci_lower"]):
                mae_str = f"{best[f'{target}_mae']:.2f} ({best[f'{target}_mae_ci_lower']:.2f}-{best[f'{target}_mae_ci_upper']:.2f})"
            else:
                mae_str = f"{best[f'{target}_mae']:.2f}"
            
            # r with CI
            if has_target_r_ci and f"{target}_r_ci_lower" in best and pd.notna(best[f"{target}_r_ci_lower"]):
                r_str = f"{best[f'{target}_r']:.3f} ({best[f'{target}_r_ci_lower']:.3f}-{best[f'{target}_r_ci_upper']:.3f})"
            else:
                r_str = f"{best[f'{target}_r']:.3f}"
            
            lines.append(
                f"| {target} | {mae_str} | "
                f"{best[f'{target}_rmse']:.2f} | {r_str} | "
                f"{best[f'{target}_r2']:.3f} |"
            )
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", 
                       default="/home/youliang/youliang_data2/bp/grid_search_results",
                       help="Base directory with experiment results")
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"Error: Directory not found: {base_path}")
        return
    
    print(f"Collecting results from: {base_path}")
    print("="*70)
    
    df = collect_all_results(base_path)
    
    if len(df) == 0:
        print("\n[Warning] No results found!")
        return
    
    # 保存 CSV
    csv_path = base_path / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved: {csv_path}")
    
    # 生成报告
    report_path = base_path / "REPORT.md"
    generate_report(df, report_path)
    print(f"✓ Report saved: {report_path}")
    
    # 打印总结
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(df)}")
    
    best = df.nsmallest(1, "test_avg_mae").iloc[0]
    print(f"\nBest Model: {best['exp_name']}")
    
    has_mae_ci = "avg_mae_ci_lower" in df.columns and pd.notna(best.get("avg_mae_ci_lower"))
    has_r = "avg_r" in df.columns and pd.notna(best.get("avg_r"))
    has_r_ci = "avg_r_ci_lower" in df.columns and pd.notna(best.get("avg_r_ci_lower"))
    
    if has_mae_ci:
        print(f"  Average MAE: {best['test_avg_mae']:.2f} mmHg (95% CI: {best['avg_mae_ci_lower']:.2f}-{best['avg_mae_ci_upper']:.2f})")
    else:
        print(f"  Average MAE: {best['test_avg_mae']:.2f} mmHg")
    
    if has_r_ci:
        print(f"  Average Pearson r: {best['avg_r']:.3f} (95% CI: {best['avg_r_ci_lower']:.3f}-{best['avg_r_ci_upper']:.3f})")
    elif has_r:
        print(f"  Average Pearson r: {best['avg_r']:.3f}")
    
    print(f"  Modality: {best['modality']}")
    print(f"  Best Epoch: {best['best_epoch']}")
    
    print("\nResults by modality:")
    for modality in sorted(df["modality"].unique()):
        df_mod = df[df["modality"] == modality]
        best_mod = df_mod.nsmallest(1, "test_avg_mae").iloc[0]
        
        mae_ci_str = ""
        r_str = ""
        
        if has_mae_ci and pd.notna(best_mod.get("avg_mae_ci_lower")):
            mae_ci_str = f" (CI: {best_mod['avg_mae_ci_lower']:.2f}-{best_mod['avg_mae_ci_upper']:.2f})"
        
        if has_r_ci and pd.notna(best_mod.get("avg_r_ci_lower")):
            r_str = f", r={best_mod['avg_r']:.3f} (CI: {best_mod['avg_r_ci_lower']:.3f}-{best_mod['avg_r_ci_upper']:.3f})"
        elif has_r and pd.notna(best_mod.get("avg_r")):
            r_str = f", r={best_mod['avg_r']:.3f}"
        
        print(f"  {modality:>5}: Best MAE = {best_mod['test_avg_mae']:.2f}{mae_ci_str}{r_str} ({best_mod['exp_name']})")
    
    print("="*70)


if __name__ == "__main__":
    main()