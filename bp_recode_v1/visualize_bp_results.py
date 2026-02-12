#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BP预测结果可视化脚本 - 同时生成 Raw 和 Cal 版本
Usage:
    python /home/youliang/youliang_data2/bp/bp_recode_v1/visualize_bp_results.py --results_dir /home/youliang/youliang_data2/bp/bp_recode_v1/bp_finetune_sweeps/alltargets_6each_parallel_20260202_022212
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats
from scipy.stats import norm
import argparse

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'ppg': '#E74C3C',
    'ecg': '#3498DB',
    'both': '#2ECC71',
    'true': '#34495E',
    'pred': '#E74C3C',
    'raw': '#3498DB',
    'cal': '#E74C3C',
}

BP_CATEGORIES = {
    'SBP': [
        (0, 120, 'Normal', '#2ECC71'),
        (120, 130, 'Elevated', '#F1C40F'),
        (130, 140, 'Stage 1', '#E67E22'),
        (140, 180, 'Stage 2', '#E74C3C'),
        (180, 300, 'Crisis', '#8E44AD'),
    ],
    'DBP': [
        (0, 80, 'Normal', '#2ECC71'),
        (80, 90, 'Stage 1', '#E67E22'),
        (90, 120, 'Stage 2', '#E74C3C'),
        (120, 200, 'Crisis', '#8E44AD'),
    ],
    'PP': [
        (0, 40, 'Low', '#3498DB'),
        (40, 60, 'Normal', '#2ECC71'),
        (60, 80, 'Elevated', '#F1C40F'),
        (80, 200, 'High', '#E74C3C'),
    ],
}

TARGETS = [
    'right_arm_sbp', 'right_arm_dbp', 'right_arm_mbp', 'right_arm_pp',
    'left_arm_sbp', 'left_arm_dbp', 'left_arm_mbp', 'left_arm_pp'
]


def load_all_results(results_dir):
    csv_path = results_dir / 'all_results.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_best_per_target(results_dir):
    csv_path = results_dir / 'best_per_target.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['target'])
        return df
    return None


def load_predictions_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'records' not in data:
        return None
    
    records = data['records']
    y_true = np.array([r['y_true'][0] if isinstance(r['y_true'], list) else r['y_true'] for r in records])
    y_pred_raw = np.array([r['y_pred_raw'][0] if isinstance(r['y_pred_raw'], list) else r['y_pred_raw'] for r in records])
    y_pred_cal = np.array([r['y_pred_cal'][0] if isinstance(r['y_pred_cal'], list) else r['y_pred_cal'] for r in records]) if 'y_pred_cal' in records[0] else y_pred_raw
    
    return {
        'y_true': y_true,
        'y_pred_raw': y_pred_raw,
        'y_pred_cal': y_pred_cal,
    }


def get_bp_type(target):
    if 'sbp' in target.lower():
        return 'SBP'
    elif 'dbp' in target.lower():
        return 'DBP'
    elif 'pp' in target.lower():
        return 'PP'
    return 'SBP'


def plot_distribution(results_dir, best_df, output_dir, use_raw=False):
    """绘制BP分布图"""
    suffix = 'raw' if use_raw else 'cal'
    pred_key = 'y_pred_raw' if use_raw else 'y_pred_cal'
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        
        row = best_df[best_df['target'] == target]
        if row.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(target.replace('_', ' ').title())
            continue
        
        metrics_path = row.iloc[0]['metrics_path']
        modality = row.iloc[0]['modality']
        
        data = load_predictions_from_json(metrics_path)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_pred = data[pred_key]
        
        bins = 30
        ax.hist(y_true, bins=bins, alpha=0.6, label='True', color=COLORS['true'], density=True)
        ax.hist(y_pred, bins=bins, alpha=0.6, label=f'Pred ({suffix})', color=COLORS[suffix], density=True)
        
        mae = np.mean(np.abs(y_true - y_pred))
        r = np.corrcoef(y_true, y_pred)[0, 1]
        
        ax.set_title(f"{target.replace('_', ' ').title()}\n({modality})", fontsize=11, fontweight='bold')
        ax.set_xlabel('BP (mmHg)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        
        textstr = f'MAE={mae:.1f}\nr={r:.3f}\nRange: [{y_pred.min():.0f}-{y_pred.max():.0f}]'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle(f'BP Distribution ({suffix.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f'bp_distribution_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'bp_distribution_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / f'bp_distribution_{suffix}.png'}")


def plot_scatter_all(results_dir, best_df, output_dir, use_raw=False):
    """绘制散点图"""
    suffix = 'raw' if use_raw else 'cal'
    pred_key = 'y_pred_raw' if use_raw else 'y_pred_cal'
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        
        row = best_df[best_df['target'] == target]
        if row.empty:
            ax.set_title(target.replace('_', ' ').title())
            continue
        
        metrics_path = row.iloc[0]['metrics_path']
        modality = row.iloc[0]['modality']
        
        data = load_predictions_from_json(metrics_path)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_pred = data[pred_key]
        
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c=COLORS[suffix], edgecolors='none')
        
        lims = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Identity')
        
        slope, intercept, r, p, se = stats.linregress(y_true, y_pred)
        x_fit = np.array(lims)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='#2C3E50', linewidth=2, label=f'Fit')
        
        mae = np.mean(np.abs(y_true - y_pred))
        
        ax.set_title(f"{target.replace('_', ' ').title()}\n({modality})", fontsize=11, fontweight='bold')
        ax.set_xlabel('True BP (mmHg)', fontsize=9)
        ax.set_ylabel(f'Predicted BP ({suffix}) (mmHg)', fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        
        textstr = f'r = {r:.3f}\nMAE = {mae:.1f}\nslope = {slope:.2f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle(f'Scatter Plots ({suffix.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f'scatter_all_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'scatter_all_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / f'scatter_all_{suffix}.png'}")


def plot_bland_altman(results_dir, best_df, output_dir, use_raw=False):
    """Bland-Altman 图"""
    suffix = 'raw' if use_raw else 'cal'
    pred_key = 'y_pred_raw' if use_raw else 'y_pred_cal'
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        
        row = best_df[best_df['target'] == target]
        if row.empty:
            ax.set_title(target.replace('_', ' ').title())
            continue
        
        metrics_path = row.iloc[0]['metrics_path']
        modality = row.iloc[0]['modality']
        
        data = load_predictions_from_json(metrics_path)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_pred = data[pred_key]
        
        mean_vals = (y_true + y_pred) / 2
        diff_vals = y_pred - y_true
        
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)
        
        ax.scatter(mean_vals, diff_vals, alpha=0.3, s=10, c=COLORS[suffix], edgecolors='none')
        
        ax.axhline(mean_diff, color='#2C3E50', linestyle='-', linewidth=1.5, label=f'Mean={mean_diff:.1f}')
        ax.axhline(mean_diff + 1.96*std_diff, color='#E74C3C', linestyle='--', linewidth=1)
        ax.axhline(mean_diff - 1.96*std_diff, color='#E74C3C', linestyle='--', linewidth=1)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title(f"{target.replace('_', ' ').title()}\n({modality})", fontsize=11, fontweight='bold')
        ax.set_xlabel('Mean (mmHg)', fontsize=9)
        ax.set_ylabel('Pred - True (mmHg)', fontsize=9)
        
        textstr = f'Bias={mean_diff:.1f}\nLoA=[{mean_diff-1.96*std_diff:.1f}, {mean_diff+1.96*std_diff:.1f}]'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Bland-Altman Plots ({suffix.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f'bland_altman_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'bland_altman_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / f'bland_altman_{suffix}.png'}")


def plot_error_distribution(results_dir, best_df, output_dir, use_raw=False):
    """误差分布图"""
    suffix = 'raw' if use_raw else 'cal'
    pred_key = 'y_pred_raw' if use_raw else 'y_pred_cal'
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        
        row = best_df[best_df['target'] == target]
        if row.empty:
            ax.set_title(target.replace('_', ' ').title())
            continue
        
        metrics_path = row.iloc[0]['metrics_path']
        modality = row.iloc[0]['modality']
        
        data = load_predictions_from_json(metrics_path)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_pred = data[pred_key]
        
        errors = y_pred - y_true
        
        ax.hist(errors, bins=50, alpha=0.7, color=COLORS[suffix], edgecolor='black', density=True)
        
        mu, std = np.mean(errors), np.std(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        ax.plot(x, norm.pdf(x, mu, std), 'k--', linewidth=2)
        
        ax.axvline(0, color='gray', linestyle=':', alpha=0.8)
        ax.axvline(mu, color='#2C3E50', linestyle='-', linewidth=1.5)
        
        ax.set_title(f"{target.replace('_', ' ').title()}\n({modality})", fontsize=11, fontweight='bold')
        ax.set_xlabel('Prediction Error (mmHg)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        
        textstr = f'μ={mu:.1f}\nσ={std:.1f}'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Error Distribution ({suffix.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f'error_distribution_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'error_distribution_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / f'error_distribution_{suffix}.png'}")


def plot_mae_by_category(results_dir, best_df, output_dir, use_raw=False):
    """不同BP区间的MAE"""
    suffix = 'raw' if use_raw else 'cal'
    pred_key = 'y_pred_raw' if use_raw else 'y_pred_cal'
    
    targets_to_plot = [
        ('right_arm_sbp', 'SBP'),
        ('right_arm_dbp', 'DBP'),
        ('right_arm_pp', 'PP'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, (target, bp_type) in enumerate(targets_to_plot):
        ax = axes[idx]
        
        row = best_df[best_df['target'] == target]
        if row.empty:
            continue
        
        metrics_path = row.iloc[0]['metrics_path']
        modality = row.iloc[0]['modality']
        
        data = load_predictions_from_json(metrics_path)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_pred = data[pred_key]
        
        categories = BP_CATEGORIES.get(bp_type, BP_CATEGORIES['SBP'])
        
        cat_names = []
        cat_maes = []
        cat_counts = []
        cat_colors = []
        
        for low, high, name, color in categories:
            if low == high:
                continue
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 10:
                mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                cat_names.append(f'{name}\n({low}-{high})')
                cat_maes.append(mae)
                cat_counts.append(mask.sum())
                cat_colors.append(color)
        
        if not cat_names:
            continue
        
        bars = ax.bar(range(len(cat_names)), cat_maes, color=cat_colors, alpha=0.8, edgecolor='black')
        
        for i, (bar, mae, count) in enumerate(zip(bars, cat_maes, cat_counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{mae:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   f'n={count}', ha='center', va='center', fontsize=8, color='white')
        
        ax.set_xticks(range(len(cat_names)))
        ax.set_xticklabels(cat_names, fontsize=9)
        ax.set_ylabel('MAE (mmHg)', fontsize=11)
        ax.set_title(f'{bp_type} ({modality})', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(cat_maes) * 1.3)
    
    plt.suptitle(f'MAE by BP Category ({suffix.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / f'mae_by_category_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'mae_by_category_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / f'mae_by_category_{suffix}.png'}")


def plot_arm_comparison(results_dir, best_df, output_dir, use_raw=False):
    """左右臂对比"""
    suffix = 'raw' if use_raw else 'cal'
    pred_key = 'y_pred_raw' if use_raw else 'y_pred_cal'
    
    bp_types = ['sbp', 'dbp', 'mbp', 'pp']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, bp_type in enumerate(bp_types):
        ax = axes[idx]
        
        right_target = f'right_arm_{bp_type}'
        left_target = f'left_arm_{bp_type}'
        
        right_row = best_df[best_df['target'] == right_target]
        left_row = best_df[best_df['target'] == left_target]
        
        if right_row.empty or left_row.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(bp_type.upper())
            continue
        
        right_data = load_predictions_from_json(right_row.iloc[0]['metrics_path'])
        left_data = load_predictions_from_json(left_row.iloc[0]['metrics_path'])
        
        if right_data is None or left_data is None:
            continue
        
        right_mae = np.mean(np.abs(right_data['y_true'] - right_data[pred_key]))
        left_mae = np.mean(np.abs(left_data['y_true'] - left_data[pred_key]))
        right_r = np.corrcoef(right_data['y_true'], right_data[pred_key])[0, 1]
        left_r = np.corrcoef(left_data['y_true'], left_data[pred_key])[0, 1]
        
        x = np.arange(2)
        width = 0.35
        
        mae_bars = ax.bar(x - width/2, [right_mae, left_mae], width, 
                         label='MAE', color='#3498DB', alpha=0.8)
        
        ax2 = ax.twinx()
        r_bars = ax2.bar(x + width/2, [right_r, left_r], width,
                        label='r', color='#E74C3C', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Right', 'Left'])
        ax.set_ylabel('MAE (mmHg)', color='#3498DB')
        ax2.set_ylabel('r', color='#E74C3C')
        ax.set_title(bp_type.upper(), fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, max(right_mae, left_mae) * 1.3)
        ax2.set_ylim(0, 1)
        
        for bar, val in zip(mae_bars, [right_mae, left_mae]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='#3498DB')
        for bar, val in zip(r_bars, [right_r, left_r]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='#E74C3C')
    
    plt.suptitle(f'Arm Comparison ({suffix.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / f'arm_comparison_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'arm_comparison_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / f'arm_comparison_{suffix}.png'}")


def plot_summary_table(best_df, output_dir, use_raw=False):
    """汇总表格"""
    suffix = 'raw' if use_raw else 'cal'
    mae_col = 'MAE_raw' if use_raw else 'MAE_cal'
    r_col = 'r_raw' if use_raw else 'r_cal'
    r2_col = 'R2_raw' if use_raw else 'R2_cal'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    table_data = []
    for _, row in best_df.iterrows():
        if pd.isna(row['target']):
            continue
        
        # 从 json 重新计算 raw 指标
        if use_raw:
            data = load_predictions_from_json(row['metrics_path'])
            if data is not None:
                y_true = data['y_true']
                y_pred = data['y_pred_raw']
                mae = np.mean(np.abs(y_true - y_pred))
                r = np.corrcoef(y_true, y_pred)[0, 1]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            else:
                mae, r, r2 = row.get(mae_col, 0), row.get(r_col, 0), row.get(r2_col, 0)
        else:
            mae = row.get(mae_col, row.get('MAE_cal', 0))
            r = row.get(r_col, row.get('r_cal', 0))
            r2 = row.get(r2_col, row.get('R2_cal', 0))
        
        table_data.append([
            row['target'].replace('_', ' ').title(),
            row['modality'].upper(),
            row['loss'],
            f"{mae:.2f}",
            f"{r:.3f}",
            f"{r2:.3f}",
        ])
    
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Target', 'Modality', 'Loss', 'MAE (mmHg)', 'r', 'R²'],
            loc='center',
            cellLoc='center',
            colWidths=[0.22, 0.12, 0.10, 0.14, 0.12, 0.12]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#3498DB' if not use_raw else '#E67E22')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        for i in range(1, len(table_data) + 1):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ECF0F1')
    
    plt.title(f'BP Prediction Results Summary ({suffix.upper()})', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / f'summary_table_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'summary_table_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / f'summary_table_{suffix}.png'}")


def plot_raw_vs_cal_comparison(results_dir, best_df, output_dir):
    """Raw vs Cal 直接对比"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    summary_data = []
    
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        
        row = best_df[best_df['target'] == target]
        if row.empty:
            ax.set_title(target.replace('_', ' ').title())
            continue
        
        metrics_path = row.iloc[0]['metrics_path']
        modality = row.iloc[0]['modality']
        
        data = load_predictions_from_json(metrics_path)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_raw = data['y_pred_raw']
        y_cal = data['y_pred_cal']
        
        bins = 40
        ax.hist(y_true, bins=bins, alpha=0.5, label='True', color=COLORS['true'], density=True)
        ax.hist(y_raw, bins=bins, alpha=0.5, label='Raw', color=COLORS['raw'], density=True)
        ax.hist(y_cal, bins=bins, alpha=0.5, label='Cal', color=COLORS['cal'], density=True)
        
        mae_raw = np.mean(np.abs(y_true - y_raw))
        mae_cal = np.mean(np.abs(y_true - y_cal))
        r_raw = np.corrcoef(y_true, y_raw)[0, 1]
        r_cal = np.corrcoef(y_true, y_cal)[0, 1]
        
        ax.set_title(f"{target.replace('_', ' ').title()} ({modality})", fontsize=11, fontweight='bold')
        ax.set_xlabel('BP (mmHg)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        
        textstr = f'Raw: MAE={mae_raw:.1f}, r={r_raw:.3f}\nCal: MAE={mae_cal:.1f}, r={r_cal:.3f}'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        range_str = f'True: [{y_true.min():.0f}-{y_true.max():.0f}]\nRaw: [{y_raw.min():.0f}-{y_raw.max():.0f}]\nCal: [{y_cal.min():.0f}-{y_cal.max():.0f}]'
        ax.text(0.05, 0.95, range_str, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='upper center', fontsize=8)
        
        # 收集数据
        true_range = y_true.max() - y_true.min()
        raw_range = y_raw.max() - y_raw.min()
        cal_range = y_cal.max() - y_cal.min()
        
        summary_data.append({
            'target': target,
            'mae_raw': mae_raw,
            'mae_cal': mae_cal,
            'r_raw': r_raw,
            'r_cal': r_cal,
            'true_range': true_range,
            'raw_range': raw_range,
            'cal_range': cal_range,
            'raw_compression': raw_range / true_range * 100,
            'cal_compression': cal_range / true_range * 100,
        })
    
    plt.suptitle('Raw vs Calibrated Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'raw_vs_cal_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'raw_vs_cal_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / 'raw_vs_cal_distribution.png'}")
    
    # 打印汇总
    print("\n" + "=" * 100)
    print("Range Compression Summary")
    print("=" * 100)
    print(f"{'Target':<20} {'MAE_raw':>10} {'MAE_cal':>10} {'r_raw':>8} {'r_cal':>8} {'Raw%':>8} {'Cal%':>8}")
    print("-" * 100)
    for d in summary_data:
        print(f"{d['target']:<20} {d['mae_raw']:>10.2f} {d['mae_cal']:>10.2f} {d['r_raw']:>8.3f} {d['r_cal']:>8.3f} {d['raw_compression']:>7.0f}% {d['cal_compression']:>7.0f}%")
    
    return summary_data


def plot_range_compression_bar(summary_data, output_dir):
    """范围压缩柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    targets = [d['target'].replace('_', ' ').title() for d in summary_data]
    raw_comp = [d['raw_compression'] for d in summary_data]
    cal_comp = [d['cal_compression'] for d in summary_data]
    
    x = np.arange(len(targets))
    width = 0.35
    
    # 左图：范围压缩百分比
    ax = axes[0]
    bars1 = ax.bar(x - width/2, raw_comp, width, label='Raw', color=COLORS['raw'], alpha=0.8)
    bars2 = ax.bar(x + width/2, cal_comp, width, label='Cal', color=COLORS['cal'], alpha=0.8)
    
    ax.axhline(100, color='gray', linestyle='--', alpha=0.5, label='No compression')
    ax.set_ylabel('Prediction Range (% of True Range)')
    ax.set_xlabel('Target')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_title('Range Compression', fontweight='bold')
    
    # 右图：MAE对比
    ax2 = axes[1]
    mae_raw = [d['mae_raw'] for d in summary_data]
    mae_cal = [d['mae_cal'] for d in summary_data]
    
    bars3 = ax2.bar(x - width/2, mae_raw, width, label='Raw', color=COLORS['raw'], alpha=0.8)
    bars4 = ax2.bar(x + width/2, mae_cal, width, label='Cal', color=COLORS['cal'], alpha=0.8)
    
    ax2.set_ylabel('MAE (mmHg)')
    ax2.set_xlabel('Target')
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.set_title('MAE Comparison', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'raw_vs_cal_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'raw_vs_cal_comparison_bar.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / 'raw_vs_cal_comparison_bar.png'}")


def main():
    parser = argparse.ArgumentParser(description='BP Results Visualization (Raw & Cal)')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    
    best_df = load_best_per_target(results_dir)
    all_results_df = load_all_results(results_dir)
    
    if best_df is None:
        print("Error: best_per_target.csv not found!")
        return
    
    print(f"Loaded {len(best_df)} best results")
    
    # 生成 CAL 版本
    print("\n" + "=" * 60)
    print("Generating CAL figures...")
    print("=" * 60)
    
    plot_distribution(results_dir, best_df, output_dir, use_raw=False)
    plot_scatter_all(results_dir, best_df, output_dir, use_raw=False)
    plot_bland_altman(results_dir, best_df, output_dir, use_raw=False)
    plot_error_distribution(results_dir, best_df, output_dir, use_raw=False)
    plot_mae_by_category(results_dir, best_df, output_dir, use_raw=False)
    plot_arm_comparison(results_dir, best_df, output_dir, use_raw=False)
    plot_summary_table(best_df, output_dir, use_raw=False)
    
    # 生成 RAW 版本
    print("\n" + "=" * 60)
    print("Generating RAW figures...")
    print("=" * 60)
    
    plot_distribution(results_dir, best_df, output_dir, use_raw=True)
    plot_scatter_all(results_dir, best_df, output_dir, use_raw=True)
    plot_bland_altman(results_dir, best_df, output_dir, use_raw=True)
    plot_error_distribution(results_dir, best_df, output_dir, use_raw=True)
    plot_mae_by_category(results_dir, best_df, output_dir, use_raw=True)
    plot_arm_comparison(results_dir, best_df, output_dir, use_raw=True)
    plot_summary_table(best_df, output_dir, use_raw=True)
    
    # 生成对比图
    print("\n" + "=" * 60)
    print("Generating comparison figures...")
    print("=" * 60)
    
    summary_data = plot_raw_vs_cal_comparison(results_dir, best_df, output_dir)
    plot_range_compression_bar(summary_data, output_dir)
    
    print("\n" + "=" * 60)
    print("All figures saved!")
    print(f"Output: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()