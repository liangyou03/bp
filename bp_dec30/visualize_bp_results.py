#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BP预测结果可视化 - 用于报告
生成：
1. BP分布直方图（真实值 vs 预测值）
2. 散点图（带回归线和r值）
3. 不同BP区间的MAE分析
4. 左右臂对比
python visualize_bp_results.py --results_dir runs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from scipy import stats

# 设置中文字体和风格
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 颜色方案
COLORS = {
    'ppg': '#E74C3C',      # 红色
    'ecg': '#3498DB',      # 蓝色  
    'both': '#2ECC71',     # 绿色
    'true': '#34495E',     # 深灰
    'pred': '#E74C3C',     # 红色
}

# BP分类标准 (AHA)
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
        (80, 80, 'Elevated', '#F1C40F'),  # DBP没有单独的elevated
        (80, 90, 'Stage 1', '#E67E22'),
        (90, 120, 'Stage 2', '#E74C3C'),
        (120, 200, 'Crisis', '#8E44AD'),
    ],
    'MBP': [
        (0, 93, 'Normal', '#2ECC71'),
        (93, 100, 'Elevated', '#F1C40F'),
        (100, 107, 'Stage 1', '#E67E22'),
        (107, 140, 'Stage 2', '#E74C3C'),
        (140, 200, 'Crisis', '#8E44AD'),
    ],
    'PP': [
        (0, 40, 'Low', '#3498DB'),
        (40, 60, 'Normal', '#2ECC71'),
        (60, 80, 'Elevated', '#F1C40F'),
        (80, 200, 'High', '#E74C3C'),
    ],
}


def load_best_config(results_dir):
    """从 best_per_target.csv 加载每个 target 的最佳配置"""
    results_dir = Path(results_dir)
    best_csv = results_dir / 'best_per_target.csv'
    
    if not best_csv.exists():
        print(f"Warning: {best_csv} not found, will use default ppg")
        return {}
    
    df = pd.read_csv(best_csv)
    best_config = {}
    for _, row in df.iterrows():
        target = row['target']
        exp_name = row['best_exp']  # 修正列名
        modality = row['modality']
        best_config[target] = {
            'exp_name': exp_name,
            'modality': modality,
        }
    return best_config


def load_best_per_target_modality(results_dir):
    """从 best_per_target_modality.csv 加载每个 target+modality 的最佳配置"""
    results_dir = Path(results_dir)
    best_csv = results_dir / 'best_per_target_modality.csv'
    
    if not best_csv.exists():
        return {}
    
    df = pd.read_csv(best_csv)
    best_config = {}
    for _, row in df.iterrows():
        key = (row['target'], row['modality'])
        best_config[key] = {
            'exp_name': row['best_exp'],
            'MAE_cal': row['MAE_cal'],
            'r': row['r'],
        }
    return best_config


# 全局缓存
_BEST_CONFIG_CACHE = {}
_BEST_TM_CACHE = {}


def load_predictions(results_dir, target, modality=None, use_best=True):
    """加载预测结果
    
    Args:
        results_dir: 结果目录
        target: BP目标 (e.g., 'right_arm_sbp')
        modality: 指定模态 ('ppg', 'ecg', 'both')，如果为 None 则自动选择最佳
        use_best: 是否使用最佳配置
    """
    results_dir = Path(results_dir)
    
    # 加载最佳配置（带缓存）
    cache_key = str(results_dir)
    if cache_key not in _BEST_CONFIG_CACHE:
        _BEST_CONFIG_CACHE[cache_key] = load_best_config(results_dir)
    if cache_key not in _BEST_TM_CACHE:
        _BEST_TM_CACHE[cache_key] = load_best_per_target_modality(results_dir)
    
    best_config = _BEST_CONFIG_CACHE[cache_key]
    best_tm_config = _BEST_TM_CACHE[cache_key]
    
    # 确定使用哪个实验
    exp_name = None
    if modality is None and use_best and target in best_config:
        # 未指定模态，用全局最佳
        modality = best_config[target]['modality']
        exp_name = best_config[target]['exp_name']
        print(f"  [{target}] Using best: {exp_name} ({modality})")
    elif modality is not None and use_best and (target, modality) in best_tm_config:
        # 指定了模态，用该模态的最佳
        exp_name = best_tm_config[(target, modality)]['exp_name']
    elif modality is None:
        modality = 'ppg'  # 默认
    
    # 查找匹配的npz文件
    if exp_name:
        # 精确查找
        pattern = f"{exp_name}/*{target}*{modality}*records.npz"
    else:
        # 模糊查找
        pattern = f"*{target}*{modality}*records.npz"
    
    files = list(results_dir.rglob(pattern))
    
    if not files:
        # print(f"Warning: No file found for {target} {modality}")
        return None
    
    # 取最新的
    npz_path = sorted(files)[-1]
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'y_true': data['y_true'],
        'y_pred': data['y_pred_cal'] if 'y_pred_cal' in data else data['y_pred'],
        'target': target,
        'modality': modality,
    }


def load_best_results(results_dir):
    """加载所有最佳结果"""
    results_dir = Path(results_dir)
    best_csv = results_dir / 'best_per_target.csv'
    
    if best_csv.exists():
        df = pd.read_csv(best_csv)
        return df
    return None


def get_bp_type(target):
    """从target名获取BP类型"""
    if 'sbp' in target.lower():
        return 'SBP'
    elif 'dbp' in target.lower():
        return 'DBP'
    elif 'mbp' in target.lower():
        return 'MBP'
    elif 'pp' in target.lower():
        return 'PP'
    return 'SBP'


def plot_distribution(results_dir, output_dir):
    """绘制BP分布图 - 8个目标的真实值与预测值分布"""
    targets = [
        'right_arm_sbp', 'right_arm_dbp', 'right_arm_mbp', 'right_arm_pp',
        'left_arm_sbp', 'left_arm_dbp', 'left_arm_mbp', 'left_arm_pp'
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        # 自动选择最佳配置
        data = load_predictions(results_dir, target, modality=None, use_best=True)
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        y_true = data['y_true']
        y_pred = data['y_pred']
        modality = data['modality']
        
        # 绘制直方图
        bins = 30
        ax.hist(y_true, bins=bins, alpha=0.6, label='True', color=COLORS['true'], density=True)
        ax.hist(y_pred, bins=bins, alpha=0.6, label='Predicted', color=COLORS['pred'], density=True)
        
        # 添加统计信息
        mae = np.mean(np.abs(y_true - y_pred))
        r = np.corrcoef(y_true, y_pred)[0, 1]
        
        ax.set_title(f"{target.replace('_', ' ').title()}\n({modality})", fontsize=11, fontweight='bold')
        ax.set_xlabel('BP (mmHg)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        
        # 添加文本框
        textstr = f'MAE={mae:.1f}\nr={r:.3f}'
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bp_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'bp_distribution.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / 'bp_distribution.png'}")


def plot_scatter_all(results_dir, output_dir):
    """绘制所有目标的散点图"""
    targets = [
        'right_arm_sbp', 'right_arm_dbp', 'right_arm_mbp', 'right_arm_pp',
        'left_arm_sbp', 'left_arm_dbp', 'left_arm_mbp', 'left_arm_pp'
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        # 自动选择最佳配置
        data = load_predictions(results_dir, target, modality=None, use_best=True)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_pred = data['y_pred']
        modality = data['modality']
        
        # 散点图
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c=COLORS.get(modality, COLORS['ppg']), edgecolors='none')
        
        # 理想线
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Identity')
        
        # 回归线
        slope, intercept, r, p, se = stats.linregress(y_true, y_pred)
        x_fit = np.array(lims)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='#2C3E50', linewidth=2, label=f'Fit (r={r:.3f})')
        
        mae = np.mean(np.abs(y_true - y_pred))
        
        ax.set_title(f"{target.replace('_', ' ').title()}\n({modality})", fontsize=11, fontweight='bold')
        ax.set_xlabel('True BP (mmHg)', fontsize=9)
        ax.set_ylabel('Predicted BP (mmHg)', fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        
        # 统计信息
        textstr = f'r = {r:.3f}\nMAE = {mae:.1f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_all.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'scatter_all.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / 'scatter_all.png'}")


def plot_mae_by_category(results_dir, output_dir):
    """绘制不同BP区间的MAE"""
    # 主要关注SBP和DBP
    targets_to_plot = [
        ('right_arm_sbp', 'SBP'),
        ('right_arm_dbp', 'DBP'),
        ('right_arm_pp', 'PP'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, (target, bp_type) in enumerate(targets_to_plot):
        ax = axes[idx]
        
        # 自动选择最佳配置
        data = load_predictions(results_dir, target, modality=None, use_best=True)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_pred = data['y_pred']
        modality = data['modality']
        
        categories = BP_CATEGORIES.get(bp_type, BP_CATEGORIES['SBP'])
        
        cat_names = []
        cat_maes = []
        cat_counts = []
        cat_colors = []
        
        for low, high, name, color in categories:
            if low == high:  # 跳过无效区间
                continue
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 10:  # 至少10个样本
                mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                cat_names.append(f'{name}\n({low}-{high})')
                cat_maes.append(mae)
                cat_counts.append(mask.sum())
                cat_colors.append(color)
        
        if not cat_names:
            continue
        
        bars = ax.bar(range(len(cat_names)), cat_maes, color=cat_colors, alpha=0.8, edgecolor='black')
        
        # 在柱子上显示数值和样本量
        for i, (bar, mae, count) in enumerate(zip(bars, cat_maes, cat_counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{mae:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   f'n={count}', ha='center', va='center', fontsize=8, color='white')
        
        ax.set_xticks(range(len(cat_names)))
        ax.set_xticklabels(cat_names, fontsize=9)
        ax.set_ylabel('MAE (mmHg)', fontsize=11)
        ax.set_title(f'{bp_type} MAE by Category ({modality})', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(cat_maes) * 1.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_by_category.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'mae_by_category.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / 'mae_by_category.png'}")


def plot_arm_comparison(results_dir, output_dir):
    """左右臂对比"""
    bp_types = ['sbp', 'dbp', 'mbp', 'pp']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, bp_type in enumerate(bp_types):
        ax = axes[idx]
        
        # 加载左右臂数据
        right_data = load_predictions(results_dir, f'right_arm_{bp_type}', 'ppg')
        left_data = load_predictions(results_dir, f'left_arm_{bp_type}', 'ppg')
        
        if right_data is None:
            right_data = load_predictions(results_dir, f'right_arm_{bp_type}', 'both')
        if left_data is None:
            left_data = load_predictions(results_dir, f'left_arm_{bp_type}', 'both')
        
        if right_data is None or left_data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # 计算指标
        right_mae = np.mean(np.abs(right_data['y_true'] - right_data['y_pred']))
        left_mae = np.mean(np.abs(left_data['y_true'] - left_data['y_pred']))
        right_r = np.corrcoef(right_data['y_true'], right_data['y_pred'])[0, 1]
        left_r = np.corrcoef(left_data['y_true'], left_data['y_pred'])[0, 1]
        
        # 柱状图
        x = np.arange(2)
        width = 0.35
        
        mae_bars = ax.bar(x - width/2, [right_mae, left_mae], width, 
                         label='MAE', color='#3498DB', alpha=0.8)
        
        ax2 = ax.twinx()
        r_bars = ax2.bar(x + width/2, [right_r, left_r], width,
                        label='r', color='#E74C3C', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Right Arm', 'Left Arm'])
        ax.set_ylabel('MAE (mmHg)', color='#3498DB')
        ax2.set_ylabel('Correlation (r)', color='#E74C3C')
        ax.set_title(bp_type.upper(), fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, max(right_mae, left_mae) * 1.3)
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, val in zip(mae_bars, [right_mae, left_mae]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='#3498DB')
        for bar, val in zip(r_bars, [right_r, left_r]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='#E74C3C')
        
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'arm_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'arm_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / 'arm_comparison.png'}")


def plot_modality_comparison(results_dir, output_dir):
    """模态对比图 - 每个 target 取该模态最好的结果"""
    targets = ['right_arm_sbp', 'right_arm_dbp', 'right_arm_pp', 'left_arm_sbp']
    modalities = ['ppg', 'ecg', 'both']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        maes = []
        rs = []
        valid_modalities = []
        
        for mod in modalities:
            # use_best=True 会从 best_per_target_modality.csv 读取该 target+modality 的最佳实验
            data = load_predictions(results_dir, target, modality=mod, use_best=True)
            if data is not None:
                mae = np.mean(np.abs(data['y_true'] - data['y_pred']))
                r = np.corrcoef(data['y_true'], data['y_pred'])[0, 1]
                maes.append(mae)
                rs.append(r)
                valid_modalities.append(mod.upper())
        
        if not valid_modalities:
            continue
        
        x = np.arange(len(valid_modalities))
        width = 0.35
        
        colors_mae = [COLORS.get(m.lower(), '#888888') for m in valid_modalities]
        
        bars = ax.bar(x, maes, width * 2, color=colors_mae, alpha=0.8, edgecolor='black')
        
        # 添加r值标注
        for i, (bar, mae, r) in enumerate(zip(bars, maes, rs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'r={r:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   f'{mae:.1f}', ha='center', va='center', fontsize=10, 
                   color='white', fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(valid_modalities)
        ax.set_ylabel('MAE (mmHg)')
        ax.set_title(target.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(maes) * 1.4)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'modality_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'modality_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / 'modality_comparison.png'}")


def plot_summary_dashboard(results_dir, output_dir):
    """综合仪表盘"""
    fig = plt.figure(figsize=(16, 12))
    
    # 布局: 3行4列
    # 第1行: SBP散点图(右臂) + SBP散点图(左臂) + DBP散点图(右臂) + DBP散点图(左臂)
    # 第2行: PP散点图(右臂) + PP散点图(左臂) + MAE by category + 模态对比
    # 第3行: 总体指标汇总表
    
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.5], hspace=0.3, wspace=0.3)
    
    # 主要散点图
    scatter_configs = [
        ('right_arm_sbp', 0, 0), ('left_arm_sbp', 0, 1),
        ('right_arm_dbp', 0, 2), ('left_arm_dbp', 0, 3),
        ('right_arm_pp', 1, 0), ('left_arm_pp', 1, 1),
    ]
    
    for target, row, col in scatter_configs:
        ax = fig.add_subplot(gs[row, col])
        
        # 自动选择最佳配置
        data = load_predictions(results_dir, target, modality=None, use_best=True)
        if data is None:
            continue
        
        y_true = data['y_true']
        y_pred = data['y_pred']
        modality = data['modality']
        
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, c=COLORS.get(modality, COLORS['ppg']), edgecolors='none')
        
        lims = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        
        slope, intercept, r, p, se = stats.linregress(y_true, y_pred)
        x_fit = np.array(lims)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='#2C3E50', linewidth=1.5)
        
        mae = np.mean(np.abs(y_true - y_pred))
        
        ax.set_title(f"{target.replace('_', ' ').title()} ({modality})", fontsize=10, fontweight='bold')
        ax.set_xlabel('True (mmHg)', fontsize=8)
        ax.set_ylabel('Pred (mmHg)', fontsize=8)
        ax.tick_params(labelsize=7)
        
        textstr = f'r={r:.3f}\nMAE={mae:.1f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # MAE by BP level (右下)
    ax_cat = fig.add_subplot(gs[1, 2])
    
    data = load_predictions(results_dir, 'right_arm_sbp', modality=None, use_best=True)
    
    if data is not None:
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        bins = [(0, 120), (120, 140), (140, 160), (160, 300)]
        labels = ['<120', '120-140', '140-160', '>160']
        maes = []
        for low, high in bins:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                maes.append(np.mean(np.abs(y_true[mask] - y_pred[mask])))
            else:
                maes.append(0)
        
        colors = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C']
        bars = ax_cat.bar(range(len(labels)), maes, color=colors, alpha=0.8, edgecolor='black')
        ax_cat.set_xticks(range(len(labels)))
        ax_cat.set_xticklabels(labels, fontsize=8)
        ax_cat.set_ylabel('MAE (mmHg)', fontsize=9)
        ax_cat.set_title('SBP MAE by Level', fontsize=10, fontweight='bold')
        
        for bar, mae in zip(bars, maes):
            if mae > 0:
                ax_cat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                           f'{mae:.1f}', ha='center', fontsize=8)
    
    # 模态对比 (右下角) - 每个 target 取该模态最好的
    ax_mod = fig.add_subplot(gs[1, 3])
    
    # 收集每个 target 在各模态下的最佳结果
    all_best_maes = {'ppg': [], 'ecg': [], 'both': []}
    all_best_rs = {'ppg': [], 'ecg': [], 'both': []}
    
    for target in ['right_arm_sbp', 'right_arm_dbp', 'right_arm_pp', 'left_arm_sbp', 'left_arm_dbp', 'left_arm_pp']:
        for mod in ['ppg', 'ecg', 'both']:
            # use_best=True 会从 best_per_target_modality.csv 读取该 target+modality 的最佳实验
            data = load_predictions(results_dir, target, modality=mod, use_best=True)
            if data is not None:
                mae = np.mean(np.abs(data['y_true'] - data['y_pred']))
                r = np.corrcoef(data['y_true'], data['y_pred'])[0, 1]
                all_best_maes[mod].append(mae)
                all_best_rs[mod].append(r)
    
    mod_labels = ['PPG', 'ECG', 'Both']
    avg_maes = [np.mean(all_best_maes['ppg']) if all_best_maes['ppg'] else 0, 
                np.mean(all_best_maes['ecg']) if all_best_maes['ecg'] else 0, 
                np.mean(all_best_maes['both']) if all_best_maes['both'] else 0]
    avg_rs = [np.mean(all_best_rs['ppg']) if all_best_rs['ppg'] else 0, 
              np.mean(all_best_rs['ecg']) if all_best_rs['ecg'] else 0, 
              np.mean(all_best_rs['both']) if all_best_rs['both'] else 0]
    
    x = np.arange(3)
    bars = ax_mod.bar(x, avg_maes, color=[COLORS['ppg'], COLORS['ecg'], COLORS['both']], 
                     alpha=0.8, edgecolor='black')
    ax_mod.set_xticks(x)
    ax_mod.set_xticklabels(mod_labels, fontsize=9)
    ax_mod.set_ylabel('Avg MAE (mmHg)', fontsize=9)
    ax_mod.set_title('Modality Comparison', fontsize=10, fontweight='bold')
    
    for bar, mae, r in zip(bars, avg_maes, avg_rs):
        ax_mod.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{mae:.1f}\nr={r:.2f}', ha='center', fontsize=8)
    
    # 底部汇总表
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # 创建汇总数据
    table_data = []
    targets = ['right_arm_sbp', 'right_arm_dbp', 'right_arm_pp', 
               'left_arm_sbp', 'left_arm_dbp', 'left_arm_pp']
    
    for target in targets:
        data = load_predictions(results_dir, target, modality=None, use_best=True)
        if data is not None:
            mae = np.mean(np.abs(data['y_true'] - data['y_pred']))
            r = np.corrcoef(data['y_true'], data['y_pred'])[0, 1]
            modality = data['modality']
            table_data.append([target.replace('_', ' ').title(), modality.upper(), f'{mae:.2f}', f'{r:.3f}'])
    
    if table_data:
        table = ax_table.table(cellText=table_data,
                              colLabels=['Target', 'Best Modality', 'MAE (mmHg)', 'r'],
                              loc='center',
                              cellLoc='center',
                              colWidths=[0.25, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    plt.savefig(output_dir / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'dashboard.pdf', bbox_inches='tight')
    plt.close()
    print(f"[Saved] {output_dir / 'dashboard.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='BP Results Visualization')
    parser.add_argument('--results_dir', type=str, 
                       default='/home/youliang/youliang_data2/bp/bp_dec30/runs',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for figures (default: results_dir/figures)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 50)
    
    # 生成所有图表
    print("\n[1/6] Plotting BP distributions...")
    plot_distribution(results_dir, output_dir)
    
    print("\n[2/6] Plotting scatter plots...")
    plot_scatter_all(results_dir, output_dir)
    
    print("\n[3/6] Plotting MAE by category...")
    plot_mae_by_category(results_dir, output_dir)
    
    print("\n[4/6] Plotting arm comparison...")
    plot_arm_comparison(results_dir, output_dir)
    
    print("\n[5/6] Plotting modality comparison...")
    plot_modality_comparison(results_dir, output_dir)
    
    print("\n[6/6] Plotting summary dashboard...")
    plot_summary_dashboard(results_dir, output_dir)
    
    print("\n" + "=" * 50)
    print("All figures saved!")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()