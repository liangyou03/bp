#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集实验结果

使用方法:
python collect_results.py --exp_dir python collect_and_viz_sweeps.py --results_dir /home/youliang/youliang_data2/bp/bp_recode_v1/bp_finetune_sweeps/sweep_20260128_133554
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def collect_results(exp_dir):
    """收集所有实验结果"""
    exp_dir = Path(exp_dir)
    
    records = []
    
    for exp_path in sorted(exp_dir.glob("exp*")):
        if not exp_path.is_dir():
            continue
        
        results_file = exp_path / "results.json"
        config_file = exp_path / "config.json"
        
        if not results_file.exists():
            continue
        
        try:
            with open(results_file) as f:
                results = json.load(f)
            with open(config_file) as f:
                config = json.load(f)
            
            record = {
                "exp_name": exp_path.name,
                "target": config.get("target_col", ""),
                "modality": config.get("modality", ""),
                "loss": config.get("loss", ""),
                "freeze": config.get("freeze_backbone", False),
                "lr_backbone": config.get("lr_backbone", 0),
                "lr_head": config.get("lr_head", 0),
            }
            
            # Test metrics
            test_raw = results.get("test_raw", {})
            test_cal = results.get("test_calibrated", {})
            
            for metric in ["MAE", "RMSE", "r", "R2"]:
                record[f"{metric}_raw"] = test_raw.get(metric, None)
                record[f"{metric}_cal"] = test_cal.get(metric, None)
            
            # Calibration
            calib = results.get("calibration", {})
            record["calib_a"] = calib.get("a", 1.0)
            record["calib_b"] = calib.get("b", 0.0)
            
            records.append(record)
            
        except Exception as e:
            print(f"Warning: failed to parse {exp_path}: {e}")
    
    if not records:
        print("No results found!")
        return None
    
    df = pd.DataFrame(records)
    
    # 保存完整结果
    df.to_csv(exp_dir / "all_results.csv", index=False)
    
    # 按target和MAE排序
    df_sorted = df.sort_values(["target", "MAE_cal"])
    
    print("=" * 80)
    print("BEST RESULTS PER TARGET (by calibrated MAE)")
    print("=" * 80)
    
    for target in df["target"].unique():
        df_target = df_sorted[df_sorted["target"] == target]
        if len(df_target) == 0:
            continue
        best = df_target.iloc[0]
        print(f"\n{target}:")
        print(f"  Best: {best['exp_name']} ({best['modality']}, freeze={best['freeze']})")
        print(f"  MAE_cal={best['MAE_cal']:.2f}, r_cal={best['r_cal']:.3f}, R2_cal={best['R2_cal']:.3f}")
    
    # 模态对比
    print("\n" + "=" * 80)
    print("MODALITY COMPARISON")
    print("=" * 80)
    
    for modality in ["ppg", "ecg", "both"]:
        df_mod = df[df["modality"] == modality]
        if len(df_mod) > 0:
            mae_list = []
            for target in df["target"].unique():
                df_tm = df_mod[(df_mod["target"] == target)]
                if len(df_tm) > 0:
                    best = df_tm.loc[df_tm["MAE_cal"].idxmin()]
                    mae_list.append(best["MAE_cal"])
            
            if mae_list:
                import numpy as np
                print(f"\n{modality.upper()}:")
                print(f"  Avg MAE_cal: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}")
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    
    collect_results(args.exp_dir)


if __name__ == "__main__":
    main()
