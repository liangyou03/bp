#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集 BP finetune 实验结果

python collect_results.py --results_dir "/home/youliang/youliang_data2/bp/bp_dec30/runs_resampled"
python collect_results.py --results_dir "/home/youliang/youliang_data2/bp/bp_dec30/runs"
python collect_results.py --results_dir "/home/youliang/youliang_data2/bp/bp_dec30/runs_resampled_best"

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集 BP finetune 实验结果
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd

def collect_results(results_dir: str, output_csv: str = None):
    """收集所有实验的 metrics JSON 文件"""
    results_dir = Path(results_dir)
    
    records = []
    
    # 遍历所有 exp* 目录
    for exp_dir in sorted(results_dir.glob("exp*")):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # 查找 metrics JSON 文件 (records 级别)
        metrics_files = list(exp_dir.glob("metrics_test_bp_*_records.json"))
        
        for mf in metrics_files:
            try:
                with open(mf, "r") as f:
                    m = json.load(f)
                
                # 解析实验配置
                # 文件名格式: metrics_test_bp_{target}_{modality}_{loss}_{constrain}_records.json
                fname = mf.stem
                parts = fname.replace("metrics_test_bp_", "").replace("_records", "").split("_")
                
                record = {
                    "exp_name": exp_name,
                    "target_col": m.get("target_col", ""),
                    "modality": m.get("modality", ""),
                    "N_records": m.get("N_records", 0),
                    # Raw metrics
                    "MAE_raw": m.get("MAE_raw", -1),
                    "RMSE_raw": m.get("RMSE_raw", -1),
                    "r_raw": m.get("r_raw", -1),
                    "R2_raw": m.get("R2_raw", -1),
                    # Calibrated metrics
                    "MAE_cal": m.get("MAE_cal", -1),
                    "RMSE_cal": m.get("RMSE_cal", -1),
                    "r_cal": m.get("r_cal", -1),
                    "R2_cal": m.get("R2_cal", -1),
                    # Calibration params
                    "calib_a": m.get("a", 0),
                    "calib_b": m.get("b", 1),
                }
                records.append(record)
                
            except Exception as e:
                print(f"[Warning] Failed to parse {mf}: {e}")
    
    if not records:
        print("No results found!")
        return None
    
    # 创建 DataFrame
    df = pd.DataFrame(records)
    
    # 按 target_col 和 MAE_cal 排序
    df = df.sort_values(["target_col", "MAE_cal"])
    
    # 输出
    if output_csv is None:
        output_csv = results_dir / "all_results.csv"
    
    df.to_csv(output_csv, index=False)
    print(f"\n[Saved] {output_csv}")
    print(f"Total experiments: {len(df)}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("SUMMARY: Best results per target (by calibrated MAE)")
    print("=" * 80)
    
    summary_records = []
    for target in df["target_col"].unique():
        df_target = df[df["target_col"] == target]
        best = df_target.loc[df_target["MAE_cal"].idxmin()]
        
        print(f"\n{target}:")
        print(f"  Best: {best['exp_name']} ({best['modality']})")
        print(f"  MAE={best['MAE_cal']:.2f}  r={best['r_cal']:.3f}  R²={best['R2_cal']:.3f}")
        
        summary_records.append({
            "target": target,
            "best_exp": best["exp_name"],
            "modality": best["modality"],
            "MAE_cal": best["MAE_cal"],
            "r": best["r_cal"],
            "R2": best["R2_cal"],
        })
    
    # 保存摘要
    df_summary = pd.DataFrame(summary_records)
    summary_csv = results_dir / "best_per_target.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n[Saved] {summary_csv}")
    
    # ========== 新增：每个 target + modality 组合的最佳结果 ==========
    best_per_target_modality = []
    for target in df["target_col"].unique():
        for modality in ["ppg", "ecg", "both"]:
            df_tm = df[(df["target_col"] == target) & (df["modality"] == modality)]
            if len(df_tm) > 0:
                best = df_tm.loc[df_tm["MAE_cal"].idxmin()]
                best_per_target_modality.append({
                    "target": target,
                    "modality": modality,
                    "best_exp": best["exp_name"],
                    "MAE_cal": best["MAE_cal"],
                    "r": best["r_cal"],
                    "R2": best["R2_cal"],
                })
    
    df_best_tm = pd.DataFrame(best_per_target_modality)
    best_tm_csv = results_dir / "best_per_target_modality.csv"
    df_best_tm.to_csv(best_tm_csv, index=False)
    print(f"[Saved] {best_tm_csv}")
    
    # 打印模态对比
    print("\n" + "=" * 80)
    print("MODALITY COMPARISON (best per target, then average)")
    print("=" * 80)
    
    for modality in ["ppg", "ecg", "both"]:
        df_mod = df[df["modality"] == modality]
        if len(df_mod) > 0:
            # 每个 target 取该模态最好的结果
            best_mae = df_mod.groupby("target_col")["MAE_cal"].min()
            best_r = df_mod.loc[df_mod.groupby("target_col")["MAE_cal"].idxmin(), ["target_col", "r_cal"]].set_index("target_col")["r_cal"]
            
            print(f"\n{modality.upper()}:")
            print(f"  Best MAE_cal: {best_mae.mean():.2f} ± {best_mae.std():.2f}")
            print(f"  Best r:       {best_r.mean():.3f} ± {best_r.std():.3f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect BP finetune experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing exp* folders")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results_dir/all_results.csv)")
    args = parser.parse_args()
    
    collect_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()