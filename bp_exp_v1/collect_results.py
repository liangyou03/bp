#!/usr/bin/env python3
"""收集BP实验结果"""
import json
from pathlib import Path
import pandas as pd

def collect_results(exp_dir):
    exp_dir = Path(exp_dir)
    records = []
    
    for exp_path in sorted(exp_dir.glob("bp_*")):
        if not exp_path.is_dir():
            continue
        
        results_file = exp_path / "results.json"
        if not results_file.exists():
            continue
        
        try:
            with open(results_file) as f:
                results = json.load(f)
            with open(exp_path / "config.json") as f:
                config = json.load(f)
            
            # 解析实验名称: bp_500hz_{target}_{modality}
            name = exp_path.name
            parts = name.split("_")
            if len(parts) >= 4:
                target = parts[2]
                modality = parts[3]
            else:
                target = config.get("target_col", "")
                modality = config.get("modality", "")
            
            test_cal = results.get("test_calibrated", {})
            records.append({
                "exp_name": name,
                "target": target,
                "modality": modality,
                "MAE_cal": test_cal.get("MAE", 0),
                "r_cal": test_cal.get("r", 0),
                "R2_cal": test_cal.get("R2", 0),
            })
        except Exception as e:
            print(f"Warning: {exp_path}: {e}")
    
    if not records:
        print("No results found!")
        return
    
    df = pd.DataFrame(records)
    df = df.sort_values(["target", "MAE_cal"])
    
    print("=" * 60)
    print("BEST RESULTS PER TARGET")
    print("=" * 60)
    
    for target in df["target"].unique():
        df_t = df[df["target"] == target]
        if len(df_t) > 0:
            best = df_t.iloc[0]
            print(f"\n{target}:")
            print(f"  {best['modality']} | MAE={best['MAE_cal']:.2f} | r={best['r_cal']:.3f}")
    
    df.to_csv(exp_dir / "summary.csv", index=False)
    print(f"\nSaved: {exp_dir / 'summary.csv'}")

if __name__ == "__main__":
    import sys
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs"
    collect_results(exp_dir)
