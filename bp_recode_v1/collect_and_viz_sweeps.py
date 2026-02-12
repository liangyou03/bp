#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect + visualize BP finetune sweep results.

Usage:
python collect_and_viz_sweeps.py --results_dir /home/youliang/youliang_data2/bp/bp_recode_v1/bp_finetune_sweeps/alltargets_6each_parallel_20260202_022212
  python collect_and_viz_sweeps.py --results_dir /home/youliang/youliang_data2/bp/bp_recode_v1/bp_finetune_sweeps/sbp_smart_20260129_081351
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def collect_results(results_dir: str, output_csv: str = None):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    records = []

    metrics_files = sorted(results_dir.rglob("metrics_test_bp_*_records.json"))
    if len(metrics_files) == 0:
        print("[ERR] No metrics_test_bp_*_records.json found under:", results_dir)
        return None

    for mf in metrics_files:
        try:
            with open(mf, "r") as f:
                m = json.load(f)

            rel = mf.relative_to(results_dir)
            exp_name = rel.parts[0] if len(rel.parts) > 0 else mf.parent.name

            stem = mf.stem
            parts = stem.replace("metrics_test_bp_", "").replace("_records", "").split("_")

            target_col = m.get("target_col", "")
            modality = m.get("modality", "")
            loss = m.get("loss", "")

            # Best-effort fallback from filename if JSON missing fields
            if not target_col and len(parts) >= 1:
                target_col = parts[0]
            if not modality and len(parts) >= 2:
                modality = parts[1]
            if not loss and len(parts) >= 3:
                loss = parts[2]

            constrain = m.get("constrain", "")
            if not constrain and len(parts) > 3:
                constrain = "_".join(parts[3:])

            record = {
                "exp_name": exp_name,
                "metrics_path": str(mf),
                "target_col": target_col,
                "modality": modality,
                "loss": loss,
                "constrain": constrain,
                "N_records": m.get("N_records", m.get("n_records", 0)),
                "MAE_raw": m.get("MAE_raw", np.nan),
                "RMSE_raw": m.get("RMSE_raw", np.nan),
                "r_raw": m.get("r_raw", np.nan),
                "R2_raw": m.get("R2_raw", np.nan),
                "MAE_cal": m.get("MAE_cal", np.nan),
                "RMSE_cal": m.get("RMSE_cal", np.nan),
                "r_cal": m.get("r_cal", np.nan),
                "R2_cal": m.get("R2_cal", np.nan),
                "calib_a": m.get("a", np.nan),
                "calib_b": m.get("b", np.nan),
            }

            # Optional: if records exist, mark it (for later detailed plots)
            record["has_records"] = int(
                any(k in m for k in ["y_true", "y_pred", "y_true_cal", "y_pred_cal", "records"])
            )

            records.append(record)

        except Exception as e:
            print(f"[WARN] Failed to parse {mf}: {e}")

    if not records:
        print("[ERR] Parsed 0 records.")
        return None

    df = pd.DataFrame(records)

    # Normalize columns
    for c in ["MAE_raw", "RMSE_raw", "r_raw", "R2_raw", "MAE_cal", "RMSE_cal", "r_cal", "R2_cal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["target_col", "MAE_cal", "MAE_raw"], na_position="last").reset_index(drop=True)

    if output_csv is None:
        output_csv = results_dir / "all_results.csv"
    else:
        output_csv = Path(output_csv)

    df.to_csv(output_csv, index=False)
    print(f"[Saved] {output_csv}")
    print(f"Total metrics files parsed: {len(df)}")

    # Best per target
    summary_records = []
    for target in sorted(df["target_col"].dropna().unique()):
        df_t = df[df["target_col"] == target].copy()
        df_t = df_t[np.isfinite(df_t["MAE_cal"].to_numpy())]
        if len(df_t) == 0:
            continue
        best = df_t.iloc[df_t["MAE_cal"].argmin()]
        summary_records.append({
            "target": target,
            "best_exp": best["exp_name"],
            "modality": best["modality"],
            "loss": best["loss"],
            "constrain": best["constrain"],
            "MAE_cal": best["MAE_cal"],
            "RMSE_cal": best["RMSE_cal"],
            "r_cal": best["r_cal"],
            "R2_cal": best["R2_cal"],
            "metrics_path": best["metrics_path"],
        })

    df_summary = pd.DataFrame(summary_records)
    summary_csv = results_dir / "best_per_target.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")

    # Best per target + modality
    best_tm = []
    for target in sorted(df["target_col"].dropna().unique()):
        for modality in ["ppg", "ecg", "both"]:
            df_tm = df[(df["target_col"] == target) & (df["modality"] == modality)].copy()
            df_tm = df_tm[np.isfinite(df_tm["MAE_cal"].to_numpy())]
            if len(df_tm) == 0:
                continue
            best = df_tm.iloc[df_tm["MAE_cal"].argmin()]
            best_tm.append({
                "target": target,
                "modality": modality,
                "best_exp": best["exp_name"],
                "loss": best["loss"],
                "constrain": best["constrain"],
                "MAE_cal": best["MAE_cal"],
                "RMSE_cal": best["RMSE_cal"],
                "r_cal": best["r_cal"],
                "R2_cal": best["R2_cal"],
                "metrics_path": best["metrics_path"],
            })

    df_best_tm = pd.DataFrame(best_tm)
    best_tm_csv = results_dir / "best_per_target_modality.csv"
    df_best_tm.to_csv(best_tm_csv, index=False)
    print(f"[Saved] {best_tm_csv}")

    # =============== Visualization ===============
    viz_dir = results_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = viz_dir / "summary.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: MAE_cal distribution by modality (boxplot)
        df_plot = df[np.isfinite(df["MAE_cal"].to_numpy())].copy()
        if len(df_plot) > 0:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)

            groups = []
            labels = []
            for mod in ["ppg", "ecg", "both"]:
                vals = df_plot[df_plot["modality"] == mod]["MAE_cal"].dropna().to_numpy()
                if len(vals) > 0:
                    groups.append(vals)
                    labels.append(mod)
            if len(groups) > 0:
                ax.boxplot(groups, labels=labels)
                ax.set_title("MAE_cal distribution by modality (all runs)")
                ax.set_ylabel("MAE_cal")
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            pdf.savefig(fig)
            fig.savefig(viz_dir / "mae_cal_box_by_modality.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # Page 2: Scatter MAE_cal vs r_cal
        df_sc = df[np.isfinite(df["MAE_cal"].to_numpy()) & np.isfinite(df["r_cal"].to_numpy())].copy()
        if len(df_sc) > 0:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
            ax.scatter(df_sc["MAE_cal"].to_numpy(), df_sc["r_cal"].to_numpy(), s=12)
            ax.set_title("MAE_cal vs r_cal (all runs)")
            ax.set_xlabel("MAE_cal (lower is better)")
            ax.set_ylabel("r_cal (higher is better)")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            pdf.savefig(fig)
            fig.savefig(viz_dir / "scatter_mae_vs_r.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # Page 3+: For each target, top-k runs barplot (by MAE_cal)
        topk = 10
        for target in sorted(df["target_col"].dropna().unique()):
            df_t = df[(df["target_col"] == target) & np.isfinite(df["MAE_cal"].to_numpy())].copy()
            if len(df_t) == 0:
                continue
            df_t = df_t.sort_values("MAE_cal").head(topk)

            labels = [
                f"{r.exp_name}\n{r.modality}/{r.loss}"
                for r in df_t.itertuples(index=False)
            ]
            vals = df_t["MAE_cal"].to_numpy()

            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(111)
            ax.bar(np.arange(len(vals)), vals)
            ax.set_title(f"Top {min(topk, len(vals))} runs by MAE_cal: {target}")
            ax.set_ylabel("MAE_cal")
            ax.set_xticks(np.arange(len(vals)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

            pdf.savefig(fig)
            fig.savefig(viz_dir / f"top{topk}_{target}_mae_cal.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # Page: Heatmap-style table (pivot) MAE_cal by (target, modality) taking best within each exp
        df_h = df[np.isfinite(df["MAE_cal"].to_numpy())].copy()
        if len(df_h) > 0:
            # best per (exp, target, modality)
            df_h = df_h.sort_values("MAE_cal").groupby(["exp_name", "target_col", "modality"], as_index=False).first()
            pivot = df_h.pivot_table(index=["exp_name"], columns=["target_col", "modality"], values="MAE_cal", aggfunc="min")

            fig = plt.figure(figsize=(14, max(4, 0.25 * len(pivot))))
            ax = fig.add_subplot(111)
            ax.axis("off")
            tbl = ax.table(
                cellText=np.round(pivot.fillna(np.nan).to_numpy(), 3),
                rowLabels=pivot.index.tolist(),
                colLabels=[f"{a}\n{b}" for a, b in pivot.columns.tolist()],
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1.0, 1.2)
            ax.set_title("MAE_cal table (best per exp/target/modality)")
            pdf.savefig(fig)
            fig.savefig(viz_dir / "mae_cal_table.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    print(f"[Saved] {pdf_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    collect_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()