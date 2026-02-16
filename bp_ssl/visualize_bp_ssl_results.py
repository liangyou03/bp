#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGETS = [
    "right_arm_sbp", "right_arm_dbp", "right_arm_mbp", "right_arm_pp",
    "left_arm_sbp", "left_arm_dbp", "left_arm_mbp", "left_arm_pp",
]


def _safe_load_json(p: Path):
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def collect_runs(results_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(results_root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        payload = _safe_load_json(metrics_path)
        if payload is None:
            continue
        cfg = payload.get("config", {})
        test_subj = payload.get("test", {}).get("subject_level", {})
        test_rec = payload.get("test", {}).get("record_level", {})
        pred_csv = run_dir / "subject_level_predictions.csv"
        rows.append({
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "target": cfg.get("TARGET_COL"),
            "mode": cfg.get("MODE"),
            "loss_mode": cfg.get("LOSS_MODE"),
            "lr": cfg.get("LR"),
            "metrics_path": str(metrics_path),
            "pred_csv": str(pred_csv) if pred_csv.exists() else None,
            "test_subject_mae": test_subj.get("MAE"),
            "test_subject_rmse": test_subj.get("RMSE"),
            "test_subject_r": test_subj.get("r"),
            "test_subject_r2": test_subj.get("R2"),
            "test_record_mae": test_rec.get("MAE"),
            "test_record_rmse": test_rec.get("RMSE"),
            "test_record_r": test_rec.get("r"),
            "test_record_r2": test_rec.get("R2"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["target", "test_subject_mae"]).reset_index(drop=True)
    return df


def pick_best(df: pd.DataFrame) -> pd.DataFrame:
    best = []
    for t in TARGETS:
        part = df[df["target"] == t].copy()
        if part.empty:
            continue
        part = part.sort_values("test_subject_mae", ascending=True)
        best.append(part.iloc[0])
    if not best:
        return pd.DataFrame()
    return pd.DataFrame(best).reset_index(drop=True)


def _load_pred_pair(csv_path: Path, target: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, None
    if target not in df.columns or "pred_value" not in df.columns:
        return None, None
    y_true = pd.to_numeric(df[target], errors="coerce").values
    y_pred = pd.to_numeric(df["pred_value"], errors="coerce").values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def plot_scatter(best_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, t in enumerate(TARGETS):
        ax = axes[i]
        row = best_df[best_df["target"] == t]
        if row.empty:
            ax.set_title(f"{t}\nNo run")
            ax.axis("off")
            continue
        pred_csv = Path(row.iloc[0]["pred_csv"])
        y_true, y_pred = _load_pred_pair(pred_csv, t)
        if y_true is None or len(y_true) == 0:
            ax.set_title(f"{t}\nNo prediction csv")
            ax.axis("off")
            continue

        mae = np.mean(np.abs(y_true - y_pred))
        r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

        ax.scatter(y_true, y_pred, s=8, alpha=0.3)
        lo = float(min(y_true.min(), y_pred.min()) - 3)
        hi = float(max(y_true.max(), y_pred.max()) + 3)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{t}\nMAE={mae:.2f}, r={r:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")

    plt.tight_layout()
    fig.savefig(out_dir / "scatter_best_per_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_hist(best_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, t in enumerate(TARGETS):
        ax = axes[i]
        row = best_df[best_df["target"] == t]
        if row.empty:
            ax.set_title(f"{t}\nNo run")
            ax.axis("off")
            continue
        pred_csv = Path(row.iloc[0]["pred_csv"])
        y_true, y_pred = _load_pred_pair(pred_csv, t)
        if y_true is None or len(y_true) == 0:
            ax.set_title(f"{t}\nNo prediction csv")
            ax.axis("off")
            continue

        err = y_pred - y_true
        mae = np.mean(np.abs(err))
        bias = np.mean(err)
        std = np.std(err)

        ax.hist(err, bins=30, alpha=0.8)
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"{t}\nMAE={mae:.2f}, bias={bias:.2f}, sd={std:.2f}")
        ax.set_xlabel("Pred - True (mmHg)")
        ax.set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(out_dir / "error_hist_best_per_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="bp_ssl results root")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(results_dir)
    if df.empty:
        print("No metrics.json found under", results_dir)
        return

    df = df.sort_values(["target", "test_subject_mae"], ascending=[True, True]).reset_index(drop=True)
    df.to_csv(out_dir / "all_results.csv", index=False)

    best_df = pick_best(df)
    best_df.to_csv(out_dir / "best_per_target.csv", index=False)

    if not best_df.empty:
        plot_scatter(best_df, out_dir)
        plot_error_hist(best_df, out_dir)

    print("Saved:", out_dir / "all_results.csv")
    print("Saved:", out_dir / "best_per_target.csv")
    print("Done")


if __name__ == "__main__":
    main()
