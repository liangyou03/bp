#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import optuna


TARGET_ALIAS = {
    "sbp": "right_arm_sbp",
    "dbp": "right_arm_dbp",
    "map": "right_arm_mbp",
    "mbp": "right_arm_mbp",
}


def resolve_target(target: str):
    key = str(target).strip().lower()
    canonical = TARGET_ALIAS.get(key, str(target).strip())
    return canonical, key


def parse_args():
    p = argparse.ArgumentParser(description="Bayesian optimization for bp_recode_v1/finetune_bp.py")
    p.add_argument("--target", required=True)
    p.add_argument("--modality", default="both", choices=["ppg", "ecg", "both"])
    p.add_argument("--npz_dir", default="/home/youliang/youliang_data2/bp/bp_recode_v1/output/npz")
    p.add_argument("--labels_csv", default="/home/youliang/youliang_data2/bp/bp_recode_v1/output/labels.csv")
    p.add_argument("--clip_ckpt", default="/home/youliang/youliang_data2/bp/bp_recode_v1/clip_finetune_out/best.pth")
    p.add_argument("--workdir", default="/home/youliang/youliang_data2/bp/bp_recode_v1")

    p.add_argument("--n_trials", type=int, default=24)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--search_epochs", type=int, default=30)
    p.add_argument("--final_epochs", type=int, default=80)
    p.add_argument("--objective", default="val_raw", choices=["val_raw", "val_cal"])
    p.add_argument("--run_final", action="store_true")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--use_raw", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=666)

    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--study_name", default="")
    p.add_argument("--storage", default="")
    p.add_argument("--study_root", default="")
    p.add_argument("--n_trials_local", type=int, default=0)
    return p.parse_args()


def trial_command(args, trial, out_dir):
    loss = trial.suggest_categorical("loss", ["huber", "mae_pearson", "mse"])
    lr_backbone = trial.suggest_float("lr_backbone", 1e-5, 1e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 5e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    head_hidden = trial.suggest_categorical("head_hidden", [128, 256, 384])
    freeze_backbone = trial.suggest_categorical("freeze_backbone", [False, True])

    cmd = [
        sys.executable,
        "finetune_bp.py",
        "--npz_dir", args.npz_dir,
        "--labels_csv", args.labels_csv,
        "--clip_ckpt", args.clip_ckpt,
        "--out_dir", str(out_dir),
        "--target_col", args.target_col,
        "--modality", args.modality,
        "--loss", loss,
        "--lr_backbone", str(lr_backbone),
        "--lr_head", str(lr_head),
        "--weight_decay", str(weight_decay),
        "--head_hidden", str(head_hidden),
        "--epochs", str(args.search_epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--gpu", "0",
        "--patience", "10",
        "--early_on", "raw",
    ]
    if args.use_raw:
        cmd.append("--use_raw")
    if freeze_backbone:
        cmd.append("--freeze_backbone")
    if loss == "mae_pearson":
        alpha = trial.suggest_float("loss_alpha", 0.2, 1.0)
        beta = trial.suggest_float("loss_beta", 0.2, 1.0)
        cmd += ["--loss_alpha", str(alpha), "--loss_beta", str(beta)]

    trial.set_user_attr("cmd", " ".join(cmd))
    return cmd


def _read_val_metric(metrics_path: Path, objective: str):
    with metrics_path.open("r") as f:
        m = json.load(f)
    if objective == "val_raw":
        return float(m.get("MAE_raw", float("inf")))
    return float(m.get("MAE_cal", float("inf")))


def run_trial(args, trial, gpu_id, study_root):
    trial_dir = Path(study_root) / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"

    cmd = trial_command(args, trial, trial_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=args.workdir, env=env, stdout=f, stderr=subprocess.STDOUT)
    sec = time.time() - t0

    if proc.returncode != 0:
        trial.set_user_attr("status", "failed")
        trial.set_user_attr("returncode", proc.returncode)
        return float("inf")

    val_metrics = trial_dir / f"metrics_val_bp_{args.target_col}_records.json"
    if not val_metrics.exists():
        trial.set_user_attr("status", "no_val_metrics")
        return float("inf")

    score = _read_val_metric(val_metrics, args.objective)
    trial.set_user_attr("status", "ok")
    trial.set_user_attr("gpu", gpu_id)
    trial.set_user_attr("duration_sec", sec)
    trial.set_user_attr("val_metrics_path", str(val_metrics))
    trial.set_user_attr("objective", args.objective)
    return float(score)


def run_worker(args):
    study = None
    for _ in range(30):
        try:
            study = optuna.load_study(study_name=args.study_name, storage=args.storage)
            break
        except Exception:
            time.sleep(1.0)
    if study is None:
        raise RuntimeError("Worker failed to load study")

    def objective(trial):
        return run_trial(args, trial, args.gpu, args.study_root)

    study.optimize(objective, n_trials=args.n_trials_local, show_progress_bar=False)


def run_final_best(args, study_name, storage, study_root):
    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_trial
    best_dir = Path(study_root) / "best_final"
    best_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "finetune_bp.py",
        "--npz_dir", args.npz_dir,
        "--labels_csv", args.labels_csv,
        "--clip_ckpt", args.clip_ckpt,
        "--out_dir", str(best_dir),
        "--target_col", args.target_col,
        "--modality", args.modality,
        "--loss", best.params["loss"],
        "--lr_backbone", str(best.params["lr_backbone"]),
        "--lr_head", str(best.params["lr_head"]),
        "--weight_decay", str(best.params["weight_decay"]),
        "--head_hidden", str(best.params["head_hidden"]),
        "--epochs", str(args.final_epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--gpu", "0",
        "--patience", "15",
        "--early_on", "raw",
    ]
    if args.use_raw:
        cmd.append("--use_raw")
    if bool(best.params.get("freeze_backbone", False)):
        cmd.append("--freeze_backbone")
    if best.params["loss"] == "mae_pearson":
        cmd += ["--loss_alpha", str(best.params["loss_alpha"]), "--loss_beta", str(best.params["loss_beta"])]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with (best_dir / "final.log").open("w") as f:
        subprocess.run(cmd, cwd=args.workdir, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)

    # metadata-aware calibration comparison (val fit -> test apply)
    compare_cmd = [
        sys.executable,
        "compare_metadata_calibration_bp_recode.py",
        "--run_dir", str(best_dir),
        "--target", args.target,
        "--labels_csv", args.labels_csv,
        "--npz_dir", args.npz_dir,
        "--modality", args.modality,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--gpu", "0",
    ]
    if args.use_raw:
        compare_cmd.append("--use_raw")
    with (best_dir / "metadata_calibration.log").open("w") as f:
        subprocess.run(compare_cmd, cwd=args.workdir, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)


def run_main(args):
    os.chdir(args.workdir)
    if not Path(args.clip_ckpt).exists():
        raise FileNotFoundError(args.clip_ckpt)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_root = Path(args.workdir) / "bp_finetune_bayes" / f"bayes_{args.target}_{args.modality}_{ts}"
    study_root.mkdir(parents=True, exist_ok=True)

    study_name = f"bayes_{args.target}_{args.modality}_{ts}"
    storage = f"sqlite:///{(study_root / 'study.db').as_posix()}"

    optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6),
    )

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_workers = max(1, len(gpus))
    n_trials_local = int(math.ceil(args.n_trials / float(n_workers)))

    procs = []
    for gpu in gpus:
        log_fp = (study_root / f"worker_gpu{gpu}.log").open("w")
        cmd = [
            sys.executable,
            "bayes_opt_recode.py",
            "--worker",
            "--target", args.target,
            "--modality", args.modality,
            "--npz_dir", args.npz_dir,
            "--labels_csv", args.labels_csv,
            "--clip_ckpt", args.clip_ckpt,
            "--workdir", args.workdir,
            "--n_trials", str(args.n_trials),
            "--gpus", args.gpus,
            "--search_epochs", str(args.search_epochs),
            "--final_epochs", str(args.final_epochs),
            "--objective", args.objective,
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--gpu", str(gpu),
            "--study_name", study_name,
            "--storage", storage,
            "--study_root", str(study_root),
            "--n_trials_local", str(n_trials_local),
        ]
        if args.use_raw:
            cmd.append("--use_raw")
        p = subprocess.Popen(cmd, cwd=args.workdir, stdout=log_fp, stderr=subprocess.STDOUT)
        procs.append((p, log_fp))

    for p, _ in procs:
        p.wait()
    for _, fp in procs:
        fp.close()

    study = optuna.load_study(study_name=study_name, storage=storage)
    best = {
        "value": study.best_value,
        "params": study.best_params,
        "number": study.best_trial.number,
        "objective": args.objective,
    }
    with (study_root / "best.json").open("w") as f:
        json.dump(best, f, indent=2)
    study.trials_dataframe().to_csv(study_root / "trials.csv", index=False)

    if args.run_final:
        run_final_best(args, study_name, storage, study_root)

    print(f"DONE study_root={study_root}")
    print(f"BEST={best}")


def main():
    args = parse_args()
    args.target_col, args.target_key = resolve_target(args.target)
    if args.worker:
        run_worker(args)
    else:
        run_main(args)


if __name__ == "__main__":
    main()
