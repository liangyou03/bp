#!/usr/bin/env python
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


def parse_args():
    p = argparse.ArgumentParser(description="Bayesian optimization for single-target BP finetune")
    p.add_argument("--target", default="right_arm_sbp")
    p.add_argument("--mode", default="fusion", choices=["ppg", "ecg", "fusion"])
    p.add_argument("--ckpt", default="./checkpoints/ssl_multimodal_bp_600hz_v1/checkpoint_epoch_100.pth")
    p.add_argument("--n_trials", type=int, default=24)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--search_epochs", type=int, default=20)
    p.add_argument("--final_epochs", type=int, default=50)
    p.add_argument("--batch_choices", default="128,256")
    p.add_argument("--workdir", default="/home/youliang/youliang_data2/bp/bp_ssl")
    p.add_argument("--run_final", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--study_name", default="")
    p.add_argument("--storage", default="")
    p.add_argument("--study_root", default="")
    p.add_argument("--n_trials_local", type=int, default=0)
    return p.parse_args()


def trial_command(args, trial, out_dir):
    loss_mode = trial.suggest_categorical("loss_mode", ["mse", "mse+dist", "wmse", "mse*pearson"])
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [int(x) for x in args.batch_choices.split(",")])
    freeze_encoders = trial.suggest_categorical("freeze_encoders", [False, True])

    cmd = [
        sys.executable,
        "launch_finetune_job.py",
        "--ckpt", args.ckpt,
        "--output_dir", str(out_dir),
        "--target", args.target,
        "--loss_mode", loss_mode,
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--epochs", str(args.search_epochs),
        "--mode", args.mode,
    ]
    if freeze_encoders:
        cmd.append("--freeze_encoders")
    if loss_mode == "mse+dist":
        cmd += ["--lambda_dist", str(trial.suggest_float("lambda_dist", 0.05, 0.6))]
    if loss_mode == "wmse":
        cmd += ["--wmse_smoothing", str(trial.suggest_float("wmse_smoothing", 0.5, 4.0))]
    if loss_mode == "mse*pearson":
        trial.set_user_attr("pearson_alpha", trial.suggest_float("pearson_alpha", 0.2, 2.0))

    trial.set_user_attr("cmd", " ".join(cmd))
    return cmd


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

    metrics_path = trial_dir / "metrics.json"
    if not metrics_path.exists():
        trial.set_user_attr("status", "no_metrics")
        return float("inf")

    with metrics_path.open("r") as f:
        metrics = json.load(f)

    val_mae = metrics.get("val", {}).get("best_mae", None)
    if val_mae is None:
        # Backward compatibility for older runs without val metrics.
        val_mae = metrics.get("test", {}).get("subject_level", {}).get("MAE", float("inf"))

    trial.set_user_attr("status", "ok")
    trial.set_user_attr("gpu", gpu_id)
    trial.set_user_attr("duration_sec", sec)
    trial.set_user_attr("metrics_path", str(metrics_path))
    trial.set_user_attr("objective", "val.best_mae")
    return float(val_mae)


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
        "launch_finetune_job.py",
        "--ckpt", args.ckpt,
        "--output_dir", str(best_dir),
        "--target", args.target,
        "--loss_mode", best.params["loss_mode"],
        "--lr", str(best.params["lr"]),
        "--batch_size", str(best.params["batch_size"]),
        "--epochs", str(args.final_epochs),
        "--mode", args.mode,
    ]
    if bool(best.params["freeze_encoders"]):
        cmd.append("--freeze_encoders")
    if best.params["loss_mode"] == "mse+dist" and "lambda_dist" in best.params:
        cmd += ["--lambda_dist", str(best.params["lambda_dist"])]
    if best.params["loss_mode"] == "wmse" and "wmse_smoothing" in best.params:
        cmd += ["--wmse_smoothing", str(best.params["wmse_smoothing"])]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with (best_dir / "final.log").open("w") as f:
        subprocess.run(cmd, cwd=args.workdir, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)


def run_main(args):
    os.chdir(args.workdir)
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(args.ckpt)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_root = Path(args.workdir) / "results" / f"bayes_{args.target}_{ts}"
    study_root.mkdir(parents=True, exist_ok=True)

    study_name = f"bayes_{args.target}_{ts}"
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
            "bayes_opt_sbp.py",
            "--worker",
            "--target", args.target,
            "--mode", args.mode,
            "--ckpt", args.ckpt,
            "--search_epochs", str(args.search_epochs),
            "--final_epochs", str(args.final_epochs),
            "--batch_choices", args.batch_choices,
            "--workdir", args.workdir,
            "--gpu", str(gpu),
            "--study_name", study_name,
            "--storage", storage,
            "--study_root", str(study_root),
            "--n_trials_local", str(n_trials_local),
        ]
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
    if args.worker:
        run_worker(args)
    else:
        run_main(args)


if __name__ == "__main__":
    main()
