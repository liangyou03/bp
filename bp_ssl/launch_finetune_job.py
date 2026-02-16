#!/usr/bin/env python
import argparse
from age_tune_v2 import Config, main


def parse_args():
    p = argparse.ArgumentParser(description="Launch one single-target finetune job")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--loss_mode", default="mse", choices=["mse", "mse+dist", "wmse", "mse*pearson"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--mode", default="fusion", choices=["ppg", "ecg", "fusion"])
    p.add_argument("--freeze_encoders", action="store_true")
    p.add_argument("--lambda_dist", type=float, default=0.3)
    p.add_argument("--wmse_smoothing", type=float, default=2.0)
    return p.parse_args()


def main_entry():
    args = parse_args()
    Config.PRETRAINED_CKPT = args.ckpt
    Config.OUTPUT_DIR = args.output_dir
    Config.TARGET_COL = args.target
    Config.MODE = args.mode
    Config.LOSS_MODE = args.loss_mode
    Config.LR = args.lr
    Config.BATCH_SIZE = args.batch_size
    Config.EPOCHS = args.epochs
    Config.FREEZE_ENCODERS = bool(args.freeze_encoders)

    if args.loss_mode == "mse+dist":
        Config.LAMBDA_DIST = args.lambda_dist
    if args.loss_mode == "wmse":
        Config.WMSE_SMOOTHING = args.wmse_smoothing

    main()


if __name__ == "__main__":
    main_entry()
