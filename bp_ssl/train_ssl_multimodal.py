'''
# 新训
python train_ssl_multimodal.py

# 断点续训
python train_ssl_multimodal.py --resume ./checkpoints/ssl_multimodal_subject_aware_clip_v1/checkpoint_epoch_5.pth

'''
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from config import MultiModalSSLConfig
from dataset import PairedECGPPGSSLDataset
from models import ResNet18_1D, SimCLR
from losses import SubjectAwareContrastiveLoss, CrossModalClipLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal SSL: Subject-Aware (ECG/PPG) + Cross-Modal CLIP Alignment")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"✅ Saved checkpoint: {path}")


def main():
    args = parse_args()
    cfg = MultiModalSSLConfig
    set_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")
    Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    print("Initializing Paired Dataset (ECG+PPG)...")
    dataset = PairedECGPPGSSLDataset(cfg.DATA_DIR, config=cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Models (two encoders, two projectors) ----
    print("Initializing Models...")
    ppg_encoder = ResNet18_1D(input_channels=cfg.PPG_CHANNELS, feature_dim=cfg.FEATURE_DIM)
    ecg_encoder = ResNet18_1D(input_channels=cfg.ECG_CHANNELS, feature_dim=cfg.FEATURE_DIM)

    ppg_model = SimCLR(ppg_encoder, feature_dim=cfg.FEATURE_DIM, projection_dim=cfg.PROJECTION_DIM).to(device)
    ecg_model = SimCLR(ecg_encoder, feature_dim=cfg.FEATURE_DIM, projection_dim=cfg.PROJECTION_DIM).to(device)

    # ---- Losses ----
    loss_intra_ppg = SubjectAwareContrastiveLoss(temperature=cfg.TEMPERATURE_INTRA)
    loss_intra_ecg = SubjectAwareContrastiveLoss(temperature=cfg.TEMPERATURE_INTRA)
    loss_xmod = CrossModalClipLoss(temperature=cfg.TEMPERATURE_XMOD)

    # ---- Optimizer / Scheduler ----
    params = list(ppg_model.parameters()) + list(ecg_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    # ---- Resume ----
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        print(f"🔄 Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        ppg_model.load_state_dict(ckpt["ppg_model_state_dict"])
        ecg_model.load_state_dict(ckpt["ecg_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        print(f"🚀 Start from epoch {start_epoch}")

    # ---- Train ----
    print(f"Start Training: {start_epoch} -> {cfg.EPOCHS}")
    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        ppg_model.train()
        ecg_model.train()

        total_loss = 0.0
        total_ppg_acc = 0.0
        total_ecg_acc = 0.0
        total_xmod_acc = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")
        for ppg_v1, ppg_v2, ecg_v1, ecg_v2, subject_ids in pbar:
            ppg_v1 = ppg_v1.to(device, non_blocking=True)
            ppg_v2 = ppg_v2.to(device, non_blocking=True)
            ecg_v1 = ecg_v1.to(device, non_blocking=True)
            ecg_v2 = ecg_v2.to(device, non_blocking=True)
            subject_ids = subject_ids.to(device, non_blocking=True)

            B = subject_ids.shape[0]

            # ---- Intra-modal (Subject-aware) ----
            ppg_in = torch.cat([ppg_v1, ppg_v2], dim=0)   # (2B,4,L)
            ecg_in = torch.cat([ecg_v1, ecg_v2], dim=0)   # (2B,1,L)

            _, ppg_z = ppg_model(ppg_in)  # (2B, proj_dim)
            _, ecg_z = ecg_model(ecg_in)

            ppg_loss, ppg_acc = loss_intra_ppg(ppg_z, subject_ids)
            ecg_loss, ecg_acc = loss_intra_ecg(ecg_z, subject_ids)

            # ---- Cross-modal CLIP alignment (time-window paired) ----
            # 用 view1 对齐（也可以换成 mean(view1, view2)，后面你可做 ablation）
            ppg_z1 = ppg_z[:B]
            ecg_z1 = ecg_z[:B]
            xmod_loss, xmod_acc = loss_xmod(ecg_z1, ppg_z1)

            loss = cfg.W_PPG * ppg_loss + cfg.W_ECG * ecg_loss + cfg.W_XMOD * xmod_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ppg_acc += ppg_acc.item()
            total_ecg_acc += ecg_acc.item()
            total_xmod_acc += xmod_acc.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppg_acc": f"{ppg_acc.item():.2%}",
                "ecg_acc": f"{ecg_acc.item():.2%}",
                "xmod_acc": f"{xmod_acc.item():.2%}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()

        n = len(loader)
        avg_loss = total_loss / n
        avg_ppg_acc = total_ppg_acc / n
        avg_ecg_acc = total_ecg_acc / n
        avg_xmod_acc = total_xmod_acc / n

        print(
            f"Epoch {epoch} | "
            f"Loss {avg_loss:.4f} | "
            f"PPG_intra_acc {avg_ppg_acc:.2%} | "
            f"ECG_intra_acc {avg_ecg_acc:.2%} | "
            f"XMOD_acc {avg_xmod_acc:.2%} | "
            f"LR {optimizer.param_groups[0]['lr']:.6e}"
        )

        # ---- Save checkpoint ----
        if epoch % 5 == 0 or epoch == cfg.EPOCHS:
            ckpt_path = os.path.join(cfg.SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")
            state = {
                "epoch": epoch,
                "ppg_model_state_dict": ppg_model.state_dict(),
                "ecg_model_state_dict": ecg_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "avg_loss": avg_loss,
                "avg_ppg_acc": avg_ppg_acc,
                "avg_ecg_acc": avg_ecg_acc,
                "avg_xmod_acc": avg_xmod_acc,
                "config": {
                    "W_PPG": cfg.W_PPG,
                    "W_ECG": cfg.W_ECG,
                    "W_XMOD": cfg.W_XMOD,
                    "TEMP_INTRA": cfg.TEMPERATURE_INTRA,
                    "TEMP_XMOD": cfg.TEMPERATURE_XMOD,
                }
            }
            save_checkpoint(state, ckpt_path)

    # ---- Save final encoders (for downstream) ----
    ppg_encoder_path = os.path.join(cfg.SAVE_DIR, "ppg_encoder_final.pth")
    ecg_encoder_path = os.path.join(cfg.SAVE_DIR, "ecg_encoder_final.pth")
    torch.save(ppg_model.encoder.state_dict(), ppg_encoder_path)
    torch.save(ecg_model.encoder.state_dict(), ecg_encoder_path)
    print(f"✅ Training complete. Saved encoders:\n  - {ppg_encoder_path}\n  - {ecg_encoder_path}")


if __name__ == "__main__":
    main()
