# bp_backbones.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import PPGEncoderCLIP, ECGEncoderCLIP


def _as_ckpt_path(clip_ckpt: str) -> str:
    p = Path(clip_ckpt)
    if p.is_dir():
        for name in ["best.pth", "best.pt", "checkpoint_best.pth", "best_checkpoint.pth"]:
            cand = p / name
            if cand.exists():
                return str(cand)
        raise FileNotFoundError(f"Cannot find best checkpoint under dir: {clip_ckpt}")
    if p.exists():
        return str(p)
    raise FileNotFoundError(f"clip_ckpt not found: {clip_ckpt}")


def _remap_state_keys_for_current_backbones(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap older naming (first_conv/stage_list/block_list/se_fc*) to current naming
    used in backbones.py (first/stages/blocks/se1,se2).
    Also handles CBAM location: ppg_enc.backbone.cbam.* -> ppg_enc.cbam.*
    """
    mapping = {
        # common
        "first_conv": "first",
        "stage_list": "stages",
        "block_list": "blocks",
        "se_fc1": "se1",
        "se_fc2": "se2",
        # some older ECG naming (safe even if not present)
        "first_conv": "first",
        "stage_list": "stages",
        "block_list": "blocks",
        "se_fc1": "se1",
        "se_fc2": "se2",
    }

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k

        # PPG CBAM sometimes lives under backbone in older code
        nk = nk.replace("ppg_enc.backbone.cbam.", "ppg_enc.cbam.")

        for old, new in mapping.items():
            nk = nk.replace(old, new)

        out[nk] = v
    return out


class BPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BPModel(nn.Module):
    """
    Multi-output BP regression on top of CLIP encoders.

    Forward returns yhat_z (standardized target space), shape (B, K).
    Calibration (scale/bias in raw BP units) is stored as buffers and applied in bp_engine.
    """
    def __init__(
        self,
        ppg_enc: Optional[nn.Module] = None,
        ecg_enc: Optional[nn.Module] = None,
        modality: str = "both",
        n_targets: int = 8,
        head_hidden: int = 256,
        normalize_emb: bool = True,
    ):
        super().__init__()
        assert modality in ["ecg", "ppg", "both"]
        self.modality = modality
        self.n_targets = int(n_targets)
        self.normalize_emb = bool(normalize_emb)

        self.ppg_enc = ppg_enc if ppg_enc is not None else PPGEncoderCLIP(with_proj=True)
        self.ecg_enc = ecg_enc if ecg_enc is not None else ECGEncoderCLIP(with_proj=True)

        in_dim = 512 if modality != "both" else 1024
        self.head = BPHead(in_dim=in_dim, hidden=int(head_hidden), out_dim=self.n_targets)

        # post-hoc calibration in raw BP units: y_cal = scale * y + bias (per target)
        self.register_buffer("calib_scale", torch.ones(self.n_targets, dtype=torch.float32))
        self.register_buffer("calib_bias", torch.zeros(self.n_targets, dtype=torch.float32))

    @staticmethod
    def _ensure_b1l(x: torch.Tensor) -> torch.Tensor:
        # Accept (B,L) or (L,) and convert to (B,1,L)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def encode(self, ecg: torch.Tensor, ppg: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_ecg = None
        z_ppg = None

        if self.modality in ["ecg", "both"]:
            ecg = self._ensure_b1l(ecg)
            z_ecg = self.ecg_enc(ecg)  # (B,512)
            if self.normalize_emb:
                z_ecg = F.normalize(z_ecg, dim=1)

        if self.modality in ["ppg", "both"]:
            ppg = self._ensure_b1l(ppg)
            z_ppg = self.ppg_enc(ppg)  # (B,512)
            if self.normalize_emb:
                z_ppg = F.normalize(z_ppg, dim=1)

        return z_ecg, z_ppg

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor) -> torch.Tensor:
        z_ecg, z_ppg = self.encode(ecg, ppg)
        if self.modality == "ecg":
            feat = z_ecg
        elif self.modality == "ppg":
            feat = z_ppg
        else:
            feat = torch.cat([z_ecg, z_ppg], dim=1)
        yhat_z = self.head(feat)  # (B,K) in standardized target space
        return yhat_z

    @torch.no_grad()
    def load_from_clip(self, clip_ckpt: str, device: torch.device) -> None:
        ckpt_path = _as_ckpt_path(clip_ckpt)
        print(f"[load CLIP] {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        if not isinstance(sd, dict):
            raise RuntimeError("Unexpected checkpoint format: state_dict not found.")

        sd = _remap_state_keys_for_current_backbones(sd)

        own = self.state_dict()
        loadable: Dict[str, torch.Tensor] = {}

        shape_mismatch = 0
        for k, v in sd.items():
            if k not in own:
                continue
            if hasattr(v, "shape") and hasattr(own[k], "shape") and tuple(v.shape) != tuple(own[k].shape):
                shape_mismatch += 1
                continue
            loadable[k] = v

        # load
        msg = self.load_state_dict(loadable, strict=False)

        # coverage report (encoder-focused)
        loaded = len(loadable)
        total_params = len([k for k in own.keys() if k.startswith("ecg_enc.") or k.startswith("ppg_enc.") or k.startswith("head.")])
        ecg_total = len([k for k in own.keys() if k.startswith("ecg_enc.")])
        ppg_total = len([k for k in own.keys() if k.startswith("ppg_enc.")])
        head_total = len([k for k in own.keys() if k.startswith("head.")])

        ecg_loaded = len([k for k in loadable.keys() if k.startswith("ecg_enc.")])
        ppg_loaded = len([k for k in loadable.keys() if k.startswith("ppg_enc.")])
        head_loaded = len([k for k in loadable.keys() if k.startswith("head.")])

        print(f"[load CLIP] loaded={loaded} | shape_mismatch={shape_mismatch}")
        print(f"[load CLIP] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        print(f"[load CLIP] ecg_enc loaded {ecg_loaded}/{ecg_total} | ppg_enc loaded {ppg_loaded}/{ppg_total} | head loaded {head_loaded}/{head_total}")

        if ecg_loaded < max(10, int(0.2 * ecg_total)):
            print("[WARN] ECG encoder coverage is low -> check key remap / architecture match.")
        if ppg_loaded < max(10, int(0.2 * ppg_total)):
            print("[WARN] PPG encoder coverage is low -> check key remap / architecture match.")

        self.to(device)

    def set_calibration(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        # scale/bias in raw BP units, shape (K,)
        assert scale.numel() == self.n_targets and bias.numel() == self.n_targets
        self.calib_scale.copy_(scale.detach().to(self.calib_scale.device).float())
        self.calib_bias.copy_(bias.detach().to(self.calib_bias.device).float())
