#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


BP_COLS_DEFAULT = [
    "right_arm_sbp", "right_arm_mbp", "right_arm_dbp", "right_arm_pp",
    "left_arm_sbp",  "left_arm_mbp",  "left_arm_dbp",  "left_arm_pp",
]


class BPLabeledDataset(Dataset):
    """
    Returns:
      ecg: (1, L_ecg)
      ppg: (1, L_ppg)
      y:   (K,)  float32
      ssoid: str
    """
    def __init__(
        self,
        df: pd.DataFrame,
        npz_dir: str,
        targets: List[str],
        use_raw: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.npz_dir = Path(npz_dir)
        self.targets = list(targets)
        self.use_raw = bool(use_raw)

        for c in ["ssoid"] + self.targets:
            if c not in self.df.columns:
                raise ValueError(f"labels_df missing column: {c}")

        self.df["ssoid"] = self.df["ssoid"].astype(str)
        for c in self.targets:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        bad = self.df[self.targets].isna().any(axis=1)
        if bad.any():
            self.df = self.df[~bad].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        r = self.df.iloc[idx]
        ssoid = str(r["ssoid"])

        p = self.npz_dir / f"{ssoid}.npz"
        with np.load(p) as d:
            if self.use_raw:
                ecg = d["ecg_raw"].astype(np.float32)
                ppg = d["ppg_raw"].astype(np.float32)
            else:
                ecg = d["ecg"].astype(np.float32)
                ppg = d["ppg"].astype(np.float32)

        # to torch (1, L)
        ecg_t = torch.from_numpy(ecg[None, :])
        ppg_t = torch.from_numpy(ppg[None, :])

        y = np.array([float(r[c]) for c in self.targets], dtype=np.float32)
        y_t = torch.from_numpy(y)

        return ecg_t, ppg_t, y_t, ssoid
