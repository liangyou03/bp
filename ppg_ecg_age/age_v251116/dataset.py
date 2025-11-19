#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch Dataset 定义
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset

class LabeledECGPPGDataset(Dataset):
    """
    返回 (ecg_tensor, ppg_tensor, age_float, ssoid_str)
    """
    def __init__(self, df: pd.DataFrame, npz_dir: Path):
        self.df = df.reset_index(drop=True)
        self.dir = Path(npz_dir)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ssoid = str(r["ssoid"]); age = float(r["age"])
        with np.load(self.dir / f"{ssoid}.npz") as d:
            x = d["x"].astype(np.float32)   # (7500,2)
        ecg = torch.from_numpy(x[:,0:1].T) # (1,7500)
        ppg = torch.from_numpy(x[:,1:2].T) # (1,7500)
        return ecg, ppg, torch.tensor(age, dtype=torch.float32), ssoid