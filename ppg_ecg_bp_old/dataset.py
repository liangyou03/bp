#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch Dataset 定义
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset


class LabeledECGPPGDataset(Dataset):
    """
    返回 (ecg_tensor, ppg_tensor, age_float, ssoid_str)
    用于年龄预测
    """
    def __init__(self, df: pd.DataFrame, npz_dir: Path):
        self.df = df.reset_index(drop=True)
        self.dir = Path(npz_dir)
    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ssoid = str(r["ssoid"])
        age = float(r["age"])
        with np.load(self.dir / f"{ssoid}.npz") as d:
            x = d["x"].astype(np.float32)   # (7500,2)
        ecg = torch.from_numpy(x[:,0:1].T)  # (1,7500)
        ppg = torch.from_numpy(x[:,1:2].T)  # (1,7500)
        return ecg, ppg, torch.tensor(age, dtype=torch.float32), ssoid


class BPDataset(Dataset):
    """
    返回 (ecg_tensor, ppg_tensor, targets_tensor, ssoid_str)
    用于多目标 BP 预测
    """
    def __init__(self, df: pd.DataFrame, npz_dir: Path, target_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.dir = Path(npz_dir)
        self.target_cols = target_cols
    
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ssoid = str(r["ssoid"])
        targets = torch.tensor([float(r[c]) for c in self.target_cols], dtype=torch.float32)
        
        with np.load(self.dir / f"{ssoid}.npz") as d:
            x = d["x"].astype(np.float32)   # (7500,2)
        ecg = torch.from_numpy(x[:,0:1].T.copy())  # (1,7500)
        ppg = torch.from_numpy(x[:,1:2].T.copy())  # (1,7500)
        return ecg, ppg, targets, ssoid