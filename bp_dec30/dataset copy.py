#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch Dataset 定义
支持 ECG 重采样 (50 Hz -> 500 Hz)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from scipy.signal import resample_poly


class LabeledECGPPGDataset(Dataset):
    """
    返回 (ecg_tensor, ppg_tensor, label_float, ssoid_str)
    
    Parameters:
        df: DataFrame with 'ssoid' and target column
        npz_dir: directory containing {ssoid}.npz files
        target_col: column name for the label (default: 'age')
        resample_ecg: if True, resample ECG from 50 Hz to 500 Hz (3630 -> 36300)
    """
    def __init__(self, df: pd.DataFrame, npz_dir, target_col: str = "age",
                 resample_ecg: bool = False):
        self.df = df.reset_index(drop=True)
        self.dir = Path(npz_dir)
        self.target_col = target_col
        self.resample_ecg = resample_ecg
        
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ssoid = str(r["ssoid"])
        label = float(r[self.target_col])
        
        with np.load(self.dir / f"{ssoid}.npz") as d:
            x = d["x"].astype(np.float32)  # (3630, 2) @ 50 Hz
        
        ecg_raw = x[:, 0]  # 3630 @ 50 Hz
        ppg_raw = x[:, 1]  # 3630 @ 50 Hz
        
        if self.resample_ecg:
            # ECG: 50 Hz -> 500 Hz (上采样 10x，保持完整 72.6 秒时长)
            ecg = resample_poly(ecg_raw, 10, 1)  # 3630 -> 36300
        else:
            ecg = ecg_raw  # 保持原样 3630
        
        ecg = torch.from_numpy(ecg[None, :].astype(np.float32))
        ppg = torch.from_numpy(ppg_raw[None, :].astype(np.float32))
        
        return ecg, ppg, torch.tensor(label, dtype=torch.float32), ssoid