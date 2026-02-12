import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset


class LabeledECGPPGDataset(Dataset):
    """
    New data format: NPZ with separate ecg and ppg arrays
    Old format: NPZ with x array where x[:,0]=ECG, x[:,1]=PPG
    """
    def __init__(self, df: pd.DataFrame, npz_dir, target_col: str = "age"):
        self.df = df.reset_index(drop=True)
        self.dir = Path(npz_dir)
        self.target_col = target_col
        
    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ssoid = str(r["ssoid"])
        label = float(r[self.target_col])
        
        with np.load(self.dir / f"{ssoid}.npz") as d:
            # Check new format (separate arrays) or old format (combined)
            if "ecg" in d and "ppg" in d:
                ecg = d["ecg"].astype(np.float32)
                ppg = d["ppg"].astype(np.float32)
            elif "x" in d:
                x = d["x"].astype(np.float32)
                ecg = x[:, 0]
                ppg = x[:, 1]
            else:
                raise ValueError(f"Unknown npz format for {ssoid}")
        
        ecg = torch.from_numpy(ecg[None, :])
        ppg = torch.from_numpy(ppg[None, :])
        
        return ecg, ppg, torch.tensor(label, dtype=torch.float32), ssoid
