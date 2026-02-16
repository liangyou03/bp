import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class KailuanPairedSSLDataset(Dataset):
    """
    Paired ECG+PPG dataset for SSL pretraining on Kailuan data.

    Returns:
      ppg_view1, ppg_view2, ecg_view1, ecg_view2, subject_label

    ECG and PPG have different sample rates (500Hz vs 50Hz) and lengths
    (3025 vs 303). We use fraction-based synchronized cropping to maintain
    temporal alignment between modalities.
    """

    def __init__(self, config):
        self.cfg = config
        self.npz_dir = Path(config.DATA_DIR)

        # Load labels.csv for subject_uid mapping and file list
        df = pd.read_csv(config.LABEL_CSV)
        df["ssoid"] = df["ssoid"].astype(str)

        if not config.USE_ALL_DATA:
            df = df[df["split"] == "train"].reset_index(drop=True)

        # Filter to existing NPZ files
        existing = []
        for _, row in df.iterrows():
            p = self.npz_dir / f"{row['ssoid']}.npz"
            if p.exists():
                existing.append(row)
        self.df = pd.DataFrame(existing).reset_index(drop=True)

        # Build subject_uid -> int mapping
        unique_uids = sorted(self.df["subject_uid"].unique())
        self.uid_map = {uid: i for i, uid in enumerate(unique_uids)}
        self.labels = [self.uid_map[uid] for uid in self.df["subject_uid"]]

        print(f"KailuanPairedSSLDataset initialized:")
        print(f"  Records: {len(self.df)}")
        print(f"  Unique subjects: {len(unique_uids)}")
        print(f"  ECG crop: {config.ECG_CROP_LEN} ({config.ECG_NATIVE_LEN} native)")
        print(f"  PPG crop: {config.PPG_CROP_LEN} ({config.PPG_NATIVE_LEN} native)")

    def __len__(self):
        return len(self.df)

    def _load_signals(self, ssoid: str):
        p = self.npz_dir / f"{ssoid}.npz"
        with np.load(p) as d:
            if self.cfg.USE_ZSCORE:
                ecg = d["ecg"].astype(np.float32)
                ppg = d["ppg"].astype(np.float32)
            else:
                ecg = d["ecg_raw"].astype(np.float32)
                ppg = d["ppg_raw"].astype(np.float32)

        # Reshape to (L, 1)
        ecg = ecg.reshape(-1, 1)
        ppg = ppg.reshape(-1, 1)
        return ecg, ppg

    def _crop_sync(self, ecg, ppg):
        """
        Fraction-based synchronized crop.
        Pick a random start fraction and apply proportionally to both modalities.
        """
        ecg_len = ecg.shape[0]
        ppg_len = ppg.shape[0]
        ecg_crop = self.cfg.ECG_CROP_LEN
        ppg_crop = self.cfg.PPG_CROP_LEN

        max_frac = 1.0 - self.cfg.CROP_FRAC
        start_frac = np.random.uniform(0, max(max_frac, 1e-6))

        ecg_start = min(int(start_frac * ecg_len), max(0, ecg_len - ecg_crop))
        ppg_start = min(int(start_frac * ppg_len), max(0, ppg_len - ppg_crop))

        ecg_out = ecg[ecg_start:ecg_start + ecg_crop]
        ppg_out = ppg[ppg_start:ppg_start + ppg_crop]

        # Pad if needed (defensive)
        if ecg_out.shape[0] < ecg_crop:
            pad = ecg_crop - ecg_out.shape[0]
            ecg_out = np.pad(ecg_out, ((0, pad), (0, 0)), mode="constant")
        if ppg_out.shape[0] < ppg_crop:
            pad = ppg_crop - ppg_out.shape[0]
            ppg_out = np.pad(ppg_out, ((0, pad), (0, 0)), mode="constant")

        return ecg_out, ppg_out

    def _augment(self, data):
        """
        Augmentation for both ECG and PPG (1 channel each):
        - Gaussian noise (p=0.5, std=0.05)
        - Random scale (p=0.5, [0.8, 1.2])
        """
        out = data.copy()

        if np.random.rand() < 0.5:
            out = out + np.random.normal(0, 0.05, out.shape).astype(np.float32)

        if np.random.rand() < 0.5:
            out = out * np.random.uniform(0.8, 1.2)

        # (L, C) -> (C, L)
        out = out.transpose(1, 0)
        return torch.from_numpy(out.copy()).float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ssoid = str(row["ssoid"])
        subject_label = self.labels[idx]

        try:
            ecg, ppg = self._load_signals(ssoid)
            ecg_crop, ppg_crop = self._crop_sync(ecg, ppg)

            ppg_view1 = self._augment(ppg_crop)
            ppg_view2 = self._augment(ppg_crop)
            ecg_view1 = self._augment(ecg_crop)
            ecg_view2 = self._augment(ecg_crop)

            return (
                ppg_view1, ppg_view2,
                ecg_view1, ecg_view2,
                torch.tensor(subject_label, dtype=torch.long),
            )

        except Exception as e:
            print(f"Error loading {ssoid}: {e}")
            ppg_dummy = torch.zeros((self.cfg.PPG_CHANNELS, self.cfg.PPG_CROP_LEN))
            ecg_dummy = torch.zeros((self.cfg.ECG_CHANNELS, self.cfg.ECG_CROP_LEN))
            return (
                ppg_dummy, ppg_dummy,
                ecg_dummy, ecg_dummy,
                torch.tensor(0, dtype=torch.long),
            )


class KailuanBPDataset(Dataset):
    """
    Labeled ECG+PPG dataset for BP regression finetuning.

    Returns:
      ppg: (1, PPG_CROP_LEN)
      ecg: (1, ECG_CROP_LEN)
      target: scalar BP value
      ssoid: str
    """

    def __init__(self, df: pd.DataFrame, config, target_col: str, is_train: bool = True):
        self.cfg = config
        self.npz_dir = Path(config.DATA_DIR)
        self.target_col = target_col
        self.is_train = is_train

        self.df = df.copy().reset_index(drop=True)
        self.df["ssoid"] = self.df["ssoid"].astype(str)
        self.df[target_col] = pd.to_numeric(self.df[target_col], errors="coerce")

        # Drop rows with NaN target
        bad = self.df[target_col].isna()
        if bad.any():
            print(f"  Dropping {bad.sum()} rows with NaN {target_col}")
            self.df = self.df[~bad].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_signals(self, ssoid: str):
        p = self.npz_dir / f"{ssoid}.npz"
        with np.load(p) as d:
            if self.cfg.USE_ZSCORE:
                ecg = d["ecg"].astype(np.float32)
                ppg = d["ppg"].astype(np.float32)
            else:
                ecg = d["ecg_raw"].astype(np.float32)
                ppg = d["ppg_raw"].astype(np.float32)

        ecg = ecg.reshape(-1, 1)
        ppg = ppg.reshape(-1, 1)
        return ecg, ppg

    def _crop(self, ecg, ppg):
        """
        Random crop (train) or center crop (val/test).
        Fraction-based to handle different sample rates.
        """
        ecg_len = ecg.shape[0]
        ppg_len = ppg.shape[0]
        ecg_crop = self.cfg.ECG_CROP_LEN
        ppg_crop = self.cfg.PPG_CROP_LEN

        max_frac = 1.0 - self.cfg.CROP_FRAC

        if self.is_train:
            start_frac = np.random.uniform(0, max(max_frac, 1e-6))
        else:
            start_frac = max_frac / 2.0  # center crop

        ecg_start = min(int(start_frac * ecg_len), max(0, ecg_len - ecg_crop))
        ppg_start = min(int(start_frac * ppg_len), max(0, ppg_len - ppg_crop))

        ecg_out = ecg[ecg_start:ecg_start + ecg_crop]
        ppg_out = ppg[ppg_start:ppg_start + ppg_crop]

        # Pad if needed
        if ecg_out.shape[0] < ecg_crop:
            pad = ecg_crop - ecg_out.shape[0]
            ecg_out = np.pad(ecg_out, ((0, pad), (0, 0)), mode="constant")
        if ppg_out.shape[0] < ppg_crop:
            pad = ppg_crop - ppg_out.shape[0]
            ppg_out = np.pad(ppg_out, ((0, pad), (0, 0)), mode="constant")

        return ecg_out, ppg_out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ssoid = str(row["ssoid"])
        target = np.float32(row[self.target_col])

        try:
            ecg, ppg = self._load_signals(ssoid)
            ecg_crop, ppg_crop = self._crop(ecg, ppg)

            # (L, C) -> (C, L)
            ecg_t = torch.from_numpy(ecg_crop.transpose(1, 0).copy()).float()
            ppg_t = torch.from_numpy(ppg_crop.transpose(1, 0).copy()).float()

            return ppg_t, ecg_t, torch.tensor(target, dtype=torch.float32), ssoid

        except Exception as e:
            print(f"Error loading {ssoid}: {e}")
            return (
                torch.zeros((self.cfg.PPG_CHANNELS, self.cfg.PPG_CROP_LEN)),
                torch.zeros((self.cfg.ECG_CHANNELS, self.cfg.ECG_CROP_LEN)),
                torch.tensor(target, dtype=torch.float32),
                ssoid,
            )
