import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy import signal
from pathlib import Path
from config import SSLConfig, ECGSSLConfig, MultiModalSSLConfig


class PPGSSLDataset(Dataset):
    def __init__(self, data_dir, config=SSLConfig, transform=None):
        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.glob("*.npz"))
        self.config = config

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir}")

        print("Parsing subject IDs from filenames...")
        self.subject_ids = [f.name.split('_')[0] for f in self.files]
        unique_ids = list(set(self.subject_ids))
        unique_ids.sort()
        self.id_map = {sid: i for i, sid in enumerate(unique_ids)}
        self.labels = [self.id_map[sid] for sid in self.subject_ids]

        print(f"Dataset initialized:")
        print(f"  - Total Files: {len(self.files)}")
        print(f"  - Unique Subjects: {len(unique_ids)}")

    def __len__(self):
        return len(self.files)

    def _resample(self, data):
        if self.config.ORIGINAL_FS == self.config.TARGET_FS:
            return data
        num_samples_target = int(data.shape[0] * self.config.TARGET_FS / self.config.ORIGINAL_FS)
        return signal.resample(data, num_samples_target, axis=0)

    def _augment(self, data):
        seq_len, num_channels = data.shape
        crop_len = self.config.CROP_LEN

        if seq_len > crop_len:
            start = np.random.randint(0, seq_len - crop_len + 1)
            crop = data[start:start + crop_len, :]
        else:
            pad_len = crop_len - seq_len
            crop = np.pad(data, ((0, pad_len), (0, 0)), 'constant')

        if np.random.rand() < 0.3:
            mask_idx = np.random.randint(0, num_channels)
            crop[:, mask_idx] = 0

        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.05, crop.shape)
            crop = crop + noise

        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            crop = crop * scale

        crop = crop.transpose(1, 0)  # (C, L)
        return torch.FloatTensor(crop)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        subject_label = self.labels[idx]
        try:
            loaded = np.load(file_path)
            raw_data = loaded['x']  # (Time, 5)

            ppg_data = raw_data[:, 1:5]
            ppg_resampled = self._resample(ppg_data)

            view1 = self._augment(ppg_resampled)
            view2 = self._augment(ppg_resampled)

            return view1, view2, torch.tensor(subject_label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            dummy = torch.zeros((self.config.INPUT_CHANNELS, self.config.CROP_LEN))
            return dummy, dummy, torch.tensor(0, dtype=torch.long)


class ECGSSLDataset(Dataset):
    def __init__(self, data_dir, config=ECGSSLConfig, transform=None):
        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.glob("*.npz"))
        self.config = config

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir}")

        print("Parsing subject IDs from filenames (ECG)...")
        self.subject_ids = [f.name.split('_')[0] for f in self.files]
        unique_ids = list(set(self.subject_ids))
        unique_ids.sort()
        self.id_map = {sid: i for i, sid in enumerate(unique_ids)}
        self.labels = [self.id_map[sid] for sid in self.subject_ids]
        print(f"ECG Dataset initialized: {len(self.files)} files, {len(unique_ids)} subjects.")

    def __len__(self):
        return len(self.files)

    def _resample(self, data):
        if self.config.ORIGINAL_FS == self.config.TARGET_FS:
            return data
        num_samples_target = int(data.shape[0] * self.config.TARGET_FS / self.config.ORIGINAL_FS)
        return signal.resample(data, num_samples_target, axis=0)

    def _augment(self, data):
        seq_len, num_channels = data.shape
        crop_len = self.config.CROP_LEN

        if seq_len > crop_len:
            start = np.random.randint(0, seq_len - crop_len + 1)
            crop = data[start:start + crop_len, :]
        else:
            pad_len = crop_len - seq_len
            crop = np.pad(data, ((0, pad_len), (0, 0)), 'constant')

        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.05, crop.shape)
            crop = crop + noise

        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            crop = crop * scale

        crop = crop.transpose(1, 0)  # (C, L)
        return torch.FloatTensor(crop)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        subject_label = self.labels[idx]
        try:
            loaded = np.load(file_path)
            raw_data = loaded['x']  # (Time, 5)

            ecg_data = raw_data[:, 0:1]
            ecg_resampled = self._resample(ecg_data)

            view1 = self._augment(ecg_resampled)
            view2 = self._augment(ecg_resampled)

            return view1, view2, torch.tensor(subject_label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            dummy = torch.zeros((self.config.INPUT_CHANNELS, self.config.CROP_LEN))
            return dummy, dummy, torch.tensor(0, dtype=torch.long)


# === 新增：配对多模态数据集（同一条记录、同一 time-window 对齐） ===
class PairedECGPPGSSLDataset(Dataset):
    """
    返回：
      ppg_view1, ppg_view2, ecg_view1, ecg_view2, subject_id
    关键：ECG/PPG 的 crop 起点一致（time-window 对齐）
    """
    def __init__(self, data_dir, config=MultiModalSSLConfig):
        self.data_dir = Path(data_dir)
        self.config = config

        if hasattr(config, "LABEL_CSV") and Path(config.LABEL_CSV).exists():
            df = pd.read_csv(config.LABEL_CSV)
            df["ssoid"] = df["ssoid"].astype(str)
            if (not getattr(config, "USE_ALL_DATA", True)) and ("split" in df.columns):
                df = df[df["split"] == "train"].reset_index(drop=True)

            keep = []
            for _, row in df.iterrows():
                p = self.data_dir / f"{row['ssoid']}.npz"
                if p.exists():
                    keep.append((str(p), str(row.get("subject_uid", row["ssoid"]))))
            if not keep:
                raise RuntimeError(f"No matched npz/labels in {data_dir}")
            self.files = [x[0] for x in keep]
            subject_keys = [x[1] for x in keep]
        else:
            self.files = sorted([str(p) for p in self.data_dir.glob("*.npz")])
            if len(self.files) == 0:
                raise RuntimeError(f"No .npz files found in {data_dir}")
            subject_keys = [Path(p).stem for p in self.files]

        unique_ids = sorted(set(subject_keys))
        self.id_map = {sid: i for i, sid in enumerate(unique_ids)}
        self.labels = [self.id_map[sid] for sid in subject_keys]

        print("Paired Dataset initialized:")
        print(f"  - Total Files: {len(self.files)}")
        print(f"  - Unique Subjects: {len(unique_ids)}")
        print(f"  - ECG crop len: {self.config.ECG_CROP_LEN}")
        print(f"  - PPG crop len: {self.config.PPG_CROP_LEN}")

    def __len__(self):
        return len(self.files)

    def _load_signals(self, file_path):
        with np.load(file_path) as d:
            if getattr(self.config, "USE_ZSCORE", True):
                ecg = d["ecg"].astype(np.float32).reshape(-1, 1)
                ppg = d["ppg"].astype(np.float32).reshape(-1, 1)
            else:
                ecg = d["ecg_raw"].astype(np.float32).reshape(-1, 1)
                ppg = d["ppg_raw"].astype(np.float32).reshape(-1, 1)
        return ecg, ppg

    def _crop_sync(self, ecg, ppg):
        ecg_len = ecg.shape[0]
        ppg_len = ppg.shape[0]
        ecg_crop = self.config.ECG_CROP_LEN
        ppg_crop = self.config.PPG_CROP_LEN

        max_frac = 1.0 - getattr(self.config, "CROP_FRAC", 0.85)
        start_frac = np.random.uniform(0, max(max_frac, 1e-6))
        ecg_start = min(int(start_frac * ecg_len), max(0, ecg_len - ecg_crop))
        ppg_start = min(int(start_frac * ppg_len), max(0, ppg_len - ppg_crop))

        ecg_out = ecg[ecg_start:ecg_start + ecg_crop]
        ppg_out = ppg[ppg_start:ppg_start + ppg_crop]

        if ecg_out.shape[0] < ecg_crop:
            ecg_out = np.pad(ecg_out, ((0, ecg_crop - ecg_out.shape[0]), (0, 0)), "constant")
        if ppg_out.shape[0] < ppg_crop:
            ppg_out = np.pad(ppg_out, ((0, ppg_crop - ppg_out.shape[0]), (0, 0)), "constant")
        return ecg_out, ppg_out

    def _augment(self, data):
        out = data.copy()
        if np.random.rand() < 0.5:
            out = out + np.random.normal(0, 0.05, out.shape).astype(np.float32)
        if np.random.rand() < 0.5:
            out = out * np.random.uniform(0.8, 1.2)
        return torch.FloatTensor(out.transpose(1, 0))  # (1, L)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        subject_label = self.labels[idx]
        try:
            ecg, ppg = self._load_signals(file_path)
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
            print(f"Error loading {file_path}: {e}")
            ppg_dummy = torch.zeros((self.config.PPG_CHANNELS, self.config.PPG_CROP_LEN))
            ecg_dummy = torch.zeros((self.config.ECG_CHANNELS, self.config.ECG_CROP_LEN))
            return ppg_dummy, ppg_dummy, ecg_dummy, ecg_dummy, torch.tensor(0, dtype=torch.long)
