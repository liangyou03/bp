import torch

class SSLConfig:
    DATA_DIR = "/home/notebook/data/personal/S9061270/pwv_multichannel/processed_zscore/unlabeled"
    SAVE_DIR = "./checkpoints/ssl_subject_aware_v1"

    ORIGINAL_FS = 250.0
    TARGET_FS = 64.0
    INPUT_CHANNELS = 4

    CROP_LEN_SEC = 30
    CROP_LEN = int(CROP_LEN_SEC * TARGET_FS)

    FEATURE_DIM = 256
    PROJECTION_DIM = 128

    BATCH_SIZE = 2048
    NUM_WORKERS = 8
    EPOCHS = 100
    LR = 3e-4
    TEMPERATURE = 0.1
    WEIGHT_DECAY = 1e-4
    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ECGSSLConfig:
    DATA_DIR = "/home/notebook/data/personal/S9061270/pwv_multichannel/processed_zscore/unlabeled"
    SAVE_DIR = "./checkpoints/ssl_ecg_subject_aware_v1"

    ORIGINAL_FS = 250.0
    TARGET_FS = 64.0
    INPUT_CHANNELS = 1

    CROP_LEN_SEC = 30
    CROP_LEN = int(CROP_LEN_SEC * TARGET_FS)

    FEATURE_DIM = 256
    PROJECTION_DIM = 128

    BATCH_SIZE = 1024
    NUM_WORKERS = 8
    EPOCHS = 100
    LR = 3e-4
    TEMPERATURE = 0.1
    WEIGHT_DECAY = 1e-4
    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === 新增：ECG+PPG 多任务预训练配置 ===
class MultiModalSSLConfig:
    # BP project paths (kailuan BP npz + labels)
    DATA_DIR = "/home/youliang/youliang_data2/bp/bp_recode_v1/output_600hz/npz"
    LABEL_CSV = "/home/youliang/youliang_data2/bp/bp_recode_v1/output_600hz/labels.csv"
    SAVE_DIR = "./checkpoints/ssl_multimodal_bp_600hz_v1"

    # BP npz already has resampled signals:
    # ecg length ~3025 @500Hz, ppg length ~303 @50Hz
    ECG_NATIVE_LEN = 3630
    PPG_NATIVE_LEN = 3630

    PPG_CHANNELS = 1
    ECG_CHANNELS = 1

    # synchronized fraction crop on each modality's own length
    CROP_FRAC = 1.0
    ECG_CROP_LEN = int(ECG_NATIVE_LEN * CROP_FRAC)
    PPG_CROP_LEN = int(PPG_NATIVE_LEN * CROP_FRAC)

    # load zscore keys by default (set False to use *_raw)
    USE_ZSCORE = True
    USE_ALL_DATA = True

    FEATURE_DIM = 256
    PROJECTION_DIM = 128

    BATCH_SIZE = 64
    NUM_WORKERS = 8
    EPOCHS = 100
    LR = 3e-4
    TEMPERATURE_INTRA = 0.1   # 单模态 subject-aware
    TEMPERATURE_XMOD = 0.07   # 跨模态 CLIP 通常 0.07~0.1 都行
    WEIGHT_DECAY = 1e-4
    SEED = 42

    # 三个任务的权重（很关键：先不要让跨模态压过单模态）
    W_PPG = 1.0
    W_ECG = 1.0
    W_XMOD = 0.5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
