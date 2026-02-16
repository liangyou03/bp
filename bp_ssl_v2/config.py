import torch


class MultiModalSSLConfig:
    # ---- Paths ----
    DATA_DIR = "/home/youliang/youliang_data2/bp/bp_recode_v1/output/npz"
    LABEL_CSV = "/home/youliang/youliang_data2/bp/bp_recode_v1/output/labels.csv"
    SAVE_DIR = "./checkpoints/ssl_multimodal_kailuan_v1"

    # ---- Signal parameters (native, no resampling) ----
    ECG_FS = 500.0
    PPG_FS = 50.0
    ECG_NATIVE_LEN = 3025   # samples at 500Hz (~6.05s)
    PPG_NATIVE_LEN = 303    # samples at 50Hz  (~6.06s)

    PPG_CHANNELS = 1
    ECG_CHANNELS = 1

    # ---- Crop (fraction-based, applied proportionally to both modalities) ----
    CROP_FRAC = 0.85
    ECG_CROP_LEN = int(ECG_NATIVE_LEN * CROP_FRAC)  # 2571
    PPG_CROP_LEN = int(PPG_NATIVE_LEN * CROP_FRAC)  # 257

    # ---- Which NPZ keys to use ----
    USE_ZSCORE = True   # True -> 'ecg'/'ppg', False -> 'ecg_raw'/'ppg_raw'

    # ---- Model dimensions ----
    FEATURE_DIM = 256
    PROJECTION_DIM = 128

    # ---- Training hyperparameters ----
    BATCH_SIZE = 256
    NUM_WORKERS = 8
    EPOCHS = 100
    LR = 3e-4
    TEMPERATURE_INTRA = 0.1
    TEMPERATURE_XMOD = 0.07
    WEIGHT_DECAY = 1e-4
    SEED = 42

    # ---- Loss weights ----
    W_PPG = 1.0
    W_ECG = 1.0
    W_XMOD = 0.5

    # ---- SSL data scope ----
    USE_ALL_DATA = True   # True = use all records (no split filtering)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BPFinetuneConfig:
    # ---- Paths ----
    DATA_DIR = "/home/youliang/youliang_data2/bp/bp_recode_v1/output/npz"
    LABEL_CSV = "/home/youliang/youliang_data2/bp/bp_recode_v1/output/labels.csv"
    PRETRAINED_CKPT = "./checkpoints/ssl_multimodal_kailuan_v1/checkpoint_epoch_100.pth"
    OUTPUT_DIR = "./results/bp_finetune_from_ssl_v1"

    # ---- Signal parameters (must match SSL pretraining) ----
    ECG_FS = 500.0
    PPG_FS = 50.0
    ECG_NATIVE_LEN = 3025
    PPG_NATIVE_LEN = 303
    PPG_CHANNELS = 1
    ECG_CHANNELS = 1

    CROP_FRAC = 0.85
    ECG_CROP_LEN = int(ECG_NATIVE_LEN * CROP_FRAC)
    PPG_CROP_LEN = int(PPG_NATIVE_LEN * CROP_FRAC)

    USE_ZSCORE = True

    # ---- Model ----
    FEATURE_DIM = 256
    MODE = "fusion"         # "ppg" | "ecg" | "fusion"

    # ---- BP target ----
    TARGET_COL = "right_arm_sbp"

    # ---- Training ----
    SEED = 42
    BATCH_SIZE = 256
    EPOCHS = 80
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    FREEZE_ENCODERS = False

    # ---- Loss ----
    LOSS_MODE = "mse"       # "mse" | "mse+dist" | "wmse" | "mse*pearson"
    LAMBDA_DIST = 0.2
    DIST_KDE_SIGMA = 2.0
    DIST_ASSIGN_SIGMA = 2.0
    DIST_INV_WEIGHT = True
    DIST_BIN_WIDTH = 1.0
    WMSE_SMOOTHING = 1.0
    WMSE_BIN_WIDTH = 1.0
    PEARSON_ALPHA = 1.0
    Y_MIN = None
    Y_MAX = None

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
