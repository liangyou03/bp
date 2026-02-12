# BP预测实验 - 新版

## 目录结构

```
bp_exp_v1/
├── backbones.py         # ECG/PPG encoder模型 (复用)
├── pretrain_clip.py     # CLIP预训练脚本
├── finetune_bp.py       # BP微调脚本
├── run_experiments.sh   # 完整实验流程
├── collect_results.py   # 结果收集
├── runs/                # 实验结果目录
│   ├── clip_pretrain_500hz/    # CLIP预训练 (ECG 500Hz)
│   ├── clip_pretrain_600hz/    # CLIP预训练 (ECG 600Hz)
│   └── bp_500hz_*/              # BP微调结果
└── data_500hz/          # 数据 (ECG 500Hz版本)
```

## 实验流程

### 1. CLIP预训练

```bash
# ECG 500Hz版本
python pretrain_clip.py \
    --data_dir ../bp_recode_v1/data_500hz \
    --out_dir ./runs/clip_pretrain_500hz \
    --epochs 100 \
    --batch_size 128 \
    --gpu 0
```

### 2. BP微调

```bash
# 单个实验
python finetune_bp.py \
    --data_dir ../bp_recode_v1/data_500hz \
    --target_col right_arm_sbp \
    --modality both \
    --clip_ckpt ./runs/clip_pretrain_500hz/clip_best.pth \
    --epochs 50 \
    --gpu 0 \
    --out_dir ./runs/bp_test
```

### 3. 收集结果

```bash
python collect_results.py ./runs
```

## 关键参数

### pretrain_clip.py
| 参数 | 说明 |
|------|------|
| --data_dir | 数据目录 (包含labels.csv和npz/) |
| --ecg_ckpt | ECGFounder预训练权重 (可选) |
| --ppg_ckpt | PPGFounder预训练权重 (可选) |
| --freeze_ecg | 冻结ECG backbone |
| --epochs | 默认100 |

### finetune_bp.py
| 参数 | 说明 |
|------|------|
| --clip_ckpt | CLIP预训练权重 (必需) |
| --target_col | BP目标列 |
| --modality | ecg/ppg/both |
| --freeze_backbone | 冻结encoder |
| --loss | mse/mae/mae_pearson |

## BP目标列

- right_arm_sbp, right_arm_dbp, right_arm_mbp, right_arm_pp
- left_arm_sbp, left_arm_dbp, left_arm_mbp, left_arm_pp
