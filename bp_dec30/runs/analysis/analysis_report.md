# BP Prediction Analysis Report

## Overall Performance

| Target | Best Model | Modality | MAE (mmHg) | r | R² |
|--------|------------|----------|------------|---|----|
| Left DBP | exp32_left_arm_dbp_ppg_maepearson | ppg | 7.28 | 0.632 | 0.399 |
| Left MBP | exp26_left_arm_mbp_both_maepearson | both | 8.77 | 0.652 | 0.425 |
| Left PP | exp37_left_arm_pp_ppg_maepearson | ppg | 5.02 | 0.796 | 0.634 |
| Left SBP | exp22_left_arm_sbp_ppg_maepearson | ppg | 9.45 | 0.739 | 0.547 |
| Right DBP | exp12_right_arm_dbp_ppg_maepearson | ppg | 7.23 | 0.631 | 0.398 |
| Right MBP | exp10_right_arm_mbp_both_lowlr | both | 8.86 | 0.651 | 0.424 |
| Right PP | exp16_right_arm_pp_both_maepearson | both | 3.04 | 0.911 | 0.830 |
| Right SBP | exp02_right_arm_sbp_ppg_maepearson | ppg | 8.62 | 0.788 | 0.620 |

## Key Findings

- **Best performing target**: Right PP (MAE=3.04, r=0.911)
- **Most challenging target**: Left SBP (MAE=9.45, r=0.739)
- **Modality preference**: PPG-only wins in 5/8 targets

## Binned MAE Analysis

### Right SBP

| Category | Range | N | MAE | RMSE | r |
|----------|-------|---|-----|------|---|
| Normal | 0-120 | 1009 | 9.70 | 12.60 | 0.380 |
| Elevated | 120-130 | 791 | 6.87 | 8.82 | 0.186 |
| Hypertension Stage 1 | 130-140 | 762 | 6.72 | 8.60 | 0.180 |
| Hypertension Stage 2 | 140-180 | 1148 | 9.38 | 12.07 | 0.534 |
| Hypertensive Crisis | 180-300 | 56 | 23.74 | 28.86 | 0.035 |

### Left SBP

| Category | Range | N | MAE | RMSE | r |
|----------|-------|---|-----|------|---|
| Normal | 0-120 | 1038 | 10.24 | 12.92 | 0.367 |
| Elevated | 120-130 | 808 | 7.31 | 9.41 | 0.150 |
| Hypertension Stage 1 | 130-140 | 764 | 6.83 | 8.76 | 0.151 |
| Hypertension Stage 2 | 140-180 | 1114 | 11.19 | 14.17 | 0.417 |
| Hypertensive Crisis | 180-300 | 42 | 32.67 | 36.18 | 0.117 |

### Right DBP

| Category | Range | N | MAE | RMSE | r |
|----------|-------|---|-----|------|---|
| Normal | 0-80 | 1944 | 7.07 | 9.62 | 0.334 |
| Elevated | 80-85 | 605 | 4.98 | 6.21 | 0.182 |
| Hypertension Stage 1 | 85-90 | 497 | 5.72 | 7.06 | 0.047 |
| Hypertension Stage 2 | 90-120 | 713 | 10.30 | 12.32 | 0.335 |
| Hypertensive Crisis | 120-200 | 7 | 39.93 | 41.75 | 0.293 |

### Left DBP

| Category | Range | N | MAE | RMSE | r |
|----------|-------|---|-----|------|---|
| Normal | 0-80 | 1966 | 6.99 | 9.16 | 0.374 |
| Elevated | 80-85 | 604 | 5.11 | 6.38 | 0.086 |
| Hypertension Stage 1 | 85-90 | 507 | 5.98 | 7.33 | 0.052 |
| Hypertension Stage 2 | 90-120 | 680 | 10.68 | 12.63 | 0.323 |
| Hypertensive Crisis | 120-200 | 9 | 31.60 | 32.82 | -0.227 |


## Clinical Implications

- SBP predictions show strong correlation (r > 0.7), suitable for screening
- DBP predictions are more challenging, typical of BP prediction tasks
- Pulse pressure (PP) predictions are highly accurate, reflecting strong PPG-BP relationship
