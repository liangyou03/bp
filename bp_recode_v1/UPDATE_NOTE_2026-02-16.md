# Update Note (2026-02-16)

## Completed
- Finished Bayesian optimization queue for `bp_recode_v1` on 9 settings:
  - targets: `sbp`, `dbp`, `map`
  - modalities: `ppg`, `ecg`, `both`
  - each setting: 24 trials, 4-GPU worker mode
- Fixed `bayes_opt_recode.py` runtime blockers:
  - missing dependency (`optuna`) in runtime env
  - target name mapping (`sbp/dbp/map` -> `right_arm_sbp/right_arm_dbp/right_arm_mbp`)
- Ensured runs were launched in `bp` conda env.

## Final Bayes Results (bp_recode_v1, val_raw MAE best)
- `sbp`: `ppg 7.7583`, `ecg 13.7423`, `both 8.1413`
- `dbp`: `ppg 7.0829`, `ecg 9.2481`, `both 7.1782`
- `map`: `ppg 7.4492`, `ecg 11.6837`, `both 7.6055`

## SSL vs CLIP-chain Comparison (right arm targets)
- Compared `bp_ssl` best-final calibrated metrics vs `bp_recode_v1` Bayes best-per-target.
- Result: SSL pipeline outperformed on all three right-arm targets in this batch.
  - SBP: SSL better
  - DBP: SSL better
  - MBP/MAP: SSL better

## Notes
- In this run set, `ppg` was consistently stronger than `ecg`; `both` was close to `ppg` but usually not better.
- Queue log indicates full completion timestamp: `2026-02-16 21:26:37`.
