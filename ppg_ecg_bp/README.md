# BP Multi-target Regression from ECG/PPG

This folder contains everything needed to finetune the pre-trained ECG/PPG encoders on the Kailuan dataset to predict multiple blood pressure targets simultaneously.

## Contents

- `prepare_kailuan_npz.py`: converts `kailuan_dataset.h5` into per-record `.npz` files (`x` with stacked ECG+PPG) and a CSV of BP labels.
- `dataset.py`, `backbones.py`, `engine.py`, `losses.py`, `utils.py`: reusable modules for loading data, defining the model, and running training/evaluation.
- `train_bp.py`: main training script with CLI arguments for data paths, model options, and training hyper-parameters.

## Dependencies

Tested with Python 3.9+. Install the common requirements (PyTorch + data libs):

```bash
pip install torch torchvision torchaudio \
    numpy pandas tqdm h5py
```

If you plan to run on GPU, install the CUDA-enabled PyTorch build that matches your driver.

## Step 1 – Convert Kailuan HDF5 to NPZ + CSV

```bash
python prepare_kailuan_npz.py \
  --h5_path /home/youliang/youliang_data2/bp/kailuan_dataset.h5 \
  --id_key new_id \
  --target_cols right_arm_dbp left_arm_mbp right_arm_pp right_arm_sbp left_arm_sbp \
  --output_npz /path/to/data/npz \
  --output_csv /path/to/data/labels.csv \
  --target_seq_len 7500
```

What it does:

1. Reads all metadata + signals from the HDF5 file.
2. Uses `id_key` to name each record (becomes `<ssoid>.npz`).
3. Resamples both ECG and PPG to `target_seq_len` samples (default 7 500) so they can be stacked into the `(seq, 2)` array expected by the model.
4. Writes one compressed `.npz` per record (`np.savez_compressed(..., x=stacked_signal)`).
5. Builds `labels.csv` with `ssoid` plus the requested BP targets.

Inspect the console output to ensure the number of exported rows matches your expectations.

## Step 2 – Train the BP Regressor

```bash
python train_bp.py \
  --npz_dir /path/to/data/npz \
  --labels_csv /path/to/data/labels.csv \
  --pretrain /path/to/1_lead_ECGFounder.pth \
  --out_dir /path/to/output/bp_run1 \
  --target_cols right_arm_dbp left_arm_mbp right_arm_pp right_arm_sbp left_arm_sbp \
  --modality both \
  --epochs 40 \
  --batch_size 256 \
  --lr_backbone 1e-5 \
  --lr_head 3e-4
```

Important flags:

- `--npz_dir`: directory containing the `.npz` files produced in Step 1.
- `--labels_csv`: the CSV with `ssoid` + BP columns.
- `--pretrain`: path to the pre-trained encoders (same file you used for age finetuning, e.g., `1_lead_ECGFounder.pth`).
- `--target_cols`: BP columns to predict; the order defines the output order.
- `--modality`: `ecg`, `ppg`, or `both`.
- `--freeze_backbone`: add this flag if you want a linear probe (only the regression head trains).
- `--constrain`: optional (`clip`, `sigmoid`, `tanh`) bounded output activation; default `none`.
- `--y_margin`: padding (in BP units) added to the train-set min/max before applying constraints.

The script performs a subject-wise split (train/val/test), trains with AMP if CUDA is available, tracks per-target metrics, saves the best checkpoint (`bp_best.pth`), and exports:

- `val_predictions.csv` / `test_predictions.csv`: record-level ground-truth vs predictions for every target.
- `final_metrics.json`: per-target MAE/RMSE/Pearson/R² for validation and test sets.

## Step 3 – Review Outputs

Inside `out_dir` you will find:

- `bp_best.pth`: weights + training stats + target metadata.
- `args.json`: (inside checkpoint) CLI arguments for reproducibility.
- `val_predictions.csv` / `test_predictions.csv`: each row has `ssoid`, `<target>_true`, `<target>_pred`.
- `final_metrics.json`: summary of validation/test losses and metrics.

Use these files to plot learning curves, analyze errors, or resume experiments.

## Tips & Troubleshooting

- Signals must be resampled to the same length before stacking; use `prepare_kailuan_npz.py` or follow the same logic in your own preprocessing pipeline.
- Ensure the IDs used in the CSV exactly match the `.npz` filenames (`<ssoid>.npz`). The training script filters labels to whatever exists in `npz_dir`.
- `train_bp.py` defaults to GPU if available; otherwise it runs on CPU (slower but functional).
- Adjust `--val_ratio` / `--test_ratio` for different subject splits; they are applied before shuffling with the provided `--seed`.
- To add/remove targets later, regenerate the CSV with the desired columns and pass the matching `--target_cols` list.
