



1. backbones.py
   目标：让原来的 AgeModel 支持多目标输出，同时保持原来的预训练加载方式不变（CLIP-style，一次性加载 ECG+PPG）。

改动：

* AgeModel 构造函数新增参数 `target_dim`，默认 1。
* head 最后一层从 `Linear(512,1)` 改为 `Linear(512,target_dim)`。
* forward：当 `target_dim==1` 时仍返回 `(B,)`（保持兼容原 age 训练脚本）；当 `target_dim>1` 返回 `(B,K)`。

不改：

* ECGEncoderCLIP / PPGEncoderCLIP 结构不动。
* `load_from_pretrain(ckpt_path)` 逻辑不动（仍从同一个 pretrain ckpt 加载两个 encoder 权重）。

2. dataset.py（你贴出来那份）
   目标：多目标 BP 数据集输出 `(B,K)`。

改动建议（非必须但推荐）：

* 如果你确认所有样本长度固定为 3630：不用改逻辑；建议加一个 assert 防止混入异常长度。

  * `if x.shape[0] != 3630: raise ...`

不改：

* 返回结构 `(ecg, ppg, targets, ssoid)` 不变。
* `targets` 仍是 `torch.tensor([col1,...,colK])` shape `(K,)`。

3. engine.py
   目标：让训练/评估同时支持单目标与多目标，并正确处理 BP 的 `mu/sigma` 向量。

改动：

* 在 `train_one_epoch()` 和 `evaluate_with_ids()` 中：

  * 把 `y` 统一成二维：`(B,) -> (B,1)`。
  * 把模型输出 `raw` 统一成二维：`(B,) -> (B,1)`。
  * 把 `mu/sigma` 转成 device tensor，并 reshape 成 `(1,K)`（标量也变 `(1,1)`），保证 `(y-mu)/sigma` 广播正确。
* `evaluate_with_ids()` 遇到 NaN 不再 `break`，改成直接 `raise`，避免后面 `np.concatenate` 崩溃。
* 增加一个内部 `_compute_metrics_dict()`：按每个 target 输出 mae/rmse/r/r2（这是为了多目标输出统一格式）。

不改：

* 训练流程、AMP、optimizer step、loss_type 分支都保留。
* `apply_constraint` 仍用 scalar `y_min/y_max`（为了最小改动；如果未来要每个 target 一个范围才需要改 utils）。

4. 训练脚本（finetune_age.py -> finetune_bp.py 或 train_bp.py 改造）
   目标：让训练脚本用 BP 多目标、并继续用同一个 pretrain ckpt 加载两个 encoder。

改动（关键几行）：

* labels：从 `age` 列改为 `--target_cols` 多列。
* dataset：从 `LabeledECGPPGDataset` 换成 `BPDataset(df, npz_dir, target_cols)`。
* model：从 `AgeModel(..., target_dim=1)` 改成 `AgeModel(..., target_dim=K)`。
* 统计量：`mu/sigma` 用训练集 target_cols 计算，得到 `(K,)`。
* evaluate：传 `target_names=args.target_cols`，并用 “平均 MAE” 做 early stopping / best ckpt。

不改：

* subject-wise split 的方式不变。
* optimizer/scaler/early stop 框架不变。
* 仍然调用 `model.load_from_pretrain(args.pretrain)`，保持“像之前一样 load 两个预训练”。

补充：你提的“load 两个预训练 model”
在你目前这套“和之前一致”的框架里，真正稳定的方式是：用你们原来的 CLIP 预训练 ckpt（里面同时包含 ecg_enc 和 ppg_enc 的权重），一次 `load_from_pretrain()` 就把两套都 load 进来。
如果你只有“单独的 ECGFounder ckpt + 单独的 PPGFounder ckpt”，那就是另一条路线，需要写一个“合并/映射”加载器（你之前那版就是在干这个），但那会显著增加风险和改动量，不符合你现在“改动不要太多”的要求。

如果你把你当前实际在跑的训练脚本文件名确认一下（是 `train_bp.py` 还是你要基于 `finetune_age.py` 改），我可以把“需要改的具体行”按你文件内容精确列出来（不重写整份脚本，只给 diff 级别改动）。
