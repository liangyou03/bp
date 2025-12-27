import torch
import numpy as np
from pathlib import Path

# 加载模型
ckpt_path = "/home/youliang/youliang_data2/bp/bp_run2/bp_both_mae_pearson_best.pth"
state = torch.load(ckpt_path, map_location="cuda:1")

# 计算 MSE
def mse_np(y, yhat):
    return float(np.mean((y - yhat) ** 2))

# 加载预测结果
import pandas as pd
df = pd.read_csv("/home/youliang/youliang_data2/bp/bp_run2/test_predictions.csv")

target_cols = ["right_arm_dbp", "left_arm_mbp", "right_arm_pp", "right_arm_sbp", "left_arm_sbp"]
for col in target_cols:
    y = df[f"{col}_true"].values
    yhat = df[f"{col}_pred"].values
    print(f"{col}: MSE={mse_np(y, yhat):.2f}")