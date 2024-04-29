import torch.nn.init as init
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from dreamy.data import UniversalDataset
from dreamy.utils import utils

import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
import torch
import torch.nn as nn
import torch.nn.functional as F


# initial settings
device = torch.device('cpu')
torch.manual_seed(7)

lookback = 13 # inputs size
horizon = 3 # predicts size

permute = False
target_feat_idx = 0
target_idx = horizon-1

# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()

# preprocessing
features, mean, std = utils.normalize(dataset.x)
adj_norm = utils.normalize_adj(dataset.graph)

features = features.to(device)
adj_norm = adj_norm.to(device)

# prepare datasets
train_rate = 0.6 
val_rate = 0.2

split_line1 = int(features.shape[0] * train_rate)
split_line2 = int(features.shape[0] * (train_rate + val_rate))

train_original_data = features[:split_line1, :, :]
val_original_data = features[split_line1:split_line2, :, :]
test_original_data = features[split_line2:, :, :]

train_input, train_target = dataset.generate_dataset(X = train_original_data, Y = train_original_data[:, :, 0], lookback_window_size = lookback, horizon_size = horizon, permute = permute)
val_input, val_target = dataset.generate_dataset(X = val_original_data, Y = val_original_data[:, :, 0], lookback_window_size = lookback, horizon_size = horizon, permute = permute)
test_input, test_target = dataset.generate_dataset(X = test_original_data, Y = test_original_data[:, :, 0], lookback_window_size = lookback, horizon_size = horizon, permute = permute)

# Selecting the first region for both input and target
train_input = train_input[:, :, 0, :]  # Selecting the first region across all timesteps and features
train_target = train_target[:, :, 0]  # Selecting the first region for the target

val_input = val_input[:, :, 0, :]
val_target = val_target[:, :, 0]

test_input = test_input[:, :, 0, :]
test_target = test_target[:, :, 0]

train_input_np = train_input.numpy()
train_target_np = train_target.numpy()
val_input_np = val_input.numpy()
val_target_np = val_target.numpy()
test_input_np = test_input.numpy()
test_target_np = test_target.numpy()


# 将NumPy数组转换为Pandas DataFrame
def numpy_to_dataframe(data, feature_names):
    num_samples, num_timesteps = data.shape[0], data.shape[1]
    return [pd.DataFrame(data[i, :, :], columns=feature_names) for i in range(num_samples)]

# 创建特征名称
num_features = train_input_np.shape[2]
feature_names = [f'Feature_{i+1}' for i in range(num_features)]

# 转换数据
train_dfs = numpy_to_dataframe(train_input_np, feature_names)
val_dfs = numpy_to_dataframe(val_input_np, feature_names)
test_dfs = numpy_to_dataframe(test_input_np, feature_names)


num_samples = len(train_dfs)  # 样本数量


"""
VARMAX Model:

For each sample, a VARMAX model is trained, which is then used to predict future time points for that specific sample.

Model parameters:
The setting order=(1, 0) indicates that the model is a first-order autoregressive model and does not include a moving average component.
First-order Autoregressive: The model predicts the current value of each variable as a linear function based solely on its own value at the previous time point.
No Moving Average Component: The model relies exclusively on the autoregressive part for predictions and does not utilize the moving average of error terms to adjust the forecasts.

"""

num_output_timesteps = test_target_np.shape[1]  # 输出时间步数

all_forecasts = []  # 存储所有样本的预测结果

subset_samples = 10
# for i in range(num_samples):
for i in range(subset_samples):
    model = VARMAX(train_dfs[i], order=(1, 0), trend='c')
    results = model.fit(disp=False)
    forecast = results.forecast(steps=num_output_timesteps)
    average_forecast = forecast.mean(axis=1).values  # 计算特征平均
    all_forecasts.append(average_forecast)  # 添加到预测结果列表
   

final_predictions = np.array(all_forecasts)
print("Final Predictions Shape:", final_predictions.shape)

train_target_subset = train_target_np[:subset_samples]  
mse_loss = np.mean((train_target_subset - final_predictions) ** 2)
print("MSE Loss:", mse_loss)

