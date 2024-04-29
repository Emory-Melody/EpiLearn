import torch.nn.init as init
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from dreamy.data import UniversalDataset
from dreamy.utils import utils

import pandas as pd
import torch

import xgboost as xgb
from sklearn.metrics import mean_squared_error

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



X_train = train_input_np.reshape(train_input_np.shape[0], -1)  # 形状变为 (308, 52)
X_val = val_input_np.reshape(val_input_np.shape[0], -1)        
X_test = test_input_np.reshape(test_input_np.shape[0], -1)     


y_train = train_target_np
y_val = val_target_np
y_test = test_target_np



model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=40, 
    learning_rate=0.1, 
    max_depth=5,
    reg_lambda=1.0,  # L2正则化系数
    reg_alpha=0.1    # L1正则化系数
)
# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# 计算MSE
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Training MSE: ", train_mse)
print("Validation MSE: ", val_mse)
print("Test MSE: ", test_mse)

