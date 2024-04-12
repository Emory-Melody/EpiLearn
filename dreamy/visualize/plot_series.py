import torch
import numpy as np
import seaborn as sb

def plot_series(x : np.array, columns : list):
    data = dict.fromkeys(columns)
    for i ,s in enumerate(columns):
        data[s] = x[:, i]
    sb.relplot(data, kind = "line", legend = columns)