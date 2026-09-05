#!/usr/bin/env python
# coding: utf-8
"""
Forecast pipeline (EpiLearn 0.1.0).

A forecast experiment needs three things:

1. a ``Dataset``,
2. a ``transforms.Compose`` describing the preprocessing,
3. ``Forecast.rolling_train(...)``, which for every rolling-origin fold trains the
   model, scores it on the held-out test window, and calibrates a conformal
   prediction interval on the validation window.

Run it with::

    python examples/forecast_task.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
# load_toy_dataset() and the built-in dataset names resolve their paths relative
# to the working directory, so these scripts chdir to the repository root.
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.SpatialTemporal.MepoGNN import MepoGNN
from epilearn.models.SpatialTemporal.EpiGNN import EpiGNN
from epilearn.models.SpatialTemporal.DASTGN import DASTGN
from epilearn.models.SpatialTemporal.ColaGNN import ColaGNN
from epilearn.models.SpatialTemporal.EpiColaGNN import EpiColaGNN
from epilearn.models.SpatialTemporal.CNNRNN_Res import CNNRNN_Res
from epilearn.models.SpatialTemporal.ATMGNN import MPNN_LSTM, ATMGNN

from epilearn.models.Temporal.Dlinear import DlinearModel
from epilearn.models.Temporal.LSTM import LSTMModel
from epilearn.models.Temporal.GRU import GRUModel

from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.tasks.forecast import Forecast


# ### Configs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(7)

lookback = 12   # input size
horizon = 3     # prediction size

epochs = 5      # training epochs (kept small so the example runs quickly)
batch_size = 16


# ### Initialize dataset

dataset = Dataset()              # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
# The toy dynamic graph ships as (T, N, N, 1); models that consume it (DASTGN,
# MepoGNN, ...) want (T, N, N), i.e. dataset.dynamic_graph.squeeze(-1). EpiGNN
# only needs the static graph, and dropping the dynamic one keeps every sliding
# window small, so this example runs in seconds.
dataset.dynamic_graph = None
print(dataset)


# ### Add transformations to the dataset
# The stats of normalize_feat / normalize_target are re-fitted on the training
# window of every fold, so do NOT apply them now: just register them.

transformation = transforms.Compose({
    'features': [transforms.normalize_feat()],
    'target': [transforms.normalize_target()],
    'graph': [transforms.normalize_adj()],
    'states': [],
})
dataset.set_transforms(transformation)


# ### Initialize the task
# prototype accepts every model imported at the top of this file.

task = Forecast(prototype=EpiGNN,
                dataset=None,
                lookback=lookback,
                horizon=horizon,
                device=device)


# ### Train + evaluate
# rolling_train replaces the old train_model(dataset=..., train_rate=..., val_rate=...)
# call: sizes are timestep counts, and every fold is scored for you.
# For EpiColaGNN use train_loss='epi_cola' instead of 'mse'.

result = task.rolling_train(dataset=dataset,
                            train_size=350,
                            val_size=60,
                            test_size=60,
                            train_loss='mse',
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=1e-3,
                            model_args={'nhids': 16},
                            max_folds=2,
                            conformal_alpha=0.1,
                            verbose=False)

print("\naggregate metrics:")
for key, value in result['aggregate_metrics'].items():
    print(f"  {key}: {value:.4f}")


# ### Re-score a single fold
# Metrics from rolling_train are in normalized units because normalize_target()
# was used. evaluate_model(inverse_normalize=True) puts them back in case counts.

last_fold = result['fold_results'][-1]
evaluation = task.evaluate_model(dataset=last_fold['test_split'],
                                 process_history=last_fold['process_history'],
                                 inverse_normalize=True)
print("\nlast fold, original units:")
for key in ['mse', 'mae', 'rmse']:
    print(f"  {key}: {float(evaluation[key]):.4f}")

# Plot the predictions of that fold with their conformal interval.
task.plot_preds(evaluation, n_show=40, region_idx=0, horizon_idx=-1)


# ### Try more datasets
# The built-in COVID datasets are named Covid_<Country>; the first channel of the
# feature tensor is the infection count, which we use as the forecast target.

for country in ['Brazil', 'Austria', 'China']:
    print(f"\n########## {country} ##########")
    country_ds = Dataset(name=f'Covid_{country}', root='./datasets/')
    country_ds.y = country_ds.x[:, :, 0]
    country_ds.set_transforms(transformation)

    task = Forecast(prototype=EpiGNN,
                    dataset=None,
                    lookback=lookback,
                    horizon=horizon,
                    device=device)
    country_result = task.rolling_train(dataset=country_ds,
                                        train_size=60,
                                        val_size=20,
                                        test_size=25,
                                        train_loss='mse',
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        lr=1e-3,
                                        model_args={'nhids': 16},
                                        max_folds=1)
    print(f"{country} MAE: {country_result['aggregate_metrics']['mae_mean']:.4f}")


# ### Try temporal models
# A temporal model sees a single region, so slice one region out of the
# spatiotemporal tensor. Keep the target 2-D: (timesteps, 1).

temporal_transformation = transforms.Compose({
    'features': [transforms.normalize_feat()],
    'target': [transforms.normalize_target()],
})

mae_list, rmse_list = [], []
for region in range(3):   # 47 regions in the toy dataset; 3 keeps it quick
    print(f"\n########## region {region} ##########")
    region_ds = Dataset(x=dataset.x[:, region, :],
                        y=dataset.y[:, region:region + 1])
    region_ds.set_transforms(temporal_transformation)

    task = Forecast(prototype=LSTMModel,
                    dataset=None,
                    lookback=lookback,
                    horizon=horizon,
                    device='cpu')
    region_result = task.rolling_train(dataset=region_ds,
                                       train_size=350,
                                       val_size=60,
                                       test_size=60,
                                       train_loss='mse',
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       lr=1e-3,
                                       model_args={'nhid': 32},
                                       max_folds=1)
    mae_list.append(region_result['aggregate_metrics']['mae_mean'])
    rmse_list.append(region_result['aggregate_metrics']['rmse_mean'])

mae = torch.FloatTensor(mae_list)
rmse = torch.FloatTensor(rmse_list)
print(f"\nmae : {mae.mean():.4f} +/- {mae.std():.4f}")
print(f"rmse: {rmse.mean():.4f} +/- {rmse.std():.4f}")
