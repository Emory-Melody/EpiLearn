"""Forecast task demo: one rolling-window protocol, several models and datasets.

Builds three Dataset objects from datasets/benchmark.pt (Brazil / Austria / China
COVID counts on a region graph) and scores them with Forecast.rolling_train,
which trains, evaluates and calibrates conformal prediction intervals per fold.

Run from the repo root:

    python tests/forecast.py
"""

import torch

from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.SpatialTemporal.EpiGNN import EpiGNN
from epilearn.models.SpatialTemporal.CNNRNN_Res import CNNRNN_Res
from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.tasks.forecast import Forecast

torch.manual_seed(7)

# settings
lookback = 12    # input window length
horizon = 3      # steps predicted
epochs = 15
batch_size = 8
window = 25      # size of the val and test window of every rolling fold

# load the three benchmark countries into Dataset objects
raw_data = torch.load("datasets/benchmark.pt", weights_only=False)
datasets = {}
for name in ['Austria', 'Brazil', 'China']:
    country = raw_data[name]
    feats = country['features'].float()    # (time, region, [infect, recover, death])
    dataset = Dataset(x=feats,
                      y=feats[:, :, 0],            # forecast the infection count
                      graph=country['graph'].float(),
                      feature_names=country['feature_names'])
    # Transforms are fitted on each fold's training window only, then applied to
    # val/test -- so the reported metrics are in normalized units.
    dataset.set_transforms(transforms.Compose({
        "features": [transforms.normalize_feat()],
        "target": [transforms.normalize_target()],
        "graph": [transforms.normalize_adj()]}))
    datasets[name] = dataset
    print(f"{name:8s} {dataset}")


def run(prototype, dataset, model_args={}):
    """Rolling-window train + evaluate. Same call for every model."""
    task = Forecast(prototype=prototype,
                    dataset=None,
                    lookback=lookback,
                    horizon=horizon,
                    device='cpu')
    return task.rolling_train(dataset=dataset,
                              train_size=dataset.n_timesteps - 3 * window,
                              val_size=window,
                              test_size=window,
                              train_loss='mse',
                              epochs=epochs,
                              batch_size=batch_size,
                              model_args=model_args)


rows = []
# same model, three datasets
for name, dataset in datasets.items():
    rows.append(('STGCN', name, run(STGCN, dataset)))
# same dataset, three models
for prototype, model_args in [(EpiGNN, {}), (CNNRNN_Res, {'nhid': 16})]:
    rows.append((prototype.__name__, 'Austria', run(prototype, datasets['Austria'], model_args)))

print(f"\n{'model':12s} {'dataset':8s} folds    RMSE     MAE     90% interval coverage")
for model_name, data_name, result in rows:
    m = result['aggregate_metrics']
    print(f"{model_name:12s} {data_name:8s} {m['n_folds']:5d}   {m['rmse_mean']:.4f}  "
          f"{m['mae_mean']:.4f}   {m['coverage_mean'] * 100:.1f}%")
print("\n(metrics are in normalized units; coverage should sit near the 90% target)")
