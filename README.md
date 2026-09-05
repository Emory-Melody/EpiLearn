
<p align="center">
<img center src="https://raw.githubusercontent.com/Emory-Melody/EpiLearn/main/asset/logo/logo_2_new.png" width = "600" alt="EpiLearn">
</p>


## <p align="center">Epidemic Modeling with Python</p>
<!-- [![Documentation Status](https://readthedocs.org/projects/epilearn-doc/badge/?version=latest)](https://epilearn-doc.readthedocs.io/en/latest/) -->
[![Documentation Status](https://readthedocs.org/projects/epilearn-doc/badge/?version=latest)](https://epilearn-doc.readthedocs.io/en/latest/)
[![License MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Emory-Melody/EpiLearn/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/epilearn)](https://pepy.tech/project/epilearn)
[![Feedback](https://img.shields.io/badge/Feedback-8A2BE2)](https://join.slack.com/t/epilearn/shared_invite/zt-2uq9tdbe8-thaXoYN~8UIWjwDqKm8vgg)



**[Documentation](https://epilearn-doc.readthedocs.io/en/latest/) | [Paper](https://arxiv.org/abs/2406.06016)**

**EpiLearn** is a Python machine learning toolkit for epidemic data modeling and analysis. We provide numerous features including:

- Implementation of Epidemic Models — 65 models spanning mechanistic, statistical, deep temporal, spatiotemporal graph, and time-series foundation models
- Unified Pipeline for Epidemic Tasks — forecasting, nowcasting, scenario modeling and source detection behind one interface
- Rolling-Window Evaluation with Uncertainty — walk-forward folds and conformal prediction intervals for every task
- Config-Driven Benchmark — compare any set of models from a single YAML file
- Simulation of Epidemic Spreading
- Visualization of Epidemic Data
  
For more machine models in epidemic modeling, feel free to check out our curated paper list [Awesome-Epidemic-Modeling-Papers](https://github.com/Emory-Melody/awesome-epidemic-modeling-papers).


Announcement
==============
**EpiLearn 0.1.0 is here.** The PyPI upload is pending, so for now install it from
source (see [Installation](#installation)); `pip install epilearn` still gives 0.0.19.
Highlights:

- 65 models, up from 26 (foundation models, modern deep time-series, scikit-learn and statistical baselines)
- Two new tasks: **nowcasting** (reporting-delay correction) and **scenario modeling**
- Rolling-window evaluation with conformal prediction intervals, and per-fold Optuna tuning
- A config-driven [benchmark](https://github.com/Emory-Melody/EpiLearn/blob/main/benchmark.md): `python -m epilearn.benchmark --config x.yaml`

Upgrading from 0.0.x? Two APIs changed — `UniversalDataset` is now `Dataset` (the old name still
works), and `train_model` now takes explicit splits. See [MIGRATION.md](https://github.com/Emory-Melody/EpiLearn/blob/main/MIGRATION.md) and the
full [CHANGELOG.md](https://github.com/Emory-Melody/EpiLearn/blob/main/CHANGELOG.md).

If you have any suggestions, please feel free to click the feedback button on top and join our slack channel!

Encounter Any Issues?
====
If you experience any issues, please don’t hesitate to open a **[GitHub Issue](https://github.com/Emory-Melody/EpiLearn/issues)**. We will do our best to address it within **three business days**. You are also warmly invited to join our **[User Slack Channel](https://join.slack.com/t/epilearn/shared_invite/zt-2uq9tdbe8-thaXoYN~8UIWjwDqKm8vgg)** for more efficient communication. Alternatively, reaching out to us via email is also perfectly fine!


Installation
==============
## From Source
```bash
git clone https://github.com/Emory-Melody/EpiLearn.git
cd EpiLearn

conda create -n epilearn python=3.10
conda activate epilearn

pip install .
```
This installs everything EpiLearn needs, including `torch` and `torch_geometric`.
A source checkout is also what gives you `configs/`, `datasets/`, `examples/` and
`tests/` — the wheel contains only the `epilearn` package.

## From Pypi
```bash
pip install epilearn
```
> **Note:** PyPI currently serves **0.0.19**. The snippets below use the 0.1.0 API, so
> until the 0.1.0 upload lands please install from source, or straight from git:
> `pip install git+https://github.com/Emory-Melody/EpiLearn.git`

For a CUDA build of PyTorch, install it first following [pytorch.org](https://pytorch.org/),
then `pip install .` — pip will keep the build you already have.
<!-- EpiLearn also requires pytorch>=1.20, torch_geometric and torch_scatter. For cpu version, we simply use *pip install torch*, *pip install torch_geometric* and *pip install torch_scatter*. For the GPU version, please refer to [Pytorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and [torch_scatter](https://pytorch-geometric.com/whl/torch-1.5.0.html). -->



Tutorial
==============
A full tutorial lives in our [documentation](https://epilearn-doc.readthedocs.io/en/latest/): [quickstart](https://epilearn-doc.readthedocs.io/en/latest/Quickstart.html), [pipelines](https://epilearn-doc.readthedocs.io/en/latest/tutorials/task_building.html), [simulations](https://epilearn-doc.readthedocs.io/en/latest/tutorials/simulation.html), [utilities](https://epilearn-doc.readthedocs.io/en/latest/tutorials/utils.html) and the [benchmark](https://epilearn-doc.readthedocs.io/en/latest/Benchmark.html). For the overall framework of EpiLearn, please check our [paper](https://arxiv.org/abs/2406.06016).

Runnable code lives in two places: the [examples](https://github.com/Emory-Melody/EpiLearn/tree/main/examples)
folder (notebooks and scripts, one per topic) and the
[tests](https://github.com/Emory-Melody/EpiLearn/tree/main/tests) folder (short per-model and
per-task demo scripts — `python tests/forecast.py`, `tests/nowcast.py`, `tests/scenario.py`, …).
Run every demo at once with `python tests/run_all_demos.py`.

Below we also offer a quick start on how to use EpiLearn for forecast, detection and nowcasting tasks.

## Forecast Pipeline
```python
from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.tasks.forecast import Forecast
# initialize settings
lookback = 12 # inputs size
horizon = 3 # predicts size
# load toy dataset
dataset = Dataset()
dataset.load_toy_dataset()
# Adding Transformations
transformation = transforms.Compose({
                "features": [transforms.normalize_feat()],
                "target": [transforms.normalize_target()],
                "graph": [transforms.normalize_adj()]})
dataset.transforms = transformation
# Initialize Task
task = Forecast(prototype=STGCN,
                dataset=None, 
                lookback=lookback, 
                horizon=horizon, 
                device='cpu')
# Training: rolling-window evaluation, with conformal intervals per fold
result = task.rolling_train(dataset=dataset,
                            train_size=400,
                            val_size=50,
                            test_size=50,
                            train_loss='mse',
                            epochs=10,
                            batch_size=5)
# Evaluation: rolling_train already scored every fold
print(result['aggregate_metrics'])
```
This trains STGCN over two rolling folds and takes well under a minute on CPU.

## Detection Pipeline
```python
from epilearn.models.Spatial.GCN import GCN
from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.tasks.detection import Detection
# initialize settings
lookback = 1 # inputs size
horizon = 2 # number of classes
# load toy dataset
dataset = Dataset()
dataset.load_toy_dataset()
dataset.y = (dataset.y > dataset.y.median(dim=1, keepdim=True).values).long() # per-node class labels
# Adding Transformations
transformation = transforms.Compose({
                "features": [transforms.normalize_feat()],
                "graph": [transforms.normalize_adj()]})
dataset.set_transforms(transformation, apply_now=True)
# Build sliding-window splits (pass adj so the graph reaches the model)
def make_split(start, end):
    return dataset.generate_dataset(X=dataset.x[start:end], Y=dataset.y[start:end],
                                    adj=dataset.graph,
                                    lookback_window_size=lookback, horizon_size=1)
train_split, val_split, test_split = make_split(0, 300), make_split(300, 400), make_split(400, 539)
# Initialize Task
task = Detection(prototype=GCN, 
                 dataset=dataset, 
                 lookback=lookback, 
                 horizon=horizon, 
                 device='cpu')
# Training
result = task.train_model(train_split=train_split,
                          val_split=val_split,
                          test_split=test_split,
                          train_loss='ce',
                          val_loss='ce',
                          epochs=50, 
                          batch_size=5)
# Evaluation
evaluation = task.evaluate_model(dataset=test_split)
```

## Nowcasting Pipeline
Nowcasting corrects for reporting delay: recent counts are still incomplete, and the
task learns how much each day will be revised upward.
```python
import numpy as np
from epilearn.models.Temporal import GRUModel
from epilearn.tasks.nowcast import NowcastTask
# initialize settings
lookback = 14 # inputs size
horizon = 7 # nowcast the last 7 days, whose reports are still incomplete
# build a reporting triangle: day t's cases trickle in over delays 1..9
rng = np.random.default_rng(0)
delays = np.arange(1, 10)
final = np.round(100 + 60 * np.sin(np.arange(260) / 18) + rng.normal(0, 4, 260))
share = np.diff(1 - np.exp(-np.r_[0, delays] / 2.5)); share /= share.sum()
triangle = np.cumsum([rng.multinomial(int(c), share) for c in final], axis=1)
# Initialize Task
task = NowcastTask(prototype=GRUModel,
                   lookback=lookback,
                   horizon=horizon,
                   min_delay=1,
                   max_delay=9,
                   device='cpu')
dataset = task.create_dataset(triangle, final, delays=delays)
# Training
result = task.rolling_train(dataset,
                            train_size=140,
                            val_size=40,
                            test_size=40,
                            epochs=60,
                            batch_size=32,
                            lr=1e-2)
# Evaluation: compare against simply trusting the latest report
print("nowcast MAE      :", result['aggregate_metrics']['mae_mean'])
print("latest-report MAE:", task.compute_naive_baseline(dataset)['naive_mae'])
```

## Benchmark
To compare many models under one rolling-window protocol, describe the run in a YAML
file instead of writing a script:
```bash
python -m epilearn.benchmark --config configs/quick_test_config.yaml
```
Per-model metrics, conformal coverage, Optuna trials and raw predictions are written to
`benchmark_results/<task>/models_<timestamp>/`. See [benchmark.md](https://github.com/Emory-Melody/EpiLearn/blob/main/benchmark.md) for the
config schema and the full model list.

Citing
==============
If you find this work useful, please cite: [EpiLearn: A Python Library for Machine Learning in Epidemic Modeling](https://arxiv.org/abs/2406.06016)

    @article{liu2024epilearn,
    title={EpiLearn: A Python Library for Machine Learning in Epidemic Modeling},
    author={Liu, Zewen and Li, Yunxiao and Wei, Mingyang and Wan, Guancheng and Lau, Max SY and Jin, Wei},
    journal={arXiv e-prints},
    pages={arXiv--2406},
    year={2024}
    }

Acknowledgement
==============
Some algorithms are adopted from the papers' implmentation and the original links can be easily found on top of each file. We also appreciate the datasets from various sources, which will be highlighted in the [dataset](https://github.com/Emory-Melody/EpiLearn/tree/main/datasets) file.

Thanks to their great work and contributions!
