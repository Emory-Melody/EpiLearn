
<p align="center">
<img center src="./asset/logo/logo_2_new.png" width = "600" alt="EpiLearn">
</p>


## <p align="center">Epidemic Modeling with Pytorch</p>

[![Documentation Status](https://readthedocs.org/projects/exe/badge/?version=latest)](https://epilearn-doc.readthedocs.io/en/latest/)
[![License MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Emory-Melody/EpiLearn/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/epilearn)](https://pepy.tech/project/epilearn)
<!-- [![PyPI downloads](https://img.shields.io/pypi/dm/epilearn)](https://pypi.org/project/epilearn/) -->

**[Documentation](https://epilearn-doc.readthedocs.io/en/latest/) | [Paper](https://arxiv.org/abs/2406.06016)** 

**EpiLearn** is a Pytorch-based machine learning tool-kit for epidemic data modeling and analysis. We provide numerour features including:

- Implementation of Epidemic Models
- Simulation of Epidemic Spreading
- Visualization of Epidemic Data
- Unified Pipeline for Epidemic Tasks
  
For more models, please refer to our [Awesome-Epidemic-Modeling-Papers](https://github.com/Emory-Melody/awesome-epidemic-modeling-papers)


Installation
==============
## From Source ##
```bash
git clone https://github.com/Emory-Melody/EpiLearn.git
cd EpiLearn

conda create -n epilearn python=3.9
conda activate epilearn

python setup.py install
```
## From Pypi ##
```bash
pip install epilearn
```

EpiLearn also   requires pytorch>=1.20, torch_geometric and torch_scatter. For cpu version, we simply use *pip install torch*, *pip install torch_geometric* and *pip install torch_scatter*. For GPU version, please refer to [Pytorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and [torch_scatter](https://pytorch-geometric.com/whl/torch-1.5.0.html).

Tutorial
==============
We provide a complete tutorial of EpiLearn in our [documentation](https://epilearn-doc.readthedocs.io/en/latest/), including [pipelines](https://epilearn-doc.readthedocs.io/en/latest/tutorials/task_building.html), [simulations](https://epilearn-doc.readthedocs.io/en/latest/tutorials/simulation.html) and other [utilities](https://epilearn-doc.readthedocs.io/en/latest/tutorials/utils.html). For more examples, please refer to the [example](https://github.com/Emory-Melody/EpiLearn/tree/main/examples) folder. For the overal framework of EpiLearn, please check our [paper](https://arxiv.org/abs/2406.06016).

Here we also offer a quickstart of how to use the EpiLearn for forecast and detection task.

## Forecast Pipeline ##
```python
from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.data import UniversalDataset
from epilearn.utils import transforms
from epilearn.tasks.forecast import Forecast
# initialize settings
lookback = 12 # inputs size
horizon = 3 # predicts size
# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()
# Adding Transformations
transformation = transforms.Compose({
                "features": [transforms.normalize_feat()],
                "graph": [transforms.normalize_adj()]})
dataset.transforms = transformation
# Initialize Task
task = Forecast(prototype=STGCN,
                dataset=None, 
                lookback=lookback, 
                horizon=horizon, 
                device='cpu')
# Training
result = task.train_model(dataset=dataset, 
                          loss='mse', 
                          epochs=50, 
                          batch_size=5, 
                          permute_dataset=True)
# Evaluation
evaluation = task.evaluate_model()
```

## Detection Pipeline ##
```python
from epilearn.models.Spatial.GCN import GCN
from epilearn.data import UniversalDataset
from epilearn.utils import transforms
from epilearn.tasks.detection import Detection
# initialize settings
lookback = 1 # inputs size
horizon = 2 # predicts size; also seen as number of classes
# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()
# Adding Transformations
transformation = transforms.Compose({
                " features": [],
                " graph": []})
dataset.transforms = transformation
# Initialize Task
task = Detection(prototype=GCN, 
                 dataset=None, 
                 lookback=lookback, 
                 horizon=horizon, 
                 device='cpu')
# Training
result = task.train_model(dataset=dataset, 
                          loss='ce', 
                          epochs=50, 
                          batch_size=5)
# Evaluation
evaluation = task.evaluate_model()
```

## Web Interface ##

Our web application can be initiated using:
```bash
python -m streamlit run interface/app.py to activate the interface
```

Citing
==============
If you find this work useful, please cite: [EpiLearn: A Python Library for Machine Learning in Epidemic Modeling](https://arxiv.org/abs/2406.06016)

    @article{liu2024epilearn,
    title={EpiLearn: A Python Library for Machine Learning in Epidemic Modeling},
    author={Liu, Zewen and Li, Yunxiao and Wei, Mingyang and Wan, Guancheng and Lau, Max S.Y. and Jin, Wei},
    journal={arXiv preprint arXiv:2406.06016},
    year={2024}
    }

Acknowledgement
==============
Some algorithms are adopted from the papers' implmentation and the original links can be easily found on top of each file. We also appreciate the datasets from various sources, which will be highlighted in the [dataset](https://github.com/Emory-Melody/EpiLearn/tree/main/datasets) file.

Thanks to their great work and contributions!