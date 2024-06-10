
<p align="center">
<img center src="./asset/logo/logo_1_new.png" width = "600" alt="EpiLearn">
</p>


## <p align="center">Epidemic Modeling with Pytorch</p>

[![Documentation Status](https://readthedocs.org/projects/exe/badge/?version=latest)](https://epilearn-doc.readthedocs.io/en/latest/)
[![License MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Emory-Melody/EpiLearn/blob/main/LICENSE)
<!-- [![PyPI downloads](https://static.pepy.tech/personalized-badge/torchdrug?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pypi.org/project/torchdrug/) -->
**[Documentation](https://epilearn-doc.readthedocs.io/en/latest/)** 

**EpiLearn** is a Pytorch-based machine learning tool-kit for epidemic data modeling and analysis. We provide numerour features including:

- Implementation of Epidemic Models
- Simulation of Epidemic Spreading
- Visualization of Epidemic Data
- Unified Pipeline for Epidemic Tasks


Installation
==============
## From Source ##
```bash
git clone https://github.com/Emory-Melody/EpiLearn.git
cd EpiLearn

conda create -n epilearn python=3.9
conda activate epilearn

python setup.py install
pip install pytorch_geometric
```
## From Pypi ##
```bash
pip install epilearn
```

Tutorial
==============
We provide a complete tutorial of EpiLearn in our [documentation](https://epilearn-doc.readthedocs.io/en/latest/) and the overal framework in our paper. For more examples, please refer to the *examples* folder.

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
...

