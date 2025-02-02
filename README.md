
<p align="center">
<img center src="./asset/logo/logo_2_new.png" width = "600" alt="EpiLearn">
</p>


## <p align="center">Epidemic Modeling with Python</p>
<!-- [![Documentation Status](https://readthedocs.org/projects/exe/badge/?version=latest)](https://epilearn-doc.readthedocs.io/en/latest/) -->
[![Documentation Status](https://readthedocs.org/projects/exe/badge/?version=latest)](https://epilearn-doc.readthedocs.io/en/latest/)
[![License MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Emory-Melody/EpiLearn/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/epilearn)](https://pepy.tech/project/epilearn)
[![Web Interface](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://epilearn.streamlit.app/)
<a target="_blank" href="https://colab.research.google.com/drive/13D5U-S-U2DhR9OKXdsGuGE2gR1fO2Y4T">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
[![Feedback](https://img.shields.io/badge/Feedback-8A2BE2)](https://join.slack.com/t/epilearn/shared_invite/zt-2uq9tdbe8-thaXoYN~8UIWjwDqKm8vgg)



<!-- **[Documentation](https://epilearn-doc.readthedocs.io/en/latest/) | [Paper](https://arxiv.org/abs/2406.06016) | [Web Interface](https://epilearn.streamlit.app/)**
 -->
**[Documentation](https://epilearn-doc.readthedocs.io/en/latest/) | [Paper](https://arxiv.org/abs/2406.06016) | [Web Interface](https://epilearn.streamlit.app/)**

**[Colab Tutorial](https://colab.research.google.com/drive/13D5U-S-U2DhR9OKXdsGuGE2gR1fO2Y4T#scrollTo=ISt5vlDKlJ1R)**

**EpiLearn** is a Python machine learning toolkit for epidemic data modeling and analysis. We provide numerous features including:

- Implementation of Epidemic Models
- Simulation of Epidemic Spreading
- Visualization of Epidemic Data
- Unified Pipeline for Epidemic Tasks
  
For more machine models in epidemic modeling, feel free to check out our curated paper list [Awesome-Epidemic-Modeling-Papers](https://github.com/Emory-Melody/awesome-epidemic-modeling-papers).


Announcement
==============
To install the latest version, please use "pip install epilearn==0.0.15" --- 11/21/2024

EpiLearn is currently updating. We will release a new version very soon! --- 11/13/2024

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

conda create -n epilearn python=3.9
conda activate epilearn

python setup.py install
```
## From Pypi 
```bash
pip install epilearn
```

#### Installing Dependencies (CPU)
```bash
pip install torch==2.5
pip install torch_geometric
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

EpiLearn also requires pytorch>=1.20, torch_geometric and torch_scatter. For cpu version, we simply use *pip install torch*, *pip install torch_geometric* and *pip install torch_scatter*. For the GPU version, please refer to [Pytorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and [torch_scatter](https://pytorch-geometric.com/whl/torch-1.5.0.html).



Tutorial
==============
We provide a quick tutorial of EpiLearn in [Google Colab](https://colab.research.google.com/drive/13D5U-S-U2DhR9OKXdsGuGE2gR1fO2Y4T#scrollTo=qv6sHNCee68F). A more completed tutorial can be found in our [documentation](https://epilearn-doc.readthedocs.io/en/latest/), including [pipelines](https://epilearn-doc.readthedocs.io/en/latest/tutorials/task_building.html), [simulations](https://epilearn-doc.readthedocs.io/en/latest/tutorials/simulation.html) and other [utilities](https://epilearn-doc.readthedocs.io/en/latest/tutorials/utils.html). For more examples, please refer to the [example](https://github.com/Emory-Melody/EpiLearn/tree/main/examples) folder. For the overall framework of EpiLearn, please check our [paper](https://arxiv.org/abs/2406.06016).

Below we also offer a quick start on how to use EpiLearn for forecast and detection tasks.

## Forecast Pipeline
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

## Detection Pipeline
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

## Web Interface

Our [web application](https://epilearn.streamlit.app/) is deployed online using [streamlit](https://streamlit.io/). But it also can be initiated using:
```bash
python -m streamlit run interface/app.py to activate the interface
```

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
