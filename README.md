
[![TorchDrug](asset/torchdrug_logo_full.svg)](https://torchdrug.ai/)
<h1 align="center">
</h1>


[![Contributions](https://img.shields.io/badge/contributions-welcome-blue)](https://github.com/DeepGraphLearning/torchdrug/blob/master/CONTRIBUTING.md)
[![License MIT](https://img.shields.io/github/license/DeepGraphLearning/torchdrug?color=blue)](https://github.com/DeepGraphLearning/torchdrug/blob/master/LICENSE)
[![PyPI downloads](https://static.pepy.tech/personalized-badge/torchdrug?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pypi.org/project/torchdrug/)
[![TorchDrug Twitter](https://img.shields.io/twitter/url?label=TorchDrug&style=social&url=https%3A%2F%2Ftwitter.com%2FDrugTorch)](https://twitter.com/DrugTorch)


**EpiLearn** is a Pytorch-based machine learning tool-kit for epidemic aata modeling and analysis. We provide numerour features including:

- Implementation of Epidemic Models
- Simulation of Epidemic Spreading
- Visualization of Epidemic Data
- Unified Pipeline for Epidemic Tasks


Installation
------------
### From Source ###
```bash
git clone https://github.com/Emory-Melody/EpiLearn.git
cd EpiLearn

conda create -n epilearn python=3.9
conda activate epilearn

python setup.py install
pip install pytorch_geometric
```
### From Pypi ###
```bash
pip install epilearn
```

Tutorial
------------
We provide brief tutorial of EpiLearn in .... Please also see our documentation at ...

Our web application can be initiated using:
```bash
python -m streamlit run interface/app.py to activate the interface
```

Citing
------------
      


<!-- ### Updates
* Updated the interface. Now we can visualize the graph data and also the simulations using a web app based on streamlit and pyvis. 05/18/2024


            python setup.py install
            pip install pytorch_geometric
            
            Use python -m streamlit run interface/app.py to activate the interface

  
* Merged code and updated transformation module. 05/17/2024
* Updated SIR simulation on graphs(See examples/data_simulation.ipynb for more details). 05/06/2024
* Use examples/forecast_task.ipynb to try epidemic models as well as other baselines (STAN is currently not available). 05/06/2024 -->