# Datasets

We collect epidemic data from various sources including the followings:

### Temporal Data 
   * **[Tycho_v1.0.0](https://www.tycho.pitt.edu/data/)**: Including eight diseases collected across 50 US states and 122 US cities from 1916 to 2009.
   * **[Measles](https://github.com/msylau/measles_competing_risks/tree/master)**: Contains measles infections in England and Wales across 954 urban centers (cities and towns) from 1944 to 1964.

### Spatial&Temporal Data
   * **Covid_static**: Contains covid infections with static graph [[1]](https://github.com/littlecherry-art/DASTGN/tree/master).
   * **Covid_dynamic**: Contains covid infections with dynamic graph [[2]](https://github.com/HySonLab/pandemic_tgnn/tree/main)[[3]](https://github.com/deepkashiwa20/MepoGNN/tree/main).

### Dataset Loading
Loading Measle and Tycho Datasets:
```python
from epilearn.data import UniversalDataset
tycho_dataset = UniversalDataset(name='Tycho_v1', root='./tmp/')
measle_dataset = UniversalDataset(name='Measle', root='./tmp/')

```

For covid data, we support the Dataset from Johns Hopkings University:
```python
from epilearn.data import UniversalDataset
jhu_dataset = UniversalDataset(name='JHU_covid', root='./tmp/')
```

For other countries, please use 'Covid_'+'country' to acquire the correspnding covid dataset. Currently, we support countries like China, Brazil, Austria, England, France, Italy, Newzealand, and Spain.
```python
from epilearn.data import UniversalDataset
covid_dataset = UniversalDataset(name='Covid_Brazil', root='./tmp/')
```
