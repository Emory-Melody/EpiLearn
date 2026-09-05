# Datasets

We collect epidemic data from various sources including the followings:

### Temporal Data 
   * **[Tycho_v1.0.0](https://www.tycho.pitt.edu/data/)**: Including eight diseases collected across 50 US states and 122 US cities from 1916 to 2009.
   * **[Measles](https://github.com/msylau/measles_competing_risks/tree/master)**: Contains measles infections in England and Wales across 954 urban centers (cities and towns) from 1944 to 1964.

### Spatial&Temporal Data
   * **Covid_static**: Contains covid infections with static graph [[1]](https://github.com/littlecherry-art/DASTGN/tree/master).
   * **Covid_dynamic**: Contains covid infections with dynamic graph [[2]](https://github.com/HySonLab/pandemic_tgnn/tree/main)[[3]](https://github.com/deepkashiwa20/MepoGNN/tree/main).

### Dataset Loading
As of 0.1.0 the loader class is `Dataset` (`UniversalDataset` still works as a
deprecated alias). The `name` / `root` arguments are unchanged: a named dataset is
downloaded into `root` on first use and reused afterwards. Several copies are
already bundled in this folder, so `root='./datasets'` loads `Measles`,
`JHU_covid` and `Covid_<country>` without touching the network.

Loading Measles and Tycho Datasets:
```python
import torch
from epilearn.data import Dataset

measle_dataset = Dataset(name='Measles', root='./tmp/')

# Tycho_v1 holds one variable-length series per disease, so it cannot be stacked into a
# single tensor: Dataset(name='Tycho_v1', ...) downloads ./tmp/Tycho_v1.pt and *then*
# raises ValueError. Read the archive and pick a disease instead.
try:
    Dataset(name='Tycho_v1', root='./tmp/')
except ValueError:
    pass
raw = torch.load('./tmp/Tycho_v1.pt', weights_only=False)
# ['DIPHTHERIA', 'HEPATITIS A', 'MEASLES', 'MUMPS', 'SMALLPOX']
tycho_measles = Dataset(x=raw['MEASLES'].unsqueeze(-1), y=raw['MEASLES'])
```

For covid data, we support the Dataset from Johns Hopkings University:
```python
from epilearn.data import Dataset
jhu_dataset = Dataset(name='JHU_covid', root='./tmp/')
```

For other countries, please use 'Covid_'+'country' to acquire the correspnding covid dataset. Currently, we support China, Brazil, Austria, England, France, Italy, NewZealand and Spain. The names are case-sensitive — note the capital `Z` in `Covid_NewZealand`.
```python
from epilearn.data import Dataset
covid_dataset = Dataset(name='Covid_Brazil', root='./tmp/')
```

A small toy dataset (47 regions, 539 timesteps, with both a static and a dynamic
graph) ships with the repository and is what the README examples use:
```python
from epilearn.data import Dataset
dataset = Dataset()
dataset.load_toy_dataset()   # reads ./datasets/features.npy and ./datasets/graphs.npy
print(dataset.x.shape, dataset.y.shape, dataset.graph.shape)
```

After loading, the data lives on `dataset.x`, `.y`, `.states`, `.graph` and
`.dynamic_graph`; column and time labels are on `dataset.feature_names`,
`.target_names`, `.timestamps` and `.regions`. Loaders that carry extra
information expose it as a `dataset.metadata` dict — e.g. `Measles` provides
`anual_population`, `anual_birth` and `coordinates` there (in 0.0.x these were
top-level attributes).

To load your own long-format CSV instead:
```python
from epilearn.data import Dataset
dataset = Dataset.from_csv(file_path='./datasets/toy_features.csv',
                           timestamp_col='time',
                           region_col='node',
                           feature_cols=['f0', 'f1', 'f2', 'f3'],
                           target_cols=['y'],
                           graph_file='./datasets/toy_edges.csv')
print(dataset.x.shape, dataset.y.shape, dataset.feature_names)
```
`toy_features.csv` / `toy_edges.csv` in this folder are a runnable example of the
expected layout: one row per (timestamp, region) with the features as columns, and
a `source,target` edge list for the graph.

## Nowcasting reporting triangle (not redistributed)

The nowcasting benchmark configs read a COVID-19 *reporting triangle* — how each
day's count was revised as later reports arrived. That dataset comes from the
**CMU Delphi Epidata API** (`hospital-admissions / smoothed_adj_covid19_from_claims`,
COVID-19 hospital admissions estimated from Change Healthcare insurance claims),
and EpiLearn does not redistribute it. Regenerate it from the original source:

```bash
python datasets/build_nowcast_triangle.py -o epilearn/data/nowcast_ready_data.npz
```

That writes the four arrays `NowcastTask.load_triangle()` expects — `triangle`
`(n_days, n_delays)`, `final_counts`, `delays`, `time_values`. Run it with a short
range first (`--start 20220101 --end 20220301 -o /tmp/t.npz`) to check
connectivity. See the script's docstring for the API endpoint, the optional API
key, and Delphi's licensing and citation terms:
<https://cmu-delphi.github.io/delphi-epidata/api/covidcast.html>

Nowcasting needs no external data to try out — `tests/nowcast.py` and the README's
nowcasting example build a synthetic reporting triangle in a few lines.
