# Model Fitting Tests

These tests check that models actually *learn* — not merely that a forward pass
returns a tensor of the right shape. Each model is fitted on synthetic data with a
known, learnable pattern and then scored on it; a model that cannot drive R² up on
data it was just trained on is broken.

## Test Structure

```
tests/model_fit_tests/
├── test_config.py               # shared settings + synthetic data generators
├── test_temporal_nn.py          # temporal neural networks
├── test_sklearn_models.py       # scikit-learn regressors
├── test_statsmodels.py          # statistical models (ARIMA, VARMAX)
├── test_spatiotemporal.py       # graph-based spatiotemporal models
├── test_foundation_models.py    # opt-in: pretrained Chronos / Moirai (downloads weights)
├── run_all_tests.py             # runner + JSON report
└── outputs/                     # generated plots and test_report.json
```

## Running Tests

```bash
conda activate epilearn

# from the repository root
python tests/model_fit_tests/run_all_tests.py

# subsets
python tests/model_fit_tests/run_all_tests.py temporal   # temporal NNs only
python tests/model_fit_tests/run_all_tests.py sklearn    # sklearn only
python tests/model_fit_tests/run_all_tests.py stats      # statsmodels only
python tests/model_fit_tests/run_all_tests.py spatial    # spatiotemporal only

# individual files also run standalone
python tests/model_fit_tests/test_temporal_nn.py
```

`run_all_tests.py` exits 0 when nothing FAILs (WARNs are tolerated) and writes
`test_report.json` next to the plots.

Plots and the report go to `tests/model_fit_tests/outputs/` by default. Set
`EPILEARN_TEST_OUTPUT_DIR` to send them somewhere else and keep the working tree
clean:

```bash
EPILEARN_TEST_OUTPUT_DIR=/tmp/epilearn_tests python tests/model_fit_tests/run_all_tests.py
```

The foundation-model test is **not** part of `run_all_tests.py`: it downloads
several GB of pretrained weights from HuggingFace and needs the optional extras
(one extra per backend, e.g. `pip install epilearn[chronos]`). Run it explicitly when you want it:

```bash
python tests/model_fit_tests/test_foundation_models.py
```

## What Each Test Does

1. **Creates synthetic data** with a learnable pattern (sine/cosine mixtures; for
   graph models, per-node phases plus a neighbour-influence term).
2. **Trains/fits the model** on the training data.
3. **Predicts** on that same data (this is a fit test, not a generalization test).
4. **Visualizes the fit**: training loss curve (NN models), predictions vs actual,
   scatter with R², residual histogram.
5. **Saves plots** to the output directory.

## Success Criteria

| category | PASS threshold |
|---|---|
| temporal NNs, scikit-learn | R² > 0.5 |
| statsmodels, spatiotemporal | R² > 0.3 |

Below the threshold is reported as WARN (model may need tuning or more data); an
exception is a FAIL and is the only thing that makes the runner exit non-zero.

## Latest Run

23 models, 21 PASS / 2 WARN / 0 FAIL, ~143 s on CPU (the suite uses CUDA
automatically when it is available). The numbers below are indicative: the data
generators are not seeded, so R² moves by ~0.01 between runs.

### Temporal Neural Networks — all passed
| Model | R² | MSE |
|-------|-----|-----|
| GRUModel | 0.898 | 0.0191 |
| LSTMModel | 0.904 | 0.0180 |
| CNNModel | 0.916 | 0.0157 |
| MLPModel | 0.940 | 0.0112 |
| DlinearModel | 0.884 | 0.0218 |
| PatchTSTModel | 0.819 | 0.0341 |

### Scikit-Learn Models — all passed
| Model | R² | MSE |
|-------|-----|-----|
| LinearRegressionModel | 0.938 | 0.0113 |
| RidgeModel | 0.935 | 0.0118 |
| LassoModel | 0.897 | 0.0188 |
| ElasticNetModel | 0.912 | 0.0161 |
| RandomForestModel | 0.987 | 0.0023 |
| GradientBoostingModel | 1.000 | 0.0000 |
| SVRModel | 0.954 | 0.0085 |
| KNNModel | 0.944 | 0.0103 |
| DecisionTreeModel | 1.000 | 0.0000 |

### Statistical Models — WARN, expected
| Model | R² | MSE | Note |
|-------|-----|-----|------|
| ARIMAModel | -0.692 | 0.1730 | designed for longer univariate series |
| VARMAXModel | -0.033 | 0.1056 | designed for longer multivariate series |

### Spatiotemporal Models — all passed
| Model | R² | MSE |
|-------|-----|-----|
| STGCN | 0.914 | 0.0157 |
| GraphWaveNet | 0.919 | 0.0147 |
| ColaGNN | 0.920 | 0.0146 |
| EpiGNN | 0.910 | 0.0163 |
| DCRNN | 0.773 | 0.0413 |
| ATMGNN | 0.360 | 0.1165 |

## Bugs Found By These Tests

1. **DCRNN**: `calculate_scaled_laplacian()` raised an ARPACK error on edge cases.
   Fixed with error handling and a fallback `lambda_max`.
2. **ATMGNN**: output tensor was moved to CPU while the target stayed on CUDA.
   Fixed by removing the forced `x = x.cpu()`.

## Notes

- ARIMA/VARMAX score below zero on short synthetic windows; this is expected and
  is why they are WARN rather than FAIL.
- ATMGNN fits the weakest of the graph models on this data but its loss does come
  down, so it is above the 0.3 spatiotemporal threshold.
