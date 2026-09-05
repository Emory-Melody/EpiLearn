# Tests

```
tests/
├── forecast.py  nowcast.py  scenario.py  detection.py  detection_task.py
│                             # one demo per task, using the high-level task API
├── ensemble.py               # regime labelling + ensembling (numpy only)
├── test.py  test_cola.py  test_epi.py  test_epi_cola.py
│           test_dastgn.py  test_mepo.py  test_stan.py  test_spatial.py
│                             # per-model demos using the low-level model.fit path
├── test_readme_examples.py   # the README's python blocks must still run
└── model_fit_tests/          # every model must be able to learn a known pattern
```

Every script in here runs. There is no known-broken file left in this directory.

Everything here runs with plain `python` — there is no pytest dependency, though
`test_readme_examples.py` also exposes a `test_readme_examples()` function that
pytest will collect if you have it installed.

Every script must be run **from the repository root**, because
`Dataset.load_toy_dataset()` resolves `./datasets` from the working directory.

```bash
conda activate epilearn

# every demo in this folder (~2.5 min) -- the one command that answers
# "does all of this still work?"
python tests/run_all_demos.py

# the homepage examples still work (~20 s)
python tests/test_readme_examples.py

# all models can still fit data (~3 min)
python tests/model_fit_tests/run_all_tests.py

# or all three at once
python tests/run_all_demos.py --readme --fit
```

Each exits 0 on success and non-zero on failure.

## Task demos

Each one is self-contained, prints its own metrics, and exits non-zero on
failure. Approximate CPU runtimes:

| script | what it shows | ~time |
|---|---|---|
| `forecast.py` | `Forecast.rolling_train` over three COVID datasets from `datasets/benchmark.pt` | 27 s |
| `nowcast.py` | `NowcastTask` on a synthetic reporting triangle, vs. the latest-report baseline | 9 s |
| `scenario.py` | `ScenarioTask` counterfactual interventions, scored with PEHE / ATE error | 16 s |
| `detection.py` | shortest working `Detection` pipeline (`train_model` with hand-built splits) | 4 s |
| `detection_task.py` | same task with two graph encoders passed via `pretrained=`, plus bootstrap CIs | 4 s |
| `ensemble.py` | `epilearn.regime` + `epilearn.strategies`: combining five weak forecasters | 3 s |

## Per-model demos

These skip the task layer and call `model.fit` / `model.predict` directly, after
building the sliding windows with `Dataset.generate_dataset` — the manual path
that the tasks wrap. Useful as a template when you add a model.

| script | model | ~time |
|---|---|---|
| `test.py` | `STGCN` | 8 s |
| `test_cola.py` | `ColaGNN` | 15 s |
| `test_epi.py` | `EpiGNN` | 9 s |
| `test_epi_cola.py` | `EpiColaGNN` | 12 s |
| `test_dastgn.py` | `DASTGN` | 25 s |
| `test_mepo.py` | `MepoGNN` (SIR states + dynamic OD graph) | 27 s |
| `test_stan.py` | `STAN` (data-driven + SIR physics head) | 6 s |
| `test_spatial.py` | `GCN`, `SAGE` (node-level classification) | 5 s |

## test_readme_examples.py

Extracts every ` ```python ` block from `README.md` and runs it verbatim in a
fresh subprocess with the repository root as the working directory — i.e. exactly
what a user gets by copy-pasting from the homepage. It also asserts that the
forecast / detection / nowcast examples are still *present* (by API marker), so
deleting an example cannot silently reduce coverage.

The one rewrite it applies is lowering `epochs=N`, because the README's forecast
example is 50 epochs of STGCN (~9 min on CPU).

```bash
python tests/test_readme_examples.py                # all blocks, epochs=2 (~20 s)
python tests/test_readme_examples.py --full         # exactly as published (~10 min)
python tests/test_readme_examples.py --epochs 10    # custom epoch budget
python tests/test_readme_examples.py forecast       # only blocks matching "forecast"
python tests/test_readme_examples.py --list         # show what would run
```

## model_fit_tests/

See [model_fit_tests/README.md](./model_fit_tests/README.md). 23 models across
temporal NNs, scikit-learn regressors, statsmodels and spatiotemporal GNNs; the
runner takes an optional subset argument (`temporal`, `sklearn`, `stats`,
`spatial`) and writes plots plus `test_report.json` to
`tests/model_fit_tests/outputs/` (override with `EPILEARN_TEST_OUTPUT_DIR`).

