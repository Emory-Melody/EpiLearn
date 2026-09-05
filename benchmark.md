# EpiLearn Benchmark

A config-driven benchmarking framework for epidemic forecasting, nowcasting and
scenario models. Every model in the config is trained with the same rolling-origin
protocol, the same hyperparameter budget and the same conformal calibration, so
the numbers are comparable.

**This file is the short version.** The complete reference — config schema,
hyperparameter-range encoding, output files and the list of supported model names —
lives in one place, the
[Benchmark page](https://epilearn-doc.readthedocs.io/en/latest/Benchmark.html) of the
documentation.

## Quick start

```bash
# as a module
python -m epilearn.benchmark --config configs/benchmark_config.yaml

# as a console script (installed with the package)
epilearn-benchmark --config configs/benchmark_config.yaml

# two-model smoke test; -c/-o are the short forms of --config/--output
epilearn-benchmark -c configs/quick_test_config.yaml -o ./benchmark_results/smoke/
```

`--config`/`-c` is required; `--output`/`-o` overrides `output.save_path` from the
config. Paths inside the config are relative to the working directory, so run the
command from the repository root.

Ready-made configs in `configs/`: `quick_test_config.yaml` (two-model smoke test),
`benchmark_config.yaml` (full forecasting sweep), `nowcast_benchmark_config.yaml`,
`scenario_benchmark_config.yaml` and `epi_models_config.yaml` (EINN / EpiDeep /
CALI-Net).

## Features

- **Three task types** — `forecast`, `nowcast`, `scenario` (set with `task:`)
- **Rolling evaluation** with expanding train window and per-fold metrics
- **Conformal prediction intervals** calibrated on each fold's validation window
- **Per-fold Optuna tuning**, including tuning the lookback window itself
- **Parallel execution** across multiple GPUs (auto-detected), one process per model
- **Crash-tolerant**: results are written as each model finishes, and a model that
  fails or is missing an optional dependency becomes a row instead of aborting the run
- Supports temporal, spatiotemporal, scikit-learn, statsmodels and foundation models

## Results

Each run writes a timestamped set of CSVs under `output.save_path` — aggregate
metrics per model, per-fold detail with the tuned hyperparameters, the Optuna
history — plus an `.npz` of raw predictions and conformal calibration data per
model (exact file list on the docs page). `load_predictions` and
`evaluate_from_saved` in `epilearn.benchmark` re-score a finished run without
retraining. Foundation models need their own extra, e.g. `pip install epilearn[chronos]`; without it
they are reported as `status: skipped_dependency` rather than failing the run.
