"""Scenario-modelling demo: simulate a counterfactual intervention with a
compartmental model, then train ScenarioTask to predict intervention effects and
score it with PEHE / ATE error against a "the policy does nothing" baseline.

Run from the repository root:

    python tests/scenario.py

Needs no data at all -- both halves are simulated from epilearn's compartmental
models.
"""

import numpy as np
import torch

from epilearn.models.Temporal import DlinearModel, GRUModel
from epilearn.tasks.scenario_modeling import ScenarioTask
from epilearn.utils.compartmental_models import SIRModel

torch.manual_seed(0)
np.random.seed(0)
# These models are tiny; on a many-core box the default thread pool costs far
# more in synchronisation than it saves.
torch.set_num_threads(4)

# ===========================================================================
# Part 1 -- one counterfactual, by hand.
#
# `parameter_schedule` may be a callable (step_idx, t, state) -> parameter
# overrides, so a time-limited policy is just a function of t. Here a lockdown
# between day 10 and day 90 cuts the transmission rate by 65%.
# ===========================================================================
beta, gamma = 0.30, 0.10                                   # R0 = beta / gamma
model = SIRModel(beta=beta, gamma=gamma)
initial_state = torch.tensor([999_000.0, 1_000.0, 0.0])    # S, I, R


def lockdown(step_idx, t, state):
    """65% contact reduction while the policy is active."""
    return {'beta': beta * (0.35 if 10 <= t < 90 else 1.0)}


runs = {
    'no intervention': model.simulate(initial_state, steps=180, dt=1.0),
    'lockdown d10-d90': model.simulate(initial_state, steps=180, dt=1.0,
                                       parameter_schedule=lockdown),
}

print("=" * 70)
print(f"COUNTERFACTUAL SIMULATION (SIR, R0 = {beta / gamma:.1f}, population 1e6)")
print("=" * 70)
print(f"{'scenario':<18} {'peak I':>10} {'peak day':>9} {'ever infected':>15}")
peaks, peak_days = {}, {}
for name, run in runs.items():
    I = run['trajectory'][:, model.compartments.index('I')]
    peaks[name], peak_days[name] = float(I.max()), int(I.argmax())
    ever_infected = float(run['trajectory'][-1, model.compartments.index('R')])
    print(f"{name:<18} {peaks[name]:>10,.0f} {peak_days[name]:>9} {ever_infected:>15,.0f}")

print(f"\neffect of the lockdown: peak I "
      f"{100 * (peaks['lockdown d10-d90'] / peaks['no intervention'] - 1):+.1f}%, "
      f"peak delayed by "
      f"{peak_days['lockdown d10-d90'] - peak_days['no intervention']} days")
print("(the peak lands after day 90 -- the epidemic rebounds once the policy lifts,")
print(" because suppression leaves a larger susceptible pool behind)")

# ===========================================================================
# Part 2 -- learn intervention effects with ScenarioTask.
#
# Every sample is one epidemic whose history is shared by `n_scenarios`
# futures; the scenarios differ only in the vaccination / isolation policy that
# begins once the history ends. Scenario 0 is the reference, and the quantity of
# interest is the treatment effect tau_s = I_s - I_0 of each other policy.
#
# Compartments are simulated as population fractions (population=1.0), which
# keeps every feature O(1) -- the built-in normalize_feat() is global rather
# than per-feature and would flatten the small policy columns to nothing.
# ===========================================================================
lookback = 15        # days of shared history
horizon = 8          # days of divergent future
n_scenarios = 4

task = ScenarioTask(prototype=DlinearModel, lookback=lookback, horizon=horizon,
                    n_scenarios=n_scenarios, baseline_scenario_idx=0,
                    target_compartment='I', device='cpu')
dataset = task.generate_dataset(n_samples=200, population=1.0, seed=42,
                                dynamic_vacc_rate_range=(0.0, 0.20),
                                dynamic_isol_rate_range=(0.0, 0.50))

N = task.N                            # number of compartments
comp = task.target_compartment_idx    # index of I, the compartment tau is measured on
baseline_idx = task.baseline_scenario_idx
print("\n" + "=" * 70)
print(f"SCENARIO DATASET (simulated with {task.comp_model.name})")
print("=" * 70)
print(f"compartments : {task.comp_model.compartments}")
print(f"features     : {tuple(dataset.x.shape)} = (samples, lookback, scenarios,"
      f" {N} compartments + 4 policy dims)")
print(f"targets      : {tuple(dataset.y.shape)} = (samples, horizon, scenarios,"
      f" compartments)")
print("\nsample 0 -- the policy of each scenario and the future it produces:")
print(f"{'scenario':>8} {'vacc rate':>10} {'isol rate':>10} {'final I':>10}")
for s in range(n_scenarios):
    vacc, isol = dataset.x[0, -1, s, N], dataset.x[0, -1, s, N + 2]
    print(f"{s:>8} {float(vacc):>10.3f} {float(isol):>10.3f} "
          f"{float(dataset.y[0, -1, s, comp]):>10.4f}"
          + ("   <- reference" if s == baseline_idx else ""))


def zero_effect_pehe(fold):
    """PEHE of predicting tau = 0, i.e. "interventions change nothing".

    Computed from the fold's own test targets, so it is directly comparable
    with the model's PEHE on that fold.
    """
    targets = fold['test_split']['targets'].reshape(-1, n_scenarios, horizon, N)
    reference = targets[:, baseline_idx, :, comp]
    tau_true = torch.stack([targets[:, s, :, comp] - reference
                            for s in range(n_scenarios)
                            if s != baseline_idx], dim=1)
    return torch.sqrt((tau_true ** 2).mean()).item()


# PEHE / ATE error are scenario-specific metrics and live only in fold_results,
# so rolling_train has to be told to report them.
scores = {}
for name, prototype, model_args in [('DLinear', DlinearModel, {}),
                                    ('GRU', GRUModel, {'nhids': 64, 'dropout': 0.0})]:
    model_task = ScenarioTask(prototype=prototype, lookback=lookback, horizon=horizon,
                              n_scenarios=n_scenarios, baseline_scenario_idx=baseline_idx,
                              target_compartment='I', device='cpu')
    result = model_task.rolling_train(dataset,
                                      train_size=80,
                                      val_size=30,
                                      test_size=30,
                                      step_size=30,
                                      max_folds=3,
                                      epochs=60,
                                      batch_size=32,
                                      lr=1e-2,
                                      model_args=model_args,
                                      report_metrics=['pehe', 'ate_error'],
                                      verbose=False)
    scores[name] = result

print("\n" + "=" * 70)
print("TREATMENT-EFFECT ACCURACY (population fractions, lower is better)")
print("=" * 70)
baseline_pehe = np.mean([zero_effect_pehe(f)
                         for f in scores['DLinear']['fold_results']])
print(f"{'model':<10} {'PEHE':>10} {'+/-':>8} {'ATE err':>10} {'vs tau=0':>10}")
print(f"{'tau = 0':<10} {baseline_pehe:>10.5f} {'-':>8} {'-':>10} {'-':>10}")
for name, result in scores.items():
    agg = result['aggregate_metrics']
    print(f"{name:<10} {agg['pehe_mean']:>10.5f} {agg['pehe_std']:>8.5f} "
          f"{agg['ate_error_mean']:>10.5f} "
          f"{100 * (1 - agg['pehe_mean'] / baseline_pehe):>9.1f}%")

best = min(scores, key=lambda k: scores[k]['aggregate_metrics']['pehe_mean'])
print(f"\nbest: {best} over {scores[best]['aggregate_metrics']['n_folds']} folds; "
      f"per-fold PEHE "
      f"{[round(f['pehe'], 5) for f in scores[best]['fold_results']]}")
print("'vs tau=0' is how much of the treatment-effect signal the model recovers;")
print("0% means it predicts the same future no matter which policy is applied.")
