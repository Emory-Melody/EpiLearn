#!/usr/bin/env python
# coding: utf-8
"""
Mechanistic (compartmental) models (EpiLearn 0.1.0).

0.1.0 ships three different flavours of compartmental model, and it is easy to
grab the wrong one:

1. ``epilearn.models.Temporal.Compartmental`` — the original torch modules
   (SIR / SIS / SEIR and their NetworkX counterparts). They are *simulators*:
   call them with a state vector and they roll the dynamics forward. The 0.0.x
   import path ``epilearn.models.Temporal.SIR`` still works as an alias.
2. ``epilearn.utils.compartmental_models`` — new: SIRModel / SEIRModel /
   SIRSModel / SEIRVIModel with a ``simulate()`` method, RK4 integration and
   time-varying parameters via ``parameter_schedule``.
3. ``epilearn.models.Temporal.CompartmentalModel`` — new: SIRModel / SISModel /
   SEIRModel wrapped as *forecasters*, so they can be passed as ``prototype=``
   to a task and compared against the neural models. The simulators in (1) take
   no ``**kwargs`` and cannot be used that way.

Run it with::

    python examples/mechanistic.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch
import matplotlib.pyplot as plt

# 1. simulators (0.0.x path epilearn.models.Temporal.SIR is aliased to this)
from epilearn.models.Temporal.Compartmental import SIR, SIS, SEIR
from epilearn.models.SpatialTemporal.NetworkSIR import NetSIR

# 2. new ODE models with time-varying parameters
from epilearn.utils.compartmental_models import SIRModel, SEIRModel, SIRSModel
from epilearn.utils.simulation import simulate_temporal_epidemic

# 3. new compartmental *forecasters*
from epilearn.models.Temporal.CompartmentalModel import SIRModel as SIRForecaster

from epilearn.data import Dataset
from epilearn.tasks.forecast import Forecast


torch.manual_seed(7)


# ## 1. The classic simulators
# The state vector is in absolute counts: SIR is [S, I, R].

state = torch.tensor([3416.0, 210.0, 65.0])
sir = SIR(horizon=100, infection_rate=0.1, recovery_rate=0.0384)
trajectory = sir(state, steps=None)     # steps=None uses horizon
print("SIR  :", tuple(trajectory.shape), "(steps, compartments)")

sis = SIS(horizon=100, infection_rate=0.3, recovery_rate=0.1)
print("SIS  :", tuple(sis(torch.tensor([3400.0, 200.0]), steps=None).shape))

seir = SEIR(horizon=100, infection_rate=0.3, recovery_rate=0.1,
            cure_rate=0.05, latency=0.2)
print("SEIR :", tuple(seir(torch.tensor([3400.0, 100.0, 150.0, 40.0]), steps=None).shape))

# epilearn.visualize.plot_series(trajectory.numpy(), columns=[...]) draws this,
# but it writes plot_series.png into the working directory, so plot it directly:
plt.figure(figsize=(10, 4))
for i, name in enumerate(['susceptible', 'infected', 'recovered']):
    plt.plot(trajectory[:, i].numpy(), label=name)
plt.xlabel("step")
plt.ylabel("individuals")
plt.title("SIR simulation")
plt.legend()
plt.tight_layout()
plt.show()


# ## 2. ODE models with an intervention schedule
# These take fractions of the population and integrate with RK4.

model = SIRModel(beta=0.3, gamma=0.1)
baseline = model.simulate(initial_state=[0.99, 0.01, 0.0], steps=150)
infected = baseline['trajectory'][:, 1]
print(f"\nno intervention: peak {infected.max():.4f} of the population "
      f"on day {int(infected.argmax())}")

# parameter_schedule can be a dict {step: overrides} or a callable
# (step_idx, t, state) -> overrides. Here: a lockdown between day 20 and 90.
lockdown = lambda step, t, state: {'beta': 0.09} if 20 <= step < 90 else None
mitigated = model.simulate(initial_state=[0.99, 0.01, 0.0], steps=150,
                           parameter_schedule=lockdown)
infected_mitigated = mitigated['trajectory'][:, 1]
print(f"with lockdown  : peak {infected_mitigated.max():.4f} of the population "
      f"on day {int(infected_mitigated.argmax())}")

plt.figure(figsize=(10, 4))
plt.plot(infected.numpy(), label='no intervention')
plt.plot(infected_mitigated.numpy(), label='beta 0.3 -> 0.09 on days 20-90')
plt.xlabel("day")
plt.ylabel("infected fraction")
plt.title("Time-varying parameters via parameter_schedule")
plt.legend()
plt.tight_layout()
plt.show()

# simulate_temporal_epidemic adds process noise and seeding on top of simulate()
noisy = simulate_temporal_epidemic(SEIRModel(beta=0.4, gamma=0.1, sigma=0.2),
                                   initial_state=[0.98, 0.01, 0.01, 0.0],
                                   steps=150,
                                   process_noise=1e-4,
                                   seed=0)
print("SEIR + noise   :", noisy['compartments'], tuple(noisy['trajectory'].shape))

waning = SIRSModel(beta=0.3, gamma=0.1, omega=0.02)
print("SIRS           :", waning.simulate([0.99, 0.01, 0.0], steps=150)['compartments'])


# ## 3. A compartmental model used as a forecaster
# Unlike the simulators above, these accept the standard
# (num_features, num_timesteps_input, num_timesteps_output) signature, so they
# drop straight into a task. fit() is a no-op: the ODE parameters are re-fitted
# on every lookback window inside predict().

toy = Dataset()
toy.load_toy_dataset()
region = 0
region_ds = Dataset(x=toy.x[:, region, :], y=toy.y[:, region:region + 1])

task = Forecast(prototype=SIRForecaster, lookback=21, horizon=7, device='cpu')
result = task.rolling_train(dataset=region_ds,
                            train_size=350,
                            val_size=60,
                            test_size=60,
                            epochs=1,
                            batch_size=16,
                            max_folds=1)
print(f"\nSIR forecaster MAE: {result['aggregate_metrics']['mae_mean']:.4f}")


# ## 4. Network-level simulation
# NetSIR runs the same dynamics per node on a contact graph.
# See examples/data_simulation.py for the full simulation toolbox.

num_nodes = 25
graph = torch.round(torch.rand(num_nodes, num_nodes))
initial_states = torch.zeros(num_nodes, 3)
initial_states[:, 0] = 1
initial_states[3, 0], initial_states[3, 1] = 0, 1

net = NetSIR(num_nodes=num_nodes, horizon=100,
             infection_rate=0.05, recovery_rate=0.05)
net_trajectory = net(initial_states, graph, steps=None)
print("NetSIR         :", tuple(net_trajectory.shape), "(steps, nodes, compartments)")
