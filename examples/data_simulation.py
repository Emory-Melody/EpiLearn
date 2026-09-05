#!/usr/bin/env python
# coding: utf-8
"""
Simulating epidemic data (EpiLearn 0.1.0).

``epilearn.utils.simulation.Time_geo`` was REMOVED in 0.1.0. Its replacements are
three simulators built on the compartmental models in
``epilearn.utils.compartmental_models``:

    simulate_temporal_epidemic          population level, one time series
    simulate_spatiotemporal_individual  individual level, on a contact graph
    simulate_spatiotemporal_regions     region level, with mobility flows

All three return dicts of tensors that drop straight into a ``Dataset``.

Run it with::

    python examples/data_simulation.py
"""

import os
import sys

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLE_DIR)
sys.path.append(os.path.dirname(EXAMPLE_DIR))
os.chdir(REPO_ROOT)          # so "./datasets" resolves (load_toy_dataset reads it from cwd)

import torch
import matplotlib.pyplot as plt

from epilearn import visualize
from epilearn.data import Dataset
from epilearn.utils import simulation
from epilearn.utils.compartmental_models import SIRModel, SEIRModel, SIRSModel
from epilearn.utils.simulation import (
    simulate_temporal_epidemic,
    simulate_spatiotemporal_individual,
    simulate_spatiotemporal_regions,
)
from epilearn.models.SpatialTemporal.NetworkSIR import NetSIR
from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.tasks.forecast import Forecast


torch.manual_seed(7)


# ## 1. Random graphs

er = simulation.get_random_graph(num_nodes=15, connect_prob=0.4)
ba = simulation.get_random_graph(num_nodes=15, num_edges=2, graph_type='barabasi_albert')
print("erdos_renyi     :", tuple(er.shape), f"{int(er.sum())} directed edges")
print("barabasi_albert :", tuple(ba.shape), f"{int(ba.sum())} directed edges")

# a graph inferred from feature similarity instead of drawn at random
features = torch.rand(10, 20)
print("from features   :", tuple(simulation.get_graph_from_features(features=features).shape))


# ## 2. Node-level simulation with NetSIR
# The classic torch simulator: one SIR process per node, coupled by the graph.

initial_states = torch.zeros(15, 3)     # [S, I, R] one-hot per node
initial_states[:, 0] = 1
for seed_node in (3, 4):
    initial_states[seed_node, 0] = 0
    initial_states[seed_node, 1] = 1

net = NetSIR(num_nodes=er.shape[0], horizon=120,
             infection_rate=0.05, recovery_rate=0.05)
net_traj = net(initial_states, er, steps=None)
print("\nNetSIR          :", tuple(net_traj.shape), "(steps, nodes, compartments)")

# visualize the network state at one point in time
visualize.plot_graph(net_traj.argmax(2)[15].detach().numpy(),
                     er.to_sparse().indices().detach().numpy(),
                     classes=['Susceptible', 'Infected', 'Recovered'])
plt.show()


# ## 3. Population-level simulation (replaces the SIR-only path)

temporal = simulate_temporal_epidemic(SIRModel(beta=0.05, gamma=0.05),
                                      initial_state=[15.0, 2.0, 0.0],
                                      steps=190)
print("temporal        :", temporal['compartments'], tuple(temporal['trajectory'].shape))

plt.figure(figsize=(10, 4))
for i, name in enumerate(temporal['compartments']):
    plt.plot(temporal['trajectory'][:, i].numpy(), label=name)
plt.xlabel("step")
plt.ylabel("individuals")
plt.title("simulate_temporal_epidemic (SIR)")
plt.legend()
plt.tight_layout()
plt.show()


# ## 4. Individual-level simulation on a contact graph
# This is the closest replacement for the old Time_geo trace generator: it gives
# per-individual states over time plus the (possibly time-varying) contact graph.

model = SIRModel(beta=0.35, gamma=0.08)
node_states, contact_graph = simulation.create_initial_conditions_individual(
    model,
    num_individuals=60,
    p_edge=0.06,
    initial_compartment_fractions={'I': 0.05},
    seed=0)
print(f"\ninitial states  : {tuple(node_states.shape)}, contact graph {tuple(contact_graph.shape)}")

individual = simulate_spatiotemporal_individual(model,
                                                contact_graph=contact_graph,
                                                initial_states=node_states,
                                                steps=90,
                                                stochastic=True,
                                                seed=0)
print("individual keys :", sorted(individual.keys()))
print(f"  trajectory    : {tuple(individual['trajectory'].shape)} (steps, individuals)")
print(f"  node_features : {tuple(individual['node_features'].shape)} (steps, individuals, compartments)")
print(f"  dynamic_graph : {tuple(individual['dynamic_graph'].shape)}")
print(f"  counts        : {tuple(individual['counts'].shape)} -> peak infected "
      f"{int(individual['counts'][:, 1].max())}")


# ## 5. Region-level simulation with mobility
# Every region runs its own compartmental model and individuals flow along the
# adjacency graph. This is the one to use to fabricate a spatiotemporal dataset.

sirs = SIRSModel(beta=0.4, gamma=0.1, omega=0.02)
region_init = simulation.create_initial_conditions_region(sirs,
                                                          n_regions=20,
                                                          p_edge=0.15,
                                                          n_initial_infected=3,
                                                          initial_infected_size=50,
                                                          seed=0)
print("\nregion init keys:", sorted(region_init.keys()))

regions = simulate_spatiotemporal_regions(sirs,
                                          region_states=region_init['region_states'],
                                          adjacency_graph=region_init['adjacency'],
                                          steps=150,
                                          travel_rate=0.05)
print(f"  trajectory    : {tuple(regions['trajectory'].shape)} (steps, regions, compartments)")
print(f"  dynamic_graph : {tuple(regions['dynamic_graph'].shape)} (signed flows)")


# ## 6. Turning a simulation into a Dataset and forecasting on it

trajectory = regions['trajectory']
infected_idx = regions['compartments'].index('I')
simulated = Dataset(x=trajectory,
                    y=trajectory[:, :, infected_idx],
                    graph=regions['adjacency'],
                    feature_names=list(regions['compartments']),
                    target_names=['I'])
print("\nsimulated dataset:", simulated)

task = Forecast(prototype=STGCN, lookback=7, horizon=3, device='cpu')
result = task.rolling_train(dataset=simulated,
                            train_size=80,
                            val_size=25,
                            test_size=25,
                            train_loss='mse',
                            epochs=10,
                            batch_size=16,
                            max_folds=1)
print(f"STGCN on simulated data, MAE: {result['aggregate_metrics']['mae_mean']:.4f}")


# ## 7. Dynamic Message Passing
# DMP is still under maintenance: it ignores `horizon` and returns NaNs.
# from epilearn.models.SpatialTemporal.DMP import DMP
# dmp = DMP(num_nodes=25, recover_rate=torch.rand(25))
# dmp_simulation = dmp(None, er)
