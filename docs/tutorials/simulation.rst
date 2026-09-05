Simulation
===================================

In this section, we provide a tutorial of the simulation methods in **EpiLearn**. In general, we focus on the simulation of static and dynamic properties including graph structure and node features.

.. note::
   **Coming from 0.0.x?** ``Time_geo`` was removed with no replacement, and
   ``Gravity_model`` changed signature. Both changes, and everything else, are
   listed in `MIGRATION.md
   <https://github.com/Emory-Melody/EpiLearn/blob/main/MIGRATION.md>`_.

Static Properties
------------------------------

Random Static Graph
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: epilearn.utils.simulation.get_random_graph
    :no-index:

``graph_type`` selects the generator: ``'erdos_renyi'`` needs ``num_nodes`` and
``connect_prob``, ``'stochastic_blockmodel'`` takes ``block_sizes`` plus a
block-wise probability matrix, and ``'barabasi_albert'`` takes ``num_edges``
(edges attached per new node). All three return a dense ``torch.float32``
adjacency matrix, ready to pass as ``Dataset(graph=...)``.

.. code-block:: python

    from epilearn.utils.simulation import get_random_graph

    adj = get_random_graph(num_nodes=25, connect_prob=0.2, graph_type='erdos_renyi')
    sbm = get_random_graph(block_sizes=[10, 15],
                           connect_prob=[[0.4, 0.05], [0.05, 0.4]],
                           graph_type='stochastic_blockmodel')
    ba = get_random_graph(num_nodes=25, num_edges=2, graph_type='barabasi_albert')
    print(adj.shape, sbm.shape, ba.shape)
    # torch.Size([25, 25]) torch.Size([25, 25]) torch.Size([25, 25])


Static Features
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: epilearn.utils.simulation.get_graph_from_features
    :no-index:

Given fixed node features, a graph can be built from the cosine similarity between
nodes. The optional ``adj`` holds the distances between nodes: pass it and each
similarity is divided by the corresponding distance, penalizing far-apart pairs.

.. code-block:: python

    import torch
    from epilearn.utils.simulation import get_graph_from_features

    feature = torch.rand(10, 20)              # 10 nodes, 20 features each
    adj = torch.randint(10, 100, (10, 10))    # pairwise distances

    graph1 = get_graph_from_features(features=feature, adj=None)
    graph2 = get_graph_from_features(features=feature, adj=adj)


Gravity Model
~~~~~~~~~~~~~~~~~~~~~~~~
The gravity model turns populations and connection strengths into mobility flows —
in epidemics, the regional contact and transmission driven by human movement. It is
parameterized by two population exponents and a connectivity decay:

.. math::
    F_{ij} = N_i^{\rho} \cdot N_j^{\theta} \cdot \exp\!\big((w_{ij} - 1) / \delta\big)

``rho`` and ``theta`` are the source/target population exponents (typically
0.5-1.0), ``delta`` is the connectivity decay (0.2-0.5 local, 0.5-1.0 regional,
1.0-2.0 long range), and ``normalize`` divides the flow by the two regions' total
population. Setting ``rho=0, theta=0, normalize=False`` gives the **diffusive**
special case — flow is just the edge weight times the population difference —
which is what ``simulate_spatiotemporal_regions`` uses when no model is given.

.. warning::
   :math:`w_{ij}` is **connection strength**: higher means more flow, and 0 means
   no edge. That is the opposite convention from a distance matrix, so convert
   distances to strengths and normalize them to ``[0, 1]`` before use.

.. autoclass:: epilearn.utils.simulation.Gravity_model
    :members:
    :undoc-members:
    :special-members: __init__
    :no-index:


Given population numbers in each node (or say region) and a connectivity matrix,
edge weights can be obtained for a pair of nodes or for the whole graph at once:

.. code-block:: python

    import torch
    from epilearn.utils.simulation import Gravity_model

    node_populations = torch.tensor([1000., 2000., 1500.])
    connectivity = torch.tensor([[0.0, 0.8, 0.2],      # connection strength in [0, 1]
                                 [0.8, 0.0, 0.5],
                                 [0.2, 0.5, 0.0]])

    gravity = Gravity_model(rho=1.0, theta=1.0, delta=0.5, normalize=True)
    print(round(gravity.compute_flow(1000.0, 2000.0, 0.8), 2))       # 446.88
    print(round(gravity.compute_net_flow(1000.0, 2000.0, 0.8), 2))   # -148.96, i.e. j -> i
    print(gravity.compute_mobility_matrix(node_populations, connectivity))
    # tensor([[  0.0000, 446.8800, 121.1379],
    #         [446.8800,   0.0000, 315.3253],
    #         [121.1379, 315.3253,   0.0000]])

    # diffusive special case: the mobility matrix is the connectivity itself
    diffusive = Gravity_model(rho=0.0, theta=0.0, delta=1.0, normalize=False)
    print(diffusive.is_diffusive)                                    # True


Dynamic Properties
------------------------------

The three simulators below all take a compartmental model and return a dict of
tensors. They differ only in what a "unit" is:

.. list-table::
   :header-rows: 1
   :widths: 38 18 44

   * - Function
     - Level
     - ``trajectory`` shape
   * - ``simulate_temporal_epidemic``
     - one population
     - ``(steps+1, n_compartments)``
   * - ``simulate_spatiotemporal_individual``
     - individuals on a contact graph
     - ``(steps+1, n_nodes)`` of compartment indices
   * - ``simulate_spatiotemporal_regions``
     - regions with mobility
     - ``(steps+1, n_regions, n_compartments)``

The models live in ``epilearn.utils.compartmental_models``
(``SIRModel``, ``SEIRModel``, ``SIRSModel``, ``SEIRVIModel``); see :doc:`utils`
for their parameters and for the ``parameter_schedule`` /``input_schedule``
mechanism used below.


Temporal Simulation
~~~~~~~~~~~~~~~~~~~~~~~~

``simulate_temporal_epidemic`` integrates the ODE at the population level. State
vectors are **counts**, and their length must match ``model.compartments``.
Interventions enter through ``parameter_schedule``: a callable
``f(step_idx, t, state) -> dict | None``, a ``{step_idx: dict}`` mapping, or a
sequence indexed by step, where ``None`` leaves the base parameters alone.

.. code-block:: python

    import epilearn
    from epilearn.utils.compartmental_models import SIRModel
    from epilearn.utils.simulation import simulate_temporal_epidemic

    model = SIRModel(beta=0.3, gamma=0.1)
    run = simulate_temporal_epidemic(model,
                                     initial_state=[9990.0, 10.0, 0.0],   # [S, I, R]
                                     steps=160, dt=1.0)
    print(run.keys())                                  # time, trajectory, compartments
    print(run['trajectory'].shape, run['compartments'])
    # torch.Size([161, 3]) ('S', 'I', 'R')
    epilearn.visualize.plot_series(run['trajectory'].numpy(),
                                   columns=['Susceptible', 'Infected', 'Recovered'])

    def lockdown(step_idx, t, state):
        """Halve transmission between day 20 and day 70."""
        return {'beta': 0.15} if 20 <= step_idx < 70 else None

    npi = simulate_temporal_epidemic(model, [9990.0, 10.0, 0.0], steps=200,
                                     parameter_schedule=lockdown)
    for r in (run, npi):
        print(f"peak {r['trajectory'][:, 1].max():.0f} on day {r['trajectory'][:, 1].argmax()}")
    # peak 3006 on day 38     <- unmitigated
    # peak 1078 on day 86     <- with the lockdown

    noisy = simulate_temporal_epidemic(model, [9990.0, 10.0, 0.0], steps=160,
                                       process_noise=5.0, seed=42)

``process_noise`` (a scalar or a per-compartment vector, with ``seed`` for
reproducibility) adds Gaussian noise after each step and re-projects to
non-negative values, which is the easiest way to get a realistic-looking series to
train a model on. Without it, ``simulate_temporal_epidemic`` is just
``model.simulate(initial_state, steps)``.

.. autofunction:: epilearn.utils.simulation.simulate_temporal_epidemic
    :no-index:


Individual-Level Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``simulate_spatiotemporal_individual`` runs a stochastic node-level process on a
contact graph: susceptible nodes are infected at rate
:math:`1 - e^{-\beta\,\Delta t\,\sum_j A_{ij}\mathbb{1}[j \text{ infectious}]}`, then
progress and recover with the model's ``sigma`` / ``gamma`` / ``omega``.
``create_initial_conditions_individual`` builds a random Erdos-Renyi contact network
plus matching initial states.

.. code-block:: python

    from epilearn.utils.compartmental_models import SEIRModel
    from epilearn.utils.simulation import (create_initial_conditions_individual,
                                          simulate_spatiotemporal_individual)

    model = SEIRModel(beta=0.35, gamma=0.1, sigma=0.2)
    node_states, contact_graph = create_initial_conditions_individual(
        model, num_individuals=200, p_edge=0.03,
        initial_compartment_fractions={'S': 0.98, 'I': 0.02}, seed=0)
    print(node_states.shape, contact_graph.shape)
    # torch.Size([200]) torch.Size([200, 200])

    res = simulate_spatiotemporal_individual(model, contact_graph, node_states,
                                             steps=60, stochastic=True, seed=0)
    print(res.keys())
    # time, trajectory, counts, compartments, contact_graph, dynamic_graph, node_features
    print(res['trajectory'].shape, res['counts'].shape)
    # torch.Size([61, 200]) compartment index, torch.Size([61, 4]) S/E/I/R totals
    print(res['node_features'].shape, res['dynamic_graph'].shape)
    # torch.Size([61, 200, 4]) one-hot, model-ready, torch.Size([61, 200, 200])

    # node_features is already shaped [T, N, F], so this is a training set:
    from epilearn.data import Dataset
    dataset = Dataset(x=res['node_features'],
                      y=res['node_features'][:, :, model.compartments.index('I')],
                      graph=contact_graph,
                      dynamic_graph=res['dynamic_graph'])

``contact_graph`` may also be time-varying: pass a ``(steps+1, N, N)`` tensor, or
a callable ``f(t, step_idx) -> adjacency`` to rewire the network as the epidemic
runs (for example, to model contact reduction).

.. warning::
   ``create_initial_conditions_individual`` **renormalizes**
   ``initial_compartment_fractions`` so they sum to 1. Passing only the infected
   share — including the default ``{'I': 0.01}`` — therefore seeds *the whole
   population* as infectious. Always give the full distribution, e.g.
   ``{'S': 0.98, 'I': 0.02}``. The model must also have a compartment literally
   named ``'S'``, otherwise the simulator raises ``ValueError``.

.. autofunction:: epilearn.utils.simulation.simulate_spatiotemporal_individual
    :no-index:

.. autofunction:: epilearn.utils.simulation.create_initial_conditions_individual
    :no-index:


Region-Level (Metapopulation) Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``simulate_spatiotemporal_regions`` alternates two phases per step: integrate each
region's internal dynamics, then move people along the graph using a
:class:`~epilearn.utils.simulation.Gravity_model`. Flows are antisymmetric, so the
total population is conserved. ``create_initial_conditions_region`` sets up the
regions, and ``create_regional_forcing_params`` gives each one its own
multi-frequency seasonal forcing, which desynchronizes the regional waves instead
of leaving them in lock-step.

.. code-block:: python

    from epilearn.utils.compartmental_models import SIRSModel
    from epilearn.utils.simulation import (create_initial_conditions_region,
                                          create_regional_forcing_params,
                                          simulate_spatiotemporal_regions,
                                          Gravity_model)

    model = SIRSModel(beta=0.35, gamma=0.1, omega=0.02)   # SIRS never burns out
    init = create_initial_conditions_region(model, n_regions=30, p_edge=0.12,
                                            pop_range=(5000, 20000),
                                            n_initial_infected=3,
                                            initial_infected_size=50, seed=1)
    print(init['n_regions'], init['n_edges'])          # 30 59
    print(init['region_states'].shape)                 # torch.Size([30, 3])

    forcing = create_regional_forcing_params(init['n_regions'], seed=1)
    print(round(forcing[0]['amp1'], 4), forcing[0]['period1'])   # 0.1967 30.0

    res = simulate_spatiotemporal_regions(
        model,
        region_states=init['region_states'],           # counts, [n_regions, n_compartments]
        adjacency_graph=init['adjacency'], steps=120, dt=1.0,
        forcing_params=forcing, forcing_parameter='beta',
        method='euler',                                # 'euler' is the stable choice here
        gravity_model=Gravity_model(rho=1.0, theta=1.0, delta=0.5, normalize=True),
        travel_rate=0.01)                              # global mobility scaling

    print(res['trajectory'].shape)                     # torch.Size([121, 30, 3])
    print(res['dynamic_graph'].shape)                  # torch.Size([121, 30, 30])
    print(res['effective_reproduction_number'].shape)  # torch.Size([121, 30])

    # straight into a spatiotemporal task
    from epilearn.data import Dataset
    dataset = Dataset(x=res['trajectory'],                    # [T, N, n_compartments]
                      y=res['trajectory'][:, :, 1],           # infectious counts
                      graph=init['adjacency'],
                      dynamic_graph=res['directed_flow'])

Three of the returned tensors are worth calling out. ``dynamic_graph`` is the
**signed** net-flow graph per step (``F[i, j] > 0`` means i → j, and
``F == -F.T``), so feed a model ``directed_flow`` instead — the same tensor
clamped at 0. ``parameter_history`` ``(steps+1, n_regions)`` records the value of
``forcing_parameter`` actually used, and ``effective_reproduction_number``
``(steps+1, n_regions)`` is the per-region :math:`R_t` derived from it (``None``
for models without ``gamma`` or without an ``'S'`` compartment). Also returned:
``counts``, ``regional_totals``, ``adjacency``, ``adjacency_history`` and
``node_features``.

.. warning::
   ``create_initial_conditions_region`` samples the seeded regions without
   replacement, so ``n_initial_infected >= n_regions`` raises ``ValueError:
   Cannot take a larger sample than population when 'replace=False'``. The default
   is ``n_initial_infected=20``, so any run with fewer than 21 regions must pass a
   smaller value. ``ensure_connected=True`` (the default) also keeps only the
   largest connected component, so read the surviving region count back from
   ``init['n_regions']``.

.. autofunction:: epilearn.utils.simulation.simulate_spatiotemporal_regions
    :no-index:

.. autofunction:: epilearn.utils.simulation.create_initial_conditions_region
    :no-index:

.. autofunction:: epilearn.utils.simulation.create_regional_forcing_params
    :no-index:


Torch Compartmental Layers
------------------------------

Separately from the ODE utilities above, ``epilearn.models.Temporal.Compartmental``
ships ``nn.Module`` compartmental layers, useful when you want an SIR-like model
that participates in autograd. ``SIR`` takes ``horizon`` (total simulation steps),
``infection_rate`` and ``recovery_rate``; ``NetworkSIR`` spreads the disease over a
graph and adds ``num_nodes``. Both are called as ``model(states, [graph,] steps)``,
where ``steps=None`` runs the full ``horizon``. ``SIS``, ``SEIR``, ``NetworkSIS``
and ``NetworkSEIR`` live in the same module.

.. code-block:: python

    import epilearn
    import torch
    from epilearn.models.Temporal.Compartmental import SIR, NetworkSIR

    # 25 nodes, all susceptible except nodes 3 and 10; columns are [S, I, R]
    initial_states = torch.zeros(25, 3)
    initial_states[:, 0] = 1
    initial_states[[3, 10], 0] = 0
    initial_states[[3, 10], 1] = 1

    # population level: the layer takes the aggregated [S, I, R] counts
    model = SIR(horizon=190, infection_rate=0.05, recovery_rate=0.05)
    preds = model(initial_states.sum(0), steps=None)
    print(preds.shape)   # torch.Size([190, 3])
    epilearn.visualize.plot_series(preds.detach().numpy(),
                                  columns=['Susceptible', 'Infected', 'Recovered'])

    # node level: same states, plus a graph
    initial_graph = epilearn.utils.simulation.get_random_graph(num_nodes=25, connect_prob=0.20)
    net = NetworkSIR(num_nodes=initial_graph.shape[0], horizon=120,
                     infection_rate=0.05, recovery_rate=0.05)
    preds = net(initial_states, initial_graph, steps=None)
    print(preds.shape)   # torch.Size([120, 25, 3])
    epilearn.visualize.plot_graph(preds.argmax(2)[15].detach().numpy(),
                                  initial_graph.to_sparse().indices().detach().numpy(),
                                  classes=['Susceptible', 'Infected', 'Recovered'])

.. note::
   These classes moved from ``epilearn.models.Temporal.SIR`` to
   ``epilearn.models.Temporal.Compartmental`` in 0.1.0; the old path still works as
   an alias. ``epilearn.models.SpatialTemporal.NetworkSIR.NetSIR`` is the
   equivalent model in the ``SpatialTemporal`` family and takes the same
   ``(x, adj, steps)`` call.
