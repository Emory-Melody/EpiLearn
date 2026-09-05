import torch
import numpy as np
import math
import networkx as nx
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union
from collections.abc import Mapping, Sequence as SequenceCollection

from .compartmental_models import CompartmentalModel



def get_random_graph(num_nodes=None, connect_prob=None, block_sizes=None, num_edges=None, graph_type='erdos_renyi'):
    """
    Generates a random static graph using one of the supported graph types: Erdos-Renyi, Stochastic Blockmodel, or Barabasi-Albert.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    connect_prob : float, optional
        Probability of edge creation (for Erdos-Renyi and Stochastic Blockmodel graphs).
    block_sizes : list of int, optional
        Sizes of blocks (for Stochastic Blockmodel graph).
    num_edges : int, optional
        Number of edges (for Barabasi-Albert graph).
    graph_type : str
        Type of graph to generate. Options are `'erdos_renyi'`, `'stochastic_blockmodel'`, `'barabasi_albert'`. Default is `'erdos_renyi'`.

    Returns
    -------
    torch.Tensor
        Adjacency matrix of the generated graph.
    """
    if graph_type == 'erdos_renyi':
        nx_graph = nx.erdos_renyi_graph(num_nodes, connect_prob)
        adj = nx.to_numpy_array(nx_graph)
    elif graph_type == 'stochastic_blockmodel':
        nx_graph = nx.stochastic_block_model(block_sizes, connect_prob)
        adj = nx.to_numpy_array(nx_graph)
    elif graph_type == 'barabasi_albert':
        nx_graph = nx.barabasi_albert_graph(num_nodes, num_edges)
        adj = nx.to_numpy_array(nx_graph)
    else:
        raise NameError("grap type not supported")

    return torch.tensor(adj, dtype=torch.float32)

def get_graph_from_features(features, adj=None, G=1):
    """
    Generate a graph from node features using cosine similarity.

    This function generates a graph where each edge weight is computed based on the cosine similarity
    between the feature vectors of the connected nodes. If an adjacency matrix is provided, the cosine 
    similarity is adjusted by the corresponding entry in the adjacency matrix.

    Parameters
    ----------
    features : torch.Tensor
        A tensor of shape (num_nodes, feat_dim) where num_nodes is the number of nodes and feat_dim is 
        the dimensionality of the feature vectors.
    adj : torch.Tensor, optional
        A tensor of shape (num_nodes, num_nodes) representing the adjacency matrix, where adj[i, j] 
        denotes the distance or weight between node i and node j. If None, the cosine similarity is 
        used directly as the edge weight. Default is None.

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_nodes, num_nodes) representing the generated graph's adjacency matrix, 
        where each entry [i, j] contains the adjusted cosine similarity between nodes i and j.
    """
    n_nodes = len(features)
    graph = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj is not None:
                graph[i, j] = (torch.nn.functional.cosine_similarity(features[i], features[j], dim=0)/adj[i,j]).item()
            else:
                graph[i, j] = torch.nn.functional.cosine_similarity(features[i], features[j], dim=0).item()
    return graph


class Gravity_model:
    """
    Elegant gravity model for human mobility in epidemic simulations.
    
    Computes flow between regions based on population attraction and connectivity strength:
    
    .. math::
        F_{ij} = N_i^{\\rho} \\cdot N_j^{\\theta} \\cdot \\exp((w_{ij} - 1) / \\delta)
    
    where higher w_{ij} indicates stronger connection between regions.
    
    Parameters
    ----------
    rho : float
        Source population exponent (typically 0.5-1.0).
    theta : float  
        Target population exponent (typically 0.5-1.0).
    delta : float
        Connectivity decay parameter. Controls sensitivity to connectivity variations.
        Recommended: 0.2-2.0 for normalized connectivity [0, 1].
    normalize : bool
        If True, normalize flows by total population (default True).
    
    Notes
    -----
    **Unified Connectivity Semantics**:
    
    Connectivity values represent connection strength where:
    - Higher values = Stronger connection = More flow
    - Lower values = Weaker connection = Less flow
    - 0 = No connection
    
    This is consistent across both diffusive and gravity models.
    
    It is recommended to normalize your connectivity matrix to [0, 1] range:
    
    >>> # Normalize connectivity matrix (edge weights)
    >>> max_conn = connectivity_matrix.max()
    >>> normalized_connectivity = connectivity_matrix / max_conn
    >>> 
    >>> # Choose delta based on desired spatial spread
    >>> # Sharp decay (local): delta = 0.2-0.5
    >>> # Moderate (regional): delta = 0.5-1.0
    >>> # Gradual (long-range): delta = 1.0-2.0
    
    Examples
    --------
    >>> # Diffusive flow (special case: rho=0, theta=0)
    >>> diffusive = Gravity_model(rho=0, theta=0, delta=1.0, normalize=False)
    >>> 
    >>> # Gravity model with normalized connectivity
    >>> gravity = Gravity_model(rho=1.0, theta=1.0, delta=0.5, normalize=True)
    >>> 
    >>> # Using with normalized edge weights
    >>> weights_normalized = adjacency / adjacency.max()
    >>> result = simulate_spatiotemporal_regions(
    ...     model, states, weights_normalized, steps=100, gravity_model=gravity)
    """
    
    def __init__(self, rho: float = 0.0, theta: float = 0.0, delta: float = 1.0, normalize: bool = False):
        self.rho = rho
        self.theta = theta
        self.delta = delta
        self.normalize = normalize
        self.is_diffusive = (rho == 0.0 and theta == 0.0 and not normalize)
    
    def compute_flow(self, pop_i: float, pop_j: float, connectivity: float) -> float:
        """
        Compute flow between two regions.
        
        Parameters
        ----------
        pop_i, pop_j : float
            Population sizes of regions i and j.
        connectivity : float
            Connection strength between regions (recommended range: 0-1).
            Higher values = stronger connection = more flow.
            - 0: No connection
            - 1: Maximum connection strength
            
            Interpretation is unified across all models:
            - Diffusive: Direct multiplier on population flow
            - Gravity: Exponential enhancement of attraction
        
        Returns
        -------
        float
            Flow magnitude or edge weight for further computation.
        
        Notes
        -----
        **Unified Connectivity Semantics**:
        
        Connectivity always represents connection strength where higher = stronger:
        
        Diffusive model (rho=0, theta=0):
            - Formula: F = connectivity × (N_i - N_j)
            - Range: [0.0, 1.0]
            - Interpretation: Fraction of population difference that flows per timestep
            - Example: 0.1 = 10% of population difference travels per day
        
        Gravity model (rho>0, theta>0):
            - Formula: F = [N_i^ρ × N_j^θ × exp((connectivity-1)/δ)] × (N_i - N_j)/(N_i + N_j)
            - Range: [0.0, 1.0] normalized
            - connectivity=1.0 gives baseline gravity attraction
            - connectivity>1.0 enhances flow (if using unnormalized weights)
            - connectivity<1.0 reduces flow
            - Delta parameter controls sensitivity:
              * Small delta (0.2-0.5): Sharp response to connectivity differences
              * Large delta (1.0-2.0): Gradual response to connectivity differences
        
        **Normalization Strategy**:
        
        For any connectivity matrix (edge weights, similarity scores, etc.):
        1. Find max value: max_conn = connectivity_matrix.max()
        2. Normalize: connectivity_matrix = connectivity_matrix / max_conn
        3. Choose delta based on desired sensitivity:
           - Sharp (local spread): delta = 0.2-0.5
           - Moderate: delta = 0.5-1.0  
           - Gradual (long-range): delta = 1.0-2.0
        """
        if connectivity <= 0 or pop_i <= 0 or pop_j <= 0:
            return 0.0
        
        if self.is_diffusive:
            # Diffusive: flow = connectivity * (N_i - N_j)
            # connectivity in [0, 1] representing travel fraction
            return connectivity
        
        # Gravity model: connectivity represents connection strength
        # Higher connectivity = stronger connection = more flow
        # Use exp((connectivity - 1) / delta) so that:
        # - connectivity=1.0 → exp(0) = 1.0 (baseline)
        # - connectivity>1.0 → exp(+) > 1.0 (enhanced)
        # - connectivity<1.0 → exp(-) < 1.0 (reduced)
        attraction = (pop_i ** self.rho) * (pop_j ** self.theta) * math.exp((connectivity - 1.0) / self.delta)
        
        if self.normalize:
            attraction /= (pop_i + pop_j)
        
        return float(attraction)
    
    def compute_net_flow(self, pop_i: float, pop_j: float, connectivity: float) -> float:
        """
        Compute net directional flow from region i to region j.
        
        Parameters
        ----------
        pop_i, pop_j : float
            Population sizes of regions i and j.
        connectivity : float
            Connection strength between regions.
            Higher values = stronger connection = more flow (unified across all models).
        
        Returns
        -------
        float
            Net flow (positive = i→j, negative = j→i).
        """
        if connectivity <= 0 or pop_i <= 0 or pop_j <= 0:
            return 0.0
        
        if self.is_diffusive:
            # Diffusive: F = connectivity * (N_i - N_j)
            return connectivity * (pop_i - pop_j)
        
        # Gravity: compute attraction then weight by population difference
        flow_magnitude = self.compute_flow(pop_i, pop_j, connectivity)
        return flow_magnitude * (pop_i - pop_j) / (pop_i + pop_j)
    
    def compute_mobility_matrix(self, populations: torch.Tensor, connectivity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of full mobility matrix.
        
        Parameters
        ----------
        populations : torch.Tensor
            Population of each region, shape (num_regions,).
        connectivity_matrix : torch.Tensor
            Connectivity matrix, shape (num_regions, num_regions).
            Higher values = stronger connection = more flow (unified semantics).
        
        Returns
        -------
        torch.Tensor
            Mobility flow matrix, shape (num_regions, num_regions).
        """
        num_regions = populations.shape[0]
        
        # Broadcast for vectorized computation
        pop_i = populations.view(-1, 1).expand(num_regions, num_regions)
        pop_j = populations.view(1, -1).expand(num_regions, num_regions)
        
        if self.is_diffusive:
            # Diffusive model: just return connectivity (edge weights)
            return connectivity_matrix.clone()
        
        # Gravity formula: higher connectivity = stronger connection = more flow
        # Using exp((connectivity - 1) / delta) for unified semantics
        mobility = (pop_i ** self.rho) * (pop_j ** self.theta) * torch.exp((connectivity_matrix - 1.0) / self.delta)
        
        if self.normalize:
            total_pop = pop_i + pop_j
            mobility = torch.where(total_pop > 0, mobility / total_pop, torch.zeros_like(mobility))
        
        # Clean up invalid entries
        mobility = mobility.masked_fill(connectivity_matrix <= 0, 0.0)
        mobility.fill_diagonal_(0.0)
        
        return mobility


# ---------------------------------------------------------------------------
# Compartmental models and epidemic simulations
# ---------------------------------------------------------------------------

_NUMBER_TYPES = (int, float, np.floating, np.integer)


def _ensure_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, _NUMBER_TYPES):
        return float(value)
    raise TypeError(f"Unsupported parameter value type: {type(value)}")


def _resolve_schedule(schedule, step_idx, t, state):
    if schedule is None:
        return None
    if callable(schedule):
        return schedule(step_idx, t, state)
    if isinstance(schedule, Mapping):
        value = schedule.get(step_idx)
        if callable(value):
            return value(step_idx, t, state)
        return value
    if isinstance(schedule, SequenceCollection) and not isinstance(schedule, (str, bytes)):
        if step_idx < len(schedule):
            value = schedule[step_idx]
            if callable(value):
                return value(step_idx, t, state)
            return value
        return None
    return schedule


def _prepare_noise_tensor(std, reference_state):
    if std is None:
        return None
    noise = torch.as_tensor(std, dtype=reference_state.dtype)
    if noise.ndim == 0:
        return torch.full_like(reference_state, float(noise))
    if noise.shape == reference_state.shape:
        return noise.clone().to(reference_state.dtype)
    if noise.ndim == 1 and noise.shape[0] == reference_state.shape[-1]:
        return noise.clone().to(reference_state.dtype)
    raise ValueError("process_noise must be a scalar or match the compartment dimension.")


def _prepare_generator(seed=None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator


def _ensure_adjacency_tensor(graph):
    if isinstance(graph, torch.Tensor):
        tensor = graph.detach().clone().to(dtype=torch.float32)
    elif isinstance(graph, np.ndarray):
        tensor = torch.from_numpy(graph).float()
    elif isinstance(graph, nx.Graph):
        tensor = torch.from_numpy(nx.to_numpy_array(graph)).float()
    else:
        raise TypeError("contact/mobility graph must be a tensor, ndarray, or NetworkX graph.")
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Adjacency matrices must be square.")
    return tensor



def _encode_node_states(initial_states, num_nodes, compartment_index):
    default_state = compartment_index.get('S', 0)
    if isinstance(initial_states, dict):
        states = torch.full((num_nodes,), default_state, dtype=torch.long)
        for comp, nodes in initial_states.items():
            idx = compartment_index[comp]
            node_ids = torch.as_tensor(list(nodes), dtype=torch.long)
            states[node_ids] = idx
        return states
    try:
        states = torch.as_tensor(initial_states, dtype=torch.long)
    except (TypeError, ValueError):
        states = torch.tensor([compartment_index[str(label)] for label in initial_states], dtype=torch.long)
    if states.numel() != num_nodes:
        raise ValueError(f"Expected {num_nodes} initial states, received {states.numel()}.")
    return states.clone().long()


def _sample_events(probabilities, candidate_mask, rng, stochastic):
    candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
    result = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if candidate_indices.numel() == 0:
        return result
    prob_tensor = torch.as_tensor(probabilities, dtype=torch.float32)
    if prob_tensor.ndim == 0:
        probs = torch.full((candidate_indices.numel(),), float(prob_tensor))
    else:
        if prob_tensor.shape != candidate_mask.shape:
            raise ValueError("Probability tensors must align with the candidate mask.")
        probs = prob_tensor[candidate_mask]
    probs = probs.clamp(0.0, 1.0)
    if stochastic:
        draws = torch.rand(probs.shape, generator=rng)
        outcomes = draws < probs
    else:
        outcomes = probs >= 0.5
    result[candidate_indices] = outcomes.to(result.dtype)
    return result


def _get_infectious_compartments(model):
    metadata = getattr(model, "metadata", None) or {}
    infectious = metadata.get("infectious_compartments")
    if infectious:
        return tuple(infectious)
    inferred = [comp for comp in model.compartments if comp.upper().startswith('I')]
    return tuple(inferred) if inferred else (model.compartments[0],)


def _get_compartment_index(model, target):
    """
    Return the index of a compartment if it exists in the model.
    """
    try:
        return model.compartments.index(target)
    except ValueError:
        return None



def simulate_temporal_epidemic(
    model: CompartmentalModel,
    initial_state: Union[Sequence[float], torch.Tensor, np.ndarray],
    steps: int,
    dt: float = 1.0,
    parameter_schedule: Optional[Union[Mapping[int, Dict[str, float]], Sequence]] = None,
    input_schedule: Optional[Union[Mapping[int, Dict[str, float]], Sequence]] = None,
    process_noise: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    method: str = 'rk4',
    seed: Optional[int] = None,
):
    """
    Run a temporal (population-level) simulation for the provided compartmental model.
    """
    state = model.validate_state(initial_state)
    history = torch.zeros(steps + 1, len(model.compartments), dtype=state.dtype)
    history[0] = state
    noise = _prepare_noise_tensor(process_noise, state)
    rng = _prepare_generator(seed)
    for step_idx in range(steps):
        t = step_idx * dt
        overrides = _resolve_schedule(parameter_schedule, step_idx, t, state)
        inputs = _resolve_schedule(input_schedule, step_idx, t, state)
        next_state = model.step(state, t, dt, method=method, external_inputs=inputs, parameter_overrides=overrides)
        if noise is not None:
            noise_sample = torch.randn(next_state.shape, generator=rng, device=next_state.device, dtype=next_state.dtype) * noise
            next_state = model.project_state(next_state + noise_sample)
        history[step_idx + 1] = next_state
        state = next_state
    time_axis = torch.linspace(0.0, steps * dt, steps + 1)
    return {"time": time_axis, "trajectory": history, "compartments": model.compartments}



def create_initial_conditions_individual(
    model,
    num_individuals=100,
    p_edge=0.01,
    initial_compartment_fractions: dict = {"I":0.01},
    seed=42
):
    """
    Create initial conditions for individual-level simulations.
    
    Parameters
    ----------
    model : CompartmentalModel
        The compartmental model (e.g., SIRModel, SIRSModel, SEIRModel) that defines
        the disease dynamics and compartment structure
    num_individuals : int
        Number of individuals in the network
    p_edge : float
        Edge probability for Erdos-Renyi random graph (contact network)
    initial_compartment_fractions : dict
        Dictionary mapping compartment names to initial fractions.
        Example: {'I': 0.01} means 1% start infected, rest susceptible
        Example: {'S': 0.9, 'E': 0.05, 'I': 0.05} for SEIR model
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    tuple
        - node_states: torch.Tensor of shape (num_individuals,) containing compartment indices
        - adjacency: torch.Tensor of shape (num_individuals, num_individuals) contact graph
    """
    # Get model compartments
    compartments = model.compartments
    n_compartments = len(compartments)
    comp_index = {comp: idx for idx, comp in enumerate(compartments)}
    
    # Create random contact graph
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(num_individuals, p_edge, seed=seed)
    
    # Create adjacency matrix
    adjacency = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
    
    # Initialize node states
    # Default: all individuals start in first compartment (usually 'S')
    node_states = torch.zeros(num_individuals, dtype=torch.long)
    
    # Calculate number of individuals in each compartment
    compartment_counts = {}
    remaining = num_individuals
    
    # Normalize fractions to sum to 1.0
    total_fraction = sum(initial_compartment_fractions.values())
    normalized_fractions = {k: v/total_fraction for k, v in initial_compartment_fractions.items()}
    
    # Handle default susceptible compartment
    if 'S' not in normalized_fractions and sum(normalized_fractions.values()) < 1.0:
        normalized_fractions['S'] = 1.0 - sum(normalized_fractions.values())
    
    # Assign individuals to compartments
    assigned = 0
    for comp_name in compartments:
        if comp_name in normalized_fractions:
            count = int(num_individuals * normalized_fractions[comp_name])
            compartment_counts[comp_name] = count
            assigned += count
        else:
            compartment_counts[comp_name] = 0
    
    # Handle rounding errors - assign remaining individuals to susceptible
    if assigned < num_individuals:
        susceptible_comp = 'S' if 'S' in comp_index else compartments[0]
        compartment_counts[susceptible_comp] += (num_individuals - assigned)
    
    # Randomly assign individuals to compartments
    individual_indices = np.random.permutation(num_individuals)
    current_idx = 0
    
    for comp_name in compartments:
        count = compartment_counts[comp_name]
        comp_idx = comp_index[comp_name]
        
        # Assign this many individuals to this compartment
        node_states[individual_indices[current_idx:current_idx + count]] = comp_idx
        current_idx += count
    
    return node_states, adjacency






def simulate_spatiotemporal_individual(
    model: CompartmentalModel,
    contact_graph: Union[torch.Tensor, np.ndarray, nx.Graph, Callable],
    initial_states: Union[Sequence[int], Sequence[str], torch.Tensor, np.ndarray, Dict[str, Iterable[int]]],
    steps: int,
    dt: float = 1.0,
    stochastic: bool = True,
    seed: Optional[int] = None,
):
    """
    Simulate an individual-level compartmental process on a contact graph.
    Nodes follow the provided model (SIR/SEIR/SIRS) with infection pressure driven by neighbors.
    Returns per-step node features (one-hot compartment indicators) and a dynamic graph tensor of shape (time, N, N).
    
    Parameters
    ----------
    model : CompartmentalModel
        The compartmental model (e.g., SIRModel, SEIRModel, SIRSModel).
    contact_graph : Union[torch.Tensor, np.ndarray, nx.Graph, Callable]
        Contact graph specification. Can be:
        - Static: Binary adjacency matrix (num_nodes, num_nodes) or NetworkX graph
        - Time-varying: Tensor of shape (steps+1, num_nodes, num_nodes)
        - Dynamic: Callable function f(t, step_idx) -> adjacency_matrix that returns
          the contact graph at each time step
    initial_states : Union[Sequence[int], Sequence[str], torch.Tensor, np.ndarray, Dict[str, Iterable[int]]]
        Initial compartment states for each individual.
    steps : int
        Number of simulation steps.
    dt : float
        Time step size (default 1.0).
    stochastic : bool
        Whether to use stochastic transitions (default True).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Simulation results with keys:
        - 'time': Time axis (steps+1,)
        - 'trajectory': Individual state history (steps+1, num_nodes) with compartment indices
        - 'counts': Compartment counts over time (steps+1, num_compartments)
        - 'compartments': Compartment names
        - 'contact_graph': Static or initial contact graph
        - 'dynamic_graph': Time-varying contact graphs (steps+1, num_nodes, num_nodes)
        - 'node_features': One-hot encoded compartment states (steps+1, num_nodes, num_compartments)
    """
    # Determine contact graph type and prepare getter function
    is_dynamic_callable = callable(contact_graph)
    is_time_series = False
    
    if is_dynamic_callable:
        # Callable: contact_graph(t, step_idx) -> adjacency matrix
        def get_adjacency(t, step_idx):
            adj = contact_graph(t, step_idx)
            return _ensure_adjacency_tensor(adj)
        initial_adjacency = get_adjacency(0.0, 0)
    elif isinstance(contact_graph, torch.Tensor) and contact_graph.ndim == 3:
        # Pre-computed time series: (steps+1, num_nodes, num_nodes)
        is_time_series = True
        adjacency_series = contact_graph
        if adjacency_series.shape[0] != steps + 1:
            raise ValueError(f"Time-varying contact graph must have shape ({steps+1}, num_nodes, num_nodes).")
        def get_adjacency(t, step_idx):
            return adjacency_series[step_idx]
        initial_adjacency = adjacency_series[0]
    else:
        # Static: convert once
        static_adjacency = _ensure_adjacency_tensor(contact_graph)
        def get_adjacency(t, step_idx):
            return static_adjacency
        initial_adjacency = static_adjacency
    
    num_nodes = initial_adjacency.shape[0]
    compartments = model.compartments
    comp_index = {comp: idx for idx, comp in enumerate(compartments)}
    node_states = _encode_node_states(initial_states, num_nodes, comp_index)
    trajectory = torch.zeros(steps + 1, num_nodes, dtype=torch.long)
    trajectory[0] = node_states
    rng = _prepare_generator(seed)
    
    # Store dynamic contact graphs
    dynamic_graph = torch.zeros(steps + 1, num_nodes, num_nodes, dtype=initial_adjacency.dtype)
    dynamic_graph[0] = initial_adjacency
    beta = _ensure_float(model.parameters.get('beta', 0.0))
    gamma = _ensure_float(model.parameters.get('gamma', 0.0))
    sigma = _ensure_float(model.parameters.get('sigma', 0.0)) if 'E' in comp_index else 0.0
    omega = _ensure_float(model.parameters.get('omega', 0.0)) if 'R' in comp_index else 0.0
    infectious_comps = _get_infectious_compartments(model)
    susceptible_idx = comp_index.get('S')
    if susceptible_idx is None:
        raise ValueError("A susceptible compartment named 'S' is required for individual simulations.")
    has_exposed = 'E' in comp_index
    has_removed = 'R' in comp_index
    metadata = getattr(model, "metadata", None) or {}
    recovery_target = metadata.get('recovery_target', 'R' if has_removed else 'S')
    waning_target = metadata.get('waning_target', 'S')
    recovery_idx = comp_index.get(recovery_target, susceptible_idx)
    waning_idx = comp_index.get(waning_target, susceptible_idx)
    for step_idx in range(steps):
        t = step_idx * dt
        
        # Get contact graph for this timestep
        adjacency = get_adjacency(t, step_idx)
        dynamic_graph[step_idx] = adjacency
        
        current_state = node_states
        next_state = current_state.clone()
        infectious_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for comp in infectious_comps:
            infectious_mask |= current_state == comp_index[comp]
        infection_force = adjacency.matmul(infectious_mask.float())
        infection_prob = (1.0 - torch.exp(-beta * dt * infection_force)).clamp(0.0, 1.0)
        susceptible_mask = current_state == susceptible_idx
        new_infections = _sample_events(infection_prob, susceptible_mask, rng, stochastic)
        if has_exposed:
            next_state[new_infections] = comp_index['E']
        else:
            next_state[new_infections] = comp_index[infectious_comps[0]]
        if has_exposed and sigma > 0.0:
            exposed_mask = current_state == comp_index['E']
            progress_prob = 1.0 - math.exp(-sigma * dt)
            progress_events = _sample_events(progress_prob, exposed_mask, rng, stochastic)
            next_state[progress_events] = comp_index['I']
        if gamma > 0.0:
            infectious_for_recovery = torch.zeros(num_nodes, dtype=torch.bool)
            for comp in infectious_comps:
                infectious_for_recovery |= current_state == comp_index[comp]
            recovery_prob = 1.0 - math.exp(-gamma * dt)
            recoveries = _sample_events(recovery_prob, infectious_for_recovery, rng, stochastic)
            next_state[recoveries] = recovery_idx
        if has_removed and omega > 0.0 and 'R' in comp_index:
            recovered_mask = current_state == comp_index['R']
            waning_prob = 1.0 - math.exp(-omega * dt)
            waning_events = _sample_events(waning_prob, recovered_mask, rng, stochastic)
            next_state[waning_events] = waning_idx
        node_states = next_state
        trajectory[step_idx + 1] = node_states
    
    # Store final contact graph
    final_t = steps * dt
    final_adjacency = get_adjacency(final_t, steps)
    dynamic_graph[steps] = final_adjacency
    
    counts = torch.zeros(steps + 1, len(compartments), dtype=torch.float32)
    for comp, idx in comp_index.items():
        counts[:, idx] = (trajectory == idx).sum(dim=1)
    time_axis = torch.linspace(0.0, steps * dt, steps + 1)
    node_features = torch.nn.functional.one_hot(trajectory, num_classes=len(compartments)).float()
    return {
        "time": time_axis,
        "trajectory": trajectory,
        "counts": counts,
        "compartments": compartments,
        "contact_graph": initial_adjacency,
        "dynamic_graph": dynamic_graph,
        "node_features": node_features,
    }





def create_initial_conditions_region(
    model,
    n_regions=100,
    p_edge=0.01,
    pop_range=(800, 20000),
    n_initial_infected=20,
    initial_infected_size=100,
    initial_compartment_fractions=None,
    ensure_connected=True,
    seed=42
):
    """
    Create initial conditions for spatiotemporal epidemic simulation based on a compartmental model.
    
    Parameters
    ----------
    model : CompartmentalModel
        The compartmental model (e.g., SIRModel, SIRSModel, SEIRModel) that defines
        the disease dynamics and compartment structure
    n_regions : int
        Number of regions in the network
    p_edge : float
        Edge probability for Erdos-Renyi random graph
    pop_range : tuple of (int, int)
        Range for random population initialization (min_pop, max_pop)
    n_initial_infected : int
        Number of regions to seed with initial infections
    initial_infected_size : int
        Number of individuals initially infected in each seeded region
    initial_compartment_fractions : dict or None
        Optional dictionary mapping compartment names to initial fractions.
        If None, all individuals start in 'S' (susceptible).
        Example: {'S': 0.9, 'E': 0.05, 'I': 0.05} for SEIR model
    ensure_connected : bool
        If True, extract largest connected component
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'adjacency': torch.Tensor of shape (n_regions, n_regions)
        - 'region_states': torch.Tensor of shape (n_regions, n_compartments)
        - 'graph': networkx.Graph object
        - 'n_regions': int (actual number of regions after connectivity check)
        - 'n_edges': int
        - 'infected_nodes': numpy.ndarray of initially infected region indices
        - 'model': the compartmental model used
        - 'compartment_names': list of compartment names
        - 'total_population': total population across all regions
    """
    # Get model compartments
    compartment_names = model.compartments
    n_compartments = len(compartment_names)
    
    # Create random graph
    G = nx.erdos_renyi_graph(n_regions, p_edge, seed=seed)
    
    # Ensure connected if requested
    if ensure_connected and not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        n_regions_actual = len(G.nodes())
    else:
        n_regions_actual = n_regions
    
    # Create adjacency matrix
    adjacency = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
    
    # Initialize states based on compartment fractions
    region_states = torch.zeros(n_regions_actual, n_compartments)
    np.random.seed(seed)
    
    for i in range(n_regions_actual):
        pop = np.random.randint(pop_range[0], pop_range[1])
        
        if initial_compartment_fractions is None:
            # Default: all in susceptible compartment (first compartment)
            region_states[i, 0] = pop
        else:
            # Distribute population according to specified fractions
            for comp_name, fraction in initial_compartment_fractions.items():
                comp_idx = compartment_names.index(comp_name)
                region_states[i, comp_idx] = pop * fraction
    
    # Seed infections in random regions
    # Find the infected compartment index (usually 'I')
    if 'I' in compartment_names:
        infected_idx = compartment_names.index('I')
    else:
        # Fallback to second compartment if 'I' not found
        infected_idx = 1
    
    infected_nodes = np.random.choice(n_regions_actual, size=n_initial_infected, replace=False)
    for node in infected_nodes:
        pop = region_states[node].sum().item()
        infected = min(initial_infected_size, pop)
        
        # Remove from susceptible (first compartment)
        region_states[node, 0] = max(0, region_states[node, 0] - infected)
        # Add to infected compartment
        region_states[node, infected_idx] += infected
    
    return {
        'adjacency': adjacency,
        'region_states': region_states,
        'graph': G,
        'n_regions': n_regions_actual,
        'n_edges': G.number_of_edges(),
        'infected_nodes': infected_nodes,
        'total_population': region_states.sum().item(),
        'model': model,
        'compartment_names': compartment_names,
    }




def create_regional_forcing_params(
    num_regions: int,
    base_amplitudes: Sequence[float] = (0.2, 0.15, 0.1),
    periods: Sequence[float] = (30.0, 47.0, 73.0),
    amplitude_noise: float = 0.1,
    seed: Optional[int] = None,
) -> List[Dict[str, Union[List[float], float]]]:
    """
    Create region-specific multi-frequency forcing parameters.
    
    Each region gets slightly different amplitudes and random phases to
    desynchronize dynamics across regions.
    
    Parameters
    ----------
    num_regions : int
        Number of regions.
    base_amplitudes : Sequence[float]
        Base amplitudes for each frequency component.
    periods : Sequence[float]
        Periods for each frequency component (in time units).
    amplitude_noise : float
        Relative noise level for amplitude perturbations (default 0.1 = 10%).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    List[Dict[str, Union[List[float], float]]]
        List of dictionaries, one per region, with keys:
        'amp1', 'phase1', 'period1', 'amp2', 'phase2', 'period2', 'amp3', 'phase3', 'period3'.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    forcing_params = []
    n_components = len(base_amplitudes)
    
    for _ in range(num_regions):
        params = {}
        for i in range(n_components):
            idx = i + 1
            # Perturb amplitude
            perturbed_amp = base_amplitudes[i] * (1.0 + np.random.uniform(-amplitude_noise, amplitude_noise))
            # Random phase
            random_phase = np.random.uniform(0, 2 * math.pi)
            
            params[f'amp{idx}'] = float(perturbed_amp)
            params[f'phase{idx}'] = float(random_phase)
            params[f'period{idx}'] = float(periods[i])
        
        forcing_params.append(params)
    
    return forcing_params


def simulate_spatiotemporal_regions(
    model: CompartmentalModel,
    region_states: Union[torch.Tensor, np.ndarray],
    adjacency_graph: Union[torch.Tensor, np.ndarray, nx.Graph, Callable],
    steps: int,
    dt: float = 1.0,
    forcing_params: Optional[List[Dict[str, float]]] = None,
    forcing_parameter: str = 'beta',
    method: str = 'euler',
    gravity_model: Optional[Gravity_model] = None,
    travel_rate: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Simulate region-level epidemics with mobility-driven population flows.
    
    This function implements a two-step process:
    1. Internal dynamics update (with optional multi-frequency forcing)
    2. Population flow based on mobility model (diffusive or gravity-based)
    
    The flows are applied AFTER internal dynamics. Flow models are configured via
    the gravity_model parameter, which supports both diffusive and gravity-based flows.
    
    Parameters
    ----------
    model : CompartmentalModel
        Compartmental epidemic model (must be SIRS-like for infinite stability).
    region_states : Union[torch.Tensor, np.ndarray]
        Initial states for each region as counts (not fractions), shape (num_regions, num_compartments).
        For SIRS: [S, I, R] counts for each region.
    adjacency_graph : Union[torch.Tensor, np.ndarray, nx.Graph, Callable]
        Connectivity/adjacency graph specification where higher values = stronger connections.
        Edge weights represent connection strength (travel rates, similarity, interaction frequency).
        Can be:
        - Static: Weighted matrix (num_regions, num_regions) or NetworkX graph
        - Time-varying: Tensor of shape (steps+1, num_regions, num_regions)
        - Dynamic: Callable f(t, step_idx) -> adjacency_matrix
    steps : int
        Number of simulation steps.
    dt : float
        Time step size (default 1.0 for daily updates).
    forcing_params : List[Dict[str, float]], optional
        Region-specific forcing parameters with keys:
        'amp1', 'phase1', 'period1', 'amp2', 'phase2', 'period2', 'amp3', 'phase3', 'period3'.
    forcing_parameter : str
        Name of the parameter to apply forcing to (default 'beta').
    method : str
        Integration method ('euler' recommended for stability, default 'euler').
    gravity_model : Gravity_model, optional
        Mobility model instance. If None, defaults to diffusive model.
        - Diffusive: Gravity_model(rho=0, theta=0, delta=1.0, normalize=False)
        - Gravity: Gravity_model(rho=1.0, theta=1.0, delta=100.0, normalize=True)
    travel_rate : float
        Global mobility scaling factor applied to every edge-wise flow (default 1.0).
        Use this to tune the overall level of inter-regional travel without
        modifying the adjacency weights.
    
    Returns
    -------
    dict
        Simulation results with keys:
        - 'time': Time axis (steps+1,)
        - 'trajectory': Regional state history (steps+1, num_regions, num_compartments)
        - 'compartments': Compartment names
        - 'adjacency': Static adjacency or provided tensor (for backward compatibility)
        - 'adjacency_history': Adjacency tensor for every timestep (steps+1, num_regions, num_regions)
        - 'dynamic_graph': Signed flow graphs (steps+1, num_regions, num_regions)
        - 'directed_flow': Non-negative directed flows derived from `dynamic_graph`
        - 'parameter_history': Value of `forcing_parameter` used per region and timestep
        - 'effective_reproduction_number': R_t estimates when applicable
        - 'counts': Aggregate compartment counts summed across regions
        - 'regional_totals': Population of each region at every timestep
        - 'node_features': Alias for trajectory (kept for convenience)
    
    Notes
    -----
    **Flow Models**:
    
    Both models use unified connectivity semantics (higher = stronger connection):
    
    Diffusive (default):
        F(i,j) = connectivity[i,j] × (N_i - N_j)
        
    Gravity:
        F(i,j) = [N_i^ρ × N_j^θ × exp((connectivity[i,j]-1)/δ)] × (N_i - N_j)/(N_i + N_j)
    
    Both ensure population conservation through antisymmetric flows.
    
    **Connectivity Normalization**:
    
    Normalize edge weights to [0, 1] for best results:
    >>> adjacency_normalized = adjacency / adjacency.max()
    
    Examples
    --------
    Diffusive flow (default):
        >>> result = simulate_spatiotemporal_regions(
        ...     model, states, adjacency, steps=100)
    
    Diffusive flow (explicit):
        >>> diffusive = Gravity_model(rho=0, theta=0, delta=1.0, normalize=False)
        >>> result = simulate_spatiotemporal_regions(
        ...     model, states, adjacency, steps=100, gravity_model=diffusive)
    
    Gravity model:
        >>> # Normalize connectivity (edge weights)
        >>> adjacency_norm = adjacency / adjacency.max()
        >>> gravity = Gravity_model(rho=1.0, theta=1.0, delta=0.5, normalize=True)
        >>> result = simulate_spatiotemporal_regions(
        ...     model, states, adjacency_norm, steps=100, gravity_model=gravity)
    """
    # Convert inputs to tensors
    state = torch.as_tensor(region_states, dtype=torch.float32)
    if state.ndim != 2 or state.shape[1] != len(model.compartments):
        raise ValueError(f"region_states must have shape [num_regions, {len(model.compartments)}].")
    
    num_regions = state.shape[0]
    num_compartments = len(model.compartments)
    
    # Determine adjacency type and prepare getter function
    is_dynamic_callable = callable(adjacency_graph)
    is_time_series = False
    
    if is_dynamic_callable:
        # Callable: adjacency_graph(t, step_idx) -> adjacency matrix
        def get_adjacency(t, step_idx):
            adj = adjacency_graph(t, step_idx)
            return _ensure_adjacency_tensor(adj)
        initial_adjacency = get_adjacency(0.0, 0)
    elif isinstance(adjacency_graph, torch.Tensor) and adjacency_graph.ndim == 3:
        # Pre-computed time series: (steps+1, num_regions, num_regions)
        is_time_series = True
        adjacency_series = adjacency_graph
        if adjacency_series.shape[0] != steps + 1:
            raise ValueError(f"Time-varying adjacency must have shape ({steps+1}, num_regions, num_regions).")
        def get_adjacency(t, step_idx):
            return adjacency_series[step_idx]
        initial_adjacency = adjacency_series[0]
    else:
        # Static: convert once
        static_adjacency = _ensure_adjacency_tensor(adjacency_graph)
        def get_adjacency(t, step_idx):
            return static_adjacency
        initial_adjacency = static_adjacency
    
    if initial_adjacency.shape[0] != num_regions:
        raise ValueError("Adjacency graph size does not match the number of regions.")

    travel_rate = float(travel_rate)
    if travel_rate < 0.0:
        raise ValueError("travel_rate must be non-negative.")
    
    # Use default diffusive model if no gravity model provided
    if gravity_model is None:
        gravity_model = Gravity_model(rho=0.0, theta=0.0, delta=1.0, normalize=False)
    
    # Initialize history
    history = torch.zeros(steps + 1, num_regions, num_compartments, dtype=state.dtype)
    history[0] = state
    
    # Track effective flow graphs (for visualization)
    dynamic_graph_history = torch.zeros(steps + 1, num_regions, num_regions, dtype=state.dtype)
    
    # Always record adjacency history for downstream tasks
    adjacency_history = torch.zeros(steps + 1, num_regions, num_regions, dtype=state.dtype)
    adjacency_history[0] = initial_adjacency
    
    # Parameter forcing bookkeeping
    base_param_value = model.parameters.get(forcing_parameter)
    if forcing_params and base_param_value is None:
        raise ValueError(f"forcing_parameter '{forcing_parameter}' not found in model parameters.")
    parameter_history = None
    if base_param_value is not None:
        parameter_history = torch.full(
            (steps + 1, num_regions),
            float(base_param_value),
            dtype=state.dtype,
        )
    
    # Main simulation loop
    for step_idx in range(steps):
        t = (step_idx + 1) * dt  # Time at next step (test_sim.ipynb uses day as step+1)
        
        # Get adjacency for this timestep
        adjacency = get_adjacency(t, step_idx)
        adjacency_history[step_idx + 1] = adjacency
        
        # --- Step 1: Internal dynamics update ---
        next_state = torch.zeros_like(state)
        
        for region_idx in range(num_regions):
            # Apply multi-frequency forcing if provided
            param_override = {}
            current_param_value = None
            if parameter_history is not None:
                current_param_value = parameter_history[step_idx, region_idx].item()
            if forcing_params and region_idx < len(forcing_params):
                params = forcing_params[region_idx]
                forcing_sum = 0.0
                # Support up to 3 frequency components (as in test_sim.ipynb)
                for i in range(1, 4):
                    amp_key = f'amp{i}'
                    phase_key = f'phase{i}'
                    period_key = f'period{i}'
                    if amp_key in params:
                        forcing_sum += params[amp_key] * math.sin(
                            2.0 * math.pi * t / params[period_key] + params[phase_key]
                        )
                
                forced_value = base_param_value * (1.0 + forcing_sum)
                param_override[forcing_parameter] = forced_value
                current_param_value = forced_value
            if parameter_history is not None:
                parameter_history[step_idx + 1, region_idx] = float(current_param_value)
            
            # Integrate internal dynamics
            next_state[region_idx] = model.step(
                state[region_idx],
                t,
                dt,
                method=method,
                parameter_overrides=param_override if param_override else None,
            )
        
        # --- Step 2: Population flow ---
        # Compute total populations
        totals = next_state.sum(dim=1)  # (num_regions,)
        
        # Compute net flows for each region
        net_flow = torch.zeros_like(next_state)
        flow_graph = torch.zeros(num_regions, num_regions, dtype=state.dtype)
        
        # Get edge list from adjacency matrix
        edges = (adjacency > 0).nonzero(as_tuple=False)
        
        # Unified flow computation using gravity model
        for edge_idx in range(edges.shape[0]):
            i, j = edges[edge_idx]
            i, j = int(i), int(j)
            
            if i >= j:  # Process each edge only once
                continue
            
            N_i, N_j = totals[i].item(), totals[j].item()
            distance = adjacency[i, j].item()
            
            if distance <= 0 or N_i <= 0 or N_j <= 0:
                continue
            
            # Compute net flow using gravity model (handles both diffusive and gravity cases)
            F_net = gravity_model.compute_net_flow(N_i, N_j, distance) * travel_rate
            
            if abs(F_net) > 1e-8:
                # Determine source and target based on flow direction
                source, target = (i, j) if F_net > 0 else (j, i)
                F_abs = abs(F_net)
                N_source = totals[source]
                
                # Partition flow by compartment fractions
                frac_source = next_state[source] / N_source if N_source > 1e-8 else torch.zeros(num_compartments)
                compartment_flows = F_abs * frac_source
                
                net_flow[source] -= compartment_flows
                net_flow[target] += compartment_flows
                flow_graph[source, target] = F_abs
                flow_graph[target, source] = -F_abs
        
        # Apply net flows
        next_state = next_state + net_flow
        
        # Store results
        state = next_state
        history[step_idx + 1] = state
        dynamic_graph_history[step_idx + 1] = flow_graph
    
    time_axis = torch.linspace(0.0, steps * dt, steps + 1)
    node_features = history.clone()
    regional_totals = history.sum(dim=2)
    counts = history.sum(dim=1)
    directed_flow = torch.clamp(dynamic_graph_history, min=0.0)
    
    rt_history = None
    susceptible_idx = _get_compartment_index(model, 'S')
    gamma_value = model.parameters.get('gamma')
    if (
        parameter_history is not None
        and gamma_value is not None
        and susceptible_idx is not None
    ):
        susceptible = history[:, :, susceptible_idx]
        totals_safe = regional_totals.clamp(min=1.0)
        beta_history = parameter_history
        rt_history = (beta_history / gamma_value) * (susceptible / totals_safe)
    
    # Return appropriate adjacency representation
    if is_dynamic_callable or is_time_series:
        adjacency_out = adjacency_history
    else:
        adjacency_out = initial_adjacency
    
    return {
        "time": time_axis,
        "trajectory": history,
        "compartments": model.compartments,
        "adjacency": adjacency_out,
        "dynamic_graph": dynamic_graph_history,
        "directed_flow": directed_flow,
        "adjacency_history": adjacency_history,
        "parameter_history": parameter_history,
        "effective_reproduction_number": rt_history,
        "counts": counts,
        "regional_totals": regional_totals,
        "node_features": node_features,
    }
