import torch
import numpy as np
import random
import math
import time
import urllib.request

from .utils import edge_to_adj

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
    import torch_geometric
    if graph_type == 'erdos_renyi':
        edges = torch_geometric.utils.erdos_renyi_graph(num_nodes, connect_prob)
    elif graph_type == 'stochastic_blockmodel':
        edges = torch_geometric.utils.stochastic_blockmodel_graph(block_sizes, connect_prob)
    elif graph_type == 'barabasi_albert':
        edges = torch_geometric.utils.barabasi_albert_graph(num_nodes, num_edges)
    else:
        raise NameError("grap type not supported")
    
    adj = edge_to_adj(edges, num_nodes)
    return adj

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

class Gravity_model(object):
    """
    A class to represent a gravity model for mobility analysis in an epidemic context.

    The gravity model is used to estimate the influence and mobility between nodes (e.g., cities or regions)
    based on their populations and the distance between them, according to the formula:

    .. math::

        e_{v,w} = \\frac{N_v^{\\rho} N_w^{\\theta}}{\\exp(d_{vw} / \\delta)},\\forall v, w \\in \\mathcal{V}
        
        
    """
    def __init__(self, source, target, s):
        """
        Initialize the Gravity_model with specified parameters.

        Parameters
        ----------
        source : float
            Exponent for the source node population :math:`\\rho` in the gravity model.
        target : float
            Exponent for the target node population :math:`\\theta` in the gravity model.
        s : float
            Scaling factor for the distance :math:`\\delta` in the gravity model.
        """
        self.source = source
        self.target = target
        self.s = s
    
    def get_influence(self, source_population, target_population, distance):
        """
        Calculate the influence between a source and a target node based on their populations and the distance between them.

        Parameters
        ----------
        source_population : float
            Population of the source node :math:`N_v`.
        target_population : float
            Population of the target node :math:`N_w`.
        distance : float
            Distance between the source and target nodes :math:`d_{vw}`.

        Returns
        -------
        float
            The calculated influence between the source and target nodes.
        """
        return (source_population**self.source * target_population**self.target) / torch.exp(distance / self.s)

    def get_mobility_graph(self, node_populations, distance_graph):
        """
        Generate a mobility graph based on node populations and a distance graph.

        Parameters
        ----------
        node_populations : torch.Tensor
            A tensor of shape (num_nodes,) representing the population of each node.
        distance_graph : torch.Tensor
            A tensor of shape (num_nodes, num_nodes) representing the distances between nodes.

        Returns
        -------
        torch.Tensor
            A tensor of shape (num_nodes, num_nodes) representing the mobility graph, where each entry [i, j]
            represents the mobility influence from node i to node j.
        """
        num_nodes = node_populations.shape[0]
        mobility_graph = torch.zeros(num_nodes, num_nodes)

        for source in range(num_nodes):
            for target in range(num_nodes):
                mobility_graph[source, target] = (node_populations[source]**self.source * node_populations[target]**self.target) / torch.exp(distance_graph[source, target] / self.s)
        return mobility_graph
        

class Time_geo(object):
    """
    Models spatial-temporal movement patterns of individuals across different regions, simulating movement based on
    temporal rhythms, personal habits, and regional exploration probabilities. It includes mechanisms to simulate
    individual trajectories, predict location changes, and calculate movement probabilities based on historical data.

    Parameters
    ----------
    region_input : ndarray
        Input array of region coordinates.
    pop_input : ndarray
        Population distribution across regions.
    p_t_raw : ndarray, optional
        Raw probability distribution of time slots for movement. Default: Loads from "epilearn/data/rhythm.npy".
    pop_num : int, optional
        Number of individuals in the population to simulate. Default: 7.
    time_slot : int, optional
        Time resolution in minutes for one slot. Default: 10.
    rho : float, optional
        Controls the exploration probability for other regions. Default: 0.6.
    gamma : float, optional
        Attenuation parameter for exploration probability. Default: 0.41.
    alpha : float, optional
        Controls the exploration depth. Default: 1.86.
    n_w : float, optional
        Average number of tours based on home per week. Default: 6.1.
    beta1 : float, optional
        Dwell rate. Default: 3.67.
    beta2 : float, optional
        Burst rate, affecting movement speed. Default: 10.
    simu_slot : int, optional
        Total number of simulation slots. Default: 144.

    Returns
    -------
    list of dicts
        Each dictionary contains the movement trace and other metrics for an individual simulated over the simu_slot duration.
    """
    def __init__(self, region_input, pop_input, p_t_raw=None, pop_num=7, time_slot=10, rho=0.6, gamma=0.41, alpha=1.86, n_w=6.1, beta1=3.67, beta2=10, simu_slot=144):
        
        super().__init__()
        self.time_slot = time_slot # time resolution is half an hour
        self.rho = rho # it controls the exploration probability for other regions
        self.gamma = gamma # it is the attenuation parameter for exploration probability
        self.alpha = alpha # it controls the exploration depth
        self.n_w = n_w # it is the average number of tour based on home a week.
        self.beta1 = beta1 # dwell rate
        self.beta2 = beta2 # burst rate
        self.simu_slot = simu_slot
        self.pop_num = pop_num

        self.sample_region = region_input

        url_feature = "https://drive.google.com/uc?export=download&id=10VfZI-uJ-pXMxg10ETRCfw0wDD7f2VZ8"
        urllib.request.urlretrieve(url_feature, './datasets/rhythm.npy')

        p_t_raw = p_t_raw if p_t_raw is not None else np.load("./datasets/rhythm.npy", allow_pickle=True)        
        self.p_t = np.array(p_t_raw).reshape(-1, (time_slot // 10)).sum(axis=1)
        self.region_num = self.sample_region.shape[0]
        self.home_location = np.random.choice(len(pop_input), pop_num, p=pop_input)
        self.pop_info = self.trace_simulate()

    def distance(self, p1, p2):#caculate distance
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def get_p_t(self, now_time):
        now_time_tup = time.localtime(float(now_time))
        i = int((now_time_tup.tm_wday * 24 * 60 + now_time_tup.tm_hour * 60 + now_time_tup.tm_min) / self.time_slot)
        return self.p_t[i]
    
    def predict_next_place_time(self, p_t_value, current_location_type):
        p1 = 1 - self.n_w * p_t_value
        p2 = 1 - self.beta1 * self.n_w * p_t_value
        p3 = self.beta2 * self.n_w * p_t_value
        location_is_change = 0
        new_location_type = 'undefined'
        if current_location_type == 'home':
            if random.uniform(0, 1) <= p1:
                new_location_type = 'home'
                location_is_change = 0
            else:
                new_location_type = 'other'
                location_is_change = 1
        elif current_location_type == 'other':
            p = random.uniform(0, 1)
            if p <= p2:
                new_location_type = 'other'
                location_is_change = 0
            elif random.uniform(0, 1) <= p3:
                new_location_type = 'other'
                location_is_change = 1
            else:
                new_location_type = 'home'
                location_is_change = 1
        if new_location_type == 'home':
            return 0, location_is_change
        else:
            return 2, location_is_change
        
    def negative_pow(self, k):
        p_k = {}
        for i, region in enumerate(k, 1):
            p_k[region[0]] = i ** (-self.alpha)
        temp = sum(p_k.values())
        for key in p_k:
            p_k[key] = p_k[key] / temp
        return p_k

    def predict_next_place_location_simplify(self, P_new, region_history, current_region, home_region):
        rp = random.uniform(0, 1)
        prob_accum, next_region = 0, 0
        if random.uniform(0, 1) < P_new:
            # explore; the explore distance is depend on history->delta_r
            length = {}
            for i, cen in enumerate(self.sample_region):
                if i in region_history:
                    continue
                length[i] = self.distance(cen, self.sample_region[current_region])
            try:
                del length[home_region]
                del length[current_region]
            except KeyError:
                pass
            k = sorted(length.items(), key=lambda x: x[1], reverse=False)
            p_k = self.negative_pow(k)
            for i, key in enumerate(p_k):
                prob_accum += p_k[key]
                if prob_accum > rp:
                    next_region = key
                    region_history[key] = 1
                    break
                else:
                    continue
        else:
            # return
            region_history_sum = sum(region_history.values())
            for key in region_history:
                prob_accum += region_history[key]/region_history_sum
                if rp < prob_accum:
                    next_region = key
                    region_history[key] += 1
                    break
        return next_region

    def predict_next_place_location(self, region_history, current_location, home_region):
        s = len(region_history.values())
        p_new = self.rho * s ** (-self.gamma) if s != 0 else 1
        return self.predict_next_place_location_simplify(p_new, region_history, current_location, home_region)

    def individual_trace_simulate(self, info, start_time, simu_slot):
        current_location_type = 'home'
        simu_trace = [[info['home'],start_time]]
        for i in range(simu_slot - 1):
            # pt is the currently move based probability
            now_time = (i+1) * 60 * self.time_slot + start_time
            p_t_value = self.get_p_t(now_time)
            now_type, location_change = self.predict_next_place_time(p_t_value, current_location_type)
            if location_change == 1:
                current_location = simu_trace[-1][0]
                if now_type == 0:
                    next_location = info['home']
                    current_location_type = 'home'
                else:
                    next_location = self.predict_next_place_location(info['region_history'], current_location, info['home'])
                    current_location_type = 'other'
                info['feature']['move_num'] += 1
                info['feature']['move_distance'] += self.distance(self.sample_region[next_location], self.sample_region[current_location])
            else:
                next_location = simu_trace[-1][0]
            simu_trace.append([next_location, now_time])
        return simu_trace
    
    def trace_simulate(self):
        """
        Generates simulated movement traces for each individual in the population. The simulation captures
        the dynamics of movement based on initial conditions and predefined movement parameters over a set
        number of time slots.

        Returns
        -------
        list of dicts
            Each dictionary contains detailed information and metrics for an individual's simulated trace, including
            home location, total number of movements, cumulative distance traveled, and the sequence of visited
            locations across the simulation period.
        """
        pop_info = []
        for i in range(self.pop_num):
            pop_info.append({'n_w': self.n_w, 'beta1': self.beta1, 'beta2': self.beta2, 'home': self.home_location[i],
                            'feature': {'move_num': 0, 'move_distance': 0}, 'region_history': {}})
            pop_info[i]['trace'] = np.array(self.individual_trace_simulate(pop_info[i], 1621785600, self.simu_slot))
        return pop_info
