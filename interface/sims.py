import torch

def edge_to_adj(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj

def get_random_graph(num_nodes=None, connect_prob=None, block_sizes=None, num_edges=None, graph_type='erdos_renyi'):
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