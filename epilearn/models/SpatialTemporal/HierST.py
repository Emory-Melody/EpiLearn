# # this code is adopted from https://github.com/dolphin-zs/HierST/blob/main/src/hierst.py

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch_geometric.nn as PyG

# from .base import BaseModel


# #--------------------------------For Temporal Module: NBeats------------------------------

# class Block(nn.Module):

#     def __init__(self, hidden_channels, thetas_dim, backcast_length=10, forecast_length=5,share_thetas=False,
#                  nb_harmonics=None):
#         super(Block, self).__init__()
#         self.hidden_channels = hidden_channels
#         self.thetas_dim = thetas_dim
#         self.backcast_length = backcast_length
#         self.forecast_length = forecast_length
#         self.share_thetas = share_thetas

#         # (batch_size, hidden_channels, backcast_length)
#         self.conv1d_1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)   # zero_padding to align length
#         self.conv1d_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)
#         self.conv1d_3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)
#         self.conv1d_4 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)

#         # (batch_size, hidden_channels, thetas_dim)
#         if share_thetas:
#             self.theta_f_fc = self.theta_b_fc = nn.Linear(backcast_length, thetas_dim)
#         else:
#             self.theta_b_fc = nn.Linear(backcast_length, thetas_dim)
#             self.theta_f_fc = nn.Linear(backcast_length, thetas_dim)


#         self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)

#     def forward(self, x):
#         x = F.relu(self.conv1d_1(x))
#         x = F.relu(self.conv1d_2(x))
#         x = F.relu(self.conv1d_3(x))
#         x = F.relu(self.conv1d_4(x))
#         return x

# def seasonality_model(thetas, t):
#     p = thetas.size()[-1]
#     assert p <= thetas.shape[1], 'thetas_dim is too big.'
#     p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
#     s1 = torch.FloatTensor([np.cos(2 * np.pi * i * t) for i in range(p1)])  # H/2-1
#     s2 = torch.FloatTensor([np.sin(2 * np.pi * i * t) for i in range(p2)])
#     S = torch.cat([s1, s2]).to(thetas.get_device())
#     return thetas.matmul(S)


# def trend_model(thetas, t):
#     p = thetas.size()[-1]
#     assert p <= 4, 'thetas_dim is too big.'
#     T = torch.FloatTensor([t ** i for i in range(p)]).to(thetas.get_device())
#     return thetas.matmul(T)


# def linspace(backcast_length, forecast_length):
#     lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length, endpoint=False)
#     b_ls = lin_space[:backcast_length] / (backcast_length + forecast_length)
#     f_ls = lin_space[backcast_length:] / (backcast_length + forecast_length)
#     return b_ls, f_ls

# class SeasonalityBlock(Block):

#     def __init__(self, hidden_channels, thetas_dim, backcast_length=10, forecast_length=5):
#         super(SeasonalityBlock, self).__init__(hidden_channels, forecast_length, backcast_length,
#                                                    forecast_length, share_thetas=True)

#         self.forecast_pool = nn.AdaptiveMaxPool1d(1)
#         self.forecast_map = nn.Linear(hidden_channels, forecast_length)

#     def forward(self, x):
#         x = super(SeasonalityBlock, self).forward(x)
#         backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace)   # (batch_size, hidden_channels, backcast_length)
#         forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace)   # (batch_size, hidden_channels, forecast_length)
#         forecast = self.forecast_map(self.forecast_pool(forecast).squeeze())       # (batch_size, forecast_length)

#         return backcast, forecast


# class TrendBlock(Block):

#     def __init__(self, hidden_channels, thetas_dim,   backcast_length=10, forecast_length=5):
#         super(TrendBlock, self).__init__(hidden_channels, thetas_dim, backcast_length,
#                                          forecast_length, share_thetas=True)
#         self.forecast_pool = nn.AdaptiveMaxPool1d(1)
#         self.forecast_map = nn.Linear(hidden_channels, forecast_length)

#     def forward(self, x):
#         x = super(TrendBlock, self).forward(x)
#         backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace)
#         forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace)
#         forecast = self.forecast_map(self.forecast_pool(forecast).squeeze())       # (batch_size, forecast_length)

#         return backcast, forecast


# class GenericBlock(Block):

#     def __init__(self, hidden_channels, thetas_dim,  backcast_length=10, forecast_length=5):
#         super(GenericBlock, self).__init__(hidden_channels, thetas_dim, backcast_length, forecast_length)

#         self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
#         self.forecast_fc = nn.Linear(thetas_dim, forecast_length)
#         self.forecast_pool = nn.AdaptiveMaxPool1d(1)
#         self.forecast_map = nn.Linear(hidden_channels, forecast_length)


#     def forward(self, x):
#         x = super(GenericBlock, self).forward(x)

#         theta_b = F.relu(self.theta_b_fc(x))
#         theta_f = F.relu(self.theta_f_fc(x))

#         backcast = self.backcast_fc(theta_b)
#         forecast = self.forecast_map(self.forecast_pool(F.relu(self.forecast_fc(theta_f))).squeeze())

#         return backcast, forecast


# class NBeatsNet(nn.Module):

#     SEASONALITY_BLOCK = 'seasonality'
#     TREND_BLOCK = 'trend'
#     GENERIC_BLOCK = 'generic'

#     def __init__(self,
#                  input_channels = 1,
#                  stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
#                  nb_blocks_per_stack=3,
#                  forecast_length=14,
#                  backcast_length=1,
#                  thetas_dims=(4, 8),
#                  share_weights_in_stack=False,
#                  hidden_channels=64):
#         super(NBeatsNet, self).__init__()
#         self.input_channels = input_channels
#         self.forecast_length = forecast_length
#         self.backcast_length = backcast_length
#         self.hidden_channels = hidden_channels
#         self.nb_blocks_per_stack = nb_blocks_per_stack
#         self.share_weights_in_stack = share_weights_in_stack
#         self.stack_types = stack_types
#         self.stacks = []
#         self.thetas_dim = thetas_dims
#         self.parameters = []
#         self.conv1d_flatten = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)
#         self.parameters.extend(self.conv1d_flatten.parameters())
#         # print(f'| N-Beats')
#         # print(f'     | -- {self.conv1d_flatten}')
#         for stack_id in range(len(self.stack_types)):
#             self.stacks.append(self.create_stack(stack_id))
#         self.parameters = nn.ParameterList(self.parameters)

#     def create_stack(self, stack_id):
#         stack_type = self.stack_types[stack_id]
#         # print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
#         blocks = []
#         for block_id in range(self.nb_blocks_per_stack):
#             block_init = NBeatsNet.select_block(stack_type)
#             if self.share_weights_in_stack and block_id != 0:
#                 block = blocks[-1]  # pick up the last one when we share weights.
#             else:
#                 block = block_init(self.hidden_channels, self.thetas_dim[stack_id],
#                                    self.backcast_length, self.forecast_length)
#                 self.parameters.extend(block.parameters())
#             # print(f'     | -- {block}')
#             blocks.append(block)
#         return blocks

#     @staticmethod
#     def select_block(block_type):
#         if block_type == NBeatsNet.SEASONALITY_BLOCK:
#             return SeasonalityBlock
#         elif block_type == NBeatsNet.TREND_BLOCK:
#             return TrendBlock
#         else:
#             return GenericBlock

#     def forward(self, backcast):
#         forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length)).to(backcast.get_device())  # maybe batch size here.
#         backcast = self.conv1d_flatten(backcast)     # (batch_size, input_channels, backcast_length) --> (batch_size, hidden_channels, backcast_length)
#         for stack_id in range(len(self.stacks)):
#             for block_id in range(len(self.stacks[stack_id])):
#                 b, f = self.stacks[stack_id][block_id](backcast)
#                 backcast = backcast - b
#                 forecast = forecast + f
#         return backcast, forecast

# class NBeatsEncoder(nn.Module):
#     def __init__(self, num_nodes, date_emb_dim, id_emb_dim, lookahead_days, lookback_days, hidden_dim, day_fea_dim, block_size):
#         super().__init__()
#         self.week_em = nn.Embedding(7, date_emb_dim)
#         self.id_em = nn.Embedding(num_nodes, id_emb_dim)
#         self.lookahead_days = lookahead_days
#         self.lookback_days = lookback_days
#         self.hidden_dim = hidden_dim

#         day_input_dim = day_fea_dim - 1 + self.week_em.embedding_dim + self.id_em.embedding_dim

#         self.day_n_beats = NBeatsNet(
#             input_channels=day_input_dim,
#             stack_types=('generic','generic'),
#             nb_blocks_per_stack=block_size,
#             forecast_length=lookahead_days,
#             backcast_length=lookback_days,
#             thetas_dims=(4,8),
#             share_weights_in_stack=True,
#             hidden_channels=hidden_dim)


#     def add_date_embed(self, input_day):
#         # last 1 dims correspond to weekday
#         x = input_day[:, :, :, :-1]
#         weekday = self.week_em(input_day[:, :, :, -1].long())
#         # return torch.cat([x, month, day, weekday], dim=-1)
#         return torch.cat([x, weekday], dim=-1)

#     def add_id_embed(self, input, g):
#         # add cent id index
#         sz = input.size()
#         cent_id = self.id_em(g['cent_n_id'].reshape(1,sz[1],1).expand(sz[0],sz[1],sz[2]).long())
#         return torch.cat([input, cent_id], dim=-1)

#     def permute_feature(self, x):
#         sz = x.size()
#         x = x.view(-1, sz[-2], sz[-1])
#         x = x.permute(0, 2, 1)
#         return x

#     def forward(self, input_day, g):
#         # [batch_size, node_num, seq_len, fea_dim]
#         sz = input_day.size()
#         # forecast_lr = self.linear_reg(input_day)
#         input_day = self.add_date_embed(input_day)
#         input_day = self.add_id_embed(input_day, g)
#         input_day = self.permute_feature(input_day)

#         backcast_day, forecast_day = self.day_n_beats(input_day)

#         backcast_day = backcast_day.view(sz[0], sz[1], self.hidden_dim, self.lookback_days)
#         forecast_day = forecast_day.squeeze().view(sz[0], sz[1], self.lookahead_days)

#         return backcast_day, forecast_day



# #------------------------------For Spatial Module----------------------


# class BaseGNNNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def dataflow_forward(self, X, g):
#         raise NotImplementedError

#     def subgraph_forward(self, X, g):
#         raise NotImplementedError

#     def forward(self, X, g, **kwargs):
#         if g['type'] == 'dataflow':
#             return self.dataflow_forward(X, g, **kwargs)
#         elif g['type'] == 'subgraph':
#             return self.subgraph_forward(X, g, **kwargs)
#         else:
#             raise Exception('Unsupported graph type {}'.format(g['type']))
        


# class GCNConv(PyG.MessagePassing):
#     def __init__(self, gcn_in_dim, config, **kwargs):
#         super().__init__(aggr=config.gcn_aggr, node_dim=-2, **kwargs)

#         self.gcn_in_dim = gcn_in_dim
#         self.gcn_node_dim = config.gcn_node_dim
#         self.gcn_dim = config.gcn_dim

#         self.fea_map = nn.Linear(self.gcn_in_dim, self.gcn_dim)

#     def forward(self, x, edge_index, edge_weight):
#         batch_size, num_nodes, _ = x.shape
#         if edge_weight.shape[0] == x.shape[0]:
#             num_edges = edge_weight.shape[1]
#             edge_weight = edge_weight.reshape(batch_size, num_edges, 1)
#         else:
#             num_edges = edge_weight.shape[0]
#             edge_weight = edge_weight.reshape(1, num_edges, 1).expand(batch_size, -1, -1)

#         # Calculate type-aware node info
#         if isinstance(x, torch.Tensor):
#             x = self.fea_map(x)
#         else:
#             x = (self.fea_map(x[0]), self.fea_map(x[1]))

#         return self.propagate(edge_index, x=x, edge_weight=edge_weight)

#     def message(self, x_j, edge_weight):
#         return x_j * edge_weight

#     def update(self, aggr_out, x):
#         if not isinstance(x, torch.Tensor):
#             x = x[1]

#         return x + aggr_out


# class MyGATConv(PyG.MessagePassing):
#     def __init__(self, gcn_in_dim, config, **kwargs):
#         super().__init__(aggr=config.gcn_aggr, node_dim=-2, **kwargs)

#         self.gcn_in_dim = gcn_in_dim
#         self.gcn_node_dim = config.gcn_node_dim
#         self.gcn_dim = config.gcn_dim

#         self.fea_map = nn.Linear(self.gcn_in_dim, self.gcn_dim)
#         self.msg_map = nn.Linear(self.gcn_dim, self.gcn_dim)

#     def forward(self, x, edge_index, edge_weight):
#         batch_size, num_nodes, _ = x.shape
#         if edge_weight.shape[0] == x.shape[0]:
#             num_edges = edge_weight.shape[1]
#             edge_weight = edge_weight.reshape(batch_size, num_edges, 1)
#         else:
#             num_edges = edge_weight.shape[0]
#             edge_weight = edge_weight.reshape(1, num_edges, 1).expand(batch_size, -1, -1)

#         # Calculate type-aware node info
#         if isinstance(x, torch.Tensor):
#             x = self.fea_map(x)
#         else:
#             x = (self.fea_map(x[0]), self.fea_map(x[1]))

#         return self.propagate(edge_index, x=x, edge_weight=edge_weight)

#     def message(self, x_i, x_j, edge_weight):
#         x_i = self.msg_map(x_i)
#         x_j = self.msg_map(x_j)
#         gate = torch.sigmoid((x_i * x_j).sum(dim=-1, keepdim=True))
#         return x_j * edge_weight * gate

#     def update(self, aggr_out, x):
#         if not isinstance(x, torch.Tensor):
#             x = x[1]

#         return x + aggr_out

# class NodeGNN(BaseGNNNet):
#     def __init__(self, gcn_type, gcn_in_dim, gcn_layer_num):
#         super().__init__()

#         self.layer_num = gcn_layer_num
#         assert self.layer_num >= 1

#         if gcn_type == 'gcn':
#             GCNClass = GCNConv
#         elif gcn_type == 'gat':
#             GCNClass = MyGATConv
#         else:
#             raise Exception(f'Unsupported gcn_type {gcn_type}')

#         convs = [GCNClass(gcn_in_dim, )]
#         for _ in range(self.layer_num-1):
#             convs.append(GCNClass(.gcn_dim, ))
#         self.convs = nn.ModuleList(convs)

#     def subgraph_forward(self, x, g, edge_weight=None):
#         edge_index = g['edge_index']
#         if edge_weight is None:
#             # edge_weight in arguments has the highest priority
#             edge_weight = g['edge_attr']

#         for conv in self.convs:
#             # conv already implements the residual connection
#             x = conv(x, edge_index, edge_weight)

#         return x


# class EdgeGNN(NodeGNN):
#     def __init__(self, num_nodes, gcn_in_dim, gcn_dim, gcn_node_dim):
#         super().__init__(gcn_in_dim, )

#         self.gcn_dim = gcn_dim
#         self.num_nodes = num_nodes
#         self.node_dim = gcn_node_dim
#         self.edge_dim = 2*(self.gcn_dim+self.node_dim)

#         self.node_emb = nn.Embedding(self.num_nodes, self.node_dim)
#         self.edge_map = nn.Sequential(
#             nn.Linear(self.edge_dim, 1),
#             nn.ReLU(),
#         )

#     def subgraph_forward(self, x, g):
#         x = super().subgraph_forward(x, g)
#         batch_size, node_num, _ = x.shape

#         # add node-specific representations
#         n_id = g['cent_n_id']
#         x_id = self.node_emb(n_id)\
#             .reshape(1, node_num, self.node_dim)\
#             .expand(batch_size, -1, -1)
#         x = torch.cat([x, x_id], dim=-1)

#         # calculate the edge gate for each node pair
#         edge_index = g['edge_index'].permute(1, 0)  # [num_edges, 2]
#         edge_num = edge_index.shape[0]
#         edge_x = x[:, edge_index, :]\
#             .reshape(batch_size, edge_num, self.edge_dim)
#         edge_gate = self.edge_map(edge_x)

#         return edge_gate



# #-------------------------------Main Model----------------------


# class HierST(BaseModel):
#     def __init__(self, 
#                 num_nodes=None,
#                 num_features=None,
#                 num_timesteps_input=None,
#                 num_timesteps_output=None,
#                 rnn_model='nbeats',
#                 rnn_nhid = 32,
#                 use_default_edge=False,
#                 gcn_hid=32,
#                 device = 'cpu'):
#         super().__init__()
#         self.device = device
#         self.m = num_nodes
#         self.nfeat = num_features
#         self.w = num_timesteps_input
#         self.h = num_timesteps_output


#         self.rnn_type = rnn_model
#         if self.rnn_type == 'nbeats':
#             self.rnn = NBeatsEncoder(num_nodes, date_emb_dim, id_emb_dim, lookahead_days, lookback_days, hidden_dim, day_fea_dim, block_size)
#             self.rnn_hid_dim = rnn_nhid
#         else:
#             raise Exception(f'Unsupported rnn type {self.rnn_type}')

#         if use_default_edge:
#             self.edge_gnn = None
#         else:
#             self.edge_gnn = EdgeGNN(num_nodes, self.rnn_hid_dim, gcn_dim, gcn_node_dim)
#         self.node_gnn = NodeGNN(self.rnn_hid_dim, )
#         self.gnn_fc = nn.Linear(gcn_hid, )

#         self.edge_gate = None
#         self.y_t = None
#         self.y_g = None

#     def forward(self, input_day, g):
#         # rnn_out.size: [batch_size, node_num, hidden_dim]
#         # y_rnn.size: [batch_size, node_num, forecast_len]
#         if self.rnn_type == 'nbeats':
#             # nb_out.size: [batch_size, node_num, hidden_dim, seq_len]
#             nb_out, self.y_t = self.rnn(input_day, g)
#             rnn_out, _ = nb_out.max(dim=-1)
#         else:
#             raise Exception(f'Unsupported rnn type {self.rnn_type}')

#         # edge_gate.size: [batch_size, edge_num, 1]
#         # gcn_out.size: [batch_size, node_num, hidden_dim]
#         # y_gcn.size: [batch_size, node_num, forecast_len]
#         if self.edge_gnn is None:
#             gcn_out = self.node_gnn(rnn_out, g)
#         else:
#             self.edge_gate = self.edge_gnn(rnn_out, g)
#             gcn_out = self.node_gnn(rnn_out, g, edge_weight=self.edge_gate)
#         self.y_g = self.gnn_fc(gcn_out)

#         y = self.y_t + self.y_g

#         return y