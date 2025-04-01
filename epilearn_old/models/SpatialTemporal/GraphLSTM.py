# # this code is adopted from 

# import torch
# import torch.nn as nn
# from torch_geometric.nn.inits import glorot, zeros
# from typing import Union, Tuple
# from torch_geometric.typing import OptPairTensor, Adj, Size

# from torch import Tensor, randn
# from torch.nn import Linear, Parameter
# import torch.nn.functional as F
# from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn.conv import MessagePassing
# from torch import matmul

# from .base import BaseModel


# class WeightedSAGEConv(MessagePassing):
#     r"""The GraphSAGE operator from the `"Inductive Representation Learning on
#     Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

#     .. math::
#         \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
#         \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

#     Args:
#         in_channels (int or tuple): Size of each input sample. A tuple
#             corresponds to the sizes of source and target dimensionalities.
#         out_channels (int): Size of each output sample.
#         normalize (bool, optional): If set to :obj:`True`, output features
#             will be :math:`\ell_2`-normalized, *i.e.*,
#             :math:`\frac{\mathbf{x}^{\prime}_i}
#             {\| \mathbf{x}^{\prime}_i \|_2}`.
#             (default: :obj:`False`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(self, in_channels: Union[int, Tuple[int, int]],
#                  out_channels: int, normalize: bool = False,
#                  bias: bool = True, weighted: bool = True, edge_count: int = 423, **kwargs):  # yapf: disable
#         super(WeightedSAGEConv, self).__init__(aggr='mean', **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.normalize = normalize
#         self.weighted = weighted

#         if isinstance(in_channels, int):
#             in_channels = (in_channels, in_channels)

#         self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
#         self.lin_r = Linear(in_channels[1], out_channels, bias=False)

#         self.lin_m = Linear(in_channels[0], in_channels[0], bias=bias)
#         self.lin_ew = Linear(1, 1, bias=bias)

#         self.lin_edge_feature_weighting = Linear(3, 1, bias=bias)

#         self.edge_attr = Parameter(randn(edge_count, 1))

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_l.reset_parameters()
#         self.lin_r.reset_parameters()

#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: Tensor = None,
#                 size: Size = None) -> Tensor:

#         if isinstance(x, Tensor):
#             x: OptPairTensor = (x, x)

#         # propagate_type: (x: OptPairTensor)
#         out = self.propagate(edge_index=edge_index, size=size)
#         out = self.lin_l(out)

#         x_r = x[1]
#         if x_r is not None:
#             out += self.lin_r(x_r)

#         if self.normalize:
#             out = F.normalize(out, p=2., dim=-1)

#         return out

#     def message(self, x_i: Tensor, x_j: Tensor, edge_weight) -> Tensor:
#         # reduced_weight = self.lin_edge_feature_weighting(edge_weight)
#         return x_i * edge_weight # self.lin_edge_feature_weighting(edge_weight)
#         if not self.weighted:
#             return x_j
#         out = self.lin_m(x_j - x_i)
#         weight = self.lin_ew(self.edge_attr)
#         return out if self.edge_attr is None else out * weight

#     def message_and_aggregate(self, adj_t: SparseTensor,
#                               x: OptPairTensor) -> Tensor:
#         adj_t = adj_t.set_value(None, layout=None)
#         return matmul(adj_t, x[0], reduce=self.aggr)

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)

# class LSTM(torch.nn.Module):
#     # This is an adaptation of torch_geometric_temporal.nn.GConvLSTM, with ChebConv replaced by the given model.
#     """
#     Args:
#         in_channels (int): Number of input features.
#         out_channels (int): Number of output features.
#         normalization (str, optional): The normalization scheme for the graph
#             Laplacian (default: :obj:`"sym"`):

#             1. :obj:`None`: No normalization
#             :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

#             2. :obj:`"sym"`: Symmetric normalization
#             :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
#             \mathbf{D}^{-1/2}`

#             3. :obj:`"rw"`: Random-walk normalization
#             :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

#             You need to pass :obj:`lambda_max` to the :meth:`forward` method of
#             this operator in case the normalization is non-symmetric.
#             :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
#             :obj:`[num_graphs]` in a mini-batch scenario and a
#             scalar/zero-dimensional tensor when operating on single graphs.
#             You can pre-compute :obj:`lambda_max` via the
#             :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         module (torch.nn.Module, optional): The layer or set of layers used to calculate each gate.
#             Could also be a lambda function returning a torch.nn.Module when given the parameters in_channels: int, out_channels: int, and bias: bool
#     """
#     def __init__(self, in_channels: int, out_channels: int, bias: bool=True, module=WeightedSAGEConv):
#         super(LSTM, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.bias = bias
#         self.module = module
#         self._create_parameters_and_layers()
#         self._set_parameters()


#     def _create_input_gate_parameters_and_layers(self):

#         self.conv_x_i = self.module(in_features=self.in_channels,
#                                  out_features=self.out_channels,
#                                  bias=self.bias)

#         self.conv_h_i = self.module(in_features=self.out_channels,
#                                  out_features=self.out_channels,
#                                  bias=self.bias)

#         self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_i = Parameter(torch.Tensor(1, self.out_channels))


#     def _create_forget_gate_parameters_and_layers(self):

#         self.conv_x_f = self.module(in_features=self.in_channels,
#                                  out_features=self.out_channels,
#                                  bias=self.bias)

#         self.conv_h_f = self.module(in_features=self.out_channels,
#                                  out_features=self.out_channels,
#                                  bias=self.bias)

#         self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_f = Parameter(torch.Tensor(1, self.out_channels))


#     def _create_cell_state_parameters_and_layers(self):

#         self.conv_x_c = self.module(in_features=self.in_channels,
#                                  out_features=self.out_channels,
#                                  bias=self.bias)

#         self.conv_h_c = self.module(in_features=self.out_channels,
#                                  out_features=self.out_channels,
#                                  bias=self.bias)

#         self.b_c = Parameter(torch.Tensor(1, self.out_channels))


#     def _create_output_gate_parameters_and_layers(self):

#         self.conv_x_o = self.module(in_features=self.in_channels,
#                                  out_features=self.out_channels,
#                                  bias=self.bias)

#         self.conv_h_o = self.module(in_features=self.out_channels,
#                                  out_features=self.out_channels,
#                                  bias=self.bias)

#         self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
#         self.b_o = Parameter(torch.Tensor(1, self.out_channels))


#     def _create_parameters_and_layers(self):
#         self._create_input_gate_parameters_and_layers()
#         self._create_forget_gate_parameters_and_layers()
#         self._create_cell_state_parameters_and_layers()
#         self._create_output_gate_parameters_and_layers()


#     def _set_parameters(self):
#         glorot(self.w_c_i)
#         glorot(self.w_c_f)
#         glorot(self.w_c_o)
#         zeros(self.b_i)
#         zeros(self.b_f)
#         zeros(self.b_c)
#         zeros(self.b_o)


#     def _set_hidden_state(self, X, H):
#         if H is None:
#             H = torch.zeros(X.shape[0], self.out_channels)
#         return H


#     def _set_cell_state(self, X, C):
#         if C is None:
#             C = torch.zeros(X.shape[0], self.out_channels)
#         return C


#     def _calculate_input_gate(self, X, edge_index, edge_weight, H, C):
#         I = self.conv_x_i(X, edge_index, edge_weight)
#         I = I + self.conv_h_i(H, edge_index, edge_weight)
#         I = I + (self.w_c_i*C)
#         I = I + self.b_i
#         I = torch.sigmoid(I)
#         return I


#     def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C):
#         F = self.conv_x_f(X, edge_index, edge_weight)
#         F = F + self.conv_h_f(H, edge_index, edge_weight)
#         F = F + (self.w_c_f*C)
#         F = F + self.b_f
#         F = torch.sigmoid(F)
#         return F


#     def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
#         T = self.conv_x_c(X, edge_index, edge_weight)
#         T = T + self.conv_h_c(H, edge_index, edge_weight)
#         T = T + self.b_c
#         T = torch.tanh(T)
#         C = F*C + I*T
#         return C

#     def _calculate_output_gate(self, X, edge_index, edge_weight, H, C):
#         O = self.conv_x_o(X, edge_index, edge_weight)
#         O = O + self.conv_h_o(H, edge_index, edge_weight)
#         O = O + (self.w_c_o*C)
#         O = O + self.b_o
#         O = torch.sigmoid(O)
#         return O


#     def _calculate_hidden_state(self, O, C):
#         H = O * torch.tanh(C)
#         return H


#     def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None,
#                 H: torch.FloatTensor=None, C: torch.FloatTensor=None) -> torch.FloatTensor:
#         """
#         Making a forward pass. If edge weights are not present the forward pass
#         defaults to an unweighted graph. If the hidden state and cell state
#         matrices are not present when the forward pass is called these are
#         initialized with zeros.

#         Arg types:
#             * **X** *(PyTorch Float Tensor)* - Node features.
#             * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
#             * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
#             * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
#             * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

#         Return types:
#             * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
#             * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
#         """
#         H = self._set_hidden_state(X, H)
#         C = self._set_cell_state(X, C)
#         I = self._calculate_input_gate(X, edge_index, edge_weight, H, C)
#         F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C)
#         C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F)
#         O = self._calculate_output_gate(X, edge_index, edge_weight, H, C)
#         H = self._calculate_hidden_state(O, C)
#         return H, C


# class GraphLinear(torch.nn.Linear):
#     """This is the exact same as torch.nn.Linear,
#     except that it can take edge_index, edge_attr and do nothing with them.
#     Makes it interchangeable with graph neural network modules."""

#     def forward(self, input, edge_index, edge_attr):
#         return super(GraphLinear, self).forward(input)


# class GraphLSTM(BaseModel):
#     """
#     Base class for Recurrent Neural Networks (LSTM or GRU).
#     Initialization to this class contains all variables for variation of the model.
#     Consists of one of the above RNN architectures followed by an optional GNN on the final hidden state.
#     Parameters:
#         node_features: int - number of features per node
#         output: int - length of the output vector on each node
#         dim: int - number of features of embedding for each node
#         module: torch.nn.Module - to be used in the LSTM to calculate each gate
#     """
#     def __init__(self, 
#                 num_nodes, 
#                 num_features,
#                 num_timesteps_input,
#                 num_timesteps_output, 
#                 dim=32, 
#                 module=GraphLinear, 
#                 rnn=LSTM, 
#                 gnn=WeightedSAGEConv, 
#                 gnn_2=WeightedSAGEConv, 
#                 rnn_depth=1, 
#                 name="RNN", 
#                 skip_connection=True):
#         super(GraphLSTM, self).__init__()
#         self.n_feat = num_features
#         self.n_in = num_timesteps_input
#         self.m = num_nodes
#         self.dim = dim
#         self.rnn_depth = rnn_depth
#         self.name = name
#         self.skip_connection = skip_connection

#         # Ensure that matrix multiplication sizes match up based on whether GNNs and RNN are used
#         if gnn:
#             if skip_connection:
#                 self.gnn = gnn(num_features, dim)
#             else:
#                 self.gnn = gnn(num_features, dim * 2)
#             if rnn:
#                 if skip_connection:
#                     self.recurrent = rnn(dim, dim, module=module)
#                 else:
#                     self.recurrent = rnn(dim * 2, dim * 2, module=module)
#             else:
#                 self.recurrent = None
#         else:
#             self.gnn = None
#             if rnn:
#                 self.recurrent = rnn(num_features, dim, module=module)
#             else:
#                 self.recurrent = None
#         if gnn_2:
#             if gnn:
#                 self.gnn_2 = gnn_2(dim * 2, dim * 2)
#             else:
#                 self.gnn_2 = gnn_2(dim + num_features, dim * 2)
#         else:
#             self.gnn_2 = None

#         self.lin1 = torch.nn.Linear(2 * dim, dim)
#         self.lin2 = torch.nn.Linear(dim, num_timesteps_output)
#         self.act1 = torch.nn.ReLU()
#         self.act2 = torch.nn.ReLU()

#     def forward(self, X, adj, states=None, dynamic_adj=None, h=None, c=None):
#         # Get data from snapshot
#         x, edge_index = X, adj.to_sparse()

#         # First GNN Layer
#         if self.gnn:
#             x = self.gnn(x, edge_index)
#             x = F.relu(x)

#         # Initialize hidden and cell states if None
#         current_dim = self.dim
#         if not self.skip_connection:
#             current_dim = self.dim * 2
#         if h is None:
#             h = torch.zeros(x.shape[0], current_dim)
#         if c is None:
#             c = torch.zeros(x.shape[0], current_dim)

#         # RNN Layer
#         if self.recurrent:
#             for i in range(self.rnn_depth):
#                 h, c = self.recurrent(x, edge_index, edge_attr, h, c)

#         # Skip connection from first GNN
#         if self.skip_connection:
#             x = torch.cat((x, h), 1)
#         else:
#             x = h

#         # Second GNN Layer
#         if self.gnn_2:
#             x = self.gnn_2(x, edge_index, edge_attr)

#         # Readout and activation layers
#         x = self.lin1(x)
#         # x = self.act1(x)
#         x = self.lin2(x)
#         # x = self.act2(x)

#         return x, h, c
    

#     def initialize(self):
#         for layer in self.children():
#             if hasattr(layer, 'reset_parameters'):
#                 layer.reset_parameters()