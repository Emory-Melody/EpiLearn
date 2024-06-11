# this code is adopted from https://github.com/chnsh/DCRNN_PyTorch/tree/pytorch_scratch/model/pytorch

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse import linalg
import torch.nn.functional as F

from .base import BaseModel



'''
utils
'''
def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    #adj = adj.to("cpu")
    adj = adj.cpu().clone().numpy()
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    #print(adj_mx)
    #adj_mx = adj_mx.to("cpu")
    #print(adj_mx.shape)
    if undirected:
        adj_mx = torch.max(adj_mx, adj_mx.transpose(0, 1))
        #adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
        #adj_mx = np.maximum(adj_mx, adj_mx.T)
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

def calculate_random_walk_matrix(adj_mx):
    #adj_mx = adj_mx.to("cpu")
    adj_mx = adj_mx.cpu().clone().numpy()
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx



'''
pytorch version of DCRNN written by Chintan Shah.
https://github.com/chnsh/DCRNN_PyTorch

This is a revised version of Chintan Shah's.
'''

'''
dcrnn_cell.py
'''
class LayerParams:
    def __init__(self, rnn_network: nn.Module, layer_type: str, device="cpu"):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.device = device

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.empty(*shape, device=self.device))
            nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=self.device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DCGRUCell(nn.Module):
    def __init__(self, num_units, max_diffusion_step, nonlinearity='tanh',
                 filter_type="laplacian", device="cpu"):
        """
        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        """

        super().__init__()
        if nonlinearity == 'tanh':
            self._activation = torch.tanh 
        elif nonlinearity == 'relu':
            self._activation = torch.relu
        else:
            raise ValueError("Please specify an activation function in [tanh, relu]!") 
        # support other nonlinearities up here?
        # self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self.device = device
        self.filter_type = filter_type

        #self._gconv_params = LayerParams(self, 'gconv', device=device)

    def reset_parameters(self):
        self._gconv_params = LayerParams(self, 'gconv', device=self.device)
            
            
    @staticmethod
    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=self.device)
        return L

    def forward(self, inputs, hx, adj_mx, num_nodes):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * num_features)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        self._num_nodes = num_nodes
        if len(self._supports) == 0:
            supports = []
            if self.filter_type == "laplacian":
                supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
            elif self.filter_type == "random_walk":
                supports.append(calculate_random_walk_matrix(adj_mx).T)
            elif self.filter_type == "dual_random_walk":
                supports.append(calculate_random_walk_matrix(adj_mx).T)
                supports.append(calculate_random_walk_matrix(adj_mx.T).T)
            else:
                supports.append(calculate_scaled_laplacian(adj_mx))
            for support in supports:
                self._supports.append(self._build_sparse_matrix(self, support))
        
        output_size = 2 * self._num_units
        fn = self._gconv
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, num_nodes * self._num_units))
        u = torch.reshape(u, (-1, num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)


    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, num_features/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
    
    
    
    

'''
dcrnn_model.py
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, max_diffusion_step, filter_type,
                              num_rnn_layers, rnn_units):
        self.max_diffusion_step = max_diffusion_step
        self.filter_type = filter_type
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, 
                 max_diffusion_step, filter_type, num_rnn_layers, rnn_units,
                 num_features, num_timesteps_input, nonlinearity='tanh', device="cpu"):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, max_diffusion_step, filter_type,
                              num_rnn_layers, rnn_units)
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, nonlinearity=nonlinearity,
                       filter_type=self.filter_type, device=device) for _ in range(self.num_rnn_layers)])
        self.device = device

    def forward(self, inputs, hidden_state=None, adj_mx=None, num_nodes=1, dropout=0):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.num_features)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_state_size = num_nodes * self.rnn_units
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj_mx, num_nodes)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
            if layer_num != len(self.dcgru_layers) - 1:
                output = F.dropout(output, p=dropout, training=self.training)

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow
    
    def initialize(self):
        for m in self.dcgru_layers:
            m.reset_parameters()


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self,
                 max_diffusion_step, filter_type, num_rnn_layers, rnn_units,
                 num_classes, num_timesteps_output, nonlinearity='tanh', device="cpu"):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, max_diffusion_step, filter_type,
                              num_rnn_layers, rnn_units)
        self.num_classes = num_classes
        self.num_timesteps_output = num_timesteps_output
        self.projection_layer = nn.Linear(self.rnn_units, self.num_classes)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step,
                       filter_type=self.filter_type, nonlinearity=nonlinearity, device=device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None, adj_mx=None, num_nodes=1, dropout=0):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.num_classes)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.num_classes)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj_mx, num_nodes)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
            if layer_num != len(self.dcgru_layers) - 1:
                output = F.dropout(output, p=dropout, training=self.training)

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, num_nodes * self.num_classes)

        return output, torch.stack(hidden_states)
    
    
    def initialize(self):
        for m in self.dcgru_layers:
            m.reset_parameters()


class DCRNN(BaseModel, Seq2SeqAttrs):
    """
    Diffusion Convolutional Recurrent Neural Network (DCRNN) 

    Parameters
    ----------
    num_features : int
        Number of input features per node.
    num_timesteps_input : int
        Number of past time steps used as input by the network.
    num_classes : int
        Number of output features per node.
    num_timesteps_output : int
        Number of future time steps to predict.
    max_diffusion_step : int
        Maximum number of diffusion steps in the graph convolution operations. Default: 2.
    filter_type : str
        Type of filter used in graph convolutions, e.g., 'laplacian'. Default: "laplacian".
    num_rnn_layers : int
        Number of recurrent neural network layers. Default: 1.
    rnn_units : int
        Number of units per recurrent layer. Default: 1.
    nonlinearity : str
        Type of nonlinearity function used in RNN. Default: "tanh".
    dropout : float
        Dropout rate applied in the network to prevent overfitting. Default: 0.
    device : str, optional
        The device (cpu or gpu) on which the model will be run. Default: 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_nodes, horizon), representing the predicted values for each node over future timesteps.
    """
    def __init__(self,
              num_features=1,
              num_timesteps_input=5,
              num_classes=1,
              num_timesteps_output=1,
              max_diffusion_step=2, 
              filter_type="laplacian",
              num_rnn_layers=1, 
              rnn_units=1,
              nonlinearity="tanh",
              dropout=0,
              device="cpu"):

        super().__init__()
        Seq2SeqAttrs.__init__(self, max_diffusion_step=max_diffusion_step, filter_type=filter_type,
                              num_rnn_layers=num_rnn_layers, rnn_units=rnn_units)
        
        
        self.encoder_model = EncoderModel(max_diffusion_step=max_diffusion_step, filter_type=filter_type,
                                          num_rnn_layers=num_rnn_layers, rnn_units=rnn_units,
                                          num_features=num_features, num_timesteps_input=num_timesteps_input, nonlinearity=nonlinearity, device=device)
        self.decoder_model = DecoderModel(max_diffusion_step=max_diffusion_step, filter_type=filter_type,
                                          num_rnn_layers=num_rnn_layers, rnn_units=rnn_units,
                                          num_classes=num_classes, num_timesteps_output=num_timesteps_output, nonlinearity=nonlinearity, device=device)
        
        self.device = device
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_classes = num_classes
        self.num_timesteps_output = num_timesteps_output
        self.dropout = dropout



    def encoder(self, inputs, adj_mx, num_nodes):
        """
        encoder forward pass on t time steps
        :param inputs: shape (num_timesteps_input, batch_size, num_sensor * num_features)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.num_timesteps_input):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state, adj_mx, num_nodes, self.dropout)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj_mx, num_nodes):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.num_timesteps_output, batch_size, self.num_nodes * self.num_classes) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.num_timesteps_output, batch_size, self.num_nodes * self.num_classes)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, num_nodes * self.decoder_model.num_classes),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.num_timesteps_output):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state, adj_mx, num_nodes, self.dropout)
            decoder_input = decoder_output
            outputs.append(decoder_output)

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, X_batch, graph, X_states, batch_graph):
        """
        Parameters
        ----------
        X_batch : torch.Tensor
            Input tensor with shape (batch_size, num_nodes, num_timesteps_input, num_features),
            representing the input features over multiple timesteps for each node.
        graph : torch.Tensor
            Static adjacency matrix with shape (num_nodes, num_nodes),
            representing the fixed connections between nodes.
        X_states : torch.Tensor, optional
            States of the nodes if available, with the same shape as X_batch.
            Used for models that incorporate node states over time. Default: None.
        batch_graph : torch.Tensor, optional
            Dynamic adjacency matrix if available, with shape similar to graph but possibly varying over time.
            Used for models that account for changing graph structures. Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, num_nodes, num_timesteps_output),
            representing the predicted values for each node over the specified output timesteps.
        """
        
        inputs = X_batch
        batch_size = X_batch.shape[0] #data.batch[-1] + 1
        
        # reshape geometric dataloader format to real format
        '''num_graphs = data.num_graphs
        individual_graphs = []
        for i in range(num_graphs):
            mask = data.batch == i
            node_features = data.x[mask]
            individual_graphs.append(node_features)
            adj_m = data.adj_m[mask]
        new_inputs = torch.stack(individual_graphs, dim=0)'''
        
        inputs = torch.permute(inputs,(2, 0, 1, 3))
        inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[1], 
                                            inputs.shape[2]*inputs.shape[3]))
        
        
        num_nodes = graph.shape[0]
        encoder_hidden_state = self.encoder(inputs, graph, num_nodes)
        outputs = self.decoder(encoder_hidden_state, graph, num_nodes)
        
        outputs = torch.reshape(outputs, (self.num_timesteps_output, batch_size, 
                                            num_nodes, self.num_classes))
        outputs = torch.permute(outputs, (1, 2, 0, 3)).squeeze(-1)
        #outputs = torch.reshape(outputs, (-1, outputs.shape[2], outputs.shape[3]))
        return outputs
    
    
    def initialize(self):
        self.encoder_model.initialize()
        self.decoder_model.initialize()
        #pass


    
