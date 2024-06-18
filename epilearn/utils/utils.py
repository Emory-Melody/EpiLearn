import torch
import numpy as np
import torch.nn as nn

import math
import torch.fft as fft
from einops import rearrange, reduce, repeat

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize(X):
    """
    Normalizes the input tensor X to have zero mean and unit standard deviation. Handles 3D tensors.
    
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to be normalized.

    Returns
    -------
    torch.Tensor
        The normalized tensor.
    torch.Tensor
        The means of the input tensor.
    torch.Tensor
        The standard deviations of the input tensor.
    """
    if len(X.shape) == 3:
        means = torch.mean(X, axis=(0, 1))
        X = X - means.unsqueeze(0).unsqueeze(0)
        stds = torch.std(X, axis=(0, 1))
        X = X / stds.unsqueeze(0).unsqueeze(0)
    X[torch.where(torch.isnan(X))]=0
    return X, means, stds


def normalize_adj(Adj):
    """
    Returns the degree normalized adjacency matrix.
    
    Parameters
    ----------
    Adj : torch.Tensor or np.array
        The input adjacency matrix.

    Returns
    -------
    torch.Tensor
        The degree normalized adjacency matrix.
    """
    if Adj is None:
        return None
    try:
        Adj = Adj.numpy()
    except:
        pass

    if len(Adj.shape) > 2:
        A_wave = Adj
        for i in range(Adj.shape[0]):
            A = Adj[i].reshape(Adj.shape[1], Adj.shape[1])
            A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
            D = np.array(np.sum(A, axis=1)).reshape((-1,))
            D[D <= 10e-5] = 10e-5    # Prevent infs
            diag = np.reciprocal(np.sqrt(D))
            A_wave[i] = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                                diag.reshape((1, -1))).reshape(Adj[i].shape)

    else:
        A = Adj
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5    # Prevent infs
        diag = np.reciprocal(np.sqrt(D))
        A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                            diag.reshape((1, -1)))
    return torch.FloatTensor(A_wave)

def diff(features):
    """
    Computes the discrete difference along the time dimension of the input tensor.
    
    Parameters
    ----------
    features : torch.Tensor
        The input feature tensor.

    Returns
    -------
    torch.Tensor
        The tensor of differences along the time dimension.
    """
    return torch.diff(features, dim=0)


def Degree_Matrix(ST_matrix):
    """
    Computes the degree matrix for a given spatio-temporal adjacency matrix.
    
    Parameters
    ----------
    ST_matrix : torch.Tensor
        The input spatio-temporal adjacency matrix.

    Returns
    -------
    torch.Tensor
        The degree matrix.
    """

    row_sum = torch.sum(ST_matrix, 0)

    ## degree matrix
    dim = len(ST_matrix)
    D_matrix = torch.zeros(dim, dim)
    for i in range(dim):
        D_matrix[i, i] = 1 / max(torch.sqrt(row_sum[i]), 1)

    return D_matrix

"""
The binary spatio-temporal adjacency matrix used in USTGCN
"""
def Static_full(n, t, A):
    """
    Constructs the full spatio-temporal adjacency matrix using the binary spatio-temporal adjacency matrix method.
    
    Parameters
    ----------
    n : int
        The dimension of the spatial adjacency matrix.
    t : int
        The length of periods.
    A : torch.Tensor
        The spatial adjacency matrix.

    Returns
    -------
    torch.Tensor
        The full spatio-temporal adjacency matrix.
    """

    I_S = torch.diag_embed(torch.ones(n))
    I_T = torch.diag_embed(torch.ones(t))

    C_S = A
    C_T = torch.tril(torch.ones(t, t), diagonal=-1)

    S = I_S + C_S
    A_ST = kronecker(C_T, S) + kronecker(I_T, C_S)

    return A_ST


"""
Use kronecker product to construct the spatio-temporal adjacency matrix
"""
def kronecker(A, B):
    """
    Constructs the spatio-temporal adjacency matrix using the Kronecker product.
    
    Parameters
    ----------
    A : torch.Tensor
        The temporal adjacency matrix.
    B : torch.Tensor
        The spatial adjacency matrix.

    Returns
    -------
    torch.Tensor
        The adjacency matrix of one space-time neighboring block.
    """
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.contiguous().view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB

def edge_to_adj(edge_index, num_nodes):
    """
    Converts edge index representation to adjacency matrix.
    
    Parameters
    ----------
    edge_index : torch.Tensor
        The edge index tensor where each column represents an edge.
    num_nodes : int
        The number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        The adjacency matrix.
    """
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj


# def tensor_to_networkx(features, graph):
#     G = nx.Graph()
#     nodes = list(range(len(graph)))
#     G.add_nodes_from(nodes)
#     edges = []
#     for i in nodes:
#         for j in nodes:
#             if graph[i, j] > 0:
#                 edges.append((i, j))
#     G.add_edges_from(edges)

#     cmap = plt.cm.get_cmap('Reds')
#     node_weights = features.view(-1).tolist()  


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series. This module smooths the input time series data 
    using a moving average filter, which helps in capturing the underlying trend by averaging over a specified window.

    Parameters
    ----------
    kernel_size : int
        The size of the moving average window.
    stride : int
        The stride of the moving average window.
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        Applies the moving average to the input tensor 'x', padding the time series at both ends to ensure
        that the moving average is computed correctly for the entire series.

        Parameters
        ----------
        x : torch.Tensor
            The input time-series data tensor, typically of shape (batch_size, time_steps, features).

        Returns
        -------
        torch.Tensor
            The tensor smoothed using the moving average, maintaining the original shape of the input.
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    A PyTorch module for series decomposition using a single moving average kernel. This module decomposes
    a time series into its residual and moving average components, providing a straightforward approach to
    trend extraction.

    Parameters
    ----------
    kernel_size : int
        The kernel size to be used for the moving average calculation. The kernel size defines the window
        for the moving average.
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        Applies the series decomposition to the input tensor 'x'. It calculates the moving average using the specified kernel size
        and computes the residual by subtracting the moving average from the original series.

        Parameters
        ----------
        x : torch.Tensor
            The input time-series data tensor, typically of shape (batch_size, time_steps).

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor)
            A tuple containing:
            - The residual tensor after subtracting the moving average from the input.
            - The moving average tensor computed from the input.
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    A PyTorch module for series decomposition using multiple moving average kernels. This module decomposes
    a time series into its residual and moving average components, leveraging multiple kernel sizes for enhanced
    flexibility and accuracy in capturing trends.

    Parameters
    ----------
    kernel_size : list of int
        List of kernel sizes to be used for moving average calculations. Each kernel size defines the window
        for the moving average.

    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        """
        Applies the series decomposition to the input tensor 'x'. It calculates multiple moving averages
        using the initialized kernels, weights them using a linear layer followed by a softmax, and computes
        the residual by subtracting the weighted moving average from the original series.

        Parameters
        ----------
        x : torch.Tensor
            The input time-series data tensor, typically of shape (batch_size, time_steps).

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor)
            A tuple containing:
            - The residual tensor after subtracting the weighted moving average from the input.
            - The weighted moving average tensor computed from the input.
        """
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean




class FourierLayer(nn.Module):
    """
    A PyTorch module that applies Fourier Transform techniques to time-series data, capable of extracting
    and manipulating frequency components. It supports both simple Fourier Transform and attention-based
    frequency analysis, allowing for the extrapolation and prediction based on frequency patterns.

    Parameters
    ----------
    pred_len : int
        The length of the sequence to predict using the frequency components.
    k : int, optional
        The number of top frequency components to retain for the analysis. Default is None.
    low_freq : int, optional
        The lowest frequency index to consider in the analysis to avoid zero-frequency (DC) components.
        Default is 1.
    output_attention : bool, optional
        Flag to determine if the layer should output the attention weights. Default is False.

    """
    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """
        Executes the Fourier transform operations on the input data 'x'. Depending on the configuration, it performs
        a basic Fourier transform to analyze frequency components, selects top-k frequencies, and optionally performs
        an inverse transform to either reconstruct or predict time-series data.

        Parameters
        ----------
        x : torch.Tensor
            The input time-series data tensor with shape (batch_size, time_steps, features).

        Returns
        -------
        torch.Tensor, tuple(torch.Tensor, torch.Tensor)
            Depending on the `output_attention` setting, it returns either the transformed data tensor or
            a tuple containing the transformed data tensor and attention weights.
        """

        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        """
        Applies Discrete Fourier Transform (DFT) and its inverse (iDFT) to extract and use the most
        significant frequency components for extrapolation or reconstruction of signals. This method
        specifically computes a frequency-domain representation using the DFT, selects the most
        significant frequencies, and applies an inverse DFT to produce a predicted output.

        Parameters
        ----------
        x : torch.Tensor
            The input time-series data tensor with shape (batch_size, time_steps, features).

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor)
            A tuple containing:
            - The extrapolated or reconstructed time-series data based on the significant frequency components.
            - The attention weights across the frequency domain shaped for visualization or further analysis,
            indicating the importance of each frequency component.
        
        """
        T = x.size(1)

        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')



    
class series_decomp(nn.Module):
    """
    A PyTorch module for series decomposition using a single moving average kernel. This module decomposes
    a time series into its residual and moving average components, providing a straightforward approach to
    trend extraction.

    Parameters
    ----------
    kernel_size : int
        The kernel size to be used for the moving average calculation. The kernel size defines the window
        for the moving average.
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        Applies the series decomposition to the input tensor 'x'. It calculates the moving average using the specified kernel size
        and computes the residual by subtracting the moving average from the original series.

        Parameters
        ----------
        x : torch.Tensor
            The input time-series data tensor, typically of shape (batch_size, time_steps).

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor)
            A tuple containing:
            - The residual tensor after subtracting the moving average from the input.
            - The moving average tensor computed from the input.
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean




