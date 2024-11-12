import torch
import torch.nn as nn
import numpy as np
from .utils import *
import os
from tqdm import tqdm
from fastdtw import fastdtw

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms, device="cpu"):
        self.transforms = transforms
        self.device = device
        self.feat_mean = 0
        self.feat_std = 1

    def __call__(self, data):
        for key, d in data.items():
            if data[key] is not None:
                data[key] = torch.Tensor(data[key])
                
        for dt, ts in self.transforms.items():
            for t in ts:
                t = t.to(self.device)
                data[dt] = t(data[dt], device=self.device)
                if type(t).__name__ == 'normalize_feat':
                    self.feat_mean, self.feat_std = t.means, t.stds
                if type(t).__name__ == 'normalize_target':
                    self.target_mean, self.target_std = t.means, t.stds
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class normalize_feat(nn.Module):
    """
    A normalization module for feature standardization in PyTorch. This module adjusts features
    to have zero mean and unit variance along specified dimensions, handling 3D and 4D tensors.

    Parameters
    ----------
    dim : int, optional
        The dimension over which to calculate the mean and standard deviation for normalization. Default: 1.
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        self.means = None
        self.stds = None
    
    def forward(self, X, device='cpu'):
        """
        Forward pass of the normalization module that normalizes a given input tensor X.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor to be normalized. Can be either a 3D or 4D tensor.
        device : str, optional
            The device to which the normalized tensor is transferred. Default: 'cpu'.

        Returns
        -------
        torch.Tensor
            The normalized tensor, adjusted to have zero mean and unit variance along the specified dimensions,
            and transferred to the specified device.
        """
        if len(X.shape) == 2:
            means = torch.mean(X, axis=0)
            X = X - means
            stds = torch.std(X, axis=0)
            X = X / stds

        if len(X.shape) == 3:
            means = torch.mean(X, axis=(0, 1))
            X = X - means.unsqueeze(0).unsqueeze(0)
            stds = torch.std(X, axis=(0, 1))
            X = X / stds.unsqueeze(0).unsqueeze(0)
        elif len(X.shape) == 4:
            means = torch.mean(X, dim=(0, 1, 2))
            X = X - means.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            stds = torch.std(X, dim=(0, 1, 2))
            X = X / stds.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        X[torch.where(torch.isnan(X))]=0
        self.means = means
        self.stds = stds

        return X.to(device)
    
    def denorm(self, X):
        X = X*self.stds
        X = X+self.means

        return X





class normalize_adj(nn.Module):
    """
    A PyTorch module for normalizing adjacency matrices to facilitate operations in graph neural networks. 
    The normalization adjusts adjacency matrices to account for node degrees, enhancing the propagation 
    of features through the network. This class can handle both batched and single adjacency matrices.

    Parameters
    ----------
    dim : int, optional
        The dimension over which to perform normalization (not utilized in current implementation). Default: 0.
    """
    def __init__(self, dim = 0):
        super().__init__()
        self.dim = dim
    
    def forward(self, Adj, device='cpu'):
        """
        Forward pass of the normalize_adj module that computes a degree-normalized adjacency matrix 
        from the input adjacency matrix 'Adj'. 

        Parameters
        ----------
        Adj : torch.Tensor or np.array
            The input adjacency matrix, which can be a 2D matrix for a single graph or a 3D tensor for batch processing.
        device : str, optional
            The device to which the normalized adjacency matrix is transferred. Default: 'cpu'.

        Returns
        -------
        torch.Tensor
            The degree-normalized adjacency matrix, transferred to the specified device.
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
            
        return torch.FloatTensor(A_wave).to(device)




class convert_to_frequency(nn.Module):
    """
    A PyTorch module for transforming time-domain data into frequency-domain representations using either
    FFT (Fast Fourier Transform) or STFT (Short Time Fourier Transform). This module is configurable to handle
    different lengths of FFT and hop sizes for STFT, making it flexible for various signal processing tasks.

    Parameters
    ----------
    ftype : str, optional
        The type of frequency transformation to perform, either 'fft' for Fast Fourier Transform or 'stft' for
        Short Time Fourier Transform. Default: "fft".
    n_fft : int, optional
        The window size for the FFT or STFT. Default: 8.
    """

    def __init__(self, ftype="fft", n_fft=8):
        super().__init__()
        self.ftype = ftype
        self.n_fft = n_fft
        self.hop_length = n_fft // 2
        self.win_length = n_fft


    def forward(self, data, **kwarg):
        """
        Applies the configured frequency transformation (FFT or STFT) to the input data.

        Parameters
        ----------
        data : torch.Tensor
            The input data tensor, expected to be a time-domain signal. It can be a 3D tensor (batch, channels, time)
            for 'fft' or a 4D tensor (batch, nodes, time, features) for 'stft'.
        **kwargs : dict
            Additional keyword arguments, such as 'device' for specifying the computation device.

        Returns
        -------
        torch.Tensor
            The frequency-domain representation of the input data. The output format depends on the transformation
            type ('fft' returns real parts of the FFT, 'stft' returns the magnitude of the STFT as a 5D tensor).
        """
        if self.ftype == "fft":
            return torch.fft.fft(data, dim=2).real.float()
        
        elif self.ftype == "stft":
            num_graph, num_node, timestep, features = data.shape
            stft_results = []
            window = torch.hann_window(self.win_length, device=kwarg['device'])
            
            for g in range(num_graph):
                for n in range(num_node):
                    for f in range(features):
                        signal = data[g, n, :, f]
                        stft_result = torch.stft(signal, n_fft=self.n_fft, 
                                                 hop_length=self.hop_length, win_length=self.win_length,
                                                 window=window, return_complex=True).real.float()
                        stft_results.append(stft_result)
            num_frequencies = stft_results[0].shape[-2]
            num_timeframes = stft_results[0].shape[-1]

            stft_results = torch.stack(stft_results)
            stft_results = stft_results.view(num_graph, num_node, features, num_frequencies, num_timeframes)
            
            return stft_results

    

class add_time_embedding(nn.Module):
    """
    A PyTorch module that appends time-based embeddings to each feature vector in the dataset. 
    It can generate embeddings using sinusoidal functions, optionally using a Fourier transform approach, 
    enhancing the temporal aspects of data for tasks such as time series forecasting or sequence modeling.

    Parameters
    ----------
    embedding_dim : int, optional
        The dimensionality of the time embeddings. Default: 13.
    fourier : bool, optional
        Specifies whether to use a Fourier transform-based approach for time embedding. Default: False.
    """
    def __init__(self, embedding_dim=13, fourier=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fourier = fourier

    def forward(self, data, **kwarg):
        """
        Generates and appends time-based embeddings to the input data tensor.

        Parameters
        ----------
        data : torch.Tensor
            The input data tensor, which can vary in dimensions depending on the application (e.g., batch, nodes, time).
        **kwargs : dict
            Additional keyword arguments such as 'device' for specifying the computation device.

        Returns
        -------
        torch.Tensor
            The input data tensor augmented with time-based embeddings. The resulting tensor includes an additional
            dimension for the embeddings, concatenated to the last dimension of the input data tensor.
        """
        num_graphs = data.shape[0]
        num_nodes = data.shape[1]
        timesteps = data.shape[2]
        
        x = torch.arange(timesteps).float().unsqueeze(0).unsqueeze(1).to(kwarg['device'])

        a = torch.div(torch.arange(0.0, self.embedding_dim), 2, rounding_mode='floor').to(kwarg['device']) * 2 / self.embedding_dim
        b = torch.matmul(x.unsqueeze(-1), (2 * np.pi / 1440) * a.unsqueeze(0)) if not self.fourier else \
            torch.matmul(x.unsqueeze(-1), (1e-4 ** a).unsqueeze(0))
        c = torch.zeros_like(b)

        c[:, :, 0::2] = b[:, :, 0::2].sin()
        c[:, :, 1::2] = b[:, :, 1::2].cos()
        c = c.expand(num_graphs, num_nodes, -1, -1)

        if len(data.shape) == 3:
            data = data.unsqueeze(-1)
            data_with_time_embeddings = torch.cat((data, c), dim=-1) 
            #data_with_time_embeddings = torch.permute(data_with_time_embeddings, (0, 1, 3, 2))
        else:
            data_with_time_embeddings = torch.cat((data, c), dim=-1) 
        return data_with_time_embeddings
    
    
class learnable_time_embedding(nn.Module):
    """
    A PyTorch module that appends learnable time embeddings to the input data, facilitating the capture
    of temporal dependencies in models that process sequences or time-series data. The embedding is learned 
    during the training process, allowing it to adapt to specific temporal patterns observed in the dataset.

    Parameters
    ----------
    timesteps : int
        The total number of timesteps in the data sequence, which defines the number of unique time indices.
    embedding_dim : int
        The dimensionality of each time embedding vector.
    """
    def __init__(self, timesteps=13, embedding_dim=13):
        super(learnable_time_embedding, self).__init__()
        self.add_embedding = nn.Embedding(timesteps, embedding_dim)

    def forward(self, data, **kwarg):
        """
        Appends learnable time-based embeddings to the input data tensor. The embeddings are added to each timestep, 
        augmenting the feature dimensions of the data.

        Parameters
        ----------
        data : torch.Tensor
            The input data tensor, typically including dimensions for batch, nodes, and time.
        **kwargs : dict
            Additional keyword arguments such as 'device' for specifying the computation device.

        Returns
        -------
        torch.Tensor
            The input data tensor augmented with time-based embeddings, expanding the last dimension to include
            the embeddings.
        """
        
        num_graphs = data.shape[0]
        num_nodes = data.shape[1]
        timesteps = data.shape[2]
        
        
        time_indices = torch.arange(timesteps).expand(num_graphs, num_nodes, timesteps).to(kwarg['device'])
        #print(time_indices.shape)
        #print(self.embedding(time_indices).shape)
        data_with_time_embeddings = torch.cat((data, self.add_embedding(time_indices).squeeze(3)), dim=-1) 

        #print(data_with_time_embeddings.shape)  
        return data_with_time_embeddings


class seasonality_and_trend_decompose(nn.Module):
    """
    A PyTorch module designed to decompose time-series data into seasonality and trend components.
    It supports both dynamic and static decomposition methods. Dynamic decomposition leverages a Fourier
    transform approach for seasonality and convolutional filters for trend extraction. Static decomposition
    utilizes a moving average approach.

    Parameters
    ----------
    decompose_type : str, optional
        The type of decomposition to perform. "dynamic" for Fourier and convolutional methods, "static" for
        moving average based decomposition. Default: "dynamic".
    moving_avg : int, optional
        The window size for the moving average in static decomposition. Default: 25.
    kernel_size : list of int, optional
        List of kernel sizes for convolutional filters in dynamic trend decomposition. Default: [4, 8, 12].

    """
    def __init__(self, decompose_type = "dynamic", moving_avg=25, kernel_size= [4, 8, 12]):
        super().__init__()
        self.decompose_type = decompose_type
        if self.decompose_type == "dynamic":
            self.seasonality_model = FourierLayer(pred_len=0, k=3)
            self.trend_model = series_decomp_multi(kernel_size=kernel_size)
        if self.decompose_type == "static":
            self.decompsition = series_decomp(moving_avg)

    def forward(self, data, **kwarg):
        """
        Decomposes the input data tensor into seasonality and trend components. The method of decomposition
        (dynamic or static) impacts the models and techniques used.

        Parameters
        ----------
        data : torch.Tensor
            The input data tensor, typically including dimensions for batch, nodes, and time.
        **kwargs : dict
            Additional keyword arguments such as 'device' for specifying the computation device.

        Returns
        -------
        list of torch.Tensor
            A list containing the seasonality and trend components of the input data, each as a separate tensor.
        """
        num_graphs = data.shape[0]
        num_nodes = data.shape[1]
        timesteps = data.shape[2]
        time_middle_data = data.permute(0, 2, 1).contiguous()
        if self.decompose_type == "dynamic":
            _, trend = self.trend_model(time_middle_data)
            seasonality, _ = self.seasonality_model(time_middle_data)
        if self.decompose_type == "static":
            seasonality, trend = self.decompsition(time_middle_data)
        return [seasonality, trend]



class calculate_dtw_matrix(nn.Module):
    """
    A PyTorch module designed to compute the Dynamic Time Warping (DTW) distance matrix between all pairs
    of time-series in a dataset. DTW is a method that calculates an optimal match between two given sequences
    with certain restrictions. The matrix is computed once and saved for future use to avoid redundant computations.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset, used to save and retrieve the computed DTW matrix.
    """
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset = dataset_name
    def forward(self, data, **kwarg):
        """
        Computes the Dynamic Time Warping (DTW) matrix for the input data. If a precomputed matrix exists
        in the cache, it loads that matrix; otherwise, it computes a new matrix and saves it.

        Parameters
        ----------
        data : np.ndarray
            The input data array where each row represents a time step and each column a time-series node.
        **kwargs : dict
            Additional keyword arguments not utilized in this method.

        Returns
        -------
        np.ndarray
            The computed or loaded DTW distance matrix, where each element (i, j) represents the DTW distance
            between the i-th and j-th time-series.
        """
        # data format -> np.narray
        all_time = data.shape[0]
        num_nodes = data.shape[1]

        cache_path = './dtw_' + self.dataset + '.npy'
        if os.path.exists(cache_path):
            dtw_matrix = np.load(cache_path)
            print('Loaded DTW matrix from {}'.format(cache_path))
        else:
            data_mean = data.reshape(all_time,num_nodes,1)
            dtw_matrix = np.zeros((num_nodes, num_nodes))
            for i in tqdm(range(num_nodes)):
                for j in range(i, num_nodes):
                    dtw_distance, _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
                    dtw_matrix[i][j] = dtw_distance

            for i in range(num_nodes):
                for j in range(i):
                    dtw_matrix[i][j] = dtw_matrix[j][i]

            np.save(cache_path, dtw_matrix)
            print('Saved DTW matrix to {}'.format(cache_path))

        return dtw_matrix
