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
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class normalize_feat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        self.means = None
        self.stds = None
    
    def forward(self, X, device='cpu'):
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

class normalize_adj(nn.Module):
    def __init__(self, dim = 0):
        super().__init__()
        self.dim = dim
    
    def forward(self, Adj, device='cpu'):
        """
        Returns the degree normalized adjacency matrix.
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
    def __init__(self, ftype="fft", n_fft=8):
        super().__init__()
        self.ftype = ftype
        self.n_fft = n_fft
        self.hop_length = n_fft // 2
        self.win_length = n_fft


    def forward(self, data, **kwarg):
        '''frequency_data = torch.zeros_like(data, dtype=torch.complex128)
        for i in range(data.shape[0]):  
            for j in range(data.shape[1]):
                for k in range(data.shape[3]):
                    frequency_data[i, j, :, k] = torch.fft.fft(data[i, j, :, k])
                    
        return frequency_data.real.float()'''
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

    def __init__(self, embedding_dim=13, fourier=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fourier = fourier

    def forward(self, data, **kwarg):
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
    def __init__(self, timesteps=13, embedding_dim=13):
        super(learnable_time_embedding, self).__init__()
        self.add_embedding = nn.Embedding(timesteps, embedding_dim)

    def forward(self, data, **kwarg):
        
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
    def __init__(self, decompose_type = "dynamic", moving_avg=25, kernel_size= [4, 8, 12]):
        super().__init__()
        self.decompose_type = decompose_type
        if self.decompose_type == "dynamic":
            self.seasonality_model = FourierLayer(pred_len=0, k=3)
            self.trend_model = series_decomp_multi(kernel_size=kernel_size)
        if self.decompose_type == "static":
            self.decompsition = series_decomp(moving_avg)

    def forward(self, data, **kwarg):
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

    def __init__(self, dataset_name):
        super().__init__()
        self.dataset = dataset_name
    def forward(self, data, **kwarg):
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
