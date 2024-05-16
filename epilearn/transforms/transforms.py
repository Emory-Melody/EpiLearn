import torch
import torch.nn as nn
import numpy as np
import pywt
from scipy.fftpack import fft as fft_func

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

    def __call__(self, data):
        for t in self.transforms:
            t = t.to(self.device)
            data = t(data, device=self.device)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string





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
        
        


class test_seq(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, data, **kwarg):
        print("test!", data.shape)
                    
        return data[0]
    
    

class ABS_TIM_EMB(nn.Module):

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

        data_with_time_embeddings = torch.cat((data, c), dim=-1) 
        return data_with_time_embeddings
    
    
class add_time_embedding(nn.Module):
    def __init__(self, timesteps=13, embedding_dim=13):
        super(add_time_embedding, self).__init__()
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
    