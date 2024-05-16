import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self):
        super().__init__()


    def forward(self, data, **kwarg):
        frequency_data = torch.zeros_like(data, dtype=torch.complex128)
        for i in range(data.shape[0]):  
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    frequency_data[i, j, k, :] = torch.fft.fft(data[i, j, k, :])
                    
        return frequency_data.real.float()


class test_seq(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, data, **kwarg):
        print("test!", data.shape)
                    
        return data[0]
    
    

'''class ABS_TIM_EMB(nn.Module):

    def __init__(self):
        super().__init__()
        self.d_model = 13
        #self.device = kwarg['device']
        self.fourier = False

    def forward(self, data, **kwarg):
        num_graphs = data.shape[0]
        num_nodes = data.shape[1]
        num_timesteps = data.shape[2]
        embedded_times = torch.zeros(num_graphs, num_nodes, num_timesteps, data.shape[3]+13)
        
        
        embedded_data = torch.zeros(data.shape[0], data.shape[1], data.shape[2], self.d_model).to(kwarg['device'])
        for t in range(data.shape[2]):
            
            x = data[:, :, t, :]
            
            a = torch.div(torch.arange(0.0, self.d_model), 2, rounding_mode='floor').to(kwarg['device']) * 2 / self.d_model
            b = torch.matmul(x.unsqueeze(-1), (2 * np.pi / 1440) * a.unsqueeze(0)) if not self.fourier else \
                torch.matmul(x.unsqueeze(-1), (1e-4 ** a).unsqueeze(0))
            c = torch.zeros_like(b)
            c[:, :, 0::2] = b[:, :, 0::2].sin()
            c[:, :, 1::2] = b[:, :, 1::2].cos()
            print(c.shape)
            print(embedded_data.shape)
            embedded_data[:, :, t, :] = c
        print(embedded_data.shape)
        print(data.shape)


        data_with_time_embeddings = torch.cat((data, embedded_data), dim=-1) 

        print(data_with_time_embeddings.shape) 
        return embedded_times'''
    
    
class add_time_embedding(nn.Module):
    def __init__(self, max_timesteps=13, embedding_dim=13):
        super(add_time_embedding, self).__init__()
        self.embedding = nn.Embedding(max_timesteps, embedding_dim)

    def forward(self, data, **kwarg):
        
        num_graphs = data.shape[0]
        num_nodes = data.shape[1]
        num_timesteps = data.shape[2]
        
        
        time_indices = torch.arange(num_timesteps).expand(num_graphs, num_nodes, num_timesteps).to(kwarg['device'])
        #print(time_indices.shape)
        #print(self.embedding(time_indices).shape)
        enhanced_data = torch.cat((data, self.embedding(time_indices).squeeze(3)), dim=-1) 

        print(enhanced_data.shape)  
        return enhanced_data
    