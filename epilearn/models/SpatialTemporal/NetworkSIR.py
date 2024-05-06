import torch
import torch.nn as nn

class NetSIR(nn.Module):
    def __init__(self, num_nodes = None, horizon=None, infection_rate=0.01, recovery_rate=0.038, population=None):
        super(NetSIR, self).__init__()
        self.pop = population
        self.horizon = horizon
        self.num_nodes = num_nodes

        self.beta = torch.abs(torch.rand(num_nodes))
        self.gamma = torch.abs(torch.rand(num_nodes))

        if infection_rate is not None: 
            new_weights = torch.zeros_like(self.beta.data) + torch.FloatTensor([infection_rate])
            self.beta.data = new_weights

        if recovery_rate is not None:
            new_weights = torch.zeros_like(self.gamma.data) + torch.FloatTensor([recovery_rate])
            self.gamma.data = new_weights
        
        self.beta = nn.Parameter(self.beta)
        self.gamma = nn.Parameter(self.gamma)
        
    def forward(self, x, adj, steps = 1):
        '''
        Args:  x: (n_nodes, states)
               adj: (n_nodes, n_nodes)
        Returns: (time_step, n_nodes, states)
        ''' 
        if self.pop is not None:
            pop = self.pop
        else:
            pop = x.sum()
        if self.horizon is not None:
            steps = self.horizon

        # rescale = nn.Softmax(dim=0)

        output = torch.zeros(self.num_nodes*steps*3, dtype=torch.float, requires_grad = False).reshape(steps, self.num_nodes, 3)
        output.data[0] = x.data
        adj = adj.float()

        for i in range(1, steps):
            new_cases = self.beta*(output.data[i-1,:,0]*(adj @ output.data[i-1,:,1])).unsqueeze(0)
            new_recovery = self.gamma*output.data[i-1,:,1]

            output.data[i,:,0] = output.data[i-1,:,0] - new_cases
            output.data[i,:,1] = output.data[i-1,:,1] + new_cases - new_recovery
            output.data[i,:,2] = output.data[i-1,:,2] + new_recovery
            
            # output.data[i,:] = rescale(output.data[i,:])
        
        return output