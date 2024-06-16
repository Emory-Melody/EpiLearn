import torch
import torch.nn as nn

class NetSIR(nn.Module):
    """
    Network-based SIR (Susceptible-Infected-Recovered) 

    Parameters
    ----------
    num_nodes : int, optional
        Number of nodes in the graph representing individuals or groups. Default: None.
    horizon : int, optional
        Number of future time steps to simulate. If None, a single step is simulated unless overridden in the forward method.
    infection_rate : float, optional
        Initial infection rate parameter, representing the rate at which susceptible individuals become infected. Default: 0.01.
    recovery_rate : float, optional
        Initial recovery rate parameter, representing the rate at which infected individuals recover. Default: 0.038.
    population : int, optional
        Total population considered in the model. If None, the sum of the initial conditions (susceptible, infected, recovered) is used as the total population.

    Returns
    -------
    torch.Tensor
        A tensor of shape (time_step, num_nodes, 3), representing the predicted number of susceptible, infected, and recovered individuals at each timestep for each node.
        Each row corresponds to a timestep, with the columns representing the susceptible, infected, and recovered counts respectively for each node.
    """
    def __init__(self, num_nodes=None, horizon=None, infection_rate=0.01, recovery_rate=0.038, population=None):
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
        
    def forward(self, x, adj, steps=1):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (n_nodes, one-hot encoding of states).
        adj : torch.Tensor
            Static adjacency matrix of the graph with shape (num_nodes, num_nodes).
        states : torch.Tensor, optional
            States of the nodes if available, with the same shape as x. Default: None.
        dynamic_adj : torch.Tensor, optional
            Dynamic adjacency matrix if available, with shape similar to adj but possibly varying over time. Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (time_step, n_nodes, probability of states),
            representing the predicted values for each node over the specified output timesteps.
        """
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
