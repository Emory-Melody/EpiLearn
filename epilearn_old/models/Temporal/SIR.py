import torch
from torch import nn


class SIR(nn.Module):
    """
    Susceptible-Infected-Recovered (SIR) Model

    Parameters
    ----------
    horizon : int, optional
        Number of future time steps to simulate. If None, a single step is simulated unless overridden in the forward method.
    infection_rate : float, optional
        Initial infection rate parameter, representing the rate at which susceptible individuals become infected. Default: 0.01.
    recovery_rate : float, optional
        Initial recovery rate parameter, representing the rate at which infected individuals recover. Default: 0.038.
    population : int, optional
        Total population considered in the model. If None, the sum of the initial conditions (susceptible, infected, recovered) is used as the total population.

    Attributes
    ----------
    beta : torch.nn.Linear
        Linear layer with no bias to model the infection rate dynamically.
    gamma : torch.nn.Linear
        Linear layer with no bias to model the recovery rate dynamically.

    Returns
    -------
    torch.Tensor
        A tensor of shape (horizon, 3), representing the predicted number of susceptible, infected, and recovered individuals at each timestep.
        Each row corresponds to a timestep, with the columns representing susceptible, infected, and recovered counts respectively.

    """
    def __init__(self, horizon=None, infection_rate=0.01, recovery_rate=0.038, population=None):
        super(SIR, self).__init__()
        self.pop = population
        self.horizon = horizon

        self.beta = nn.Linear(1, 1, bias = False)
        self.gamma = nn.Linear(1, 1, bias = False)

        if infection_rate is not None: 
            self.beta.weight.data = torch.FloatTensor([infection_rate])

        if recovery_rate is not None:
            self.gamma.weight.data = torch.FloatTensor([recovery_rate])
        
    def forward(self, x, steps=1):
        """
        Parameters
        ----------
        x : torch.Tensor
            The initial condition tensor for the model. Expected shape is (3,), where the elements represent the 
            number of susceptible (S), infected (I), and recovered (R) individuals respectively.
        steps : int, optional
            Number of future time steps to simulate. If `horizon` is specified during initialization and not None,
            it overrides this parameter. Default is 1 if `horizon` is None.

        Returns
        -------
        torch.Tensor
            A tensor of shape (steps, 3), representing the predicted number of susceptible, infected, and recovered 
            individuals at each timestep. Each row corresponds to a timestep, with the columns representing 
            susceptible, infected, and recovered counts respectively.
        """
        if self.pop is not None:
            pop = self.pop
        else:
            pop = x.sum()
        if self.horizon is not None:
            steps = self.horizon

        assert len(x.shape) == 1 and x.shape[0] == 3

        output = torch.zeros(steps* 3, dtype=torch.float, requires_grad = False).reshape(steps, 3)
        output.data[0] = x.data

        for i in range(1, steps):
            output.data[i][0] = output.data[i-1][0] - self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop
            output.data[i][1] = output.data[i-1][1] + self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop - self.gamma(output.data[i-1][1].unsqueeze(0)) 
            output.data[i][2] = output.data[i-1][2] + self.gamma(output.data[i-1][1].unsqueeze(0))
        return output



class SIS(nn.Module):
    """
    Susceptible-Infected-Susceptible (SIS) Model

    Parameters
    ----------
    horizon : int, optional
        Number of future time steps to simulate. If None, a single step is simulated unless overridden in the forward method.
    infection_rate : float, optional
        Infection rate parameter, representing the rate at which susceptible individuals become infected. If None, must be initialized separately.
    recovery_rate : float, optional
        Recovery rate parameter, representing the rate at which infected individuals recover and return to the susceptible state. If None, must be initialized separately.
    population : int, optional
        Total population considered in the model. If None, the sum of the initial conditions (susceptible and infected) is used as the total population.

    Attributes
    ----------
    beta : torch.nn.Linear
        Linear layer with no bias to model the infection rate dynamically.
    gamma : torch.nn.Linear
        Linear layer with no bias to model the recovery rate dynamically.

    Returns
    -------
    torch.Tensor
        A tensor of shape (horizon, 2), representing the predicted number of susceptible and infected individuals at each timestep.
        Each row corresponds to a timestep, with the columns representing the susceptible and infected counts respectively.

    """
    def __init__(self, horizon=None, infection_rate = None, recovery_rate = None, population = None):
        super(SIS, self).__init__()
        self.pop = population
        self.horizon = horizon

        self.beta = nn.Linear(1, 1, bias = False)
        self.gamma = nn.Linear(1, 1, bias = False)

        if infection_rate is not None:
            self.beta.weight.data = torch.FloatTensor([infection_rate])

        if recovery_rate is not None:
            self.gamma.weight.data = torch.FloatTensor([recovery_rate])
        
    def forward(self, x, steps=1):
        """
        Parameters
        ----------
        x : torch.Tensor
            The initial condition tensor for the model, representing the initial numbers of susceptible (S) and infected (I) individuals.
            Expected shape is (2,), where x[0] is the number of susceptible and x[1] is the number of infected individuals at the start.
        steps : int, optional
            Number of future time steps to simulate. If `horizon` is specified during initialization and not None, it overrides this parameter.
            Default is 1 if `horizon` is None.

        Returns
        -------
        torch.Tensor
            A tensor of shape (steps, 2), representing the predicted number of susceptible and infected individuals at each timestep.
            Each row corresponds to a timestep, with the first column representing susceptible and the second column representing infected counts.
        """
        if self.pop is not None:
            pop = self.pop
        else:
            pop = x.sum()
        if self.horizon is not None:
            steps = self.horizon

        assert len(x.shape) == 1 and x.shape[0] == 2

        output = torch.zeros(steps* 2, dtype=torch.float, requires_grad = False).reshape(steps, 2)
        output.data[0] = x.data

        for i in range(1, steps):
            output.data[i][0] = output.data[i-1][0] - self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop + self.gamma(output.data[i-1][1].unsqueeze(0))
            output.data[i][1] = output.data[i-1][1] + self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop - self.gamma(output.data[i-1][1].unsqueeze(0)) 
            

        return output

class SEIR(nn.Module):
    """
    Susceptible-Exposed-Infected-Recovered (SEIR) Model

    Parameters
    ----------
    horizon : int, optional
        Number of future time steps to simulate. If None, a single step is simulated unless overridden in the forward method.
    infection_rate : float, optional
        Infection rate parameter, representing the rate at which susceptible individuals become exposed. If None, must be initialized separately.
    recovery_rate : float, optional
        Recovery rate parameter, representing the rate at which infected individuals recover. If None, must be initialized separately.
    cure_rate : float, optional
        Natural immunity rate parameter, representing the rate at which individuals (across S, E, I, R compartments) return to susceptible due to loss of immunity. If None, must be initialized separately.
    latency : float, optional
        Latency rate parameter, representing the rate at which exposed individuals become infected. If None, must be initialized separately.
    population : int, optional
        Total population considered in the model. If None, the sum of the initial conditions (susceptible, exposed, infected, recovered) is used as the total population.

    Attributes
    ----------
    beta : torch.nn.Linear
        Linear layer with no bias to dynamically model the infection rate.
    gamma : torch.nn.Linear
        Linear layer with no bias to dynamically model the recovery rate.
    mu : torch.nn.Linear
        Linear layer with no bias to model the natural immunity rate.
    a : torch.nn.Linear
        Linear layer with no bias to model the latency rate.

    Returns
    -------
    torch.Tensor
        A tensor of shape (horizon, 4), representing the predicted number of susceptible, exposed, infected, and recovered individuals at each timestep.
        Each row corresponds to a timestep, with the columns representing the counts of susceptible, exposed, infected, and recovered individuals respectively.

    """
    def __init__(self, horizon=None, infection_rate = None, recovery_rate = None, cure_rate = None, latency = None, population = None):
        super(SEIR, self).__init__()
        self.pop = population

        self.beta = nn.Linear(1, 1, bias = False)
        self.gamma = nn.Linear(1, 1, bias = False)
        self.mu = nn.Linear(1, 1, bias = False)
        self.a = nn.Linear(1, 1, bias = False)

        if infection_rate is not None:
            self.beta.weight.data = torch.FloatTensor([infection_rate])

        if recovery_rate is not None:
            self.gamma.weight.data = torch.FloatTensor([recovery_rate])

        if cure_rate is not None:
            self.mu.weight.data = torch.FloatTensor([cure_rate])

        if latency is not None:
            self.a.weight.data = torch.FloatTensor([latency])
        
    def forward(self, x, steps=1):
        """
        Parameters
        ----------
        x : torch.Tensor
            The initial condition tensor for the model, representing the initial numbers of susceptible (S), exposed (E), 
            infected (I), and recovered (R) individuals. Expected shape is (4,), where elements correspond to S, E, I, and R counts.
        steps : int, optional
            Number of future time steps to simulate. If `horizon` is specified during initialization and not None, 
            it overrides this parameter. Default is 1 if `horizon` is None.

        Returns
        -------
        torch.Tensor
            A tensor of shape (steps, 4), representing the predicted number of susceptible, exposed, infected, and recovered 
            individuals at each timestep. Each row corresponds to a timestep, with columns representing the counts of 
            susceptible, exposed, infected, and recovered individuals respectively.
        """
        if self.pop is not None:
            pop = self.pop
        else:
            pop = x.sum()
        if self.horizon is not None:
            steps = self.horizon

        assert len(x.shape) == 1 and x.shape[0] == 4

        output = torch.zeros(steps* 4, dtype=torch.float, requires_grad = False).reshape(steps, 4)
        output.data[0] = x.data

        for i in range(1, steps):
            output.data[i][0] = output.data[i-1][0] - self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop + self.mu(pop - output.data[i-1][0]) # S
            output.data[i][1] = output.data[i-1][1] + self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop - self.mu(output.data[i-1][1]) - self.a(output.data[i-1][1])# E
            output.data[i][2] = output.data[i-1][2] + self.a(output.data[i-1][1]) - self.gamma(output.data[i-1][2]) - self.mu(output.data[i-1][2])# I
            output.data[i][3] = output.data[i-1][3] + self.gamma(output.data[i-1][2]) - self.mu(output.data[i-1][3])# R

        return output
