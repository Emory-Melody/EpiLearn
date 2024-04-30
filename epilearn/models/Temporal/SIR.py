import torch
from torch import nn


class SIR(nn.Module):
    def __init__(self, infection_rate = None, recovery_rate = None, population = None):
        super(SIR, self).__init__()
        self.pop = population

        self.beta = nn.Linear(1, 1, bias = False)
        self.gamma = nn.Linear(1, 1, bias = False)

        if infection_rate is not None:
            assert self.beta.weight.data >= 0
            self.beta.weight.data = torch.FloatTensor([infection_rate])

        if recovery_rate is not None:
            assert self.gamma.weight.data >= 0
            self.gamma.weight.data = torch.FloatTensor([recovery_rate])
        
    def forward(self, x, steps = 1):
        if self.pop is not None:
            pop = self.pop
        else:
            pop = x.sum()

        assert len(x.shape) == 1 and x.shape[0] == 3

        output = torch.zeros(steps* 3, dtype=torch.float, requires_grad = False).reshape(steps, 3)
        output.data[0] = x.data

        for i in range(1, steps):
            output.data[i][0] = output.data[i-1][0] - self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop
            output.data[i][1] = output.data[i-1][1] + self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop - self.gamma(output.data[i-1][1].unsqueeze(0)) 
            output.data[i][2] = output.data[i-1][2] + self.gamma(output.data[i-1][1].unsqueeze(0))
            # if self.fixed_pop:
            #     if output.data[i][0] > pop:
            #         output.data[i][0] = pop
            #     if output.data[i][1] > pop:
            #         output.data[i][1] = pop
            #     if output.data[i][2] > pop:
            #         output.data[i][2] = pop
            # if output.data[i][0] < 0:
            #     output.data[i][0] = 0
            # if output.data[i][1] < 0:
            #     output.data[i][1] = 0
            # if output.data[i][2] < 0:
            #     output.data[i][2] = 0
        return output[1:]

class SIS(nn.Module):
    def __init__(self, infection_rate = None, recovery_rate = None, population = None):
        super(SIS, self).__init__()
        self.pop = population

        self.beta = nn.Linear(1, 1, bias = False)
        self.gamma = nn.Linear(1, 1, bias = False)

        if infection_rate is not None:
            assert self.beta.weight.data >= 0
            self.beta.weight.data = torch.FloatTensor([infection_rate])

        if recovery_rate is not None:
            assert self.gamma.weight.data >= 0
            self.gamma.weight.data = torch.FloatTensor([recovery_rate])
        
    def forward(self, x, steps = 1):
        if self.pop is not None:
            pop = self.pop
        else:
            pop = x.sum()

        assert len(x.shape) == 1 and x.shape[0] == 2

        output = torch.zeros(steps* 2, dtype=torch.float, requires_grad = False).reshape(steps, 2)
        output.data[0] = x.data

        for i in range(1, steps):
            output.data[i][0] = output.data[i-1][0] - self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop + self.gamma(output.data[i-1][1].unsqueeze(0))
            output.data[i][1] = output.data[i-1][1] + self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop - self.gamma(output.data[i-1][1].unsqueeze(0)) 
            

        return output[1:]

class SEIR(nn.Module):
    def __init__(self, infection_rate = None, recovery_rate = None, cure_rate = None, latency = None, population = None, ):
        super(SEIR, self).__init__()
        self.pop = population

        self.beta = nn.Linear(1, 1, bias = False)
        self.gamma = nn.Linear(1, 1, bias = False)
        self.mu = nn.Linear(1, 1, bias = False)
        self.a = nn.Linear(1, 1, bias = False)

        if infection_rate is not None:
            assert self.beta.weight.data >= 0
            self.beta.weight.data = torch.FloatTensor([infection_rate])

        if recovery_rate is not None:
            assert self.gamma.weight.data >= 0
            self.gamma.weight.data = torch.FloatTensor([recovery_rate])

        if cure_rate is not None:
            assert self.mu.weight.data >= 0
            self.mu.weight.data = torch.FloatTensor([cure_rate])

        if latency is not None:
            assert self.a.weight.data >= 0
            self.a.weight.data = torch.FloatTensor([latency])
        
    def forward(self, x, steps = 1):
        if self.pop is not None:
            pop = self.pop
        else:
            pop = x.sum()

        assert len(x.shape) == 1 and x.shape[0] == 4

        output = torch.zeros(steps* 4, dtype=torch.float, requires_grad = False).reshape(steps, 4)
        output.data[0] = x.data

        for i in range(1, steps):
            output.data[i][0] = output.data[i-1][0] - self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop + self.mu(pop - output.data[i-1][0]) # S
            output.data[i][1] = output.data[i-1][1] + self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))/pop - self.mu(output.data[i-1][1]) - self.a(output.data[i-1][1])# E
            output.data[i][2] = output.data[i-1][2] + self.a(output.data[i-1][1]) - self.gamma(output.data[i-1][2]) - self.mu(output.data[i-1][2])# I
            output.data[i][3] = output.data[i-1][3] + self.gamma(output.data[i-1][2]) - self.mu(output.data[i-1][3])# R

        return output[1:]