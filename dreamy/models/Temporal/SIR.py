import torch
from torch import nn


class SIR(nn.Module):
    def __init__(self, infection_rate = None, recovery_rate = None, fixed_population = True):
        super(SIR, self).__init__()
        self.fixed_pop = fixed_population

        self.beta = nn.Linear(1, 1, bias = False)
        self.gamma = nn.Linear(1, 1, bias = False)
        if self.gamma.weight.data < 0:
            self.gamma.weight.data = -self.gamma.weight.data
        if self.beta.weight.data < 0:
            self.beta.weight.data = -self.beta.weight.data

        if infection_rate is not None:
            self.beta.weight.data = torch.FloatTensor([infection_rate])

        if recovery_rate is not None:
            self.gamma.weight.data = torch.FloatTensor([recovery_rate])
        
    def forward(self, x, steps):
        pop = x.sum()
        assert len(x.shape) == 1 and x.shape[0] == 3

        output = torch.zeros(steps* 3, dtype=torch.float, requires_grad = False).reshape(steps, 3)
        output.data[0] = x.data

        for i in range(1, steps):
            output.data[i][0] = output.data[i-1][0] - self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0))
            output.data[i][1] = output.data[i-1][1] + self.beta((output.data[i-1][0] * output.data[i-1][1]).unsqueeze(0)) - self.gamma(output.data[i-1][1].unsqueeze(0))
            output.data[i][2] = output.data[i-1][2] + self.gamma(output.data[i-1][1].unsqueeze(0))
            if self.fixed_pop:
                if output.data[i][0] > pop:
                    output.data[i][0] = pop
                if output.data[i][1] > pop:
                    output.data[i][1] = pop
                if output.data[i][2] > pop:
                    output.data[i][2] = pop
            if output.data[i][0] < 0:
                output.data[i][0] = 0
            if output.data[i][1] < 0:
                output.data[i][1] = 0
            if output.data[i][2] < 0:
                output.data[i][2] = 0

        return output[1:]