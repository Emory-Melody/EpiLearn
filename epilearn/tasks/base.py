import torch

class BaseTask:
    def __init__(self, prototype, model, dataset, lookback, horizon, ahead, device='cpu'):
        self.model = model
        self.prototype = prototype
        self.dataset = dataset
        self.device = device
        self.lookback = lookback
        self.horizon = horizon
        self.ahead = ahead
    
    def load_model(self, model):
        pass

    def load_dataset(self, dataset):
        pass

    def print_dataset(self, dataset = None):
        pass
