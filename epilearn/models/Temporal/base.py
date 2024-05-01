import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch

from ...utils.utils import *


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def fit(self,data, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
        if initialize:
            self.initialize()

        # self.data = pyg_data[0].to(self.device)
        self.data = data.to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def fit_with_val(self, data, train_iters=1000, initialize=True, patience=100, verbose=False, **kwargs):
        if initialize:
            self.initialize()

        self.data = data.to(self.device)
        self.data.train_mask = self.data.train_mask + self.data.val1_mask
        self.data.val_mask = self.data.val2_mask
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print(f'=== training {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0

        x, edge_index = self.data.x, self.data.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            output = self.forward(x, edge_index)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 50 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(x, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = accuracy(output[val_mask], labels[val_mask])

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
                best_epoch = i
            else:
                patience -= 1

            if i > early_stopping and patience <= 0:
                break

        if verbose:
             # print('=== early stopping at {0}, loss_val = {1} ==='.format(best_epoch, best_loss_val) )
             print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val) )
        self.load_state_dict(weights)

    def predict(self, x=None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities)
        """
        self.eval()
        if x is None:
            x = self.data.x
        return self.forward(x )
