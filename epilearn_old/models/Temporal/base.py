import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch
from tqdm import tqdm

from ...utils.utils import *
from ...utils.metrics import get_loss

class BaseModel(nn.Module):
    def __init__(self, device = 'cpu'):
        super(BaseModel, self).__init__()
        self.device = device

    def fit(self, 
            train_input, 
            train_target, 
            train_states=None, 
            train_graph=None, 
            train_dynamic_graph=None,
            val_input=None, 
            val_target=None,
            val_states=None, 
            val_graph= None, 
            val_dynamic_graph=None,
            loss='mse', 
            epochs=1000, 
            batch_size=10,
            lr=1e-3, 
            initialize=True, 
            verbose=False, 
            patience=100, 
            **kwargs):
        if initialize:
            self.initialize()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = get_loss(loss)

        training_losses = []
        validation_losses = []
        early_stopping = patience
        best_val = float('inf')
        for epoch in tqdm(range(epochs)):
            
            loss = self.train_epoch(optimizer = optimizer, loss_fn = loss_fn, feature = train_input,  target = train_target, batch_size = batch_size, device = self.device)
            training_losses.append(loss)
            if val_input is not None and val_input.numel():
                val_loss, output = self.evaluate(loss_fn = loss_fn, feature = val_input,  target = val_target, device = self.device)
                validation_losses.append(val_loss)

                if best_val > val_loss:
                    best_val = val_loss
                    self.output = output
                    best_weights = deepcopy(self.state_dict())
                    patience = early_stopping
                else:
                    patience -= 1

                if epoch > early_stopping and patience <= 0:
                    break

                if verbose and epoch%50 == 0:
                    print(f"######### epoch:{epoch}")
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))
            else:
                validation_losses.append(None)
                best_weights = deepcopy(self.state_dict())
                if verbose and epoch%50 == 0:
                    print(f"######### epoch:{epoch}")
                    print("Training loss: {}".format(training_losses[-1]))
                    print("Validation loss: {}".format(validation_losses[-1]))

        print("\n")
        print("Final Training loss: {}".format(training_losses[-1]))
        print("Final Validation loss: {}".format(validation_losses[-1]))

        self.load_state_dict(best_weights)

        
    def train_epoch(self, optimizer, loss_fn, feature, target = None, batch_size = 1, device = 'cpu'):
        """
        Trains one epoch with the given data.
        :param feature: Training features of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """
        permutation = torch.randperm(feature.shape[0])

        epoch_training_losses = []
        for i in range(0, feature.shape[0], batch_size):
            self.train()
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            X_batch, y_batch = feature[indices], target[indices]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)
            
            out = self.forward(X_batch)
            loss = loss_fn(out.reshape(y_batch.shape), y_batch)
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
        return sum(epoch_training_losses)/len(epoch_training_losses)
    
    def evaluate(self, loss_fn, feature, target = None, device = 'cpu'):
        with torch.no_grad():
            self.eval()
            feature = feature.to(device=device)
            target = target.to(device=device)

            out = self.forward(feature)
            val_loss = loss_fn(out.reshape(target.shape), target)
            val_loss = val_loss.detach().cpu().numpy().item()
            
            return val_loss, out

    def predict(self, feature, graph=None, states=None, dynamic_graph=None):
        """
        Returns
        -------
        torch.FloatTensor
        """
        self.eval()
        result = self.forward(feature.to(self.device))

        return result.detach().cpu()
