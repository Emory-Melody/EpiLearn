import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from ...utils.utils import *


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = "cpu"
        self.best_model = None
        self.best_output = None

    def fit(self, train_dataset, val_dataset, epochs=1000, batch_size=10, initialize=True,
            verbose=False, patience=100, lr=1e-3, shuffle=False, weight_decay=1e-3, **kwargs):
        if initialize:
            self.initialize()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        training_losses = []
        validation_losses = []
        early_stopping = patience
        best_val = float('inf')
        es_flag = False
        for epoch in tqdm(range(epochs)):
            loss = self.train_epoch(optimizer = optimizer, loss_fn = loss_fn, dataset=train_dataset, 
                                    batch_size = batch_size, device = self.device, shuffle=shuffle)
            training_losses.append(loss)

            val_loss, output = self.evaluate(loss_fn = loss_fn, dataset=val_dataset, 
                                             batch_size = batch_size, device = self.device, shuffle=shuffle)
            validation_losses.append(val_loss)

            if best_val > val_loss:
                best_epoch = epoch
                best_train = loss
                best_val = val_loss
                self.best_output = output
                best_weights = deepcopy(self.state_dict())
                self.best_model = best_weights
                patience = early_stopping
            else:
                patience -= 1


            if verbose and epoch%1 == 0:
                print(f"######### epoch:{epoch}")
                print("Training loss: {}".format(training_losses[-1]))
                print("Validation loss: {}".format(validation_losses[-1]))

            if epoch > early_stopping and patience <= 0:
                es_flag = True
                #print(f"Early stop at Epoch {epoch}")
                break

            

        if es_flag:
            print(f"Early stop at Epoch {epoch}!")
        print("\nFinal Training loss: {}".format(training_losses[-1]))
        print("Final Validation loss: {}".format(validation_losses[-1]))
        print("Best Epoch: {}".format(best_epoch))
        print("Best Training loss: {}".format(best_train))
        print("Best Validation loss: {}".format(best_val))

        self.load_state_dict(best_weights)

        
    def train_epoch(self, optimizer, loss_fn, dataset, batch_size = 1, device = 'cpu', shuffle=False):
        """
        Trains one epoch with the given data.
        :param feature: Training features of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """
        
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        
        epoch_training_losses = []
        for batch_data in train_loader:
            self.train()
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            y_batch = batch_data.y

            out = self.forward(batch_data)
            loss = loss_fn(out, y_batch)

            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss)
        return sum(epoch_training_losses)/len(epoch_training_losses)
    
    
    def evaluate(self, loss_fn, dataset, batch_size=1, device = 'cpu', shuffle=False):
        with torch.no_grad():
            self.eval()
            val_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
            val_losses = []
            outs = []
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                y_batch = batch_data.y

                out = self.forward(batch_data)
                val_loss = loss_fn(out, y_batch)

                val_losses.append(val_loss)
                outs.append(out)
            
            return sum(val_losses)/len(val_losses), torch.reshape(torch.cat(outs, dim=0), (dataset.output_dim))

    def predict(self, dataset, batch_size=1, device = 'cpu', shuffle=False, output_dim=None):
        """
        Returns
        -------
        torch.FloatTensor
        """
        print("\nPredicting Progress...")
        self.eval()
        
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        outs = []
        for batch_data in tqdm(test_loader, total=len(test_loader)):
            batch_data = batch_data.to(device)
            y_batch = batch_data.y

            out = self.forward(batch_data)
            outs.append(out)
            
        return torch.reshape(torch.cat(outs, dim=0), (len(dataset), *output_dim))
    
