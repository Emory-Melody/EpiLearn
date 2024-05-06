import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader


from ...data import UniversalDataset
from ...utils import utils, metrics


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = "cpu"
        self.best_model = None
        self.best_output = None

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
            weight_decay=1e-3,
            initialize=True, 
            verbose=False, 
            patience=100, 
            shuffle=False,
            **kwargs):
        if initialize:
            self.initialize()

        train_dataset = UniversalDataset(x=train_input, y=train_target, graph=train_graph, dynamic_graph=train_dynamic_graph)
        val_dataset = UniversalDataset(x=val_input, y=val_target, graph=val_graph, dynamic_graph=val_dynamic_graph)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = metrics.get_loss(loss)

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
        edge_index = dataset.edge_index
        edge_weight = dataset.edge_weight
        
        epoch_training_losses = []
        for batch_data in train_loader:
            self.train()
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            y_batch = batch_data.y
            x_batch = batch_data.x

            out = self.forward(x_batch, edge_index, edge_weight)
            loss = loss_fn(out, y_batch)

            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss)
        return sum(epoch_training_losses)/len(epoch_training_losses)
    
    
    def evaluate(self, loss_fn, dataset, batch_size=1, device = 'cpu', shuffle=False):
        with torch.no_grad():
            self.eval()
            val_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
            edge_index = dataset.edge_index
            edge_weight = dataset.edge_weight
            val_losses = []
            outs = []
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                y_batch = batch_data.y
                x_batch = batch_data.x

                out = self.forward(x_batch, edge_index, edge_weight)
                val_loss = loss_fn(out, y_batch)

                val_losses.append(val_loss)
                outs.append(out)
            
            return sum(val_losses)/len(val_losses), torch.cat(outs, dim=0)

    def predict(self, feature, graph=None, states=None, dynamic_graph=None, batch_size=1, device = 'cpu', shuffle=False):
        """
        Returns
        -------
        torch.FloatTensor
        """
        print("\nPredicting Progress...")
        self.eval()
        dataset = UniversalDataset(x=feature, graph=graph, dynamic_graph=dynamic_graph)
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        edge_index = dataset.edge_index
        edge_weight = dataset.edge_weight
        outs = []
        for batch_data in tqdm(test_loader, total=len(test_loader)):
            batch_data = batch_data.to(device)
            y_batch = batch_data.y
            x_batch = batch_data.x

            out = self.forward(x_batch, edge_index, edge_weight)
            outs.append(out)
            
        return torch.stack(outs).squeeze()
    