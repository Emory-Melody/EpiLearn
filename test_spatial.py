import torch
import os
import matplotlib.pyplot as plt

#os.sys.path.append("./Dreamy")
from dreamy.models.SpatialTemporal.STGCN import STGCN
from dreamy.models.Spatial.GCN import GCN
from dreamy.models.Spatial.GAT import GAT
from dreamy.models.Spatial.SAGE import SAGE
from dreamy.models.Spatial.DCRNN import DCRNN
from dreamy.data import UniversalDataset, SpatialDataset
from dreamy.utils import utils

import torch_geometric


# initial settings
#device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(7)

lookback = 13 # inputs size
horizon = 5 # predicts size

epochs = 50 # training epochs
batch_size = 2 # training batch size

permute = True

# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()

# preprocessing
features, mean, std = utils.normalize(dataset.x)
adj_norm = utils.normalize_adj(dataset.graph)

features = features.to(device)
adj_norm = adj_norm.to(device)

# prepare datasets
train_rate = 0.6 
val_rate = 0.2

split_line1 = int(features.shape[0] * train_rate)
split_line2 = int(features.shape[0] * (train_rate + val_rate))

train_original_data = features[:split_line1, :, :]
val_original_data = features[split_line1:split_line2, :, :]
test_original_data = features[split_line2:, :, :]

train_input, train_target = dataset.generate_dataset(X = train_original_data, Y = train_original_data[:, :, 0], lookback_window_size = lookback, horizon_size = horizon, permute = permute)
val_input, val_target = dataset.generate_dataset(X = val_original_data, Y = val_original_data[:, :, 0], lookback_window_size = lookback, horizon_size = horizon, permute = permute)
test_input, test_target = dataset.generate_dataset(X = test_original_data, Y = test_original_data[:, :, 0], lookback_window_size = lookback, horizon_size = horizon, permute = permute)


print(train_input.shape)
print(train_target.shape)
print(test_input.shape)



rows, cols = torch.where(adj_norm != 0)
weights = adj_norm[rows, cols]

edge_index = torch.stack([rows.long(), cols.long()], dim=0).to(device)
edge_weight = weights.clone().detach().to(device)



train_dataset = SpatialDataset(x=train_input, y=train_target, adj_m=adj_norm)
val_dataset = SpatialDataset(x=val_input, y=val_target, adj_m=adj_norm)



'''model = GCN(input_dim=train_input.shape[2]*train_input.shape[3],
        hidden_dim=16,
        output_dim=horizon,
        nlayers=2, with_bn=True,
        dropout=0.3, device=device)'''

'''model = GAT(input_dim=train_input.shape[2]*train_input.shape[3],
        hidden_dim=16,
        output_dim=horizon,
        nlayers=2, with_bn=True, nheads=[2,3], concat=False,
        dropout=0.3, device=device)'''

'''model = SAGE(input_dim=train_input.shape[2]*train_input.shape[3],
        hidden_dim=16,
        output_dim=horizon,
        nlayers=1, with_bn=True, aggr=torch_geometric.nn.GRUAggregation,
        dropout=0.3, device=device)'''
        

model = DCRNN(input_dim=train_input.shape[3],
              seq_len=lookback,
              output_dim=1,
              horizon=horizon,
              max_diffusion_step=2, 
              filter_type="laplacian",
              num_rnn_layers=1, 
              rnn_units=1,
              nonlinearity="tanh",
              dropout=0.5,
              device=device)

model = model.to(device)


model.fit(
        train_dataset = train_dataset, 
        val_dataset = val_dataset, 
        verbose = True, 
        batch_size = batch_size,
        lr=1e-3,
        shuffle=False,
        patience=5,
        epochs = epochs)

# evaluate
output_dim=[47, horizon]
test_dataset = SpatialDataset(x=test_input, adj_m=adj_norm)
out = model.predict(dataset=test_dataset, batch_size=batch_size, device=device, output_dim=output_dim)
print(out.shape)

preds = out.detach().cpu() * std[0] + mean[0]
targets = test_target.detach().cpu() * std[0] + mean[0]
# MAE
mae = utils.get_MAE(preds, targets)
print(f"MAE: {mae.item()}")




