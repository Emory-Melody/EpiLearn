import torch
import os
import matplotlib.pyplot as plt
#os.sys.path.append("../Epilearn")
os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from epilearn.models.SpatialTemporal.STGCN import STGCN
from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.GAT import GAT
from epilearn.models.Spatial.SAGE import SAGE
from epilearn.models.Spatial.GIN import GIN
from epilearn.models.SpatialTemporal.DCRNN import DCRNN
from epilearn.models.SpatialTemporal.GraphWaveNet import GraphWaveNet
from epilearn.data import UniversalDataset
from epilearn.utils import utils, metrics, transforms

import torch_geometric


# initial settings
#device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(7)

lookback = 13 # inputs size
horizon = 1 # predicts size

epochs = 50 # training epochs
batch_size = 8 # training batch size

permute = True

# load toy dataset
dataset = UniversalDataset()
dataset.load_toy_dataset()

# preprocessing
features, mean, std = utils.normalize(dataset.x)
adj_norm = utils.normalize_adj(dataset.graph)
adj_dynamic_norm = utils.normalize_adj(dataset.dynamic_graph)

features = features.to(device)
adj_norm = adj_norm.to(device)
adj_dynamic_norm = adj_dynamic_norm.to(device)

# prepare datasets
train_rate = 0.6 
val_rate = 0.2

target_feat_idx = None
target_idx = None

split_line1 = int(features.shape[0] * train_rate)
split_line2 = int(features.shape[0] * (train_rate + val_rate))

train_original_data = features[:split_line1, :, :]
val_original_data = features[split_line1:split_line2, :, :]
test_original_data = features[split_line2:, :, :]

train_original_states = dataset.states[:split_line1, :, :]
val_original_states = dataset.states[split_line1:split_line2, :, :]
test_original_states = dataset.states[split_line2:, :, :]

train_input, train_target, train_states, train_adj = dataset.generate_dataset(X = train_original_data, Y = train_original_data[..., 0], states = train_original_states, dynamic_adj = adj_dynamic_norm, lookback_window_size = lookback, horizon_size = horizon, permute = permute, feat_idx = target_feat_idx, target_idx = target_idx)
val_input, val_target, val_states, val_adj = dataset.generate_dataset(X = val_original_data, Y = val_original_data[..., 0], states = val_original_states, dynamic_adj = adj_dynamic_norm, lookback_window_size = lookback, horizon_size = horizon, permute = permute, feat_idx = target_feat_idx, target_idx = target_idx)
test_input, test_target, test_states, test_adj = dataset.generate_dataset(X = test_original_data, Y = test_original_data[..., 0], states = test_original_states, dynamic_adj = adj_dynamic_norm, lookback_window_size = lookback, horizon_size = horizon, permute = permute, feat_idx = target_feat_idx, target_idx = target_idx)

rows, cols = torch.where(adj_norm != 0)
weights = adj_norm[rows, cols]

edge_index = torch.stack([rows.long(), cols.long()], dim=0).to(device)
edge_weight = weights.clone().detach().to(device)


print("original: ", train_input.shape)
data = {"features": train_input}
'''transform_lists = torch.nn.Sequential(transforms.convert_to_frequency(ftype="fft"), transforms.test_seq())
#transform_list = [transforms.convert_to_frequency()]
transform_list = torch.nn.Sequential(transforms.learnable_time_embedding(timesteps=13,embedding_dim=10))
#transform_list = torch.nn.Sequential(transforms.add_time_embedding(embedding_dim=10))
transform = transforms.Compose(transform_list, device)'''

transform_list = {"features": [transforms.add_time_embedding(embedding_dim=10)]}
transform = transforms.Compose(transform_list, device)

data = transform(data)
#val_input = transform(val_input)

print("transform: ", data["features"].shape)


#train_dataset = UniversalDataset(x=train_input[..., 0,:], y=train_target, graph=adj_norm, edge_index=edge_index, edge_weight=edge_weight)
#val_dataset = UniversalDataset(x=val_input[..., 0,:], y=val_target, graph=adj_norm, edge_index=edge_index, edge_weight=edge_weight)



model = GCN(num_features=train_input.shape[3],
        hidden_dim=16,
        num_classes=horizon,
        nlayers=2, with_bn=True,
        dropout=0.3, device=device)

'''model = GAT(num_features=train_input.shape[3],
        hidden_dim=16,
        num_classes=horizon,
        nlayers=2, with_bn=True, nheads=[2,4], concat=True,
        dropout=0.3, device=device)'''

'''model = SAGE(num_features=train_input.shape[2]*train_input.shape[3],
        hidden_dim=16,
        num_classes=horizon,
        nlayers=1, with_bn=True, aggr=torch_geometric.nn.GRUAggregation,
        dropout=0.3, device=device)'''

'''model = GIN(num_features=train_input.shape[2]*train_input.shape[3],
        hidden_dim=16,
        num_classes=horizon,
        nlayers=2, 
        dropout=0.3, device=device)'''


'''model = DCRNN(num_features=train_input.shape[3],
              num_timesteps_input=lookback,
              num_timesteps_output=horizon,
              num_classes=1,
              max_diffusion_step=2, 
              filter_type="laplacian",
              num_rnn_layers=1, 
              rnn_units=1,
              nonlinearity="tanh",
              dropout=0.5,
              device=device)'''


'''model = GraphWaveNet(num_timesteps_input=train_input.shape[3], num_timesteps_output=horizon, 
                     adj_m=adj_norm, gcn_bool=True, 
                     addaptadj=True, aptinit=None, 
                     blocks=4, nlayers=2,
                     residual_channels=32, dilation_channels=32, 
                     skip_channels=32 * 8, end_channels=32 * 16,dropout=0.3, device=device)'''

model = model.to(device)


model.fit(
        train_input=train_input[..., 0,:], 
        train_target=train_target, 
        train_states=None, 
        train_graph=adj_norm, 
        train_dynamic_graph=None,
        val_input=val_input[..., 0,:], 
        val_target=val_target,
        val_states=None, 
        val_graph=adj_norm, 
        val_dynamic_graph=None,
        loss='mse', 
        epochs=100, 
        batch_size=10,
        lr=1e-3, 
        weight_decay=1e-3,
        initialize=True, 
        verbose=True, 
        patience=10, 
        shuffle=False,
        )
        
'''model.fit(
        train = train_dataset, 
        val_dataset = val_dataset, 
        verbose = True, 
        batch_size = batch_size,
        lr=1e-3,
        shuffle=False,
        patience=5,
        epochs = epochs)
'''


# evaluate
#output_dim=[47, horizon]
#test_dataset = UniversalDataset(x=test_input[..., 0,:], edge_index=edge_index)
#out = model.predict(dataset=test_dataset, batch_size=batch_size, device=device, output_dim=output_dim)
out = model.predict(feature=test_input[..., 0,:], 
                    graph=adj_norm, 
                    states=None, 
                    dynamic_graph=None, 
                    batch_size=1, 
                    device = device, 
                    shuffle=False)
#out = model.predict(feature=test_input, graph=adj_norm, states=None, dynamic_graph=None)
print(out.shape)

preds = out.detach().cpu() * std[0] + mean[0]
targets = test_target.detach().cpu() * std[0] + mean[0]
# MAE
mae = metrics.get_MAE(preds, targets)
print(f"MAE: {mae.item()}")