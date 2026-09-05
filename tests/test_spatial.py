"""Spatial (node-level) models demo: GCN and SAGE doing detection -- per region, is
today a high-incidence day? -- via the low-level model.fit / model.predict path.

Run it from the repo root with:

    python tests/test_spatial.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)          # load_toy_dataset() resolves './datasets' from the cwd

import torch

from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.SAGE import SAGE
# The other two node-level models are left out on purpose: in 0.1.0
# Spatial.GIN.forward raises UnboundLocalError, and Spatial.GAT normalizes its
# attention over every edge of the collated batch at once, which costs ~0.5 s per
# sample on this dense 47-region graph and collapses its output to one class.
from epilearn.data import Dataset
from epilearn.utils import transforms


# ### Configs

device = torch.device('cpu')
torch.manual_seed(7)

lookback = 1        # a spatial model sees one timestep at a time
num_classes = 2     # high incidence vs. not

epochs = 15
batch_size = 32


# ### Load the toy dataset and turn it into a classification problem

dataset = Dataset()                 # 0.0.x called this UniversalDataset
dataset.load_toy_dataset()
dataset.dynamic_graph = None        # spatial models only use the static graph
# Detection is node classification, so binarize the target: label each
# (day, region) pair by whether its case count is above the dataset median.
dataset.y = (dataset.y > dataset.y.median()).long()
print(dataset)
print(f"positive class share: {dataset.y.float().mean():.3f}")

# The toy graph is a raw OD matrix (values up to 5.4e6), so normalize it before
# feeding it to a graph model. 0.1.0: Compose.__call__ returns (data, history), a
# tuple -- set_transforms(apply_now=True) applies it and keeps the history for you.
transformation = transforms.Compose({
    'features': [transforms.normalize_feat()],
    'graph': [transforms.normalize_adj()],
})
dataset.set_transforms(transformation, apply_now=True)

features = dataset.x.to(device)
labels = dataset.y.to(device)
adj_norm = dataset.graph.to(device)


# ### Build the sliding windows
# 0.1.0: generate_dataset returns a DICT (features/targets/states/dynamic_graph/graph)
# and needs adj= passed explicitly -- it no longer falls back to dataset.graph.

def make_split(start, end):
    split = dataset.generate_dataset(X=features[start:end],
                                     Y=labels[start:end],
                                     adj=adj_norm,
                                     lookback_window_size=lookback,
                                     horizon_size=1)
    # features: (samples, lookback, nodes, channels) -> drop the length-1 time axis
    # to get the (samples, nodes, channels) a spatial model expects.
    # targets: (samples, nodes, 1), one class label per region per day.
    return split['features'][:, 0], split['targets']


train_input, train_target = make_split(0, 320)
val_input, val_target = make_split(320, 430)
test_input, test_target = make_split(430, 539)
num_features = train_input.shape[-1]
print(f"train_input {tuple(train_input.shape)} (samples, nodes, channels)")


# ### Train each model and score it on the held-out window

def train_and_score(model):
    model = model.to(device)
    model.fit(train_input=train_input,
              train_target=train_target,
              train_states=None,
              train_graph=adj_norm,
              train_dynamic_graph=None,
              val_input=val_input,
              val_target=val_target,
              val_states=None,
              val_graph=adj_norm,
              val_dynamic_graph=None,
              loss='ce',
              epochs=epochs,
              batch_size=batch_size,
              lr=1e-2,
              weight_decay=1e-3,
              initialize=True,
              verbose=False,
              patience=10,
              shuffle=False)
    # predict() returns class logits, (samples, nodes, num_classes).
    logits = model.predict(feature=test_input,
                           graph=adj_norm,
                           states=None,
                           dynamic_graph=None,
                           batch_size=batch_size,
                           device=device,
                           shuffle=False)
    preds = logits.argmax(dim=-1)
    return (preds == test_target.squeeze(-1).cpu()).float().mean().item()


results = {}
print("\n########## GCN ##########")
results['GCN'] = train_and_score(
    GCN(num_features=num_features, hidden_dim=16, num_classes=num_classes,
        nlayers=2, with_bn=True, dropout=0.3, device=device))
print("\n########## SAGE ##########")
results['SAGE'] = train_and_score(
    SAGE(num_features=num_features, hidden_dim=16, num_classes=num_classes,
         nlayers=1, with_bn=True, aggr='mean', dropout=0.3, device=device))


# ### Results

share = test_target.float().mean().item()
majority = max(share, 1 - share)
print(f"\ntest accuracy (majority-class baseline: {majority:.3f})")
for name, acc in results.items():
    print(f"  {name:4s}: {acc:.3f}")
