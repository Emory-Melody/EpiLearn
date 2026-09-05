"""Detection task demo: classify which regions are outbreak hotspots next step.

The shortest working Detection pipeline -- binarize the target into per-node class
labels, cut sliding windows with Dataset.generate_dataset, then
train_model / evaluate_model. See tests/detection_task.py for the same task with
transforms, a second graph encoder and bootstrap confidence intervals.

Run from the repo root:

    python tests/detection.py
"""

import torch

from epilearn.models.Spatial.GCN import GCN
from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.tasks.detection import Detection

device = torch.device('cpu')
torch.manual_seed(7)

# settings
lookback = 1     # input window length
horizon = 2      # for Detection this is the NUMBER OF CLASSES, not a time span
epochs = 15
batch_size = 25

# load toy dataset: x=(539, 47, 4) counts, graph=(47, 47) mobility
dataset = Dataset()
dataset.load_toy_dataset()

# Detection needs per-node CLASS labels, so binarize the raw counts: is this
# region above the median region at this timestep? A single global threshold
# would not work here -- the toy series trends upward, so it would label almost
# every late timestep the same way.
dataset.y = (dataset.y > dataset.y.median(dim=1, keepdim=True).values).long()

# z-score the features and degree-normalize the graph (raw OD values reach 5.4e6)
dataset.set_transforms(transforms.Compose({
    "features": [transforms.normalize_feat()],
    "graph": [transforms.normalize_adj()]}), apply_now=True)


def make_split(start, end):
    # horizon_size=1 is the TARGET window (one timestep of labels); the class
    # count lives in Detection(horizon=...). generate_dataset returns a dict.
    return dataset.generate_dataset(X=dataset.x[start:end], Y=dataset.y[start:end],
                                    adj=dataset.graph,
                                    lookback_window_size=lookback, horizon_size=1)


train_split = make_split(0, 300)
val_split = make_split(300, 400)
test_split = make_split(400, 539)
print(f"train {tuple(train_split['features'].shape)} -> {tuple(train_split['targets'].shape)} "
      f"labels, test {tuple(test_split['features'].shape)}")

# Initialize task. rolling_train does not support the 'ce' loss yet, so the
# single-split trainer is the supported path for Detection.
task = Detection(prototype=GCN,
                 dataset=dataset,
                 lookback=lookback,
                 horizon=horizon,
                 device=device)

# Training
result = task.train_model(train_split=train_split,
                          val_split=val_split,
                          test_split=test_split,
                          train_loss='ce',
                          val_loss='ce',
                          epochs=epochs,
                          batch_size=batch_size,
                          model_args={'num_classes': horizon})

# Evaluation, against always predicting the most common class
evaluation = task.evaluate_model(dataset=test_split, compute_bootstrap_ci=False)
labels = test_split['targets'].reshape(-1)
majority = (labels == labels.mode().values).float().mean().item()

print(f"\ntest cross-entropy : {result['loss']:.4f}")
print(f"GCN accuracy       : {evaluation['accuracy']:.4f}")
print(f"GCN macro F1       : {evaluation['macro_f1']:.4f}")
print(f"majority-class acc : {majority:.4f}")
