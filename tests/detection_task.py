"""Detection task demo: swap the graph encoder, then read the full report.

Same hotspot-classification task as tests/detection.py, but here two Spatial
encoders (GCN and SAGE) are built explicitly and handed to the task with
pretrained=, which is how you use a model whose constructor does not accept the
task's default arguments. Prints the built-in evaluation report, including
bootstrap confidence intervals.

Run from the repo root:

    python tests/detection_task.py
"""

import torch

from epilearn.models.Spatial.GCN import GCN
from epilearn.models.Spatial.SAGE import SAGE
from epilearn.data import Dataset
from epilearn.utils import transforms
from epilearn.tasks.detection import Detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)
print(f"device: {device}")

# settings
lookback = 1     # input window length
horizon = 2      # for Detection this is the NUMBER OF CLASSES, not a time span
epochs = 15
batch_size = 25

# load toy dataset and turn the counts into per-node class labels: is this region
# above the median region at this timestep?
dataset = Dataset()
dataset.load_toy_dataset()
dataset.y = (dataset.y > dataset.y.median(dim=1, keepdim=True).values).long()

# set_transforms(apply_now=True) normalizes the stored tensors right away;
# leave 'target' out, or the class labels would be normalized too
transformation = transforms.Compose({
    'features': [transforms.normalize_feat()],
    'graph': [transforms.normalize_adj()],
})
dataset.set_transforms(transformation, apply_now=True)
print(f"features mean {dataset.x.mean():.3e}, max graph weight {dataset.graph.max():.3f}")
print(f"process history: {sorted(dataset.get_process_history().keys())}")


def make_split(start, end):
    return dataset.generate_dataset(X=dataset.x[start:end], Y=dataset.y[start:end],
                                    adj=dataset.graph,
                                    lookback_window_size=lookback, horizon_size=1)


train_split = make_split(0, 300)
val_split = make_split(300, 400)
test_split = make_split(400, 539)
num_features = train_split['features'].shape[-1]

encoders = {
    'GCN': GCN(num_features=num_features, hidden_dim=16,
               num_classes=horizon, device=device),
    'SAGE': SAGE(num_features=num_features, hidden_dim=16,
                 num_classes=horizon, device=device),
}

scores = {}
for name, model in encoders.items():
    print(f"\n########## {name}")
    task = Detection(prototype=type(model),
                     model=model,
                     dataset=dataset,
                     lookback=lookback,
                     horizon=horizon,
                     device=device)
    # pretrained= skips the task's own model construction, so SAGE's
    # (num_features, hidden_dim, num_classes) signature is respected
    task.train_model(train_split=train_split,
                     val_split=val_split,
                     test_split=test_split,
                     train_loss='ce',
                     val_loss='ce',
                     epochs=epochs,
                     batch_size=batch_size,
                     pretrained=model)
    evaluation = task.evaluate_model(dataset=test_split, n_bootstrap=100)
    scores[name] = evaluation

labels = test_split['targets'].reshape(-1)
majority = (labels == labels.mode().values).float().mean().item()
print(f"\n{'encoder':8s} accuracy  macro F1  accuracy 95% CI")
for name, evaluation in scores.items():
    low, high = evaluation['bootstrap_accuracy_ci']
    print(f"{name:8s} {evaluation['accuracy']:.4f}    {evaluation['macro_f1']:.4f}    "
          f"[{low:.4f}, {high:.4f}]")
print(f"{'majority':8s} {majority:.4f}    (always predict the most common class)")
