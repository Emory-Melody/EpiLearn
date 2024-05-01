import torch
import numpy as np

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_MAE(pred, target):
    return torch.mean(torch.absolute(pred - target))

def normalize(X):
    if len(X.shape) == 3:
        means = torch.mean(X, axis=(0, 1))
        X = X - means.unsqueeze(0).unsqueeze(0)
        stds = torch.std(X, axis=(0, 1))
        X = X / stds.unsqueeze(0).unsqueeze(0)
    
    return X, means, stds

    

def normalize_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    try:
        A = A.numpy()
    except:
        pass
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return torch.FloatTensor(A_wave)
