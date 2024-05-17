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


def normalize(X):
    if len(X.shape) == 3:
        means = torch.mean(X, axis=(0, 1))
        X = X - means.unsqueeze(0).unsqueeze(0)
        stds = torch.std(X, axis=(0, 1))
        X = X / stds.unsqueeze(0).unsqueeze(0)
    X[torch.where(torch.isnan(X))]=0
    return X, means, stds


def normalize_adj(Adj):
    """
    Returns the degree normalized adjacency matrix.
    """
    if Adj is None:
        return None
    try:
        Adj = Adj.numpy()
    except:
        pass

    if len(Adj.shape) > 2:
        A_wave = Adj
        for i in range(Adj.shape[0]):
            A = Adj[i].reshape(Adj.shape[1], Adj.shape[1])
            A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
            D = np.array(np.sum(A, axis=1)).reshape((-1,))
            D[D <= 10e-5] = 10e-5    # Prevent infs
            diag = np.reciprocal(np.sqrt(D))
            A_wave[i] = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                                diag.reshape((1, -1))).reshape(Adj[i].shape)

    else:
        A = Adj
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5    # Prevent infs
        diag = np.reciprocal(np.sqrt(D))
        A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                            diag.reshape((1, -1)))
    return torch.FloatTensor(A_wave)

def diff(features):
    return torch.diff(features, dim=0)


def Degree_Matrix(ST_matrix):

    row_sum = torch.sum(ST_matrix, 0)

    ## degree matrix
    dim = len(ST_matrix)
    D_matrix = torch.zeros(dim, dim)
    for i in range(dim):
        D_matrix[i, i] = 1 / max(torch.sqrt(row_sum[i]), 1)

    return D_matrix

"""
The binary spatio-temporal adjacency matrix used in USTGCN
"""
def Static_full(n, t, A):
    """
    :param n: the dimension of the spatial adjacency matrix
    :param t: the length of periods
    :param A: the spatial adjacency matrix
    :return: the full USTGCN spatio-temporal adjacency matrix
    """
    I_S = torch.diag_embed(torch.ones(n))
    I_T = torch.diag_embed(torch.ones(t))

    C_S = A
    C_T = torch.tril(torch.ones(t, t), diagonal=-1)

    S = I_S + C_S
    A_ST = kronecker(C_T, S) + kronecker(I_T, C_S)

    return A_ST


"""
Use kronecker product to construct the spatio-temporal adjacency matrix
"""
def kronecker(A, B):
    """
    :param A: the temporal adjacency matrix
    :param B: the spatial adjacency matrix
    :return: the adjacency matrix of one space-time neighboring block
    """
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.contiguous().view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB

def edge_to_adj(edge_index, num_nodes):
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj




    





