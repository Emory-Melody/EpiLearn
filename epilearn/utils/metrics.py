import torch
import torch.nn as nn
import torch.nn.functional as F

#——————————————————————————losses

def get_loss(loss_name = 'mse'):
    """
    Retrieves the specified loss function based on the input loss name. It supports mean squared error (MSE),
    a standardized loss (stan), an epidemic-collaboration specific loss (epi_cola), and cross-entropy loss.

    Parameters
    ----------
    loss_name : str, optional
        Name of the loss function to retrieve. Default is 'mse'.

    Returns
    -------
    callable
        The corresponding loss function as specified by loss_name.
    """
    loss_name = loss_name.lower()
    if loss_name == 'mse':
        return nn.MSELoss()
    if loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'stan':
        return stan_loss
    elif loss_name == 'epi_cola':
        return epi_cola_loss
    elif loss_name == 'ce':
        return cross_entropy_loss


def stan_loss(output, label, scale=0.5):
    """
    Calculates a combined mean squared error loss on predicted and physically informed predicted values,
    scaled by a given factor.

    Parameters
    ----------
    output : tuple of torch.Tensor
        The predicted values and the physically informed predicted values.
    label : torch.Tensor
        The ground truth values.
    scale : float, optional
        Scaling factor for the physical informed loss component. Default: 0.5.

    Returns
    -------
    torch.Tensor
        The calculated total loss as a scalar tensor.
    """
    pred_IR, pred_phy_IR = output
    mse = nn.MSELoss()
    total_loss = mse(pred_IR, label) + scale*mse(pred_phy_IR, label)
    return total_loss

def epi_cola_loss(output, label, scale=0.5):
    """
    Calculates a combined L1 and mean squared error loss on the output and an epidemiological output,
    scaled by a given factor.

    Parameters
    ----------
    output : tuple of torch.Tensor
        The primary model output and the epidemiological model output.
    label : torch.Tensor
        The ground truth values.
    scale : float, optional
        Scaling factor for the epidemiological loss component. Default: 0.5.

    Returns
    -------
    torch.Tensor
        The calculated total loss as a scalar tensor.
    """
    output, epi_output = output
    mse = nn.MSELoss()
    total_loss = F.l1_loss(output, label) + scale*mse(epi_output, label)
    return total_loss

def cross_entropy_loss(output, label):
    """
    Computes the cross-entropy loss between the logits and labels, adjusting the label tensor to fit the logits dimensions.

    Parameters
    ----------
    output : torch.Tensor
        The logits from the model.
    label : torch.Tensor
        The ground truth labels, scaled to match the number of classes based on output dimensions.

    Returns
    -------
    torch.Tensor
        The cross-entropy loss as a scalar tensor.
    """
    label = (((label-label.min())/(label.max()-label.min()+1))*output.shape[-1]).int()
    ce = nn.CrossEntropyLoss()
    return ce(output.float().view(-1, output.shape[-1]), label.long().view(-1))


#--------------------metrics------------------
def get_MSE(pred, target):
    """
    Calculates the Mean Absolute Error (MAE) between predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.

    Returns
    -------
    torch.Tensor
        The MAE value as a scalar tensor.
    """
    pred = pred.reshape(target.shape)
    mse_loss = nn.MSELoss(reduction='mean')
    return mse_loss(pred, target)

def get_MAE(pred, target):
    """
    Calculates the Mean Absolute Error (MAE) between predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.

    Returns
    -------
    torch.Tensor
        The MAE value as a scalar tensor.
    """
    pred = pred.reshape(target.shape)
    return torch.mean(torch.absolute(pred - target))

def get_RMSE(pred, target):
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.

    Returns
    -------
    torch.Tensor
        The RMSE value as a scalar tensor.
    """
    pred = pred.reshape(target.shape)
    mse_loss = nn.MSELoss(reduction='mean')
    return torch.sqrt(mse_loss(pred, target))

def get_ACC(pred, target):
    """
    Calculates the accuracy of predictions by comparing them to the targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted labels.
    target : torch.Tensor
        True labels.

    Returns
    -------
    torch.Tensor
        The accuracy as a scalar tensor.
    """
    result = pred.eq(target).sum()/len(pred.reshape(-1))
    return result
