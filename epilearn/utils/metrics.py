import torch
import torch.nn as nn
import torch.nn.functional as F

#——————————————————————————losses

def get_loss(loss_name = 'mse'):
    loss_name = loss_name.lower()
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'stan':
        return stan_loss
    elif loss_name == 'epi_cola':
        return epi_cola_loss
    elif loss_name == 'ce':
        return cross_entropy_loss


def stan_loss(output, label, scale=0.5):
    pred_IR, pred_phy_IR = output
    mse = nn.MSELoss()
    total_loss = mse(pred_IR, label) + scale*mse(pred_phy_IR, label)
    return total_loss

def epi_cola_loss(output, label, scale=0.5):
    output, epi_output = output
    mse = nn.MSELoss()
    total_loss = F.l1_loss(output, label) + scale*mse(epi_output, label)
    return total_loss

def cross_entropy_loss(output, label):
    label = (((label-label.min())/(label.max()-label.min()+1))*output.shape[-1]).int()
    ce = nn.CrossEntropyLoss()
    return ce(output.float().view(-1, output.shape[-1]), label.long().view(-1))


#--------------------metrics------------------
def get_MAE(pred, target):
    return torch.mean(torch.absolute(pred - target))

def get_RMSE(pred, target):
    mse_loss = nn.MSELoss(reduction='mean')
    return torch.sqrt(mse_loss(pred, target))

def get_ACC(pred, target):
    result = pred.eq(target).sum()/len(pred)
    return result