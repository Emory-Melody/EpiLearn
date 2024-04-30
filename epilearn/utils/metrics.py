import torch
import torch.nn as nn


def get_loss(loss_name = 'mse'):
    loss_name = loss_name.lower()
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'stan':
        return stan_loss


def stan_loss(output, label, scale = 0.5):
    pred_IR, pred_phy_IR = output
    mse = nn.MSELoss()
    total_loss = mse(pred_IR, label) + scale*mse(pred_phy_IR, label)
    return total_loss