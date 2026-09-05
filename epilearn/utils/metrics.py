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
    if type(loss_name) is not str:
        # print("using custom loss function")
        return loss_name

    loss_name = loss_name.lower()
    if loss_name == 'mse': 
        return nn.MSELoss(reduction='mean')
    if loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'stan':
        return stan_loss
    elif loss_name == 'epi_cola':
        return epi_cola_loss
    elif loss_name == 'ce':
        return cross_entropy_loss
    else:
        raise ValueError(f"Loss function '{loss_name}' is not supported.")




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

def get_MAPE(pred, target, eps: float = 1e-8):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between predictions and targets.
    """
    pred = pred.reshape(target.shape)
    target = target.reshape(pred.shape)
    denominator = torch.clamp(target.abs(), min=eps)
    return torch.mean(torch.abs((target - pred) / denominator)) * 100.0

def get_R2(pred, target, eps: float = 1e-8):
    """
    Calculates the coefficient of determination (R^2 score).
    """
    pred = pred.reshape(target.shape)
    target = target.reshape(pred.shape)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - (ss_res / (ss_tot + eps))

def get_NRMSE(pred, target, eps: float = 1e-8):
    """
    Calculates the Normalized Root Mean Squared Error (NRMSE).

    NRMSE = RMSE / σ_target

    This is a scale-invariant metric that allows fair comparison across
    datasets with different variances. NRMSE values:
    - < 0.1: Excellent
    - 0.1-0.2: Good
    - 0.2-0.5: Acceptable
    - > 0.5: Poor

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.
    eps : float, optional
        Small value to avoid division by zero. Default: 1e-8.

    Returns
    -------
    torch.Tensor
        The NRMSE value as a scalar tensor.
    """
    pred = pred.reshape(target.shape)
    target = target.reshape(pred.shape)

    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    target_std = torch.std(target)

    # Avoid division by zero
    target_std = torch.clamp(target_std, min=eps)

    return rmse / target_std

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


# Registry of available metrics
METRIC_REGISTRY = {
    'mse': get_MSE,
    'mae': get_MAE,
    'rmse': get_RMSE,
    'nrmse': get_NRMSE,
    'mape': get_MAPE,
    'r2': get_R2,
    'acc': get_ACC,
    'accuracy': get_ACC,
}


def get_metric(metric_name):
    """
    Retrieves the specified metric function based on the input metric name.
    
    Parameters
    ----------
    metric_name : str or callable
        Name of the metric function to retrieve, or a custom callable.
        Supported names: 'mse', 'mae', 'rmse', 'mape', 'r2', 'acc'/'accuracy'
        
    Returns
    -------
    callable
        The corresponding metric function.
        
    Raises
    ------
    ValueError
        If metric_name is not a supported string and not callable.
    """
    if callable(metric_name):
        return metric_name
    
    if not isinstance(metric_name, str):
        raise ValueError(f"metric_name must be a string or callable, got {type(metric_name)}")
    
    metric_name_lower = metric_name.lower()
    if metric_name_lower in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric_name_lower]
    
    raise ValueError(
        f"Metric '{metric_name}' is not supported. "
        f"Available metrics: {list(METRIC_REGISTRY.keys())}"
    )


def compute_metrics(preds, targets, metric_names, residual_fn=None):
    """
    Compute multiple metrics on predictions and targets.
    
    Parameters
    ----------
    preds : torch.Tensor or dict
        Model predictions. Can be a dict (e.g., {'mean': tensor, 'std': tensor})
        if the metric function is designed to handle it.
    targets : torch.Tensor
        Ground truth targets
    metric_names : list of str or callable
        List of metric names or custom callables to compute.
        Custom callables should handle dict predictions if the model outputs dicts.
    residual_fn : callable, optional
        Custom residual function (preds, targets) -> residuals.
        Used for computing MSE/MAE/RMSE when preds is a dict.
        
    Returns
    -------
    dict
        Dictionary mapping metric names to their computed values (as floats)
    """
    results = {}
    
    for name in metric_names:
        try:
            metric_fn = get_metric(name)
            
            # Handle custom residual function for dict predictions
            if residual_fn is not None and name in ['mse', 'mae', 'rmse']:
                # For MSE/MAE/RMSE with dict predictions, use residual_fn
                residuals = residual_fn(preds, targets)
                if name == 'mse':
                    value = torch.mean(residuals ** 2).item()
                elif name == 'mae':
                    value = torch.mean(torch.abs(residuals)).item()
                elif name == 'rmse':
                    value = torch.sqrt(torch.mean(residuals ** 2)).item()
            else:
                # Let the metric function handle the predictions directly
                # Custom metrics should be designed to handle dict outputs if needed
                value = metric_fn(preds, targets)
                if hasattr(value, 'item'):
                    value = value.item()
            
            # Use the string name for known metrics, or generate name for callables
            key = name if isinstance(name, str) else getattr(name, '__name__', 'custom_metric')
            results[key] = value
            
        except Exception as e:
            key = name if isinstance(name, str) else getattr(name, '__name__', 'custom_metric')
            results[key] = float('nan')
            print(f"Warning: Failed to compute metric '{key}': {e}")
    
    return results
