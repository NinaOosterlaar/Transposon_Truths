import torch

def l1_regularization(parameters):
    """
    Compute L1 regularization (sum of absolute values of all parameters).
    
    Parameters:
    -----------
    parameters : iterable of torch.Tensor
        Model parameters to regularize
    
    Returns:
    --------
    torch.Tensor
        Scalar L1 penalty
    """
    l1_penalty = 0.0
    for param in parameters:
        l1_penalty += torch.sum(torch.abs(param))
    return l1_penalty

def zinb_nll(x, mu, theta, pi, eps=1e-8, reduction='sum'):
    """
    Zero-Inflated Negative Binomial Negative Log-Likelihood loss.
    
    Parameters:
    -----------
    x : torch.Tensor
        Observed counts (raw counts, not normalized)
    mu : torch.Tensor
        Mean parameter of NB distribution (after size factor correction)
    theta : torch.Tensor
        Dispersion parameter of NB distribution (positive)
    pi : torch.Tensor
        Zero-inflation probability (between 0 and 1)
    eps : float
        Small constant for numerical stability
    reduction : str
        'sum', 'mean', or 'none'. Default='sum' for consistency with PyTorch losses
    
    Returns:
    --------
    torch.Tensor
        Negative log-likelihood. Shape depends on reduction:
        - 'sum': scalar (sum over all elements)
        - 'mean': scalar (mean over all elements)
        - 'none': same shape as input (per-element loss)
    """
    # Clamp inputs to safe ranges to prevent numerical issues
    theta = torch.clamp(theta, min=eps)
    mu    = torch.clamp(mu,    min=eps)
    pi    = torch.clamp(pi,    min=eps, max=1.0 - eps)
    
    # Check for NaN/Inf in inputs
    # if torch.isnan(mu).any() or torch.isinf(mu).any():
    #     print(f"WARNING: NaN/Inf detected in mu! min={mu.min()}, max={mu.max()}")
    # if torch.isnan(theta).any() or torch.isinf(theta).any():
    #     print(f"WARNING: NaN/Inf detected in theta! min={theta.min()}, max={theta.max()}")
    # if torch.isnan(pi).any() or torch.isinf(pi).any():
    #     print(f"WARNING: NaN/Inf detected in pi! min={pi.min()}, max={pi.max()}")

    # log NB pmf - use numerically stable computations
    # For lgamma, clamp inputs to prevent overflow
    t1 = (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )
    t2 = theta * (torch.log(theta ) - torch.log(theta + mu ))
    t3 = x * (torch.log(mu ) - torch.log(theta + mu ))
    log_nb = t1 + t2 + t3
    
    # # Clamp log_nb to prevent extreme values
    # log_nb = torch.clamp(log_nb, min=-50, max=50)

    is_zero = (x == 0).float()

    # For zero-inflated component
    is_zero = (x == 0)

    log_pi = torch.log(pi)
    log_1_minus_pi = torch.log1p(-pi)

    log_prob_zero = torch.logaddexp(log_pi, log_1_minus_pi + log_nb)
    # log_prob_zero = log_pi
    log_prob_nonzero = log_1_minus_pi + log_nb

    log_prob = torch.where(is_zero, log_prob_zero, log_prob_nonzero)
    nll = -log_prob
    
    # Check for NaN/Inf in output
    # if torch.isnan(nll).any() or torch.isinf(nll).any():
    #     print(f"WARNING: NaN/Inf detected in NLL output!")
    #     print(f"  mu stats: min={mu.min():.4f}, max={mu.max():.4f}, mean={mu.mean():.4f}")
    #     print(f"  theta stats: min={theta.min():.4f}, max={theta.max():.4f}, mean={theta.mean():.4f}")
    #     print(f"  pi stats: min={pi.min():.4f}, max={pi.max():.4f}, mean={pi.mean():.4f}")
    #     # Replace NaN/Inf with large but finite value
    #     nll = torch.where(torch.isnan(nll) | torch.isinf(nll), torch.tensor(50.0, device=nll.device), nll)
    
    if reduction == 'sum':
        return nll.sum()
    elif reduction == 'mean':
        return nll.mean()
    elif reduction == 'none':
        return nll
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'sum', 'mean', or 'none'.")
    
def mae_loss(x, mu, pi, pi_threshold, reduction='sum'):
    """
    Mean Absolute Error loss between observed counts and mean parameter.
    
    Parameters:
    -----------
    x : torch.Tensor
        Observed counts (raw counts, not normalized)
    mu : torch.Tensor
        Mean parameter of NB distribution (after size factor correction)
    theta : torch.Tensor
        Dispersion parameter of NB distribution (not used in MAE)
    pi : torch.Tensor
        Zero-inflation probability (not used in MAE)
    eps : float
        Small constant for numerical stability (not used in MAE)
    reduction : str
        'sum', 'mean', or 'none'. Default='sum' for consistency with PyTorch losses
        
    Returns:
    --------
    torch.Tensor
        MAE loss. Shape depends on reduction:
        - 'sum': scalar (sum over all elements)
        - 'mean': scalar (mean over all elements)
        - 'none': same shape as input (per-element loss)
    """
    reconstruction = mu * (pi < pi_threshold).float()
    mae = torch.abs(x - reconstruction)
    
    if reduction == 'sum':
        return mae.sum()
    elif reduction == 'mean':
        return mae.mean()
    elif reduction == 'none':
        return mae
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'sum', 'mean', or 'none'.")
    
    
def reconstruct_masked_values(x, mu, pi, mask, pi_threshold):
    """
    Loss that measures how well the model reconstructs masked values to not be zero.
    
    Parameters:
    -----------
    x : torch.Tensor
        Observed counts (raw counts, not normalized)
    mu : torch.Tensor
        Mean parameter of NB distribution (after size factor correction)
    pi : torch.Tensor
        Zero-inflation probability (between 0 and 1)
    mask : torch.Tensor
        Boolean mask indicating which values to reconstruct (True = reconstruct)
    pi_threshold : float
        Threshold for zero-inflation probability to consider a value as non-zero
    
    Returns:
    --------
    torch.Tensor
        Tensor with masked values reconstructed.
    """
    masked_values = x[mask]
    if masked_values.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    reconstruction = mu[mask] * (pi[mask] < pi_threshold).float()
    loss = torch.abs(masked_values - reconstruction)
    return loss.mean()

