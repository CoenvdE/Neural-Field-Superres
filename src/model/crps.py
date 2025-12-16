"""
CRPS (Continuous Ranked Probability Score) loss functions.

CRPS is a proper scoring rule for probabilistic forecasts that generalizes
Mean Absolute Error to probabilistic predictions. It measures the integrated
squared difference between the predicted CDF and the empirical CDF of the
observation.

Key properties:
- For deterministic forecasts, CRPS reduces to MAE
- CRPS has the same units as the target variable (unlike NLL)
- CRPS is always non-negative (unlike NLL which can be negative)
- CRPS is more robust to miscalibration than NLL

This module provides:
1. crps_gaussian: Closed-form CRPS for Gaussian predictions
2. crps_ensemble: CRPS for ensemble predictions (multiple samples)

To integrate into training:
1. Import desired function in module.py
2. Add "crps" as loss_fn option in config
3. Call crps_gaussian/crps_ensemble in _calculate_loss method

Example usage:
    from src.model.crps import crps_gaussian
    
    # For Gaussian predictions
    mu = model_output[..., :D]  # means
    sigma = model_output[..., D:]  # std devs (after softplus)
    loss = crps_gaussian(mu, sigma, targets).mean()

References:
- Gneiting & Raftery (2007): "Strictly Proper Scoring Rules, Prediction, 
  and Estimation"
- Matheson & Winkler (1976): "Scoring Rules for Continuous Probability 
  Distributions"
"""

import math
import torch
import torch.nn as nn


def crps_gaussian(
    mu: torch.Tensor, 
    sigma: torch.Tensor, 
    y: torch.Tensor
) -> torch.Tensor:
    """Compute CRPS for Gaussian predictions using closed-form solution.
    
    The closed-form CRPS for a Gaussian N(μ, σ²) evaluated at observation y is:
    
        CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    
    where z = (y - μ) / σ, φ is standard normal PDF, Φ is standard normal CDF.
    
    This is equivalent to:
        CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    
    where X, X' ~ N(μ, σ²) are independent draws.
    
    Args:
        mu: Predicted means, shape [..., D] or broadcastable
        sigma: Predicted standard deviations, shape [..., D] or broadcastable
            Must be positive
        y: Observations, shape [..., D] or broadcastable
        
    Returns:
        CRPS values with same shape as broadcasted inputs.
        Lower is better. Units are same as y.
        
    Example:
        >>> mu = torch.tensor([0.0, 1.0])
        >>> sigma = torch.tensor([1.0, 0.5])
        >>> y = torch.tensor([0.5, 1.2])
        >>> crps = crps_gaussian(mu, sigma, y)
        >>> print(crps)  # tensor([0.2345, 0.1234])
    """
    # Standardized error
    z = (y - mu) / sigma
    
    # Standard normal PDF: φ(z) = exp(-z²/2) / √(2π)
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    
    # Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z/√2))
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    
    # Closed-form CRPS for Gaussian
    # CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    
    return crps


def crps_ensemble(
    ensemble: torch.Tensor, 
    y: torch.Tensor,
    reduce_ensemble: bool = True
) -> torch.Tensor:
    """Compute CRPS for ensemble predictions.
    
    For M ensemble members, the CRPS can be computed as:
    
        CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
             = (1/M) Σᵢ |xᵢ - y| - (1/2M²) Σᵢⱼ |xᵢ - xⱼ|
    
    This is useful when:
    - Your model produces samples (e.g., from VAE, diffusion, or MC dropout)
    - You want to evaluate without assuming a parametric distribution
    
    The second term measures ensemble spread and encourages diversity.
    
    Args:
        ensemble: Ensemble predictions, shape [M, ..., D] where M is number
            of ensemble members. First dimension is the ensemble dimension.
        y: Observations, shape [..., D] (will be broadcast)
        reduce_ensemble: If True, return scalar CRPS per location.
            If False, return both terms separately for diagnostics.
        
    Returns:
        If reduce_ensemble=True:
            CRPS values with shape [..., D]
        If reduce_ensemble=False:
            Tuple of (reliability_term, resolution_term) each with shape [..., D]
            where CRPS = reliability_term - resolution_term
            
    Note:
        The resolution term encourages ensemble spread. If all ensemble members
        are identical, resolution_term = 0 and CRPS = MAE.
        
    Example:
        >>> # 10 ensemble members, batch of 32, 100 points, 5 variables
        >>> ensemble = torch.randn(10, 32, 100, 5)
        >>> y = torch.randn(32, 100, 5)
        >>> crps = crps_ensemble(ensemble, y)
        >>> print(crps.shape)  # torch.Size([32, 100, 5])
    """
    M = ensemble.shape[0]
    
    # Term 1: Mean absolute error between ensemble and observation
    # (1/M) Σᵢ |xᵢ - y|
    reliability = torch.mean(torch.abs(ensemble - y), dim=0)
    
    # Term 2: Mean pairwise difference within ensemble  
    # This measures ensemble spread/diversity
    # (1/2M²) Σᵢⱼ |xᵢ - xⱼ|
    # 
    # Efficient computation: for each pair (i,j), compute |xᵢ - xⱼ|
    # Note: This is O(M²) in memory, could be optimized for large M
    
    # Expand for pairwise computation: [M, 1, ...] and [1, M, ...]
    ensemble_i = ensemble.unsqueeze(1)  # [M, 1, ..., D]
    ensemble_j = ensemble.unsqueeze(0)  # [1, M, ..., D]
    
    # Pairwise absolute differences: [M, M, ..., D]
    pairwise_diff = torch.abs(ensemble_i - ensemble_j)
    
    # Average over both ensemble dimensions
    resolution = pairwise_diff.sum(dim=(0, 1)) / (2 * M * M)
    
    if reduce_ensemble:
        return reliability - resolution
    else:
        return reliability, resolution


def crps_ensemble_efficient(
    ensemble: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """Memory-efficient CRPS for large ensembles using sorting.
    
    For sorted ensemble members x₍₁₎ ≤ x₍₂₎ ≤ ... ≤ x₍ₘ₎, the CRPS can be
    computed in O(M log M) time and O(M) memory:
    
        CRPS = (1/M) Σᵢ |x₍ᵢ₎ - y| - (1/M²) Σᵢ (2i - 1 - M) * x₍ᵢ₎
    
    This is mathematically equivalent to crps_ensemble but more efficient
    for large ensemble sizes (M > 100).
    
    Args:
        ensemble: Ensemble predictions, shape [M, ..., D]
        y: Observations, shape [..., D]
        
    Returns:
        CRPS values with shape [..., D]
        
    Note:
        This implementation sorts along the ensemble dimension, which may
        not preserve gradients correctly for all use cases. Use crps_ensemble
        if you need to backpropagate through the ensemble generation process.
    """
    M = ensemble.shape[0]
    
    # Sort ensemble members
    sorted_ensemble, _ = torch.sort(ensemble, dim=0)  # [M, ..., D]
    
    # Term 1: MAE
    reliability = torch.mean(torch.abs(ensemble - y), dim=0)
    
    # Term 2: Gini-like spread term using sorted order
    # weights[i] = (2i + 1 - M) / M² for i in 0..M-1
    indices = torch.arange(M, device=ensemble.device, dtype=ensemble.dtype)
    weights = (2 * indices + 1 - M) / (M * M)
    
    # Reshape weights for broadcasting: [M, 1, 1, ...]
    for _ in range(ensemble.dim() - 1):
        weights = weights.unsqueeze(-1)
    
    # Weighted sum gives spread term
    resolution = (weights * sorted_ensemble).sum(dim=0)
    
    return reliability - resolution


class CRPSLoss(nn.Module):
    """CRPS loss module for integration with PyTorch training.
    
    This module wraps the CRPS functions for easy use as a loss function.
    
    Args:
        reduction: How to reduce the loss ('mean', 'sum', 'none')
        distribution: Type of prediction ('gaussian' or 'ensemble')
        
    Example:
        >>> loss_fn = CRPSLoss(reduction='mean', distribution='gaussian')
        >>> 
        >>> # Model outputs mean and log-variance
        >>> pred = model(x)  # [B, N, 2*D]
        >>> mu = pred[..., :D]
        >>> sigma = F.softplus(pred[..., D:]) ** 0.5 + 1e-3
        >>> 
        >>> loss = loss_fn(mu, sigma, targets)
    """
    
    def __init__(
        self, 
        reduction: str = 'mean',
        distribution: str = 'gaussian'
    ):
        super().__init__()
        self.reduction = reduction
        self.distribution = distribution
        
    def forward(
        self, 
        mu_or_ensemble: torch.Tensor,
        sigma_or_y: torch.Tensor,
        y: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute CRPS loss.
        
        For Gaussian:
            forward(mu, sigma, y) -> CRPS
            
        For Ensemble:
            forward(ensemble, y) -> CRPS (y goes in second position)
        """
        if self.distribution == 'gaussian':
            if y is None:
                raise ValueError("Must provide y for Gaussian CRPS")
            crps = crps_gaussian(mu_or_ensemble, sigma_or_y, y)
        elif self.distribution == 'ensemble':
            # For ensemble, sigma_or_y is actually y
            crps = crps_ensemble(mu_or_ensemble, sigma_or_y)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        if self.reduction == 'mean':
            return crps.mean()
        elif self.reduction == 'sum':
            return crps.sum()
        elif self.reduction == 'none':
            return crps
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# =============================================================================
# Integration example (DO NOT UNCOMMENT - for reference only)
# =============================================================================
#
# To integrate CRPS into module.py, you would:
#
# 1. Add import at top of module.py:
#    from .crps import crps_gaussian, CRPSLoss
#
# 2. Add "crps" option to loss_fn handling in __init__:
#    elif self.loss_fn == "crps":
#        if loss_fn_config.get("heteroscedastic", False):
#            self.likelihood = HeteroscedasticGaussianLikelihood(...)
#        else:
#            self.likelihood = GaussianLikelihood(...)
#
# 3. Modify _calculate_loss to handle CRPS:
#    if self.loss_fn == "crps":
#        if isinstance(self.likelihood, HeteroscedasticGaussianLikelihood):
#            n_vars = predictions.shape[-1] // 2
#            mu = predictions[..., :n_vars]
#            log_var = predictions[..., n_vars:]
#            sigma = (F.softplus(log_var) ** 0.5 + 1e-3)
#        else:
#            mu = predictions
#            sigma = self.likelihood.noise.expand_as(predictions)
#        
#        crps_values = crps_gaussian(mu, sigma, targets)
#        loss = crps_values.mean()
#
# 4. Add config option:
#    loss_fn: "crps"
#    loss_fn_config:
#        heteroscedastic: true  # or false for homoscedastic
# =============================================================================
