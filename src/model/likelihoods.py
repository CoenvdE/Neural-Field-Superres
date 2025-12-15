"""
Likelihood modules for probabilistic predictions.

Inspired by gridded-tnp's likelihood implementation for Gaussian distributions.
Converts deterministic model outputs into probability distributions for computing
negative log-likelihood loss.
"""

import torch
import torch.nn as nn
import torch.distributions as td
from abc import ABC, abstractmethod


class Likelihood(ABC, nn.Module):
    """Abstract base class for likelihood functions.
    
    Likelihood modules convert model outputs into probability distributions.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> td.Distribution:
        """Convert model output to a probability distribution.
        
        Args:
            x: Model output tensor
            
        Returns:
            A torch.distributions.Distribution object
        """
        pass


class GaussianLikelihood(Likelihood):
    """Homoscedastic Gaussian likelihood with learned global noise parameter.
    
    The model predicts only the mean. The variance is a single learned parameter
    shared across all predictions (constant/homoscedastic noise).
    
    This is appropriate when you believe the prediction uncertainty is roughly
    constant across the spatial domain (e.g., uniform sensor noise).
    
    Args:
        noise: Initial noise standard deviation
        train_noise: Whether to learn the noise parameter during training
    """
    
    def __init__(self, noise: float = 0.1, train_noise: bool = True):
        super().__init__()
        
        # Store noise as log to ensure positivity
        self.log_noise = nn.Parameter(
            torch.as_tensor(noise).log(), 
            requires_grad=train_noise
        )
    
    @property
    def noise(self) -> torch.Tensor:
        """Get the noise standard deviation."""
        return self.log_noise.exp()
    
    @noise.setter
    def noise(self, value: float):
        """Set the noise standard deviation."""
        self.log_noise = nn.Parameter(torch.as_tensor(value).log())
    
    def forward(self, x: torch.Tensor) -> td.Normal:
        """Convert model output to Gaussian distribution with fixed noise.
        
        Args:
            x: Model predictions [B, N, D] - interpreted as means
            
        Returns:
            Normal distribution with mean=x and learned constant std
        """
        return td.Normal(x, self.noise)


class HeteroscedasticGaussianLikelihood(Likelihood):
    """Heteroscedastic Gaussian likelihood with input-dependent noise.
    
    The model predicts both mean and log-variance, allowing the uncertainty
    to vary spatially. The model output must have even-numbered channels,
    where the first half represents means and the second half log-variances.
    
    This is appropriate when prediction uncertainty varies across the domain
    (e.g., higher uncertainty over mountains, lower over oceans).
    
    Mathematical formulation:
        - Model outputs: [μ₁, μ₂, ..., μₙ, log(σ₁²), log(σ₂²), ..., log(σₙ²)]
        - Distribution: N(μᵢ, σᵢ²) where σᵢ = softplus(log(σᵢ²))^0.5 + min_noise
    
    Args:
        min_noise: Minimum noise floor to prevent overconfident predictions
    """
    
    def __init__(self, min_noise: float = 1e-3):
        super().__init__()
        self.min_noise = min_noise
    
    def forward(self, x: torch.Tensor) -> td.Normal:
        """Convert model output to Gaussian with spatially-varying uncertainty.
        
        Args:
            x: Model predictions [B, N, 2*D] where first D channels are means,
               last D channels are log-variances
               
        Returns:
            Normal distribution with input-dependent mean and std
            
        Raises:
            AssertionError: If input doesn't have even number of channels
        """
        assert x.shape[-1] % 2 == 0, \
            f"Heteroscedastic likelihood requires even channels, got {x.shape[-1]}"
        
        # Split into mean and log-variance
        loc = x[..., :x.shape[-1] // 2]
        log_var = x[..., x.shape[-1] // 2:]
        
        # Convert log-variance to standard deviation
        # softplus ensures positivity, min_noise prevents numerical issues
        scale = (
            nn.functional.softplus(log_var) ** 0.5 + self.min_noise
        )
        
        return td.Normal(loc, scale)
