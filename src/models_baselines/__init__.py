"""
Baseline Models for Super-Resolution Comparison.

Simple interpolation and ML baselines for comparing against neural field models.
"""

from .linear_interpolation import LinearInterpolationBaseline
from .cubic_interpolation import CubicInterpolationBaseline
from .random_forest import RandomForestBaseline

__all__ = [
    'LinearInterpolationBaseline',
    'CubicInterpolationBaseline', 
    'RandomForestBaseline',
]
