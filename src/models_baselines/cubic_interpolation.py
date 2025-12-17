"""
Cubic Interpolation Baseline for Super-Resolution.

Uses scipy's RegularGridInterpolator with cubic method to upsample
low-resolution Aurora predictions or latent features to HRES resolution.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Tuple, Optional
import torch


class CubicInterpolationBaseline:
    """
    Bicubic interpolation baseline for super-resolution.
    
    Interpolates low-resolution gridded data to high-resolution query positions
    using cubic spline interpolation for smoother results than linear.
    """
    
    def __init__(self, method: str = "cubic"):
        """
        Args:
            method: Interpolation method ('cubic' for bicubic interpolation)
        """
        self.method = method
    
    def interpolate(
        self,
        low_res_data: np.ndarray,
        low_res_lat: np.ndarray,
        low_res_lon: np.ndarray,
        query_lat: np.ndarray,
        query_lon: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate low-res data to high-res query positions.
        
        Args:
            low_res_data: [lat, lon] or [lat, lon, channels] low-res data
            low_res_lat: 1D array of low-res latitudes
            low_res_lon: 1D array of low-res longitudes
            query_lat: 1D or 2D array of query latitudes
            query_lon: 1D or 2D array of query longitudes
            
        Returns:
            Interpolated values at query positions
        """
        # Handle multi-channel data
        if low_res_data.ndim == 3:
            n_channels = low_res_data.shape[-1]
            results = []
            for c in range(n_channels):
                interp = self._interpolate_single(
                    low_res_data[:, :, c],
                    low_res_lat, low_res_lon,
                    query_lat, query_lon
                )
                results.append(interp)
            return np.stack(results, axis=-1)
        else:
            return self._interpolate_single(
                low_res_data, low_res_lat, low_res_lon,
                query_lat, query_lon
            )
    
    def _interpolate_single(
        self,
        data: np.ndarray,
        low_res_lat: np.ndarray,
        low_res_lon: np.ndarray,
        query_lat: np.ndarray,
        query_lon: np.ndarray,
    ) -> np.ndarray:
        """Interpolate single-channel data."""
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (low_res_lat, low_res_lon),
            data,
            method=self.method,
            bounds_error=False,
            fill_value=np.nan,
        )
        
        # Create query points
        if query_lat.ndim == 1 and query_lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(query_lon, query_lat)
            query_points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
            result = interpolator(query_points)
            return result.reshape(len(query_lat), len(query_lon))
        else:
            query_points = np.stack([query_lat.ravel(), query_lon.ravel()], axis=-1)
            return interpolator(query_points)
    
    def __call__(
        self,
        low_res_data: np.ndarray,
        low_res_lat: np.ndarray,
        low_res_lon: np.ndarray,
        query_lat: np.ndarray,
        query_lon: np.ndarray,
    ) -> np.ndarray:
        """Alias for interpolate()."""
        return self.interpolate(
            low_res_data, low_res_lat, low_res_lon,
            query_lat, query_lon
        )
    
    def evaluate_on_dataset(
        self,
        dataset,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate interpolation baseline on a dataset.
        
        Args:
            dataset: EraPredictionHresDataset or EraLatentHresDataset
            num_samples: Number of samples to evaluate (None = all)
            
        Returns:
            Dict with MSE and RMSE metrics per variable
        """
        from tqdm import tqdm
        
        n_samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        
        # Get coordinate arrays from dataset
        low_res_lat = dataset.pred_lat if hasattr(dataset, 'pred_lat') else dataset.latent_lat
        low_res_lon = dataset.pred_lon if hasattr(dataset, 'pred_lon') else dataset.latent_lon
        hres_lat = dataset.hres_lat
        hres_lon = dataset.hres_lon
        
        all_mse = []
        
        for idx in tqdm(range(n_samples), desc="Cubic interpolation"):
            sample = dataset[idx]
            
            # Get low-res features
            if 'low_res_features' in sample:
                low_res_data = sample['low_res_features'].numpy()
                low_res_data = low_res_data.reshape(len(low_res_lat), len(low_res_lon), -1)
            else:
                low_res_data = sample['latents'].numpy()
                low_res_data = low_res_data.reshape(len(low_res_lat), len(low_res_lon), -1)
            
            # Get targets
            query_fields = sample['query_fields'].numpy()
            
            # Interpolate
            pred = self.interpolate(
                low_res_data, low_res_lat, low_res_lon,
                hres_lat, hres_lon
            )
            pred = pred.reshape(-1, pred.shape[-1])
            
            # Compute MSE
            n_vars = min(pred.shape[-1], query_fields.shape[-1])
            mse = np.mean((pred[:, :n_vars] - query_fields[:, :n_vars]) ** 2)
            all_mse.append(mse)
        
        mean_mse = np.mean(all_mse)
        
        return {
            "mse": float(mean_mse),
            "rmse": float(np.sqrt(mean_mse)),
        }
