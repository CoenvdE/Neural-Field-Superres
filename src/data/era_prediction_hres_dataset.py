"""
Dataset for loading Aurora predictions and HRES surface field values from Zarr.

Reconstructs HRES high-resolution surface fields (targets) given Aurora predictions.
Surface-only: 2D coordinates (lat, lon) for 2 surface variables (2t, msl).

Each sample returns:
- low_res_features: [L, num_vars] - Aurora prediction values at low-res grid positions
- low_res_pos: [L, 2] - positions of low-res grid (lat, lon)
- query_pos: [Q, 2] - HRES grid positions to predict at
- query_fields: [Q, num_vars] - target field values at query positions
- query_auxiliary_features: [Q, num_aux] - optional static features (z, lsm, slt)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import xarray as xr
import json
from pathlib import Path

class EraPredictionHresDataset(Dataset):
    """
    Dataset for neural field super-resolution from Aurora predictions to HRES surface fields.
    
    Args:
        prediction_zarr_path: Path to Zarr with Aurora predictions
        hres_zarr_path: Path to Zarr with HRES field values
        variables: List of surface variable names (e.g., ['2t', 'msl'])
        num_query_samples: Number of HRES positions to sample per batch (None = all)
        normalize_coords: Whether to normalize coordinates to [-1, 1]
        split: 'train', 'val', or None. Splits by time.
        val_months: Number of months at the end to use for validation
        static_zarr_path: Optional path to static features zarr
        use_static_features: Whether to load static features
        region_bounds: Optional dict with lat_min, lat_max, lon_min, lon_max for sub-region filtering
    """
    
    def __init__(
        self,
        prediction_zarr_path: str,
        hres_zarr_path: str,
        variables: Optional[List[str]] = None,
        static_zarr_path: Optional[str] = None,
        static_variables: Optional[List[str]] = None,
        static_statistics_path: Optional[str] = None,
        use_static_features: bool = False,
        normalize_static_features: bool = True,
        num_query_samples: Optional[int] = None,
        normalize_coords: bool = True,
        statistics_path: Optional[str] = None,
        prediction_statistics_path: Optional[str] = None,  # Statistics for Aurora predictions
        split: Optional[str] = None,
        val_months: int = 3,
        region_bounds: Optional[Dict[str, float]] = None,
        zarr_format: Optional[int] = None,  # None=auto, 2=v2, 3=v3
    ):
        super().__init__()
        
        self.prediction_zarr_path = prediction_zarr_path
        self.hres_zarr_path = hres_zarr_path
        self.normalize_coords = normalize_coords
        self.num_query_samples = num_query_samples
        self.split = split
        self.val_months = val_months
        self.static_zarr_path = static_zarr_path
        self.static_variables = static_variables
        self.static_statistics_path = static_statistics_path
        self.use_static_features = use_static_features
        self.normalize_static_features = normalize_static_features
        self.region_bounds = region_bounds
        self.statistics_path = statistics_path
        self.prediction_statistics_path = prediction_statistics_path
        self.zarr_format = zarr_format
        
        # Open Zarr
        self.prediction_ds = self._open_zarr(prediction_zarr_path)
        self.hres_ds = self._open_zarr(hres_zarr_path)
        
        # Get original coordinates
        self._pred_lat_orig = self.prediction_ds['lat'].values
        self._pred_lon_orig = self.prediction_ds['lon'].values
        self.pred_times = self.prediction_ds['time'].values
        
        self._hres_lat_orig = self.hres_ds['latitude'].values
        self._hres_lon_orig = self.hres_ds['longitude'].values
        self.hres_times = self.hres_ds['time'].values
        
        # Apply grid clipping and region filtering
        self._clip_hres_to_pred_bounds()
        self._apply_region_bounds()
        
        # Setup time indices
        self._setup_time_indices()
        
        # Determine variables
        if variables is None:
            self.variables = [v for v in self.hres_ds.data_vars if v not in self.hres_ds.coords]
        else:
            self.variables = variables
        
        # Load normalization statistics for target variables
        self._load_statistics()
        
        # Compute bounds for normalization (based on prediction grid)
        self._compute_bounds()
        
        # Build position grids
        self._build_hres_grid()
        self._build_pred_grid()
        
        # Load static features
        self._load_static_features()
        
        self._print_info()
    
    def _open_zarr(self, zarr_path: str) -> Tuple:
        """Open a Zarr store with v3 fallback support."""
        if self.zarr_format == 3:
            ds = xr.open_zarr(zarr_path, consolidated=False, zarr_format=3)
        else:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        return ds
    
    def _clip_hres_to_pred_bounds(self):
        """Clip HRES grid to only include points within prediction spatial coverage."""
        lat_min, lat_max = self._pred_lat_orig.min(), self._pred_lat_orig.max()
        lon_min, lon_max = self._pred_lon_orig.min(), self._pred_lon_orig.max()
        
        # Find HRES indices within prediction bounds
        hres_lat_mask = (self._hres_lat_orig >= lat_min) & (self._hres_lat_orig <= lat_max)
        hres_lon_mask = (self._hres_lon_orig >= lon_min) & (self._hres_lon_orig <= lon_max)
        
        self.hres_lat_indices = np.where(hres_lat_mask)[0]
        self.hres_lon_indices = np.where(hres_lon_mask)[0]
        self.hres_lat = self._hres_lat_orig[hres_lat_mask]
        self.hres_lon = self._hres_lon_orig[hres_lon_mask]
        
        # Prediction uses full grid initially
        self.pred_lat = self._pred_lat_orig.copy()
        self.pred_lon = self._pred_lon_orig.copy()
        self.pred_lat_indices = np.arange(len(self.pred_lat))
        self.pred_lon_indices = np.arange(len(self.pred_lon))
    
    def _apply_region_bounds(self):
        """Optionally filter both grids to a user-specified geographic sub-region."""
        if self.region_bounds is None:
            return
        
        lat_min = self.region_bounds.get("lat_min", -90)
        lat_max = self.region_bounds.get("lat_max", 90)
        lon_min = self.region_bounds.get("lon_min", -180)
        lon_max = self.region_bounds.get("lon_max", 180)
        
        # Filter prediction grid
        pred_lat_mask = (self.pred_lat >= lat_min) & (self.pred_lat <= lat_max)
        pred_lon_mask = (self.pred_lon >= lon_min) & (self.pred_lon <= lon_max)
        self.pred_lat_indices = np.where(pred_lat_mask)[0]
        self.pred_lon_indices = np.where(pred_lon_mask)[0]
        self.pred_lat = self.pred_lat[pred_lat_mask]
        self.pred_lon = self.pred_lon[pred_lon_mask]
        
        # Filter HRES grid
        hres_lat_mask = (self.hres_lat >= lat_min) & (self.hres_lat <= lat_max)
        hres_lon_mask = (self.hres_lon >= lon_min) & (self.hres_lon <= lon_max)
        self.hres_lat_indices = self.hres_lat_indices[hres_lat_mask]
        self.hres_lon_indices = self.hres_lon_indices[hres_lon_mask]
        self.hres_lat = self.hres_lat[hres_lat_mask]
        self.hres_lon = self.hres_lon[hres_lon_mask]
    
    def _setup_time_indices(self):
        """Setup train/val time indices."""
        n_common = min(len(self.pred_times), len(self.hres_times))
        
        pred_times_common = pd.to_datetime(self.pred_times[:n_common])
        hres_times_common = pd.to_datetime(self.hres_times[:n_common])
        
        if not (pred_times_common == hres_times_common).all():
            mismatch_idx = np.where(pred_times_common != hres_times_common)[0][0]
            raise ValueError(
                f"Timestamp mismatch between prediction and HRES datasets at index {mismatch_idx}:\n"
                f"  Prediction: {pred_times_common[mismatch_idx]}\n"
                f"  HRES:       {hres_times_common[mismatch_idx]}\n"
            )
        
        all_indices = list(range(n_common))
        
        if self.split is None:
            self.time_indices = all_indices
            self.split_info = f"all ({n_common} timesteps)"
        else:
            times = pd.to_datetime(self.pred_times[:n_common])
            cutoff = times[-1] - pd.DateOffset(months=self.val_months)
            
            if self.split == 'train':
                self.time_indices = np.where(times <= cutoff)[0].tolist()
                self.split_info = f"train ({len(self.time_indices)} timesteps)"
            else:
                self.time_indices = np.where(times > cutoff)[0].tolist()
                self.split_info = f"val ({len(self.time_indices)} timesteps)"
    
    def _load_statistics(self):
        """Load pre-computed statistics for normalization."""
        self.target_mean = {}
        self.target_std = {}
        self.pred_mean = {}
        self.pred_std = {}
        self.static_mean = {}
        self.static_std = {}
        
        # Load HRES target statistics
        if self.statistics_path is None:
            hres_path = Path(self.hres_zarr_path)
            stats_path = hres_path.parent / f"{hres_path.stem}_statistics.json"
        else:
            stats_path = Path(self.statistics_path)
        
        if not stats_path.exists():
            raise ValueError(f"Statistics file not found at {stats_path}")
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        for var in self.variables:
            if var in stats:
                self.target_mean[var] = float(stats[var]['mean'])
                self.target_std[var] = float(stats[var]['std'])
            else:
                raise ValueError(f"Statistics not found for variable {var}")
        
        # Load Aurora prediction statistics
        if self.prediction_statistics_path is None:
            pred_path = Path(self.prediction_zarr_path)
            pred_stats_path = pred_path.parent / f"{pred_path.stem}_statistics.json"
        else:
            pred_stats_path = Path(self.prediction_statistics_path)
        
        if not pred_stats_path.exists():
            raise ValueError(f"Prediction statistics file not found at {pred_stats_path}")
        
        with open(pred_stats_path, 'r') as f:
            pred_stats = json.load(f)
        
        for var in self.variables:
            if var in pred_stats:
                self.pred_mean[var] = float(pred_stats[var]['mean'])
                self.pred_std[var] = float(pred_stats[var]['std'])
            else:
                raise ValueError(f"Prediction statistics not found for variable {var}")
        
        # Load statistics for static variables if using static features
        if self.use_static_features:
            if self.static_statistics_path:
                static_stats_path = Path(self.static_statistics_path)
            elif self.static_zarr_path:
                static_path = Path(self.static_zarr_path)
                static_stats_path = static_path.parent / f"{static_path.stem}_statistics.json"
            else:
                static_stats_path = None
            
            try:
                with open(static_stats_path, 'r') as f:
                    static_stats = json.load(f)
                
                for var in self.static_variables:
                    if var in static_stats:
                        self.static_mean[var] = float(static_stats[var]['mean'])
                        self.static_std[var] = float(static_stats[var]['std'])
            except FileNotFoundError:
                raise ValueError(
                    f"Static statistics file not found at {static_stats_path}. "
                )

    def _compute_bounds(self):
        """Compute lat/lon bounds for normalization based on prediction grid coverage."""
        self.lat_min, self.lat_max = self.pred_lat.min(), self.pred_lat.max()
        self.lon_min, self.lon_max = self.pred_lon.min(), self.pred_lon.max()
    
    def _normalize(self, arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Normalize to [-1, 1]."""
        if not self.normalize_coords:
            return arr
        return 2 * (arr - vmin) / (vmax - vmin + 1e-8) - 1
    
    def _build_hres_grid(self):
        """Build HRES query grid positions."""
        lat_norm = self._normalize(self.hres_lat, self.lat_min, self.lat_max)
        lon_norm = self._normalize(self.hres_lon, self.lon_min, self.lon_max)
        
        lon_grid, lat_grid = np.meshgrid(lon_norm, lat_norm)
        self.hres_positions = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=-1).astype(np.float32)
        self.num_hres_points = self.hres_positions.shape[0]
    
    def _build_pred_grid(self):
        """Build prediction grid positions."""
        lat_norm = self._normalize(self.pred_lat, self.lat_min, self.lat_max)
        lon_norm = self._normalize(self.pred_lon, self.lon_min, self.lon_max)
        
        lon_grid, lat_grid = np.meshgrid(lon_norm, lat_norm)
        self.pred_positions = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=-1).astype(np.float32)
    
    def _load_static_features(self):
        """Load optional static features and normalize them."""
        if not self.use_static_features:
            self.static_features = None
            return
        
        if self.static_zarr_path is None:
            raise ValueError("static_zarr_path must be specified when use_static_features=True")
        
        static_ds = xr.open_zarr(self.static_zarr_path, consolidated=True)
        
        available = []
        if self.static_variables is None:
            self.static_variables = static_ds.data_vars.keys()
        else:
            for v in self.static_variables:
                if v in static_ds.data_vars or v in static_ds.coords:
                    available.append(v)
        
        if not available:
            raise ValueError(f"No static variables found")
        
        features = []
        for var in available:
            data = static_ds[var].values
            
            if len(data.shape) == 2:
                data = data[np.ix_(self.hres_lat_indices, self.hres_lon_indices)]
            
            data = data.flatten().astype(np.float32)
            
            if self.normalize_static_features:
                if var not in self.static_mean or var not in self.static_std:
                    raise ValueError(f"Cannot normalize static feature '{var}'")
                
                std_val = self.static_std[var]
                if abs(std_val) < 1e-8:
                    raise ValueError(f"Cannot normalize static feature '{var}': std is {std_val}")
                
                data = (data - self.static_mean[var]) / std_val
            
            features.append(data)
        
        self.static_features = np.stack(features, axis=-1)
    
    def _print_info(self):
        """Print dataset info."""
        pass  # Disabled verbose output
    
    def __len__(self) -> int:
        return len(self.time_indices)
    
    @property
    def hres_shape(self) -> Tuple[int, int]:
        """Return HRES grid shape (lat, lon) for visualization."""
        return (len(self.hres_lat), len(self.hres_lon))
    
    @property
    def geo_bounds(self) -> Dict[str, float]:
        """Return geographic bounds for visualization."""
        return {
            "lat_min": float(self.hres_lat.min()),
            "lat_max": float(self.hres_lat.max()),
            "lon_min": float(self.hres_lon.min()),
            "lon_max": float(self.hres_lon.max()),
        }
    
    @property
    def statistics(self) -> Dict:
        """Return statistics dict for denormalization."""
        return {
            var: {"mean": self.target_mean[var], "std": self.target_std[var]}
            for var in self.variables
        }
    
    def get_timestamp(self, idx: int) -> str:
        """Get formatted timestamp string for a sample index."""
        time_idx = self.time_indices[idx]
        timestamp = pd.to_datetime(self.pred_times[time_idx])
        return timestamp.strftime("%Y-%m-%d %H:%M")
    
    def denormalize_targets(self, normalized_data: torch.Tensor, var_idx: int) -> torch.Tensor:
        """Denormalize target data back to original scale."""
        var = self.variables[var_idx]
        if var not in self.target_mean:
            raise ValueError(f"Cannot denormalize variable '{var}'")
        
        mean = self.target_mean[var]
        std = self.target_std[var]
        
        return normalized_data * std + mean
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        time_idx = self.time_indices[idx]
        
        # Load Aurora predictions with region slicing
        pred_features = []
        for var in self.variables:
            data = self.prediction_ds[var].isel(
                time=time_idx,
                lat=self.pred_lat_indices,
                lon=self.pred_lon_indices,
            ).values.flatten()
            
            # Normalize predictions
            if var in self.pred_mean:
                data = (data - self.pred_mean[var]) / self.pred_std[var]
            
            pred_features.append(data)
        
        # Stack to [L, num_vars]
        low_res_features = np.stack(pred_features, axis=-1).astype(np.float32)
        
        # Load HRES targets with clipping/region slicing
        fields = []
        for var in self.variables:
            data = self.hres_ds[var].isel(
                time=time_idx,
                latitude=self.hres_lat_indices,
                longitude=self.hres_lon_indices,
            ).values.flatten()
            
            if var in self.target_mean:
                data = (data - self.target_mean[var]) / self.target_std[var]
            
            fields.append(data)
        query_fields = np.stack(fields, axis=-1).astype(np.float32)
        
        # Get positions
        low_res_pos = self.pred_positions.copy()
        query_pos = self.hres_positions.copy()
        aux_feats = self.static_features.copy() if self.static_features is not None else None
        
        # Sample subset if needed
        if self.num_query_samples and self.num_query_samples < self.num_hres_points:
            indices = np.random.choice(self.num_hres_points, self.num_query_samples, replace=False)
            query_pos = query_pos[indices]
            query_fields = query_fields[indices]
            if aux_feats is not None:
                aux_feats = aux_feats[indices]
        
        result = {
            'low_res_features': torch.from_numpy(low_res_features).float(),
            'low_res_pos': torch.from_numpy(low_res_pos).float(),
            'query_pos': torch.from_numpy(query_pos).float(),
            'query_fields': torch.from_numpy(query_fields).float(),
            # Grid metadata for analytical KNN
            'low_res_grid_shape': torch.tensor([len(self.pred_lat), len(self.pred_lon)], dtype=torch.long),
        }
        
        if aux_feats is not None:
            result['query_auxiliary_features'] = torch.from_numpy(aux_feats).float()
        
        return result
