"""
Dataset for loading ERA5 latents and HRES surface field values from Zarr.

Reconstructs HRES high-resolution surface fields (targets) given latents from ERA5.
Surface-only: 2D coordinates (lat, lon) for 2 surface variables (2t, msl).

Each sample returns:
- latents: [Z, D_latent] - latent features at latent grid positions
- latent_pos: [Z, 2] - positions of latents (lat, lon)
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

class EraLatentHresDataset(Dataset):
    """
    Dataset for neural field super-resolution from ERA5 latents to HRES surface fields.
    
    Args:
        latent_zarr_path: Path to Zarr with ERA5 latents
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
        latent_zarr_path: str,
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
        split: Optional[str] = None,
        val_months: int = 3,
        region_bounds: Optional[Dict[str, float]] = None,
        zarr_format: Optional[int] = None,  # None=auto, 2=v2, 3=v3
    ):
        super().__init__()
        
        self.latent_zarr_path = latent_zarr_path
        self.hres_zarr_path = hres_zarr_path
        self.normalize_coords = normalize_coords #TODO: what when
        self.num_query_samples = num_query_samples #TODO: FIGURE OUT HOW MUCH
        self.split = split
        self.val_months = val_months
        self.static_zarr_path = static_zarr_path
        self.static_variables = static_variables
        self.static_statistics_path = static_statistics_path
        self.use_static_features = use_static_features
        self.normalize_static_features = normalize_static_features
        self.region_bounds = region_bounds
        self.statistics_path = statistics_path
        self.zarr_format = zarr_format
        
        # Open Zarr (with v3 fallback support)
        self.latent_ds = self._open_zarr(latent_zarr_path)
        self.hres_ds = self._open_zarr(hres_zarr_path)
        
        # Get original coordinates
        self._latent_lat_orig = self.latent_ds['lat'].values
        self._latent_lon_orig = self.latent_ds['lon'].values
        self.latent_times = self.latent_ds['time'].values
        
        self._hres_lat_orig = self.hres_ds['latitude'].values
        self._hres_lon_orig = self.hres_ds['longitude'].values
        self.hres_times = self.hres_ds['time'].values
        
        # Apply grid clipping and region filtering
        self._clip_hres_to_latent_bounds()
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
        
        # Compute bounds for normalization (based on latent grid)
        self._compute_bounds()
        
        # Build position grids
        self._build_hres_grid()
        self._build_latent_grid()
        
        # Load static features
        self._load_static_features()
        
        self._print_info()
    
    def _open_zarr(self, zarr_path: str) -> Tuple:
        """Open a Zarr store with v3 fallback support.
        
        Tries xarray first. If that fails (e.g., for zarr v3 with sharding),
        falls back to direct zarr access wrapped in an xarray-like interface.
        
        Returns:
            Tuple of (xarray.Dataset or ZarrDatasetWrapper, zarr.Group or None)
        """
        if self.zarr_format == 3:
            # Explicitly open as v3
            ds = xr.open_zarr(zarr_path, consolidated=False, zarr_format=3)
        else:
            # Auto-detect format
            ds = xr.open_zarr(zarr_path, consolidated=True)
        return ds
    
    def _clip_hres_to_latent_bounds(self):
        """Clip HRES grid to only include points within latent spatial coverage."""
        lat_min, lat_max = self._latent_lat_orig.min(), self._latent_lat_orig.max()
        lon_min, lon_max = self._latent_lon_orig.min(), self._latent_lon_orig.max()
        
        # Find HRES indices within latent bounds
        hres_lat_mask = (self._hres_lat_orig >= lat_min) & (self._hres_lat_orig <= lat_max)
        hres_lon_mask = (self._hres_lon_orig >= lon_min) & (self._hres_lon_orig <= lon_max)
        
        self.hres_lat_indices = np.where(hres_lat_mask)[0]
        self.hres_lon_indices = np.where(hres_lon_mask)[0]
        self.hres_lat = self._hres_lat_orig[hres_lat_mask]
        self.hres_lon = self._hres_lon_orig[hres_lon_mask]
        
        # Latent uses full grid initially
        self.latent_lat = self._latent_lat_orig.copy()
        self.latent_lon = self._latent_lon_orig.copy()
        self.latent_lat_indices = np.arange(len(self.latent_lat))
        self.latent_lon_indices = np.arange(len(self.latent_lon))
    
    def _apply_region_bounds(self):
        """Optionally filter both grids to a user-specified geographic sub-region."""
        if self.region_bounds is None:
            return
        
        lat_min = self.region_bounds.get("lat_min", -90)
        lat_max = self.region_bounds.get("lat_max", 90)
        lon_min = self.region_bounds.get("lon_min", -180)
        lon_max = self.region_bounds.get("lon_max", 180)
        
        # Filter latent grid
        latent_lat_mask = (self.latent_lat >= lat_min) & (self.latent_lat <= lat_max)
        latent_lon_mask = (self.latent_lon >= lon_min) & (self.latent_lon <= lon_max)
        self.latent_lat_indices = np.where(latent_lat_mask)[0]
        self.latent_lon_indices = np.where(latent_lon_mask)[0]
        self.latent_lat = self.latent_lat[latent_lat_mask]
        self.latent_lon = self.latent_lon[latent_lon_mask]
        
        # Filter HRES grid
        hres_lat_mask = (self.hres_lat >= lat_min) & (self.hres_lat <= lat_max)
        hres_lon_mask = (self.hres_lon >= lon_min) & (self.hres_lon <= lon_max)
        # Update indices relative to original arrays
        self.hres_lat_indices = self.hres_lat_indices[hres_lat_mask]
        self.hres_lon_indices = self.hres_lon_indices[hres_lon_mask]
        self.hres_lat = self.hres_lat[hres_lat_mask]
        self.hres_lon = self.hres_lon[hres_lon_mask]
    
    def _setup_time_indices(self):
        """
        Setup train/val time indices.
        NOTE: this assumes that the latent and HRES datasets have the same timestamps 
        and are in the same order (or a subset in the same order).
        """
        n_common = min(len(self.latent_times), len(self.hres_times))
        
        # Validate that timestamps actually match between datasets
        latent_times_common = pd.to_datetime(self.latent_times[:n_common])
        hres_times_common = pd.to_datetime(self.hres_times[:n_common])
        
        if not (latent_times_common == hres_times_common).all():
            # Find first mismatch
            mismatch_idx = np.where(latent_times_common != hres_times_common)[0][0]
            raise ValueError(
                f"Timestamp mismatch between latent and HRES datasets at index {mismatch_idx}:\\n"
                f"  Latent: {latent_times_common[mismatch_idx]}\\n"
                f"  HRES:   {hres_times_common[mismatch_idx]}\\n"
                f"Datasets must have matching timestamps for the first {n_common} timesteps."
            )
        
        all_indices = list(range(n_common))
        
        if self.split is None:
            self.time_indices = all_indices
            self.split_info = f"all ({n_common} timesteps)"
        else:
            times = pd.to_datetime(self.latent_times[:n_common])
            cutoff = times[-1] - pd.DateOffset(months=self.val_months)
            
            if self.split == 'train':
                self.time_indices = np.where(times <= cutoff)[0].tolist()
                self.split_info = f"train ({len(self.time_indices)} timesteps)"
            else:
                self.time_indices = np.where(times > cutoff)[0].tolist()
                self.split_info = f"val ({len(self.time_indices)} timesteps)"
    
    def _load_statistics(self):
        """Load pre-computed statistics for target variable normalization."""
        self.target_mean = {}
        self.target_std = {}
        self.static_mean = {}
        self.static_std = {}
        
        # Try to load statistics file
        if self.statistics_path is None:
            # Try default path based on hres_zarr_path
            hres_path = Path(self.hres_zarr_path)
            stats_path = hres_path.parent / f"{hres_path.stem}_statistics.json"
        else:
            stats_path = Path(self.statistics_path)
        
        if not stats_path.exists():
            raise ValueError(f"Statistics file not found at {stats_path}")
        
        # Load statistics
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Extract mean and std for each target variable
        for var in self.variables:
            if var in stats:
                self.target_mean[var] = float(stats[var]['mean'])
                self.target_std[var] = float(stats[var]['std'])
            else:
                raise ValueError(f"Statistics not found for variable {var}")
        
        # NOTE: print to be sure
        # print(f"Loaded target statistics from {stats_path}")
        # for var in self.variables:
        #     print(f"  {var}: mean={self.target_mean[var]:.2f}, std={self.target_std[var]:.2f}")
        
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
                    f"Please create this file with mean/std for {self.static_variables} or set use_static_features=false."
                )
        
        # NOTE: print to be sure
        # print(f"Loaded static feature statistics:")
        # for var in self.static_mean.keys():
        #     print(f"  {var}: mean={self.static_mean[var]:.2f}, std={self.static_std[var]:.2f}")

    def _compute_bounds(self):
        """Compute lat/lon bounds for normalization based on latent grid coverage."""
        # Use latent bounds as the base - HRES is already clipped to these
        self.lat_min, self.lat_max = self.latent_lat.min(), self.latent_lat.max()
        self.lon_min, self.lon_max = self.latent_lon.min(), self.latent_lon.max()
    
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
    
    def _build_latent_grid(self):
        """Build latent grid positions."""
        lat_norm = self._normalize(self.latent_lat, self.lat_min, self.lat_max)
        lon_norm = self._normalize(self.latent_lon, self.lon_min, self.lon_max)
        
        lon_grid, lat_grid = np.meshgrid(lon_norm, lat_norm)
        self.latent_positions = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=-1).astype(np.float32)
    
    def _load_static_features(self):
        """Load optional static features and normalize them."""
        if not self.use_static_features:
            self.static_features = None
            return
        
        if self.static_zarr_path is None:
            raise ValueError("static_zarr_path must be specified when use_static_features=True")
        
        # print(f"Loading static features from {self.static_zarr_path}...")
        static_ds = xr.open_zarr(self.static_zarr_path, consolidated=True)
        
        # Check both data_vars and coordinates for static variables
        available = []
        if self.static_variables is None:
            self.static_variables = static_ds.data_vars.keys()
        else:
            for v in self.static_variables:
                if v in static_ds.data_vars or v in static_ds.coords:
                    available.append(v)
        
        if not available:
            raise ValueError(f"No static variables found")
        
        # Load and optionally normalize each static variable
        features = []
        for var in available:
            data = static_ds[var].values
            
            # Slice 2D spatial grid to match HRES region using advanced indexing
            # np.ix_ creates a mesh from 1D index arrays for efficient 2D slicing
            if len(data.shape) == 2:
                data = data[np.ix_(self.hres_lat_indices, self.hres_lon_indices)]
            
            data = data.flatten().astype(np.float32)
            
            # Apply Z-score normalization if statistics available
            if self.normalize_static_features:
                if var not in self.static_mean or var not in self.static_std:
                    raise ValueError(
                        f"Cannot normalize static feature '{var}': "
                        f"statistics not found in {self.static_statistics_path}. "
                        f"Available: {list(self.static_mean.keys())}"
                    )
                
                std_val = self.static_std[var]
                if abs(std_val) < 1e-8:
                    raise ValueError(
                        f"Cannot normalize static feature '{var}': "
                        f"standard deviation is {std_val} (constant or near-constant values). "
                        f"Consider excluding this feature or disabling normalization."
                    )
                
                data = (data - self.static_mean[var]) / std_val
                # print(f"  {var}: normalized (mean={self.static_mean[var]:.2f}, std={std_val:.2f})")
            # else:
                # print(f"  {var}: NOT normalized")
            
            features.append(data)
        
        self.static_features = np.stack(features, axis=-1)
        # print(f"  Loaded static features: {available}, shape={self.static_features.shape}")
    
    def _print_info(self):
        """Print dataset info."""
        pass  # Disabled verbose output
        # print(f"EraLatentHresDataset: {self.split_info}")
        # print(f"  Latent: ({len(self.latent_lat)}, {len(self.latent_lon)})")
        # print(f"  HRES: ({len(self.hres_lat)}, {len(self.hres_lon)})")
        # print(f"  Variables: {self.variables}")
        # print(f"  Samples: {len(self)}")
    
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
    
    def denormalize_targets(self, normalized_data: torch.Tensor, var_idx: int) -> torch.Tensor:
        """
        Denormalize target data back to original scale.
        
        Args:
            normalized_data: [..., num_vars] normalized data
            var_idx: Index of variable to denormalize
            
        Returns:
            Denormalized data in original units
        """
        var = self.variables[var_idx]
        if var not in self.target_mean:
            raise ValueError(f"Cannot denormalize variable '{var}': statistics not found")
        
        mean = self.target_mean[var]
        std = self.target_std[var]
        
        return normalized_data * std + mean
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        time_idx = self.time_indices[idx]
        
        # Load latents with region slicing
        latent_data = self.latent_ds['surface_latents'].isel(
            time=time_idx,
            lat=self.latent_lat_indices,
            lon=self.latent_lon_indices,
        ).values
        # Reshape from (lat, lon, channel) to (lat*lon, channel)
        latent_data = latent_data.reshape(-1, latent_data.shape[-1]).astype(np.float32)
        
        # Load HRES targets with clipping/region slicing
        fields = []
        for var in self.variables:
            data = self.hres_ds[var].isel(
                time=time_idx,
                latitude=self.hres_lat_indices,
                longitude=self.hres_lon_indices,
            ).values.flatten()
            
            # Apply Z-score normalization if enabled
            if var in self.target_mean:
                data = (data - self.target_mean[var]) / self.target_std[var]
            else:
                raise ValueError(f"Cannot normalize variable '{var}': statistics not found")
            
            fields.append(data)
        query_fields = np.stack(fields, axis=-1).astype(np.float32)
        
        # Get positions
        # NOTE: copy is needed to avoid modifying the original data
        latent_pos = self.latent_positions.copy()
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
            'latents': torch.from_numpy(latent_data).float(),
            'latent_pos': torch.from_numpy(latent_pos).float(),
            'query_pos': torch.from_numpy(query_pos).float(),
            'query_fields': torch.from_numpy(query_fields).float(),
            # Grid metadata for analytical KNN (avoids torch.cdist)
            'latent_grid_shape': torch.tensor([len(self.latent_lat), len(self.latent_lon)], dtype=torch.long),
        }
        
        if aux_feats is not None:
            result['query_auxiliary_features'] = torch.from_numpy(aux_feats).float()
        
        return result

