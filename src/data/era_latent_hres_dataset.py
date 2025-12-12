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


class EraLatentHresDataset(Dataset):
    """
    Dataset for neural field super-resolution from ERA5 latents to HRES surface fields.
    
    Args:
        latent_zarr_path: Path to Zarr store with ERA5 latents
        hres_zarr_path: Path to Zarr store with HRES field values
        variables: List of surface variable names (e.g., ['2t', 'msl'])
        num_query_samples: Number of HRES positions to sample per batch (None = all)
        normalize_coords: Whether to normalize coordinates to [-1, 1]
        split: 'train', 'val', or None. Splits by time.
        val_months: Number of months at the end to use for validation
        static_zarr_path: Optional path to static features zarr
        use_static_features: Whether to load static features
    """
    
    def __init__(
        self,
        latent_zarr_path: str,
        hres_zarr_path: str,
        variables: Optional[List[str]] = None,
        static_zarr_path: Optional[str] = None,
        static_variables: Optional[List[str]] = None,
        use_static_features: bool = False,
        num_query_samples: Optional[int] = None,
        normalize_coords: bool = True,
        split: Optional[str] = None,
        val_months: int = 3,
    ):
        super().__init__()
        
        self.latent_zarr_path = latent_zarr_path
        self.hres_zarr_path = hres_zarr_path
        self.normalize_coords = normalize_coords
        self.num_query_samples = num_query_samples
        self.split = split
        self.val_months = val_months
        self.static_zarr_path = static_zarr_path
        self.static_variables = static_variables or ['z', 'lsm', 'slt']
        self.use_static_features = use_static_features
        
        # Open Zarr stores
        self.latent_ds = xr.open_zarr(latent_zarr_path, consolidated=True)
        self.hres_ds = xr.open_zarr(hres_zarr_path, consolidated=True)
        
        # Get coordinates (assume standard names)
        self.latent_lat = self.latent_ds['lat'].values
        self.latent_lon = self.latent_ds['lon'].values
        self.latent_times = self.latent_ds['time'].values
        
        self.hres_lat = self.hres_ds['latitude'].values
        self.hres_lon = self.hres_ds['longitude'].values
        self.hres_times = self.hres_ds['time'].values
        
        # Setup time indices
        self._setup_time_indices()
        
        # Determine variables
        if variables is None:
            self.variables = [v for v in self.hres_ds.data_vars if v not in self.hres_ds.coords]
        else:
            self.variables = variables
        
        # Compute bounds for normalization
        self._compute_bounds()
        
        # Build position grids
        self._build_hres_grid()
        self._build_latent_grid()
        
        # Load static features
        self._load_static_features()
        
        self._print_info()
    
    def _setup_time_indices(self):
        """Setup train/val time indices."""
        n_common = min(len(self.latent_times), len(self.hres_times))
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
    
    def _compute_bounds(self):
        """Compute lat/lon bounds for normalization."""
        all_lats = np.concatenate([self.latent_lat, self.hres_lat])
        all_lons = np.concatenate([self.latent_lon, self.hres_lon])
        self.lat_min, self.lat_max = all_lats.min(), all_lats.max()
        self.lon_min, self.lon_max = all_lons.min(), all_lons.max()
    
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
        """Load optional static features."""
        if not self.use_static_features or self.static_zarr_path is None:
            self.static_features = None
            return
        
        print(f"Loading static features from {self.static_zarr_path}...")
        static_ds = xr.open_zarr(self.static_zarr_path, consolidated=True)
        
        available = [v for v in self.static_variables if v in static_ds.data_vars]
        if not available:
            print(f"  Warning: No static variables found")
            self.static_features = None
            return
        
        features = [static_ds[v].values.flatten().astype(np.float32) for v in available]
        self.static_features = np.stack(features, axis=-1)
        print(f"  Loaded: {available}, shape={self.static_features.shape}")
    
    def _print_info(self):
        """Print dataset info."""
        print(f"EraLatentHresDataset: {self.split_info}")
        print(f"  Latent: ({len(self.latent_lat)}, {len(self.latent_lon)})")
        print(f"  HRES: ({len(self.hres_lat)}, {len(self.hres_lon)})")
        print(f"  Variables: {self.variables}")
        print(f"  Samples: {len(self)}")
    
    def __len__(self) -> int:
        return len(self.time_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        time_idx = self.time_indices[idx]
        
        # Load latents - use surface_latents (not the first var which might be lat_bounds)
        latent_data = self.latent_ds['surface_latents'].isel(time=time_idx).values
        # Reshape from (lat, lon, channel) to (lat*lon, channel)
        latent_data = latent_data.reshape(-1, latent_data.shape[-1]).astype(np.float32)
        
        # Load HRES targets
        fields = []
        for var in self.variables:
            data = self.hres_ds[var].isel(time=time_idx).values.flatten()
            fields.append(data)
        query_fields = np.stack(fields, axis=-1).astype(np.float32)
        
        # Get positions
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
        }
        
        if aux_feats is not None:
            result['query_auxiliary_features'] = torch.from_numpy(aux_feats).float()
        
        return result


def create_superres_dataloader(
    latent_zarr_path: str,
    hres_zarr_path: str,
    variables: Optional[List[str]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create a single DataLoader for the super-resolution dataset."""
    dataset = EraLatentHresDataset(
        latent_zarr_path=latent_zarr_path,
        hres_zarr_path=hres_zarr_path,
        variables=variables,
        **kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_train_val_dataloaders(
    latent_zarr_path: str,
    hres_zarr_path: str,
    variables: Optional[List[str]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    val_months: int = 3,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_dataset = EraLatentHresDataset(
        latent_zarr_path=latent_zarr_path,
        hres_zarr_path=hres_zarr_path,
        variables=variables,
        split='train',
        val_months=val_months,
        **kwargs,
    )
    val_dataset = EraLatentHresDataset(
        latent_zarr_path=latent_zarr_path,
        hres_zarr_path=hres_zarr_path,
        variables=variables,
        split='val',
        val_months=val_months,
        **kwargs,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
