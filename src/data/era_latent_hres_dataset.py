"""
Dataset and DataLoader for loading ERA5 latents and HRES field values from Zarr.

Goal: Reconstruct HRES high-resolution field values (targets) given latents from ERA5.

The dataset loads:
- Latents: from ERA5 low-res Zarr (lat, lon, timesteps, levels/channels)
- Targets: HRES high-res field values (lat, lon, timesteps, variables)

Supports:
- Surface-only training (2D: lat, lon)
- Atmospheric-only training (3D: lat, lon, level)
- Combined surface + atmospheric training

Each sample returns:
- latents: [Z, D_latent] - latent features at latent grid positions
- latent_pos: [Z, coord_dim] - positions of latents (lat, lon) or (lat, lon, level)
- query_pos: [Q, coord_dim] - HRES grid positions to predict
- query_features: [Q, num_vars] - target field values at query positions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Literal, Union
import xarray as xr


class EraLatentHresDataset(Dataset):
    """
    Dataset for neural field super-resolution from ERA5 latents to HRES fields.
    
    Args:
        latent_zarr_path: Path to Zarr store with ERA5 latents
        hres_zarr_path: Path to Zarr store with HRES field values
        variables: List of variable names to load from HRES (e.g., ['2t', '10u', '10v'])
                   If None, all data variables will be loaded.
        mode: Training mode - 'surface', 'atmospheric', or 'all'
              - 'surface': 2D coords (lat, lon), surface-level latents only
              - 'atmospheric': 3D coords (lat, lon, level), atmospheric latents only
              - 'all': 3D coords, all levels including surface
        num_query_samples: Number of HRES positions to sample per batch (None = all)
        normalize_coords: Whether to normalize coordinates to [-1, 1]
        surface_level_idx: Index of surface level in latent data (default: 0)
        atmospheric_level_range: Tuple (start, end) for atmospheric level indices
        split: 'train', 'val', or None. If specified, automatically splits by time.
        val_months: Number of months at the end to use for validation (default: 3)
        time_indices: Optional explicit list of time indices to use (overrides split)
    """
    
    def __init__(
        self,
        latent_zarr_path: str,
        hres_zarr_path: str,
        variables: Optional[List[str]] = None, #TODO: not sure
        mode: Literal['surface', 'atmospheric', 'all'] = 'all',
        num_query_samples: Optional[int] = None, #TODO: not sure
        normalize_coords: bool = True, #TODO: not sure
        surface_level_idx: int = 0,
        atmospheric_level_range: Optional[Tuple[int, int]] = None, #TODO: not sure
        split: Optional[Literal['train', 'val']] = None,
        val_months: int = 3,
        time_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.latent_zarr_path = latent_zarr_path
        self.hres_zarr_path = hres_zarr_path
        self.normalize_coords = normalize_coords
        self.mode = mode
        self.num_query_samples = num_query_samples
        self.surface_level_idx = surface_level_idx
        self.atmospheric_level_range = atmospheric_level_range
        self.split = split
        self.val_months = val_months
        
        # Determine coordinate dimensionality based on mode
        self.coord_dim = 2 if mode == 'surface' else 3
        
        # Open Zarr stores with xarray for easier coordinate handling
        self.latent_ds = xr.open_zarr(latent_zarr_path, consolidated=True)
        self.hres_ds = xr.open_zarr(hres_zarr_path, consolidated=True)
        
        # Detect coordinate names for latents
        self.latent_lat_name = self._detect_coord(self.latent_ds, ['lat', 'latitude'])
        self.latent_lon_name = self._detect_coord(self.latent_ds, ['lon', 'longitude'])
        self.latent_time_name = self._detect_coord(self.latent_ds, ['time', 'timestep', 'timesteps'])
        self.latent_level_name = self._detect_coord(self.latent_ds, ['level', 'levels', 'pressure_level', 'isobaricInhPa'])
        
        # Detect coordinate names for HRES
        self.hres_lat_name = self._detect_coord(self.hres_ds, ['lat', 'latitude'])
        self.hres_lon_name = self._detect_coord(self.hres_ds, ['lon', 'longitude'])
        self.hres_time_name = self._detect_coord(self.hres_ds, ['time', 'timestep', 'timesteps'])
        self.hres_level_name = self._detect_coord(self.hres_ds, ['level', 'levels', 'pressure_level', 'isobaricInhPa'])
        
        # Store coordinates
        self.latent_lat = self.latent_ds[self.latent_lat_name].values
        self.latent_lon = self.latent_ds[self.latent_lon_name].values
        self.latent_times = self.latent_ds[self.latent_time_name].values
        self.latent_levels = (
            self.latent_ds[self.latent_level_name].values 
            if self.latent_level_name else np.array([0])
        )
        
        self.hres_lat = self.hres_ds[self.hres_lat_name].values
        self.hres_lon = self.hres_ds[self.hres_lon_name].values
        self.hres_times = self.hres_ds[self.hres_time_name].values
        self.hres_levels = (
            self.hres_ds[self.hres_level_name].values
            if self.hres_level_name else np.array([0])
        )
        
        # Setup time indices for train/val split
        self._setup_time_indices(time_indices)
        
        # Determine which levels to use based on mode
        self._setup_level_selection()
        
        # Determine variables to load
        if variables is None:
            # Auto-detect data variables (exclude coordinates)
            self.variables = [
                v for v in self.hres_ds.data_vars 
                if v not in self.hres_ds.coords
            ]
        else:
            self.variables = variables
        
        # Compute coordinate bounds for normalization
        self._compute_bounds()
        
        # Pre-compute HRES grid positions
        self._build_hres_grid() #TODO: hceck if needed
        
        # Pre-compute latent grid positions
        self._build_latent_grid() #TODO: check if needed
        
        self._print_info()
        
    def _setup_time_indices(self, time_indices: Optional[List[int]] = None):
        """
        Setup time indices for train/val splitting.
        
        For split='train': use all timesteps EXCEPT last val_months
        For split='val': use ONLY last val_months
        For split=None: use all timesteps
        """
        # Find common timesteps between latent and HRES
        n_common = min(len(self.latent_times), len(self.hres_times))
        all_indices = list(range(n_common))
        
        if time_indices is not None:
            # Use explicit time indices
            self.time_indices = time_indices
            self.split_info = f"custom ({len(time_indices)} timesteps)"
        elif self.split is None:
            # Use all timesteps
            self.time_indices = all_indices
            self.split_info = f"all ({n_common} timesteps)"
        else:
            # Compute split based on val_months
            # Convert times to pandas for easy month extraction
            times = pd.to_datetime(self.latent_times[:n_common])
            
            # Find the cutoff: last val_months
            last_time = times[-1]
            cutoff_time = last_time - pd.DateOffset(months=self.val_months)
            
            train_mask = times <= cutoff_time
            val_mask = times > cutoff_time
            
            train_indices = np.where(train_mask)[0].tolist()
            val_indices = np.where(val_mask)[0].tolist()
            
            if self.split == 'train':
                self.time_indices = train_indices
                self.split_info = f"train ({len(train_indices)} timesteps, up to {cutoff_time.strftime('%Y-%m-%d')})"
            else:  # 'val'
                self.time_indices = val_indices
                self.split_info = f"val ({len(val_indices)} timesteps, from {cutoff_time.strftime('%Y-%m-%d')})"
    
    def _print_info(self):
        """Print dataset information."""
        print(f"EraLatentHresDataset initialized:")
        print(f"  Split: {self.split_info}")
        print(f"  Mode: {self.mode} (coord_dim={self.coord_dim})")
        print(f"  Latent grid: ({len(self.latent_lat)}, {len(self.latent_lon)}, "
              f"{len(self.selected_latent_levels)}) levels at {len(self.latent_times)} total timesteps")
        if self.mode == 'surface':
            print(f"    Using surface level idx: {self.surface_level_idx}")
        elif self.mode == 'atmospheric':
            print(f"    Using atmospheric levels: {self.atmospheric_level_range}")
        print(f"  HRES grid: ({len(self.hres_lat)}, {len(self.hres_lon)}) "
              f"at {len(self.hres_times)} total timesteps")
        print(f"  Variables: {self.variables}")
        print(f"  Samples in this split: {len(self)}")
        if self.num_query_samples:
            print(f"  Query sampling: {self.num_query_samples} points per sample")
        
    def _setup_level_selection(self): #TODO: check this is wrong, surface doesnt have latent
        """Setup which levels to use based on mode."""
        if self.mode == 'surface':
            # Use only surface level
            self.selected_latent_levels = np.array([self.latent_levels[self.surface_level_idx]])
            self.selected_latent_level_indices = [self.surface_level_idx]
        elif self.mode == 'atmospheric':
            # Use atmospheric levels (exclude surface)
            if self.atmospheric_level_range is not None:
                start, end = self.atmospheric_level_range
            else:
                # Default: all levels except surface (assumed at idx 0)
                start, end = 1, len(self.latent_levels)
            self.selected_latent_levels = self.latent_levels[start:end]
            self.selected_latent_level_indices = list(range(start, end))
        else:  # 'all'
            # Use all levels
            self.selected_latent_levels = self.latent_levels
            self.selected_latent_level_indices = list(range(len(self.latent_levels)))
            
    def _detect_coord(self, ds: xr.Dataset, names: List[str]) -> Optional[str]:
        """Detect coordinate name from a list of possible names."""
        for name in names:
            if name in ds.coords or name in ds.dims:
                return name
        return None
    
    def _compute_bounds(self):
        """Compute lat/lon/level bounds for normalization."""
        all_lats = np.concatenate([self.latent_lat, self.hres_lat])
        all_lons = np.concatenate([self.latent_lon, self.hres_lon])
        
        self.lat_min, self.lat_max = all_lats.min(), all_lats.max()
        self.lon_min, self.lon_max = all_lons.min(), all_lons.max()
        
        if self.latent_level_name:
            self.level_min = self.selected_latent_levels.min()
            self.level_max = self.selected_latent_levels.max()
        else: #TODO: check this
            self.level_min, self.level_max = 0, 1
            
    def _normalize_lat(self, lat: np.ndarray) -> np.ndarray:
        """Normalize latitude to [-1, 1]."""
        if not self.normalize_coords:
            return lat
        return 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min + 1e-8) - 1
    
    def _normalize_lon(self, lon: np.ndarray) -> np.ndarray:
        """Normalize longitude to [-1, 1]."""
        if not self.normalize_coords:
            return lon
        return 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min + 1e-8) - 1
    
    def _normalize_level(self, level: np.ndarray) -> np.ndarray:
        """Normalize level to [-1, 1]."""
        if not self.normalize_coords:
            return level
        # Handle single level case
        if self.level_max == self.level_min:
            return np.zeros_like(level)
        return 2 * (level - self.level_min) / (self.level_max - self.level_min + 1e-8) - 1
        
    def _build_hres_grid(self):
        """Build HRES query grid positions."""
        hres_lat_norm = self._normalize_lat(self.hres_lat)
        hres_lon_norm = self._normalize_lon(self.hres_lon)
        
        if self.coord_dim == 2:
            # 2D grid for surface mode
            lon_grid, lat_grid = np.meshgrid(hres_lon_norm, hres_lat_norm)
            self.hres_positions = np.stack([
                lat_grid.flatten(),
                lon_grid.flatten()
            ], axis=-1).astype(np.float32)
        else:
            # 3D grid - HRES typically surface only, so level=0
            lon_grid, lat_grid = np.meshgrid(hres_lon_norm, hres_lat_norm)
            level_norm = self._normalize_level(np.array([0.0]))  # Surface level
            self.hres_positions = np.stack([
                lat_grid.flatten(),
                lon_grid.flatten(),
                np.full(lat_grid.size, level_norm[0])  # All at surface level
            ], axis=-1).astype(np.float32)
        
        self.hres_shape = (len(self.hres_lat), len(self.hres_lon))
        self.num_hres_points = self.hres_positions.shape[0]
        
    def _build_latent_grid(self):
        """Build latent grid positions."""
        latent_lat_norm = self._normalize_lat(self.latent_lat)
        latent_lon_norm = self._normalize_lon(self.latent_lon)
        
        if self.coord_dim == 2:
            # 2D grid for surface mode
            lon_grid, lat_grid = np.meshgrid(latent_lon_norm, latent_lat_norm)
            self.latent_positions = np.stack([
                lat_grid.flatten(),
                lon_grid.flatten()
            ], axis=-1).astype(np.float32)
            self.latent_shape = (len(self.latent_lat), len(self.latent_lon))
        else:
            # 3D grid with levels
            latent_level_norm = self._normalize_level(self.selected_latent_levels.astype(np.float32))
            
            # Create 3D meshgrid
            lon_grid, lat_grid, level_grid = np.meshgrid(
                latent_lon_norm, latent_lat_norm, latent_level_norm, indexing='ij'
            )
            # Reorder to (lat, lon, level) for consistency
            lat_grid = lat_grid.transpose(1, 0, 2)
            lon_grid = lon_grid.transpose(1, 0, 2)
            level_grid = level_grid.transpose(1, 0, 2)
            
            self.latent_positions = np.stack([
                lat_grid.flatten(),
                lon_grid.flatten(),
                level_grid.flatten()
            ], axis=-1).astype(np.float32)
            self.latent_shape = (
                len(self.latent_lat), 
                len(self.latent_lon), 
                len(self.selected_latent_levels)
            )
        
        self.num_latent_points = self.latent_positions.shape[0]
        
    def __len__(self) -> int:
        """Number of samples = number of timesteps in this split."""
        return len(self.time_indices)
    
    def _get_latent_data(self, idx: int) -> np.ndarray:
        """Load latent data for a specific sample index."""
        # Map sample index to actual time index
        time_idx = self.time_indices[idx]
        time_val = self.latent_times[time_idx]
        
        # Detect the main latent data variable
        latent_var = self._detect_latent_variable()
        
        # Load data at this timestep
        data = self.latent_ds[latent_var].sel({self.latent_time_name: time_val})
        
        # Select levels based on mode
        if self.latent_level_name:
            level_vals = [self.latent_levels[i] for i in self.selected_latent_level_indices]
            if len(level_vals) == 1:
                data = data.sel({self.latent_level_name: level_vals[0]})
            else:
                data = data.sel({self.latent_level_name: level_vals})
        
        data_np = data.values
        
        # Reshape to [num_spatial_points, features]
        # Expected shapes:
        # - 2D mode: [lat, lon, features] -> [lat*lon, features]
        # - 3D mode: [lat, lon, level, features] or [level, lat, lon, features] -> [lat*lon*level, features]
        
        if data_np.ndim == 2:
            # [lat, lon] -> [lat*lon, 1]
            return data_np.reshape(-1, 1).astype(np.float32)
        elif data_np.ndim == 3:
            if self.mode == 'surface':
                # [lat, lon, features] -> [lat*lon, features]
                return data_np.reshape(-1, data_np.shape[-1]).astype(np.float32)
            else:
                # Could be [lat, lon, level] or [level, lat, lon]
                # Flatten all spatial dims
                return data_np.reshape(-1, 1).astype(np.float32)
        else:
            # [lat, lon, level, features] or similar
            # Flatten to [lat*lon*level, features]
            n_features = data_np.shape[-1] if data_np.ndim > 3 else 1
            return data_np.reshape(-1, n_features).astype(np.float32)
    
    def _detect_latent_variable(self) -> str:
        """Detect the main latent data variable name."""
        possible_names = ['latent', 'latents', 'embedding', 'embeddings', 'features', 'data']
        for name in possible_names:
            if name in self.latent_ds.data_vars:
                return name
        # Return first data variable if no match
        data_vars = list(self.latent_ds.data_vars)
        if data_vars:
            return data_vars[0]
        raise ValueError(f"No data variables found in latent dataset. Available: {list(self.latent_ds.variables)}")
    
    def _get_hres_data(self, idx: int) -> np.ndarray:
        """Load HRES field data for a specific sample index."""
        # Map sample index to actual time index
        time_idx = self.time_indices[idx]
        time_val = self.hres_times[time_idx]
        
        fields = []
        for var in self.variables:
            data = self.hres_ds[var].sel({self.hres_time_name: time_val})
            
            # Handle case where HRES has levels too
            if self.hres_level_name and self.hres_level_name in data.dims:
                # Select surface level for HRES if it has multiple levels
                data = data.isel({self.hres_level_name: 0})
            
            fields.append(data.values.flatten())
        
        # Stack variables: [H*W, num_vars]
        return np.stack(fields, axis=-1).astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample for the given timestep index.
        
        Returns:
            dict with:
                - latents: [Z, D_latent] latent features
                - latent_pos: [Z, coord_dim] latent positions (2D or 3D based on mode)
                - query_pos: [Q, coord_dim] query positions (HRES grid)
                - query_features: [Q, num_vars] target field values
        """
        # Load latent data
        latent_data = self._get_latent_data(idx)
        
        # Load HRES target data
        hres_data = self._get_hres_data(idx)
        
        # Get positions
        latent_pos = self.latent_positions.copy()
        query_pos = self.hres_positions.copy()
        query_features = hres_data
        
        # Sample subset of query positions if specified
        if self.num_query_samples is not None and self.num_query_samples < self.num_hres_points:
            indices = np.random.choice(
                self.num_hres_points, 
                size=self.num_query_samples, 
                replace=False
            )
            query_pos = query_pos[indices]
            query_features = query_features[indices]
        
        return {
            'latents': torch.from_numpy(latent_data).float(),
            'latent_pos': torch.from_numpy(latent_pos).float(),
            'query_pos': torch.from_numpy(query_pos).float(),
            'query_features': torch.from_numpy(query_features).float(),
        }


def create_superres_dataloader(
    latent_zarr_path: str,
    hres_zarr_path: str,
    batch_size: int = 1,
    variables: Optional[List[str]] = None,
    mode: Literal['surface', 'atmospheric', 'all'] = 'all',
    num_query_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> Tuple[EraLatentHresDataset, DataLoader]:
    """
    Create dataset and dataloader for super-resolution training.
    
    Args:
        latent_zarr_path: Path to ERA5 latents Zarr
        hres_zarr_path: Path to HRES fields Zarr
        batch_size: Batch size (typically 1 since each sample is a full field)
        variables: Variables to load from HRES (None = auto-detect all)
        mode: 'surface', 'atmospheric', or 'all'
        num_query_samples: Number of query points to sample (None = all)
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        **dataset_kwargs: Additional args passed to EraLatentHresDataset
        
    Returns:
        (dataset, dataloader) tuple
    """
    dataset = EraLatentHresDataset(
        latent_zarr_path=latent_zarr_path,
        hres_zarr_path=hres_zarr_path,
        variables=variables,
        mode=mode,
        num_query_samples=num_query_samples,
        **dataset_kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return dataset, dataloader


def create_train_val_dataloaders(
    latent_zarr_path: str,
    hres_zarr_path: str,
    batch_size: int = 1,
    variables: Optional[List[str]] = None,
    mode: Literal['surface', 'atmospheric', 'all'] = 'all',
    num_query_samples: Optional[int] = None,
    val_months: int = 3,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[Tuple[EraLatentHresDataset, DataLoader], Tuple[EraLatentHresDataset, DataLoader]]:
    """
    Create train and validation datasets and dataloaders.
    
    Splits temporally: last `val_months` months are validation, rest is training.
    
    Example with 2018-2020 data and val_months=3:
        - Train: 2018-01-01 to 2020-09-30
        - Val: 2020-10-01 to 2020-12-31
    
    Args:
        latent_zarr_path: Path to ERA5 latents Zarr
        hres_zarr_path: Path to HRES fields Zarr
        batch_size: Batch size
        variables: Variables to load from HRES
        mode: 'surface', 'atmospheric', or 'all'
        num_query_samples: Number of query points to sample
        val_months: Number of months at the end for validation (default: 3)
        num_workers: DataLoader workers
        **dataset_kwargs: Additional args passed to EraLatentHresDataset
        
    Returns:
        ((train_dataset, train_dataloader), (val_dataset, val_dataloader))
    """
    # Create training dataset/dataloader
    train_dataset = EraLatentHresDataset(
        latent_zarr_path=latent_zarr_path,
        hres_zarr_path=hres_zarr_path,
        variables=variables,
        mode=mode,
        num_query_samples=num_query_samples,
        split='train',
        val_months=val_months,
        **dataset_kwargs
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    # Create validation dataset/dataloader
    val_dataset = EraLatentHresDataset(
        latent_zarr_path=latent_zarr_path,
        hres_zarr_path=hres_zarr_path,
        variables=variables,
        mode=mode,
        num_query_samples=num_query_samples,
        split='val',
        val_months=val_months,
        **dataset_kwargs
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return (train_dataset, train_dataloader), (val_dataset, val_dataloader)


if __name__ == "__main__":
    # Example usage with placeholder paths
    LATENT_ZARR = "/path/to/latents_europe.zarr"
    HRES_ZARR = "/path/to/hres_europe.zarr"
    
    print("=" * 60)
    print("EraLatentHresDataset - Example Usage")
    print("=" * 60)
    
    print("\n--- Surface Mode (2D) ---")
    print("""
    dataset, dataloader = create_superres_dataloader(
        latent_zarr_path=LATENT_ZARR,
        hres_zarr_path=HRES_ZARR,
        mode='surface',           # Only surface level, 2D coords (lat, lon)
        variables=['2t', 'msl'],  # Surface variables
        num_query_samples=1000,
    )
    # latent_pos: [B, Z, 2], query_pos: [B, Q, 2]
    """)
    
    print("\n--- Atmospheric Mode (3D) ---")
    print("""
    dataset, dataloader = create_superres_dataloader(
        latent_zarr_path=LATENT_ZARR,
        hres_zarr_path=HRES_ZARR,
        mode='atmospheric',       # Atmospheric levels, 3D coords (lat, lon, level)
        variables=['t', 'q'],     # Atmospheric variables
        atmospheric_level_range=(1, 13),  # Levels 1-12
        num_query_samples=2000,
    )
    # latent_pos: [B, Z, 3], query_pos: [B, Q, 3]
    """)
    
    print("\n--- All Levels Mode (3D) ---")
    print("""
    dataset, dataloader = create_superres_dataloader(
        latent_zarr_path=LATENT_ZARR,
        hres_zarr_path=HRES_ZARR,
        mode='all',               # All levels, 3D coords
        variables=None,           # Auto-detect all variables
        num_query_samples=5000,
    )
    # latent_pos: [B, Z, 3], query_pos: [B, Q, 3]
    """)
    
    # Uncomment below to test with real data:
    # try:
    #     dataset, dataloader = create_superres_dataloader(
    #         latent_zarr_path=LATENT_ZARR,
    #         hres_zarr_path=HRES_ZARR,
    #         mode='surface',
    #         num_query_samples=1000,
    #     )
    #     
    #     sample = dataset[0]
    #     print("\nSample shapes:")
    #     for key, value in sample.items():
    #         print(f"  {key}: {value.shape}")
    #         
    # except Exception as e:
    #     print(f"\nCould not load data: {e}")
