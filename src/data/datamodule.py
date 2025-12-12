"""
PyTorch Lightning DataModule for Neural Field Super-Resolution.

Surface-only: 2D (lat, lon) for surface variables (2t, msl).
"""

import lightning as L
from torch.utils.data import DataLoader
from typing import Optional, List, Dict

from .era_latent_hres_dataset import EraLatentHresDataset


class NeuralFieldDataModule(L.LightningDataModule):
    """DataModule for ERA5 latents -> HRES surface field reconstruction."""
    
    def __init__(
        self,
        # Data paths
        latent_zarr_path: str,
        hres_zarr_path: str,
        
        # Variables
        variables: Optional[List[str]] = None,
        
        # Query sampling
        num_query_samples: Optional[int] = None,
        
        # Train/val split
        val_months: int = 3,
        
        # DataLoader
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        
        # Normalization
        normalize_coords: bool = True,
        normalize_targets: bool = True,
        statistics_path: Optional[str] = None,
        
        # Static/auxiliary features
        static_zarr_path: Optional[str] = None,
        static_variables: Optional[List[str]] = None,
        use_static_features: bool = False,
        
        # Region filtering - individual floats for CLI compatibility
        region_lat_min: Optional[float] = None,
        region_lat_max: Optional[float] = None,
        region_lon_min: Optional[float] = None,
        region_lon_max: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.latent_zarr_path = latent_zarr_path
        self.hres_zarr_path = hres_zarr_path
        self.variables = variables
        self.num_query_samples = num_query_samples
        self.val_months = val_months
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize_coords = normalize_coords
        self.normalize_targets = normalize_targets
        self.statistics_path = statistics_path
        self.static_zarr_path = static_zarr_path
        self.static_variables = static_variables
        self.use_static_features = use_static_features
        
        # Build region_bounds dict from individual params if any are set
        if all(v is not None for v in [region_lat_min, region_lat_max, region_lon_min, region_lon_max]):
            self.region_bounds = {
                "lat_min": region_lat_min,
                "lat_max": region_lat_max,
                "lon_min": region_lon_min,
                "lon_max": region_lon_max,
            }
        else:
            self.region_bounds = None
        
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = EraLatentHresDataset(
                latent_zarr_path=self.latent_zarr_path,
                hres_zarr_path=self.hres_zarr_path,
                variables=self.variables,
                num_query_samples=self.num_query_samples,
                normalize_coords=self.normalize_coords,
                normalize_targets=self.normalize_targets,
                statistics_path=self.statistics_path,
                split="train",
                val_months=self.val_months,
                static_zarr_path=self.static_zarr_path,
                static_variables=self.static_variables,
                use_static_features=self.use_static_features,
                region_bounds=self.region_bounds,
            )
            
            self.val_dataset = EraLatentHresDataset(
                latent_zarr_path=self.latent_zarr_path,
                hres_zarr_path=self.hres_zarr_path,
                variables=self.variables,
                num_query_samples=self.num_query_samples,
                normalize_coords=self.normalize_coords,
                normalize_targets=self.normalize_targets,
                statistics_path=self.statistics_path,
                split="val",
                val_months=self.val_months,
                static_zarr_path=self.static_zarr_path,
                static_variables=self.static_variables,
                use_static_features=self.use_static_features,
                region_bounds=self.region_bounds,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )
    
    @property
    def hres_shape(self) -> tuple:
        """Get HRES grid shape (lat, lon) for visualization."""
        if self.train_dataset is not None:
            return self.train_dataset.hres_shape
        elif self.val_dataset is not None:
            return self.val_dataset.hres_shape
        return None
    
    @property
    def geo_bounds(self) -> dict:
        """Get geographic bounds for visualization."""
        if self.train_dataset is not None:
            return self.train_dataset.geo_bounds
        elif self.val_dataset is not None:
            return self.val_dataset.geo_bounds
        return None
    
    def denormalize_targets(self, normalized_data, var_idx: int):
        """Denormalize target data back to original scale."""
        if self.val_dataset is not None:
            return self.val_dataset.denormalize_targets(normalized_data, var_idx)
        elif self.train_dataset is not None:
            return self.train_dataset.denormalize_targets(normalized_data, var_idx)
        return normalized_data
