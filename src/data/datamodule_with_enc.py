"""
PyTorch Lightning DataModule for Neural Field Super-Resolution with Encoder.

Uses Aurora predictions as input instead of ERA5 latents.
Surface-only: 2D (lat, lon) for surface variables (2t, msl).
"""

import lightning as L
from torch.utils.data import DataLoader
from typing import Optional, List, Dict

from .era_prediction_hres_dataset import EraPredictionHresDataset


class NeuralFieldEncoderDataModule(L.LightningDataModule):
    """DataModule for Aurora predictions -> HRES surface field reconstruction."""
    
    def __init__(
        self,
        # Data paths
        prediction_zarr_path: str,
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
        statistics_path: Optional[str] = None,
        prediction_statistics_path: Optional[str] = None,
        
        # Static/auxiliary features
        static_zarr_path: Optional[str] = None,
        static_variables: Optional[List[str]] = None,
        static_statistics_path: Optional[str] = None,
        use_static_features: bool = False,
        normalize_static_features: bool = True,
        
        # Region filtering - individual floats for CLI compatibility
        region_lat_min: Optional[float] = None,
        region_lat_max: Optional[float] = None,
        region_lon_min: Optional[float] = None,
        region_lon_max: Optional[float] = None,
        
        # Dataloader optimization
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        
        # Zarr format (None=auto, 2=v2, 3=v3 with sharding support)
        zarr_format: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.prediction_zarr_path = prediction_zarr_path
        self.hres_zarr_path = hres_zarr_path
        self.variables = variables
        self.num_query_samples = num_query_samples
        self.val_months = val_months
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize_coords = normalize_coords
        self.statistics_path = statistics_path
        self.prediction_statistics_path = prediction_statistics_path
        self.static_zarr_path = static_zarr_path
        self.static_variables = static_variables
        self.static_statistics_path = static_statistics_path
        self.use_static_features = use_static_features
        self.normalize_static_features = normalize_static_features
        
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
        
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.zarr_format = zarr_format
        
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = EraPredictionHresDataset(
                prediction_zarr_path=self.prediction_zarr_path,
                hres_zarr_path=self.hres_zarr_path,
                variables=self.variables,
                num_query_samples=self.num_query_samples,
                normalize_coords=self.normalize_coords,
                statistics_path=self.statistics_path,
                prediction_statistics_path=self.prediction_statistics_path,
                split="train",
                val_months=self.val_months,
                static_zarr_path=self.static_zarr_path,
                static_variables=self.static_variables,
                static_statistics_path=self.static_statistics_path,
                use_static_features=self.use_static_features,
                normalize_static_features=self.normalize_static_features,
                region_bounds=self.region_bounds,
                zarr_format=self.zarr_format,
            )
            
            self.val_dataset = EraPredictionHresDataset(
                prediction_zarr_path=self.prediction_zarr_path,
                hres_zarr_path=self.hres_zarr_path,
                variables=self.variables,
                num_query_samples=self.num_query_samples,
                normalize_coords=self.normalize_coords,
                statistics_path=self.statistics_path,
                prediction_statistics_path=self.prediction_statistics_path,
                split="val",
                val_months=self.val_months,
                static_zarr_path=self.static_zarr_path,
                static_variables=self.static_variables,
                static_statistics_path=self.static_statistics_path,
                use_static_features=self.use_static_features,
                normalize_static_features=self.normalize_static_features,
                region_bounds=self.region_bounds,
                zarr_format=self.zarr_format,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
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
    
    @property
    def target_variables(self):
        """Get target variable names for visualization."""
        if self.train_dataset is not None:
            return self.train_dataset.variables
        elif self.val_dataset is not None:
            return self.val_dataset.variables
        return self.variables
    
    @property
    def target_statistics(self):
        """Get target statistics for uncertainty denormalization."""
        if self.train_dataset is not None:
            return self.train_dataset.statistics
        elif self.val_dataset is not None:
            return self.val_dataset.statistics
        return None
