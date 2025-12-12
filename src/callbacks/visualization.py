"""
HRES Visualization Callback for training monitoring.

Generates side-by-side visualizations of ground truth and predicted HRES fields
during training, using cartopy for geographic projection with land mask overlay.
"""

import torch
import numpy as np
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
from PIL import Image

# Try to import cartopy (optional but recommended)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Geographic visualization will use basic imshow.")


class HRESVisualizationCallback(L.Callback):
    """
    Callback to visualize HRES predictions during validation.
    
    Creates side-by-side plots of:
    - Ground truth HRES field
    - Predicted HRES field  
    - Difference (error) map
    
    Uses cartopy for proper geographic projection with land mask overlay.
    Logs visualizations to WandB as images.
    """
    
    def __init__(
        self,
        log_every_n_epochs: int = 5,
        num_samples: int = 4,
        variable_name: str = "2t",
        colormap: str = "RdYlBu_r",  # Good for temperature
        error_colormap: str = "RdBu_r",
        figsize: Tuple[int, int] = (18, 5),
        dpi: int = 100,
        land_alpha: float = 0.3,  # Opacity of land mask
        use_cartopy: bool = True,  # Set to False to disable cartopy
    ):
        """
        Args:
            log_every_n_epochs: Frequency of visualization logging.
            num_samples: Number of samples to visualize per epoch.
            variable_name: Name of the variable being visualized (for title).
            colormap: Matplotlib colormap for field values.
            error_colormap: Colormap for error/difference maps.
            figsize: Figure size (width, height) in inches.
            dpi: Figure DPI.
            land_alpha: Opacity of land mask overlay (0-1).
            use_cartopy: Whether to use cartopy for geographic projection.
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.variable_name = variable_name
        self.colormap = colormap
        self.error_colormap = error_colormap
        self.figsize = figsize
        self.dpi = dpi
        self.land_alpha = land_alpha
        self.use_cartopy = use_cartopy and HAS_CARTOPY
        
    def on_validation_epoch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule
    ) -> None:
        """Generate and log full-grid visualizations at end of validation epoch."""
        # Check if we should log this epoch (always log epoch 0 for pre-training viz)
        if trainer.current_epoch == 0:
            pass  # Always visualize at epoch 0
        elif (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        
        # Get WandB logger
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            return
        
        # Get datamodule for full-grid access
        if trainer.datamodule is None:
            return
        
        # Get HRES grid info
        hres_shape = self._get_hres_shape(trainer)
        geo_bounds = self._get_geo_bounds(trainer)
        
        if hres_shape is None:
            return
        
        # Get the validation dataset for full-grid samples
        val_dataset = trainer.datamodule.val_dataset
        if val_dataset is None:
            return
        
        # Store original num_query_samples and temporarily disable subsampling
        original_num_query_samples = val_dataset.num_query_samples
        val_dataset.num_query_samples = None  # Use full grid
        
        device = pl_module.device
        images = []
        
        try:
            # Generate visualizations for a few samples
            for sample_idx in range(min(self.num_samples, len(val_dataset))):
                sample = val_dataset[sample_idx]
                
                # Move to device and add batch dimension
                latents = sample['latents'].unsqueeze(0).to(device)          # [1, Z, D]
                latent_pos = sample['latent_pos'].unsqueeze(0).to(device)    # [1, Z, 2]
                query_pos = sample['query_pos'].unsqueeze(0).to(device)      # [1, Q, 2]
                query_fields = sample['query_fields'].unsqueeze(0)            # [1, Q, V]
                
                # Optional auxiliary features
                query_aux = None
                if 'query_auxiliary_features' in sample:
                    query_aux = sample['query_auxiliary_features'].unsqueeze(0).to(device)
                
                # Use chunked inference for full grid (memory-efficient)
                with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cpu', 
                                        enabled=device.type == 'cuda'):
                    predictions = pl_module.chunked_forward(
                        query_pos=query_pos,
                        latents=latents,
                        latent_pos=latent_pos,
                        query_auxiliary_features=query_aux,
                        chunk_size=8192,
                    )
                
                # Move to CPU for visualization
                pred = predictions[0, :, 0].cpu().numpy()    # [Q] - first variable
                target = query_fields[0, :, 0].numpy()       # [Q]
                
                # Denormalize predictions and targets back to original scale
                if hasattr(trainer.datamodule, 'denormalize_targets'):
                    var_idx = 0  # First variable
                    pred = trainer.datamodule.denormalize_targets(
                        torch.from_numpy(pred).unsqueeze(-1), var_idx
                    ).squeeze(-1).numpy()
                    target = trainer.datamodule.denormalize_targets(
                        torch.from_numpy(target).unsqueeze(-1), var_idx
                    ).squeeze(-1).numpy()
                
                # Reshape to 2D grid
                try:
                    pred_2d = pred.reshape(hres_shape)
                    target_2d = target.reshape(hres_shape)
                except ValueError:
                    continue
                
                # Create figure
                fig = self._create_comparison_figure(
                    target_2d, 
                    pred_2d, 
                    geo_bounds=geo_bounds,
                    sample_idx=sample_idx,
                    epoch=trainer.current_epoch,
                )
                
                # Convert to image
                img = self._fig_to_image(fig)
                images.append(img)
                plt.close(fig)
                
        finally:
            # Restore original setting
            val_dataset.num_query_samples = original_num_query_samples
        
        # Log to WandB
        if images:
            import wandb
            wandb_logger.experiment.log({
                f"val/hres_predictions": [
                    wandb.Image(img, caption=f"Sample {i}") 
                    for i, img in enumerate(images)
                ],
                "epoch": trainer.current_epoch,
            })
    
    def _create_comparison_figure(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        geo_bounds: Optional[Dict[str, float]],
        sample_idx: int,
        epoch: int,
    ) -> plt.Figure:
        """Create a side-by-side comparison figure with geographic context."""
        
        # Compute common color scale
        vmin = min(target.min(), prediction.min())
        vmax = max(target.max(), prediction.max())
        
        # Compute error
        diff = prediction - target
        err_max = max(abs(diff.min()), abs(diff.max()))
        
        # Create figure with or without cartopy
        if self.use_cartopy and geo_bounds is not None:
            fig = self._create_cartopy_figure(
                target, prediction, diff,
                geo_bounds, vmin, vmax, err_max,
                sample_idx, epoch
            )
        else:
            fig = self._create_basic_figure(
                target, prediction, diff,
                geo_bounds, vmin, vmax, err_max,
                sample_idx, epoch
            )
        
        return fig
    
    def _create_cartopy_figure(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        diff: np.ndarray,
        geo_bounds: Dict[str, float],
        vmin: float, vmax: float, err_max: float,
        sample_idx: int,
        epoch: int,
    ) -> plt.Figure:
        """Create figure using cartopy for geographic projection."""
        # Extract bounds
        lon_min = geo_bounds["lon_min"]
        lon_max = geo_bounds["lon_max"]
        lat_min = geo_bounds["lat_min"]
        lat_max = geo_bounds["lat_max"]
        
        # Create extent for imshow [lon_min, lon_max, lat_min, lat_max]
        extent = [lon_min, lon_max, lat_min, lat_max]
        
        # Create figure with PlateCarree projection
        projection = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            1, 3, 
            figsize=self.figsize, 
            dpi=self.dpi,
            subplot_kw={'projection': projection}
        )
        
        # Plot data and add features to each axis
        data_list = [target, prediction, diff]
        titles = [
            f"Ground Truth ({self.variable_name})",
            f"Prediction ({self.variable_name})",
            "Error (Pred - GT)"
        ]
        cmaps = [self.colormap, self.colormap, self.error_colormap]
        vmins = [vmin, vmin, -err_max]
        vmaxs = [vmax, vmax, err_max]
        
        for ax, data, title, cmap, v_min, v_max in zip(
            axes, data_list, titles, cmaps, vmins, vmaxs
        ):
            # Set extent
            ax.set_extent(extent, crs=projection)
            
            # Plot data
            im = ax.imshow(
                data,
                cmap=cmap,
                vmin=v_min,
                vmax=v_max,
                extent=extent,
                origin="upper",
                transform=projection,
            )
            
            # Add land/ocean features
            ax.add_feature(cfeature.LAND, alpha=self.land_alpha, facecolor='darkgray')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', linestyle='--')
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_title(title)
            
            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation='horizontal')
        
        # Add metrics to suptitle
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))
        fig.suptitle(
            f"Epoch {epoch + 1} | Sample {sample_idx} | "
            f"Region: ({lat_min:.1f}°N to {lat_max:.1f}°N, {lon_min:.1f}°E to {lon_max:.1f}°E) | "
            f"RMSE: {rmse:.4f} | MAE: {mae:.4f}",
            fontsize=11,
            fontweight="bold",
        )
        
        plt.tight_layout()
        return fig
    
    def _create_basic_figure(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        diff: np.ndarray,
        geo_bounds: Optional[Dict[str, float]],
        vmin: float, vmax: float, err_max: float,
        sample_idx: int,
        epoch: int,
    ) -> plt.Figure:
        """Create basic figure without cartopy (fallback)."""
        fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)
        
        # Determine extent if geo_bounds available
        if geo_bounds:
            extent = [
                geo_bounds["lon_min"], geo_bounds["lon_max"],
                geo_bounds["lat_min"], geo_bounds["lat_max"]
            ]
        else:
            extent = None
        
        # Ground truth
        im0 = axes[0].imshow(
            target, 
            cmap=self.colormap, 
            vmin=vmin, 
            vmax=vmax,
            aspect="auto",
            origin="upper",
            extent=extent,
        )
        axes[0].set_title(f"Ground Truth ({self.variable_name})")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Prediction
        im1 = axes[1].imshow(
            prediction, 
            cmap=self.colormap, 
            vmin=vmin, 
            vmax=vmax,
            aspect="auto",
            origin="upper",
            extent=extent,
        )
        axes[1].set_title(f"Prediction ({self.variable_name})")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference (error)
        im2 = axes[2].imshow(
            diff, 
            cmap=self.error_colormap, 
            vmin=-err_max, 
            vmax=err_max,
            aspect="auto",
            origin="upper",
            extent=extent,
        )
        axes[2].set_title(f"Error (Pred - GT)")
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Add metrics to title
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))
        
        if geo_bounds:
            region_str = (
                f"Region: ({geo_bounds['lat_min']:.1f}°N to {geo_bounds['lat_max']:.1f}°N, "
                f"{geo_bounds['lon_min']:.1f}°E to {geo_bounds['lon_max']:.1f}°E) | "
            )
        else:
            region_str = ""
        
        fig.suptitle(
            f"Epoch {epoch + 1} | Sample {sample_idx} | {region_str}"
            f"RMSE: {rmse:.4f} | MAE: {mae:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        
        plt.tight_layout()
        return fig
    
    def _fig_to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert matplotlib figure to PIL Image."""
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return Image.open(buf)
    
    def _get_wandb_logger(self, trainer: L.Trainer) -> Optional[WandbLogger]:
        """Get WandB logger from trainer."""
        if trainer.logger is None:
            return None
        
        if isinstance(trainer.logger, WandbLogger):
            return trainer.logger
        
        # Check if it's a list of loggers
        if hasattr(trainer.logger, "_loggers"):
            for logger in trainer.logger._loggers:
                if isinstance(logger, WandbLogger):
                    return logger
        
        return None
    
    def _get_hres_shape(self, trainer: L.Trainer) -> Optional[Tuple[int, int]]:
        """Get HRES grid shape from datamodule."""
        if trainer.datamodule is None:
            return None
        
        if hasattr(trainer.datamodule, "hres_shape"):
            return trainer.datamodule.hres_shape
        
        return None
    
    def _get_geo_bounds(self, trainer: L.Trainer) -> Optional[Dict[str, float]]:
        """Get geographic bounds from datamodule."""
        if trainer.datamodule is None:
            return None
        
        if hasattr(trainer.datamodule, "geo_bounds"):
            return trainer.datamodule.geo_bounds
        
        return None
