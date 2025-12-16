"""
HRES Visualization Callback for training monitoring.

Generates side-by-side visualizations of ground truth and predicted HRES fields
during training, using cartopy for geographic projection with land mask overlay.
Supports uncertainty visualization when using likelihood-based loss functions.
"""

import torch
import numpy as np
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
import wandb

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.model.likelihoods import GaussianLikelihood, HeteroscedasticGaussianLikelihood

class HRESVisualizationCallback(L.Callback):
    """
    Callback to visualize HRES predictions during validation.
    
    Visualizes 1 sample with all variables in the dataset, showing:
    - Ground truth HRES field for each variable
    - Predicted HRES field for each variable
    
    Always uses cartopy for geographic projection with land mask overlay.
    Static/auxiliary features are visualized once on the first validation epoch.
    Logs all visualizations to WandB as images.
    """
    
    def __init__(
        self,
        log_every_n_epochs: int = 5,
        chunk_size: int = 8192,
    ):
        """
        Args:
            log_every_n_epochs: Frequency of visualization logging.
            chunk_size: Chunk size for memory-efficient inference on full grid.
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.chunk_size = chunk_size
        self.static_features_plotted = False  # Track if static features have been plotted
        
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
            raise ValueError("WandB logger is None")
        
        # Get datamodule for full-grid access
        if trainer.datamodule is None:
            raise ValueError("Trainer datamodule is None")
        
        # Get HRES grid info
        hres_shape = self._get_hres_shape(trainer)
        geo_bounds = self._get_geo_bounds(trainer)
        
        if hres_shape is None:
            raise ValueError("HRES shape is None")
        
        if geo_bounds is None:
            raise ValueError("Geographic bounds are None - dataset not properly initialized")
        
        # Get configured region bounds for title display (if available)
        configured_region = self._get_configured_region(trainer)
        
        # Get the validation dataset for full-grid samples
        val_dataset = trainer.datamodule.val_dataset
        if val_dataset is None:
            raise ValueError("Validation dataset is None")
        
        # Store original num_query_samples and temporarily disable subsampling
        original_num_query_samples = val_dataset.num_query_samples
        val_dataset.num_query_samples = None  # Use full grid
        device = pl_module.device

        try:
            # Only visualize 1 sample (first sample)
            sample_idx = 0
            sample = val_dataset[sample_idx]
            
            # Get timestamp for this sample (if available)
            timestamp_str = None
            if hasattr(val_dataset, 'get_timestamp'):
                timestamp_str = val_dataset.get_timestamp(sample_idx)
            
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
            # Lightning handles precision (bf16-mixed) automatically based on trainer config
            predictions = pl_module.chunked_forward(
                query_pos=query_pos,
                latents=latents,
                latent_pos=latent_pos,
                query_auxiliary_features=query_aux,
                chunk_size=self.chunk_size,
            )
            
            # Extract uncertainty if using likelihood functions
            uncertainty = None
            if pl_module.likelihood is not None:
                if isinstance(pl_module.likelihood, HeteroscedasticGaussianLikelihood):
                    # Extract variance from predictions (heteroscedastic)
                    pred_dist = pl_module.likelihood(predictions)
                    uncertainty = pred_dist.stddev  # [B, Q, V]
                elif isinstance(pl_module.likelihood, GaussianLikelihood):
                    # Uniform noise (homoscedastic)
                    uncertainty = torch.full_like(predictions, pl_module.likelihood.noise.item())
                
                # Extract mean predictions from distribution
                pred_dist = pl_module.likelihood(predictions)
                predictions = pred_dist.mean
            
            # Automatically determine all variables from query_fields shape
            num_vars = query_fields.shape[-1]
            
            # Get variable names from datamodule if available
            if hasattr(trainer.datamodule, 'target_variables'):
                variable_names = trainer.datamodule.target_variables
            else:
                # Fallback to generic names
                variable_names = [f"var_{i}" for i in range(num_vars)]
            
            var_data = []  # List of (target_2d, pred_2d, var_name) tuples
            uncertainty_data = []  # List of uncertainty_2d arrays (if available)
            
            for var_idx in range(num_vars):
                var_name = variable_names[var_idx] if var_idx < len(variable_names) else f"var_{var_idx}"
                
                # Move to CPU for visualization
                pred = predictions[0, :, var_idx].cpu().numpy()    # [Q]
                target = query_fields[0, :, var_idx].numpy()       # [Q]
                
                # NOTE: Visualization is now in NORMALIZED space (not denormalized)
                # Denormalize predictions and targets back to original scale
                # if hasattr(trainer.datamodule, 'denormalize_targets'):
                #     pred = trainer.datamodule.denormalize_targets(
                #         torch.from_numpy(pred).unsqueeze(-1), var_idx
                #     ).squeeze(-1).numpy()
                #     target = trainer.datamodule.denormalize_targets(
                #         torch.from_numpy(target).unsqueeze(-1), var_idx
                #     ).squeeze(-1).numpy()
                # else:
                #     raise ValueError("Denormalize targets not implemented for this datamodule")
                
                # Process uncertainty if available (also in normalized space)
                if uncertainty is not None:
                    unc = uncertainty[0, :, var_idx].cpu().numpy()  # [Q]
                    # Denormalize uncertainty: multiply by variable std
                    # if hasattr(trainer.datamodule, 'target_statistics'):
                    #     var_std = trainer.datamodule.target_statistics[var_idx]['std']
                    #     unc = unc * var_std
                    uncertainty_data.append(unc)
                
                # Reshape to 2D grid
                try:
                    pred_2d = pred.reshape(hres_shape)
                    target_2d = target.reshape(hres_shape)
                    var_data.append((target_2d, pred_2d, var_name))
                except ValueError:
                    raise ValueError("Failed to reshape predictions and targets to 2D grid in visualization callback")
            
            # Reshape uncertainty to 2D
            if uncertainty_data:
                uncertainty_data_2d = [unc.reshape(hres_shape) for unc in uncertainty_data]
            else:
                uncertainty_data_2d = None
            
            # Create multi-variable figure
            if var_data:
                fig = self._create_multi_var_cartopy_figure(
                    var_data,
                    geo_bounds=geo_bounds,
                    sample_idx=sample_idx,
                    epoch=trainer.current_epoch,
                    uncertainty_data=uncertainty_data_2d,
                    title_region_bounds=configured_region,
                    timestamp_str=timestamp_str,
                )
                
                # Log to WandB via PyTorch Lightning logger
                pl_module.logger.log_image(
                    key="plots/val_hres_predictions",
                    images=[fig]
                )
                plt.close(fig)
            
            # Plot auxiliary features only once (on first call)
            if not self.static_features_plotted and hasattr(val_dataset, 'static_features') and val_dataset.static_features is not None:
                aux_fig = self._create_auxiliary_figure(
                    val_dataset.static_features,
                    val_dataset.static_variables if hasattr(val_dataset, 'static_variables') else ['z', 'lsm', 'slt'],
                    hres_shape,
                    geo_bounds,
                    trainer.current_epoch,
                )
                if aux_fig is not None:
                    # Log to WandB via PyTorch Lightning logger
                    pl_module.logger.log_image(
                        key="plots/auxiliary_features",
                        images=[aux_fig]
                    )
                    plt.close(aux_fig)
                    self.static_features_plotted = True  # Mark as plotted
                else:
                    raise ValueError("Static features not found in validation dataset")
                
        finally:
            # Restore original setting
            val_dataset.num_query_samples = original_num_query_samples
    
    def _create_multi_var_cartopy_figure(
        self,
        var_data: List[Tuple[np.ndarray, np.ndarray, str]],
        geo_bounds: Dict[str, float],
        sample_idx: int,
        epoch: int,
        uncertainty_data: Optional[List[np.ndarray]] = None,
        title_region_bounds: Optional[Dict[str, float]] = None,
        timestamp_str: Optional[str] = None,
    ) -> plt.Figure:
        """Create multi-variable figure using cartopy.
        
        Layout: Prediction | Ground Truth | Error (| Uncertainty if likelihood)
        When uncertainty_data is provided, creates 4 columns.
        Otherwise creates 3 columns: Pred, GT, Error.
        
        Args:
            title_region_bounds: Optional configured region bounds for title display.
                                 Falls back to geo_bounds if not provided.
        """
        num_vars = len(var_data)
        
        # Extract bounds
        lon_min = geo_bounds["lon_min"]
        lon_max = geo_bounds["lon_max"]
        lat_min = geo_bounds["lat_min"]
        lat_max = geo_bounds["lat_max"]
        extent = [lon_min, lon_max, lat_min, lat_max]
        
        # Determine number of columns (3 for MSE with error, 4 for likelihood with uncertainty)
        num_cols = 4 if uncertainty_data is not None else 3
        
        # Create figure
        projection = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            num_vars, num_cols,
            figsize=(6 * num_cols, 5 * num_vars),
            dpi=200,  # High DPI for sharp images
            subplot_kw={'projection': projection}
        )
        
        # Handle single variable case (axes won't be 2D)
        if num_vars == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each variable
        for row_idx, (target, pred, var_name) in enumerate(var_data):
            # Compute common color scale for this variable
            vmin = min(target.min(), pred.min())
            vmax = max(target.max(), pred.max())
            
            # Compute metrics for this variable
            diff = pred - target
            rmse = np.sqrt(np.mean(diff ** 2))
            mae = np.mean(np.abs(diff))
            
            # Column 0: Prediction (left)
            ax_pred = axes[row_idx, 0]
            ax_pred.set_extent(extent, crs=projection)
            im_pred = ax_pred.imshow(
                pred,
                cmap="RdYlBu_r",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="upper",
                transform=projection,
            )
            ax_pred.add_feature(cfeature.LAND, alpha=0.3, facecolor='darkgray')
            ax_pred.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            ax_pred.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', linestyle='--')
            gl_pred = ax_pred.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl_pred.top_labels = False
            gl_pred.right_labels = False
            ax_pred.set_title(f"Prediction ({var_name})")
            plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04, orientation='horizontal')
            
            # Column 1: Ground Truth (right)
            ax_gt = axes[row_idx, 1]
            ax_gt.set_extent(extent, crs=projection)
            im_gt = ax_gt.imshow(
                target,
                cmap="RdYlBu_r",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="upper",
                transform=projection,
            )
            ax_gt.add_feature(cfeature.LAND, alpha=0.3, facecolor='darkgray')
            ax_gt.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            ax_gt.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', linestyle='--')
            gl_gt = ax_gt.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl_gt.top_labels = False
            gl_gt.right_labels = False
            ax_gt.set_title(f"Ground Truth ({var_name})")
            plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04, orientation='horizontal')
            
            # Column 2: Error (always shown)
            ax_err = axes[row_idx, 2]
            ax_err.set_extent(extent, crs=projection)
            error_max = np.abs(diff).max()
            im_err = ax_err.imshow(
                diff,
                cmap="RdBu_r",  # Diverging: blue=negative error, red=positive error
                vmin=-error_max,
                vmax=error_max,
                extent=extent,
                origin="upper",
                transform=projection,
            )
            ax_err.add_feature(cfeature.LAND, alpha=0.3, facecolor='darkgray')
            ax_err.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            ax_err.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', linestyle='--')
            gl_err = ax_err.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl_err.top_labels = False
            gl_err.right_labels = False
            ax_err.set_title(f"Error ({var_name})\nRMSE={rmse:.4f}, MAE={mae:.4f}")
            plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04, orientation='horizontal')
            
            
            # Column 3: Uncertainty (only when using likelihood)
            if uncertainty_data is not None:
                ax_unc = axes[row_idx, 3]
                ax_unc.set_extent(extent, crs=projection)
                
                unc = uncertainty_data[row_idx]
                unc_mean = unc.mean()
                unc_std = unc.std()
                
                im_unc = ax_unc.imshow(
                    unc,
                    cmap="viridis",  # Perceptually uniform colormap for uncertainty
                    vmin=0,
                    vmax=unc.max(),
                    extent=extent,
                    origin="upper",
                    transform=projection,
                )
                ax_unc.add_feature(cfeature.LAND, alpha=0.3, facecolor='darkgray')
                ax_unc.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
                ax_unc.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', linestyle='--')
                gl_unc = ax_unc.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
                gl_unc.top_labels = False
                gl_unc.right_labels = False
                ax_unc.set_title(f"Uncertainty ({var_name})\nmean={unc_mean:.4f}, std={unc_std:.4f}")
                plt.colorbar(im_unc, ax=ax_unc, fraction=0.046, pad=0.04, orientation='horizontal')
        # Add overall title (use configured region bounds if available for cleaner display)
        title_bounds = title_region_bounds if title_region_bounds is not None else geo_bounds
        title_lat_min = title_bounds.get("lat_min", lat_min)
        title_lat_max = title_bounds.get("lat_max", lat_max)
        title_lon_min = title_bounds.get("lon_min", lon_min)
        title_lon_max = title_bounds.get("lon_max", lon_max)
        
        # Build title with optional timestamp
        title_parts = [f"Epoch {epoch + 1}"]
        if timestamp_str is not None:
            title_parts.append(f"Time: {timestamp_str}")
        title_parts.append(f"Sample {sample_idx}")
        title_parts.append(f"Region: ({title_lat_min:.1f}°N to {title_lat_max:.1f}°N, {title_lon_min:.1f}°E to {title_lon_max:.1f}°E)")
        
        fig.suptitle(
            " | ".join(title_parts),
            fontsize=12,
            fontweight="bold",
            y=0.995
        )
        
        plt.tight_layout()
        return fig
    
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
    
    def _get_configured_region(self, trainer: L.Trainer) -> Optional[Dict[str, float]]:
        """Get user-configured region bounds from dataset (if available)."""
        if trainer.datamodule is None:
            return None
        
        # Try to get from validation dataset first, then train dataset
        for dataset_attr in ['val_dataset', 'train_dataset']:
            dataset = getattr(trainer.datamodule, dataset_attr, None)
            if dataset is not None and hasattr(dataset, 'region_bounds'):
                return dataset.region_bounds
        
        return None
    
    def _create_auxiliary_figure(
        self,
        static_features: np.ndarray,
        static_variables: List[str],
        hres_shape: Tuple[int, int],
        geo_bounds: Optional[Dict[str, float]],
        epoch: int,
    ) -> Optional[plt.Figure]:
        """Create a figure showing auxiliary/static features."""
        num_vars = static_features.shape[-1]
        if num_vars == 0:
            return None
        
        # Variable-specific colormaps and labels
        var_config = {
            'z': {'cmap': 'terrain', 'label': 'Geopotential (z)'},
            'lsm': {'cmap': 'Blues_r', 'label': 'Land-Sea Mask'},
            'slt': {'cmap': 'tab10', 'label': 'Soil Type'},
        }
        
        fig_width = 6 * num_vars
        fig_height = 5
        
        # Always use cartopy
        projection = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            1, num_vars,
            figsize=(fig_width, fig_height),
            dpi=200,  # High DPI for sharp images
            subplot_kw={'projection': projection}
        )
        
        if num_vars == 1:
            axes = [axes]
        
        # Extent for imshow
        if geo_bounds:
            extent = [
                geo_bounds["lon_min"], geo_bounds["lon_max"],
                geo_bounds["lat_min"], geo_bounds["lat_max"]
            ]
        else:
            extent = None
        
        for idx, ax in enumerate(axes):
            var_name = static_variables[idx] if idx < len(static_variables) else f"var_{idx}"
            config = var_config.get(var_name, {'cmap': 'viridis', 'label': var_name})
            
            # Reshape to 2D
            try:
                data_2d = static_features[:, idx].reshape(hres_shape)
            except ValueError:
                continue
            
            # Use cartopy for all plots
            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            im = ax.imshow(
                data_2d,
                cmap=config['cmap'],
                extent=extent,
                origin="upper",
                transform=ccrs.PlateCarree(),
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', linestyle='--')
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_title(config['label'])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation='horizontal')
        
        if geo_bounds:
            region_str = (
                f"Region: ({geo_bounds['lat_min']:.1f}°N to {geo_bounds['lat_max']:.1f}°N, "
                f"{geo_bounds['lon_min']:.1f}°E to {geo_bounds['lon_max']:.1f}°E)"
            )
        else:
            region_str = ""
        
        fig.suptitle(
            f"Auxiliary Features | Epoch {epoch + 1} | {region_str}",
            fontsize=12,
            fontweight="bold",
        )
        
        plt.tight_layout()
        return fig
