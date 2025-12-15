#!/usr/bin/env python
"""
Neural Field Super-Resolution Training Script.

Usage:
    # With Lightning CLI (recommended):
    python -m src.train fit --config config/default.yaml
    
    # With experiment config:
    python -m src.train fit --config config/experiment_europe.yaml
    
    # Override specific parameters:
    python -m src.train fit --config config/default.yaml \
        --data.latent_zarr_path /path/to/latents.zarr \
        --data.hres_zarr_path /path/to/hres.zarr \
        --trainer.max_epochs 50
    
    # Quick test run:
    python -m src.train fit --config config/default.yaml --trainer.fast_dev_run true
"""

import os
import sys

# Suppress pkg_resources deprecation warnings from Lightning (must be before lightning import)
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lightning.pytorch.cli import LightningCLI

from src.model import NeuralFieldSuperResModule
from src.data import NeuralFieldDataModule

# Optimize for A100 Tensor Cores
torch.set_float32_matmul_precision('high')


class NeuralFieldCLI(LightningCLI):
    """Custom CLI with project-specific defaults."""
    
    def add_arguments_to_parser(self, parser):
        """Add custom arguments."""
        # Note: num_output_features and coord_dim are set directly in config
        # since they depend on the problem setup, not derived from data
        pass


def cli_main():
    """Main entry point for Lightning CLI."""
    cli = NeuralFieldCLI(
        NeuralFieldSuperResModule,
        NeuralFieldDataModule,
        save_config_kwargs={"overwrite": True},
        subclass_mode_model=False,
        subclass_mode_data=False,
    )


def manual_train():
    """
    Alternative manual training setup (without CLI).
    
    Useful for debugging or notebook usage.
    """
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import (
        ModelCheckpoint, 
        EarlyStopping, 
        LearningRateMonitor
    )
    from src.callbacks import HRESVisualizationCallback
    
    # Paths - cluster data directory
    LATENT_ZARR = "/projects/prjs1858/latents_europe_2018_2020.zarr"
    HRES_ZARR = "/projects/prjs1858/hres_europe_2018_2020.zarr"
    
    # Create data module
    datamodule = NeuralFieldDataModule(
        latent_zarr_path=LATENT_ZARR,
        hres_zarr_path=HRES_ZARR,
        variables=["2t"],
        mode="surface",
        batch_size=4,
        val_months=3,
        num_workers=4,
    )
    
    # Create model
    model = NeuralFieldSuperResModule(
        num_output_features=1,
        num_input_features=256,
        num_query_features=1,
        num_hidden_features=256,
        num_heads=8,
        coord_dim=2,
        coordinate_system="latlon",
        num_decoder_layers=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        use_scheduler=True,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="epoch={epoch:03d}-val_loss={val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=15,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        HRESVisualizationCallback(
            log_every_n_epochs=5,
            num_samples=4,
            variable_name="2t",
        ),
    ]
    
    # Logger
    logger = WandbLogger(
        project="neural-field-superres",
        save_dir="logs/",
    )
    
    # Trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        max_epochs=100,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    # Use Lightning CLI by default
    cli_main()
    
    # Uncomment for manual training:
    # manual_train()

# python -m src.train fit --config config/default.yaml