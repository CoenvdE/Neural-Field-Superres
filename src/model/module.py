"""
PyTorch Lightning Module for Neural Field Super-Resolution.

Wraps the NeuralFieldSuperRes model with training/validation logic,
optimizer configuration, and metric logging.

Terminology:
    - query_pos: [B, Q, coord_dim] positions to predict at (input)
    - query_auxiliary_features: [B, Q, num_aux] optional auxiliary features (z, lsm, slt)
    - query_fields: [B, Q, num_output] target field values (ground truth, for loss only)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any, Literal
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import NeuralFieldSuperRes


class NeuralFieldSuperResModule(pl.LightningModule):
    """
    Lightning module for Neural Field Super-Resolution training.
    
    Reconstructs high-resolution HRES fields from ERA5 latent representations.
    Uses decoder-only architecture (latents provided directly from dataset).
    """
    
    def __init__(
        self,
        # Model architecture (decoder-only)
        num_output_features: int = 1,
        coord_dim: int = 2,
        num_hidden_features: int = 512,
        num_heads: int = 8,
        
        # Processor options
        use_self_attention: bool = False,
        num_processor_layers: int = 1,
        
        # Decoder options
        decoder_type: str = "knn_cross",
        num_decoder_layers: int = 2,
        
        # Position encoding
        use_rope: bool = False,
        
        # Auxiliary features (optional, e.g., z/lsm/slt)
        num_auxiliary_features: int = 0,  # 0 = disabled, 3 = z/lsm/slt
        
        # Optimizer settings
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        
        # Scheduler settings
        use_scheduler: bool = True,
        scheduler_t_max: Optional[int] = None,  # Defaults to max_epochs
        scheduler_eta_min: float = 1e-6,
        
        # Loss settings
        loss_type: Literal["mse", "mae", "huber"] = "mse",
        
        # Encoder-related args (commented out / not used in decoder-only mode)
        # num_input_features: int = 256,
        # num_latents: int = 64,
        # coordinate_system: str = "latlon",
        # use_encoder: bool = False,
        # num_encoder_layers: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build model (decoder-only)
        self.model = NeuralFieldSuperRes(
            num_output_features=num_output_features,
            coord_dim=coord_dim,
            num_hidden_features=num_hidden_features,
            num_heads=num_heads,
            use_self_attention=use_self_attention,
            num_processor_layers=num_processor_layers,
            decoder_type=decoder_type,
            num_decoder_layers=num_decoder_layers,
            use_rope=use_rope,
            num_auxiliary_features=num_auxiliary_features,
        )
        
        # Loss function
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Store for visualization callback
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        
    def forward(
        self,
        query_pos: torch.Tensor,
        latents: torch.Tensor,
        latent_pos: torch.Tensor,
        query_auxiliary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(
            query_pos=query_pos,
            latents=latents,
            latent_pos=latent_pos,
            query_auxiliary_features=query_auxiliary_features,
        )
    
    def _shared_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int,
        stage: str
    ) -> Dict[str, torch.Tensor]:
        """Shared logic for training and validation steps."""
        latents = batch["latents"]               # [B, Z, D_latent]
        latent_pos = batch["latent_pos"]         # [B, Z, coord_dim]
        query_pos = batch["query_pos"]           # [B, Q, coord_dim]
        query_fields = batch["query_fields"]     # [B, Q, num_vars] (targets)
        
        # Optional auxiliary features (None if not in batch)
        query_auxiliary_features = batch.get("query_auxiliary_features")  # [B, Q, num_aux] or None
        
        # Forward pass
        predictions = self.model(
            query_pos=query_pos,
            latents=latents,
            latent_pos=latent_pos,
            query_auxiliary_features=query_auxiliary_features,
        )
        
        # Compute loss against ground truth
        loss = self.loss_fn(predictions, query_fields)
        
        # Compute additional metrics
        with torch.no_grad():
            mse = nn.functional.mse_loss(predictions, query_fields)
            mae = nn.functional.l1_loss(predictions, query_fields)
            rmse = torch.sqrt(mse)
        
        return {
            "loss": loss,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "predictions": predictions,
            "targets": query_fields,
            "query_pos": query_pos,
        }
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        outputs = self._shared_step(batch, batch_idx, "train")
        
        # Log metrics
        self.log("train/loss", outputs["loss"], prog_bar=True, sync_dist=True)
        self.log("train/mse", outputs["mse"], sync_dist=True)
        self.log("train/mae", outputs["mae"], sync_dist=True)
        self.log("train/rmse", outputs["rmse"], sync_dist=True)
        
        return outputs["loss"]
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        outputs = self._shared_step(batch, batch_idx, "val")
        
        # Log metrics
        self.log("val/loss", outputs["loss"], prog_bar=True, sync_dist=True)
        self.log("val/mse", outputs["mse"], sync_dist=True)
        self.log("val/mae", outputs["mae"], sync_dist=True)
        self.log("val/rmse", outputs["rmse"], sync_dist=True)
        
        # Store first few batches for visualization
        if batch_idx < 4:
            self.validation_step_outputs.append({
                "predictions": outputs["predictions"].detach().cpu(),
                "targets": outputs["targets"].detach().cpu(),
                "query_pos": outputs["query_pos"].detach().cpu(),
            })
        
        return outputs
    
    def on_validation_epoch_end(self) -> None:
        """Clear validation outputs after epoch."""
        # Outputs are used by visualization callback, then cleared
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        config = {"optimizer": optimizer}
        
        if self.hparams.use_scheduler:
            # Use trainer's max_epochs if not specified
            t_max = self.hparams.scheduler_t_max
            if t_max is None:
                t_max = self.trainer.max_epochs if self.trainer else 100
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=self.hparams.scheduler_eta_min,
            )
            
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        
        return config
