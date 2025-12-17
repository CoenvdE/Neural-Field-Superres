"""
PyTorch Lightning Module for Neural Field Super-Resolution with Encoder.

Wraps the NeuralFieldSuperRes encoder+decoder model with training/validation logic,
optimizer configuration, and metric logging.

This variant encodes low-resolution features (Aurora predictions) into a latent space
before decoding to high-resolution outputs.

Terminology:
    - low_res_pos: [B, L, coord_dim] positions of low-resolution inputs
    - low_res_features: [B, L, num_input_features] low-resolution input features
    - query_pos: [B, Q, coord_dim] positions to predict at (input)
    - query_auxiliary_features: [B, Q, num_aux] optional auxiliary features (z, lsm, slt)
    - query_fields: [B, Q, num_output] target field values (ground truth, for loss only)
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Optional, List, Dict, Any, Literal
from torch.optim.lr_scheduler import CosineAnnealingLR
from .likelihoods import GaussianLikelihood, HeteroscedasticGaussianLikelihood
from .model_with_enc import NeuralFieldSuperRes


class NeuralFieldSuperResEncoderModule(L.LightningModule):
    """
    Lightning module for Neural Field Super-Resolution with Encoder.
    
    Encodes low-resolution Aurora predictions into learned latent representations,
    then decodes to high-resolution HRES fields.
    """
    
    def __init__(
        self,
        # Model architecture
        num_output_features: int = 1,
        num_input_features: int = 4,  # Number of Aurora prediction variables (2t, 10u, 10v, msl)
        coord_dim: int = 2,
        num_hidden_features: int = 512,
        num_heads: int = 8,
        
        # Encoder options
        num_encoder_layers: int = 2,
        encoder_type: str = "knn_cross",
        num_latents: int = 64,  # Number of learned latent positions
        
        # Processor options
        use_self_attention: bool = False,
        num_processor_layers: int = 1,
        
        # Decoder options
        decoder_type: str = "knn_cross",
        num_decoder_layers: int = 2,
        
        # Position encoding
        use_rope: bool = False,
        pos_init_std: float = 0.02,  # For CrossAttention position encoding
        
        # KNN cross-attention
        k_nearest: int = 16,  # Number of nearest neighbors for cross-attention
        use_gridded_knn: bool = False,  # Use analytical KNN for regular grids
        roll_lon: bool = False,  # Longitude wraparound for global models
        
        # Auxiliary features (optional, e.g., z/lsm/slt)
        num_auxiliary_features: int = 0,  # 0 = disabled, 3 = z/lsm/slt
        
        # Optimizer settings
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        
        # Scheduler settings
        use_scheduler: bool = True,
        scheduler_t_max: Optional[int] = None,  # Defaults to max_epochs
        scheduler_eta_min: float = 1e-6,
        init_std: float = 1,
        
        # Loss settings
        loss_type: Literal["mse", "gaussian_nll", "heteroscedastic_nll"] = "mse",
        
        # Gaussian NLL settings (only used when loss_type contains 'nll')
        gaussian_noise_init: float = 0.1,        # Initial noise for homoscedastic
        gaussian_min_noise: float = 1e-3,        # Min noise for heteroscedastic
        gaussian_train_noise: bool = True,       # Whether to learn noise parameter
        
        # torch.compile() settings (PyTorch 2.0+)
        compile_model: bool = False,             # Enable torch.compile for speedup
        compile_mode: str = "default",           # default, reduce-overhead, max-autotune
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build model (encoder + decoder)
        # Enable variance prediction for heteroscedastic loss
        predict_variance = loss_type == "heteroscedastic_nll"
        
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
            pos_init_std=pos_init_std,
            num_auxiliary_features=num_auxiliary_features,
            predict_variance=predict_variance,
            k_nearest=k_nearest,
            use_gridded_knn=use_gridded_knn,
            roll_lon=roll_lon,
            num_encoder_layers=num_encoder_layers,
            encoder_type=encoder_type,
            num_latents=num_latents,
        )
        
        # Input projection: low_res features -> hidden_dim
        self.input_proj = nn.Linear(num_input_features, num_hidden_features)
        
        # Loss function and likelihood
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
            self.likelihood = None
        elif loss_type == "gaussian_nll":
            # Homoscedastic: model predicts mean, variance is learned parameter
            self.likelihood = GaussianLikelihood(
                noise=gaussian_noise_init,
                train_noise=gaussian_train_noise
            )
            self.loss_fn = None  # Will use NLL directly
        elif loss_type == "heteroscedastic_nll":
            # Heteroscedastic: model predicts both mean and variance
            self.likelihood = HeteroscedasticGaussianLikelihood(
                min_noise=gaussian_min_noise
            )
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Compile model for faster execution (PyTorch 2.0+)
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                print(f"✓ Model compiled with mode='{compile_mode}'")
            except Exception as e:
                print(f"⚠ torch.compile failed, using eager mode: {e}")
        
    def forward(
        self,
        low_res_pos: torch.Tensor,
        low_res_features: torch.Tensor,
        query_pos: torch.Tensor,
        query_auxiliary_features: Optional[torch.Tensor] = None,
        low_res_grid_shape: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        # Project input features to hidden dimension
        low_res_features = self.input_proj(low_res_features)
        
        return self.model(
            low_res_pos=low_res_pos,
            low_res_features=low_res_features,
            query_pos=query_pos,
            query_auxiliary_features=query_auxiliary_features,
            latent_grid_shape=low_res_grid_shape,
        )
    
    @torch.no_grad()
    def chunked_forward(
        self,
        low_res_pos: torch.Tensor,
        low_res_features: torch.Tensor,
        query_pos: torch.Tensor,
        query_auxiliary_features: Optional[torch.Tensor] = None,
        low_res_grid_shape: Optional[torch.Tensor] = None,
        chunk_size: int = 8192,
    ) -> torch.Tensor:
        """
        Memory-efficient forward pass that processes queries in chunks.
        """
        B, Q, _ = query_pos.shape
        
        # Project input features once
        low_res_features = self.input_proj(low_res_features)
        
        # Process in chunks
        predictions_list = []
        for start_idx in range(0, Q, chunk_size):
            end_idx = min(start_idx + chunk_size, Q)
            
            chunk_pos = query_pos[:, start_idx:end_idx, :]
            
            chunk_aux = None
            if query_auxiliary_features is not None:
                chunk_aux = query_auxiliary_features[:, start_idx:end_idx, :]
            
            chunk_pred = self.model(
                low_res_pos=low_res_pos,
                low_res_features=low_res_features,
                query_pos=chunk_pos,
                query_auxiliary_features=chunk_aux,
                latent_grid_shape=low_res_grid_shape,
            )
            predictions_list.append(chunk_pred)
        
        return torch.cat(predictions_list, dim=1)
    
    def _shared_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int,
        stage: str
    ) -> Dict[str, torch.Tensor]:
        """Shared logic for training and validation steps."""
        low_res_features = batch["low_res_features"]   # [B, L, num_input]
        low_res_pos = batch["low_res_pos"]             # [B, L, coord_dim]
        query_pos = batch["query_pos"]                 # [B, Q, coord_dim]
        query_fields = batch["query_fields"]           # [B, Q, num_vars] (targets)
        
        # Optional auxiliary features (None if not in batch)
        query_auxiliary_features = batch.get("query_auxiliary_features")  # [B, Q, num_aux] or None
        
        # Grid metadata for analytical KNN (use first sample, same for whole batch)
        low_res_grid_shape = batch.get("low_res_grid_shape")  # [B, 2] or None
        if low_res_grid_shape is not None:
            low_res_grid_shape = low_res_grid_shape[0]  # [2] - same for all samples
        
        # Project input features to hidden dimension
        low_res_features = self.input_proj(low_res_features)
        
        # Forward pass
        predictions = self.model(
            low_res_pos=low_res_pos,
            low_res_features=low_res_features,
            query_pos=query_pos,
            query_auxiliary_features=query_auxiliary_features,
            latent_grid_shape=low_res_grid_shape,
        )
        
        # Compute loss against ground truth
        if self.likelihood is not None:
            # Gaussian NLL loss
            pred_dist = self.likelihood(predictions)
            nll = -pred_dist.log_prob(query_fields).sum() / query_fields.numel()
            loss = nll
            pred_mean = pred_dist.mean
        else:
            # Deterministic loss (MSE, etc.)
            loss = self.loss_fn(predictions, query_fields)
            pred_mean = predictions
        
        # Compute additional metrics using mean predictions
        with torch.no_grad():
            rmse = torch.sqrt(nn.functional.mse_loss(pred_mean, query_fields))

        # For validation: on_epoch=True ensures metrics are aggregated
        on_epoch = (stage == "val")
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True, on_epoch=on_epoch)
        self.log(f"{stage}/rmse", rmse, sync_dist=True, on_epoch=on_epoch)
        
        return {"loss": loss}
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        outputs = self._shared_step(batch, batch_idx, "train")
        return outputs["loss"]
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        outputs = self._shared_step(batch, batch_idx, "val")
        return outputs["loss"]
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        config = {"optimizer": optimizer}
        
        if self.hparams.get('use_scheduler', False):
            t_max = self.hparams.get('scheduler_t_max', None)
            if t_max is None:
                t_max = self.trainer.max_epochs if self.trainer else 100
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=self.hparams.get('scheduler_eta_min', 1e-6),
            )
            
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        
        return config
