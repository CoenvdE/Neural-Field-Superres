import torch
import torch.nn as nn
from typing import Optional

from .layers import CrossAttention
from .pos_emb import CoordinateEncoder


class NeuralFieldSuperRes(nn.Module):
    """Neural Field Super-Resolution model (decoder-only, no encoder).
    
    Terminology:
        - query_pos: [B, Q, coord_dim] positions to predict at (input)
        - query_auxiliary_features: [B, Q, num_aux] optional auxiliary features (z, lsm, slt)
        - query_fields: [B, Q, num_output] target values (NOT used as input, only for loss)
    """
    
    def __init__(
        self,
        num_output_features: int,
        coord_dim: int = 2,
        num_hidden_features: int = 256,
        num_heads: int = 8,
        use_self_attention: bool = False,
        num_processor_layers: int = 1,
        decoder_type: str = 'knn_cross',
        num_decoder_layers: int = 1,
        use_processor: bool = False,
        use_rope: bool = False, 
        num_auxiliary_features: int = 0,  # 0 = disabled
        pos_init_std: float = 0.02,  # For CrossAttention position encoding
        predict_variance: bool = False,  # If True, output both mean and log-variance
        k_nearest: int = 16,  # Number of nearest neighbors for cross-attention
        use_gridded_knn: bool = False,  # Use analytical KNN for regular grids
        roll_lon: bool = False,  # Longitude wraparound for global models
        lat_ascending: bool = False,  # Set False for ERA5/Aurora (descending lat order)
    ):
        super().__init__()
        self.num_output_features = num_output_features
        self.coord_dim = coord_dim
        self.num_hidden_features = num_hidden_features
        self.num_heads = num_heads
        self.num_processor_layers = num_processor_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_type = decoder_type
        self.use_self_attention = use_self_attention
        self.use_rope = use_rope
        self.use_processor = use_processor
        self.num_auxiliary_features = num_auxiliary_features
        self.pos_init_std = pos_init_std
        self.predict_variance = predict_variance
        self.k_nearest = k_nearest
        self.use_gridded_knn = use_gridded_knn
        self.roll_lon = roll_lon
        self.lat_ascending = lat_ascending
        
        # If auxiliary features exist, project them to hidden_dim
        if num_auxiliary_features > 0:
            # Input is pos_encoding (hidden_dim) + aux_features (num_aux)
            self.query_proj = nn.Linear(
                num_auxiliary_features, num_hidden_features
            )
        else:
            self.query_proj = None  # No projection needed

        # Processor (Self-Attention on Latents)
        if use_self_attention:
            self.processor = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=num_hidden_features, nhead=num_heads, batch_first=True),
                num_layers=num_processor_layers
            )

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()  # LayerNorm for residual connections
        for _ in range(num_decoder_layers):
            if decoder_type == 'knn_cross':
                self.decoder_layers.append(
                    CrossAttention(
                        num_hidden_features,
                        num_heads,
                        coord_dim=coord_dim,
                        pos_init_std=self.pos_init_std,
                        positional_information_type="rope" if use_rope else "rff",
                        k_nearest=k_nearest,
                        use_gridded_knn=use_gridded_knn,
                        roll_lon=roll_lon,
                        lat_ascending=lat_ascending,
                    )
                )
                self.decoder_norms.append(nn.LayerNorm(num_hidden_features))
        
        # Shared position encoder for pre-computing position embeddings
        # This is much more efficient than computing in each layer
        self.shared_pos_encoder = CoordinateEncoder(
            coord_dim=coord_dim,
            embed_dim=num_hidden_features,
            coordinate_system="cartesian",
            learnable_coefficients=True,
            init_std=pos_init_std,
        )

        # Final projection to output dimension
        # If predicting variance, output is [mean, log_var] so double the channels
        output_dim = num_output_features * 2 if predict_variance else num_output_features
        self.final_proj = nn.Linear(num_hidden_features, output_dim)

        # Scale init_query_vector to prevent dominating positional encodings
        self.init_query_vector = nn.Parameter(torch.randn(1, num_hidden_features) * 0.02)

    def forward(
        self,
        query_pos: torch.Tensor,
        latents: torch.Tensor,
        latent_pos: torch.Tensor,
        query_auxiliary_features: Optional[torch.Tensor] = None,
        latent_grid_shape: Optional[torch.Tensor] = None,  # [2] for analytical KNN
    ) -> torch.Tensor:
        """
        Forward pass (decoder-only mode).
        
        Args:
            query_pos: [B, Q, coord_dim] positions to predict at
            latents: [B, Z, D_latent] latent features from Aurora encoder
            latent_pos: [B, Z, coord_dim] latent positions
            query_auxiliary_features: [B, Q, num_aux] optional auxiliary features (z, lsm, slt)
            latent_grid_shape: [2] tensor (num_lat, num_lon) for analytical KNN
        
        Returns:
            predictions: [B, Q, num_output_features] predicted field values
        """

        # Processor (self-attention on latents)
        if self.use_self_attention:
            latents, latent_pos = self.processor(latents, latent_pos)

        # Build query input
        query_hidden = self.init_query_vector.expand(query_pos.shape[0], query_pos.shape[1], -1) #NOTE: [B, Q, D]
        
        # Step 2: Concatenate with auxiliary features if present
        if query_auxiliary_features is not None:
            query_projected_auxiliary_features = self.query_proj(query_auxiliary_features)
            query_hidden = query_hidden + query_projected_auxiliary_features
        
        # Pre-compute position encodings ONCE (shared across all decoder layers)
        # This is a major optimization - previously computed per layer!
        query_pos_enc = self.shared_pos_encoder(query_pos)    # [B, Q, D]
        context_pos_enc = self.shared_pos_encoder(latent_pos)  # [B, Z, D]
        
        # Decoder: cross-attention from query positions to latents
        for i, layer in enumerate(self.decoder_layers):
            delta, _ = layer(
                query=query_hidden,
                query_pos=query_pos,
                query_pos_enc=query_pos_enc,        # Pass pre-computed encoding
                context=latents,
                context_pos=latent_pos,
                context_pos_enc=context_pos_enc,    # Pass pre-computed encoding
                context_grid_shape=latent_grid_shape,
            )
            query_hidden = self.decoder_norms[i](query_hidden + delta)

        # Project to output dimension
        return self.final_proj(query_hidden)
