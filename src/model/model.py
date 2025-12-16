import torch
import torch.nn as nn
from typing import Optional

from .layers import CrossAttention
from .pos_emb import CoordinateEncoder, PointwiseFFN


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
        aux_embed_dim: int = 64,  # Embedding dimension for auxiliary features
        pos_init_std: float = 0.02,  # For CrossAttention position encoding
        predict_variance: bool = False,  # If True, output both mean and log-variance
        k_nearest: int = 16,  # Number of nearest neighbors for cross-attention
        use_gridded_knn: bool = False,  # Use analytical KNN for regular grids
        roll_lon: bool = False,  # Longitude wraparound for global models
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
        self.aux_embed_dim = aux_embed_dim
        self.pos_init_std = pos_init_std
        self.predict_variance = predict_variance
        self.k_nearest = k_nearest
        self.use_gridded_knn = use_gridded_knn
        self.roll_lon = roll_lon
        
        # Positional encoding for query initialization
        # If auxiliary features exist, pos encoding uses (hidden - aux_embed_dim) dims
        # Otherwise, pos encoding uses full hidden dim
        if num_auxiliary_features > 0:
            pos_enc_dim = num_hidden_features - aux_embed_dim
            # Project auxiliary features to aux_embed_dim dimensions
            self.query_aux_proj = nn.Linear(num_auxiliary_features, aux_embed_dim)
        else:
            pos_enc_dim = num_hidden_features
            self.query_aux_proj = None
        
        self.query_pos_encoder = CoordinateEncoder(
            coord_dim=coord_dim,
            embed_dim=pos_enc_dim,
            coordinate_system="cartesian",
            learnable_coefficients=True,
            init_std=pos_init_std,
        )

        # Processor (Self-Attention on Latents)
        if use_self_attention:
            self.processor = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=num_hidden_features, nhead=num_heads, batch_first=True),
                num_layers=num_processor_layers
            )

        # Decoder: cross-attention + FFN blocks
        self.decoder_layers = nn.ModuleList()
        self.decoder_attn_norms = nn.ModuleList()  # LayerNorm after attention
        self.decoder_ffns = nn.ModuleList()         # Pointwise FFN
        self.decoder_ffn_norms = nn.ModuleList()    # LayerNorm after FFN
        
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
                    )
                )
                self.decoder_attn_norms.append(nn.LayerNorm(num_hidden_features))
                # FFN with 4x expansion (standard Transformer)
                self.decoder_ffns.append(
                    PointwiseFFN(num_hidden_features, num_hidden_features * 4, num_hidden_features)
                )
                self.decoder_ffn_norms.append(nn.LayerNorm(num_hidden_features))

        # Final projection to output dimension
        # If predicting variance, output is [mean, log_var] so double the channels
        output_dim = num_output_features * 2 if predict_variance else num_output_features
        self.final_proj = nn.Linear(num_hidden_features, output_dim)

        # OLD: Learned init query vector (commented out - now using pos encoding)
        # self.init_query_vector = nn.Parameter(torch.randn(1, num_hidden_features))

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

        # Build query input using positional encoding
        # OLD: query_hidden = self.init_query_vector.expand(query_pos.shape[0], query_pos.shape[1], -1)
        query_pos_enc = self.query_pos_encoder(query_pos)  # [B, Q, pos_enc_dim]
        
        # Concatenate with auxiliary features if present
        if query_auxiliary_features is not None:
            query_aux_proj = self.query_aux_proj(query_auxiliary_features)  # [B, Q, num_aux]
            query_hidden = torch.cat([query_pos_enc, query_aux_proj], dim=-1)  # [B, Q, hidden]
        else:
            query_hidden = query_pos_enc  # [B, Q, hidden]
        
        # Decoder: cross-attention + FFN blocks
        for i, layer in enumerate(self.decoder_layers):
            # Cross-attention with residual
            delta, _ = layer(
                query=query_hidden,
                query_pos=query_pos,
                context=latents,
                context_pos=latent_pos,
                context_grid_shape=latent_grid_shape,
            )
            query_hidden = self.decoder_attn_norms[i](query_hidden + delta)
            
            # FFN with residual
            ffn_out = self.decoder_ffns[i](query_hidden)
            query_hidden = self.decoder_ffn_norms[i](query_hidden + ffn_out)

        # Project to output dimension
        return self.final_proj(query_hidden)
