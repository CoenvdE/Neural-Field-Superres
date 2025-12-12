import torch
import torch.nn as nn
from typing import Optional

from .layers import CrossAttention


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
        use_processor: bool = True,
        use_rope: bool = False,
        # Auxiliary features (optional, e.g., z/lsm/slt)
        num_auxiliary_features: int = 0,  # 0 = disabled
        # Encoder-related args (commented out / not used)
        # num_input_features: int = 256,
        # num_latents: int = 64,
        # coordinate_system: str = 'cartesian',
        # embedding_freq_multiplier: tuple = (1.0, 1.0),
        # use_encoder: bool = False,
        # num_encoder_layers: int = 1,
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

        # Encoder-related attributes (commented out)
        # self.num_input_features = num_input_features
        # self.num_latents = num_latents
        # self.coordinate_system = coordinate_system
        # self.use_encoder = use_encoder
        # self.num_encoder_layers = num_encoder_layers

        # Initialize latents and positions (encoder-related, commented out)
        # self.latents = nn.Parameter(
        #     self._initialize_latents(num_latents, num_hidden_features))
        # self.latent_pos = nn.Parameter(
        #     self._initialize_grid_positions(num_latents, coord_dim, coordinate_system))

        # Input projection (encoder-related, commented out)
        # self.input_proj = nn.Linear(
        #     num_input_features, num_hidden_features) if num_input_features != num_hidden_features else nn.Identity()

        # Query projection: projects [query_pos, aux_features] to hidden dim
        # Input dim = coord_dim + num_auxiliary_features
        query_input_dim = coord_dim + num_auxiliary_features
        self.query_proj = nn.Linear(query_input_dim, num_hidden_features)

        # Encoder (commented out)
        # if use_encoder:
        #     self.encoder_layers = nn.ModuleList([
        #         CrossAttention(
        #             num_hidden_features,
        #             num_heads,
        #             coord_dim=coord_dim,
        #             coordinate_system=coordinate_system,
        #             use_rope=use_rope,
        #         )
        #         for _ in range(num_encoder_layers)
        #     ])

        # Processor (Self-Attention on Latents)
        if use_self_attention:
            self.processor = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=num_hidden_features, nhead=num_heads, batch_first=True),
                num_layers=num_processor_layers
            )

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_decoder_layers):
            if decoder_type == 'knn_cross':
                self.decoder_layers.append(
                    CrossAttention(
                        num_hidden_features,
                        num_heads,
                        # coord_dim=coord_dim,
                        # coordinate_system=coordinate_system,
                        # use_rope=use_rope,
                    )
                )
            else:
                raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # Final projection to output dimension
        self.final_proj = nn.Linear(num_hidden_features, num_output_features)

    # Encoder-related methods (commented out)
    # def _initialize_latents(self, num_latents: int, dim: int) -> torch.Tensor:
    #     """Initialize latents with random features (like gridded-tnp)."""
    #     return torch.randn(num_latents, dim)

    # def _initialize_grid_positions(self, num_points: int, dim: int, coordinate_system: str = 'cartesian') -> torch.Tensor:
    #     """Initialize positions uniformly distributed over a domain [-1, 1]."""
    #     if coordinate_system == 'cartesian':
    #         return torch.rand(num_points, dim) * 2 - 1
    #     elif coordinate_system == 'latlon':
    #         return torch.rand(num_points, dim) * 2 - 1
    #     elif coordinate_system == 'polar':
    #         return torch.rand(num_points, dim) * 2 - 1
    #     else:
    #         raise ValueError(f"Unknown coordinate_system: {coordinate_system}")

    def forward(
        self,
        query_pos: torch.Tensor,
        latents: torch.Tensor,
        latent_pos: torch.Tensor,
        query_auxiliary_features: Optional[torch.Tensor] = None,
        # Encoder-related args (commented out)
        # grid_features: Optional[torch.Tensor] = None,
        # grid_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (decoder-only mode).
        
        Args:
            query_pos: [B, Q, coord_dim] positions to predict at
            latents: [B, Z, D_latent] latent features from Aurora encoder
            latent_pos: [B, Z, coord_dim] latent positions
            query_auxiliary_features: [B, Q, num_aux] optional auxiliary features (z, lsm, slt)
        
        Returns:
            predictions: [B, Q, num_output_features] predicted field values
        """
        # Encoder logic (commented out)
        # if self.use_encoder:
        #     latents = self.latents.unsqueeze(0).expand(query_pos.shape[0], -1, -1)
        #     latent_pos = self.latent_pos.unsqueeze(0).expand(query_pos.shape[0], -1, -1)
        #     if grid_features is None or grid_pos is None:
        #         raise ValueError("Encoder requires grid_features and grid_pos inputs")
        #
        #     for layer in self.encoder_layers:
        #         delta_latents, _ = layer(
        #             query=latents,
        #             query_pos=latent_pos,
        #             context=grid_features,
        #             context_pos=grid_pos,
        #         )
        #         latents = latents + delta_latents
        # else:
        #     if latents is None or latent_pos is None:
        #         raise ValueError("Latents and latent_pos must be provided when encoder is disabled")
        #     latents = self.input_proj(latents)

        # Processor (self-attention on latents)
        if self.use_self_attention:
            latents, latent_pos = self.processor(latents, latent_pos)

        # Build query input: [query_pos] or [query_pos, auxiliary_features]
        if query_auxiliary_features is not None:
            query_input = torch.cat([query_pos, query_auxiliary_features], dim=-1)
        else:
            query_input = query_pos
        
        # Project to hidden dimension
        query_hidden = self.query_proj(query_input)
        
        # Decoder: cross-attention from query positions to latents
        for layer in self.decoder_layers:
            delta, _ = layer(
                query=query_hidden,
                query_pos=query_pos,
                context=latents,
                context_pos=latent_pos
            )
            query_hidden = query_hidden + delta

        # Project to output dimension
        return self.final_proj(query_hidden)
