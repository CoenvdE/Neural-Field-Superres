import torch
import torch.nn as nn
from typing import Optional

from .layers import CrossAttention


class NeuralFieldSuperResWithEncoder(nn.Module):
    """Neural Field Super-Resolution model with optional encoder."""
    
    def __init__(
        self,
        num_output_features: int,
        num_input_features: int,
        num_query_features: int,
        num_hidden_features: int,
        num_heads: int,
        num_latents: int = 64,
        coord_dim: int = 2,
        coordinate_system: str = 'cartesian',
        embedding_freq_multiplier: tuple = (1.0, 1.0),
        use_self_attention: bool = False,
        num_processor_layers: int = 1,
        use_encoder: bool = False,
        num_encoder_layers: int = 1,
        decoder_type: str = 'equivariant',
        num_decoder_layers: int = 1,
        use_processor: bool = True,
        use_rope: bool = False
    ):
        super().__init__()
        self.num_output_features = num_output_features
        self.num_input_features = num_input_features
        self.num_query_features = num_query_features
        self.num_hidden_features = num_hidden_features
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.num_processor_layers = num_processor_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.coord_dim = coord_dim
        self.coordinate_system = coordinate_system
        self.use_encoder = use_encoder
        self.decoder_type = decoder_type
        self.use_self_attention = use_self_attention
        self.use_rope = use_rope
        self.use_processor = use_processor

        # Initialize latents and positions
        self.latents = nn.Parameter(
            self._initialize_latents(num_latents, num_hidden_features))
        self.latent_pos = nn.Parameter(
            self._initialize_grid_positions(num_latents, coord_dim, coordinate_system))

        # Input projection
        self.input_proj = nn.Linear(
            num_input_features, num_hidden_features) if num_input_features != num_hidden_features else nn.Identity()

        self.query_features_proj = nn.Linear(
            num_query_features, num_hidden_features) if num_query_features != num_hidden_features else nn.Identity()

        # Encoder (Optional)
        if use_encoder:
            self.encoder_layers = nn.ModuleList([
                CrossAttention(
                    num_hidden_features,
                    num_heads,
                    coord_dim=coord_dim,
                    coordinate_system=coordinate_system,
                    use_rope=use_rope,
                )
                for _ in range(num_encoder_layers)
            ])

        # Processor (Self-Attention on Latents)
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.processor = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=num_hidden_features, nhead=num_heads, batch_first=True),
                num_layers=num_processor_layers
            )

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_decoder_layers):
            if decoder_type == 'equivariant':
                self.decoder_layers.append(
                    CrossAttention(
                        num_hidden_features,
                        num_heads,
                        coord_dim=coord_dim,
                        coordinate_system=coordinate_system,
                        use_rope=use_rope,
                    )
                )
            else:
                raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # Final projection
        self.final_proj = nn.Linear(num_hidden_features, num_output_features)

    def _initialize_latents(self, num_latents: int, dim: int) -> torch.Tensor:
        """Initialize latents with random features (like gridded-tnp)."""
        return torch.randn(num_latents, dim)  # TODO: initialize like GTNP

    def _initialize_grid_positions(self, num_points: int, dim: int, coordinate_system: str = 'cartesian') -> torch.Tensor:
        """Initialize positions uniformly distributed over a domain [-1, 1]."""
        if coordinate_system == 'cartesian':
            return torch.rand(num_points, dim) * 2 - 1
        elif coordinate_system == 'latlon':
            return torch.rand(num_points, dim) * 2 - 1
        elif coordinate_system == 'polar':
            return torch.rand(num_points, dim) * 2 - 1
        else:
            raise ValueError(f"Unknown coordinate_system: {coordinate_system}")

    def forward(
        self,
        query_pos: torch.Tensor,
        query_features: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        latent_pos: Optional[torch.Tensor] = None,
        grid_features: Optional[torch.Tensor] = None,
        grid_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional encoder.
        
        If use_encoder=True: uses grid_features/grid_pos to encode into latents.
        If use_encoder=False: uses provided latents/latent_pos directly.
        """
        # Encoder/getting latents
        if self.use_encoder:
            # Expand to batch size
            latents = self.latents.unsqueeze(
                0).expand(query_pos.shape[0], -1, -1)
            latent_pos = self.latent_pos.unsqueeze(
                0).expand(query_pos.shape[0], -1, -1)
            if grid_features is None or grid_pos is None:
                raise ValueError(
                    "Encoder requires grid_features and grid_pos inputs")

            for layer in self.encoder_layers:
                delta_latents, _ = layer(
                    query=latents,
                    query_pos=latent_pos,
                    context=grid_features,
                    context_pos=grid_pos,
                )
                latents = latents + delta_latents

        else:
            if latents is None or latent_pos is None:
                raise ValueError(
                    "Latents and latent_pos must be provided when encoder is disabled")
            latents = self.input_proj(latents)

        # Processor
        if self.use_self_attention:
            latents, latent_pos = self.processor(latents, latent_pos)

        # Decoder
        if query_features is None:
            raise ValueError("query_features must be provided")
        query_features = self.query_features_proj(query_features)
        
        for i, layer in enumerate(self.decoder_layers):
            delta, _ = layer(
                query=query_features,
                query_pos=query_pos,
                context=latents,
                context_pos=latent_pos
            )
            query_features = query_features + delta

        return self.final_proj(query_features)
