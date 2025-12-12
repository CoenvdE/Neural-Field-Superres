"""Attention building blocks."""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from .pos_emb import CoordinateEncoder
from .rope_stuff import RotaryEmbedding
from .rope_stuff import apply_rotary_pos_emb


class CrossAttention(nn.Module):
    """K-NN cross-attention between two sets of grid-aligned tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        coord_dim: int,
        coordinate_system: str = "cartesian",
        k_nearest: int = 16,
        dropout: float = 0.0,
        positional_information_type: str = "rff", # options: rff, rope, bi-invariant, none
        pos_hidden_dim: Optional[int] = None, #TODO: needed?
        pos_mlp_layers: int = 2, #TODO: needed?
        pos_learnable_coefficients: bool = True,
        pos_init_std: float = 1.0,
    ) -> None:
        super().__init__()
        if k_nearest <= 0:
            raise ValueError("k_nearest must be positive")

        self.embed_dim = embed_dim
        self.k_nearest = k_nearest
        self.positional_information_type = positional_information_type
        self.coordinate_system = coordinate_system

        if positional_information_type == "rff":
            self.pos_encoder = CoordinateEncoder(
                coord_dim=coord_dim,
                embed_dim=embed_dim,
                coordinate_system=coordinate_system,
                hidden_dim=pos_hidden_dim, #TODO: needed?
                num_layers=pos_mlp_layers, #TODO: needed?
                learnable_coefficients=pos_learnable_coefficients,
                init_std=pos_init_std,
            )
        elif positional_information_type == "rope":
            # self.pos_encoder = RotaryEmbedding(embed_dim) #TODO: make this for 2d matrix stuff
            raise NotImplementedError
        elif positional_information_type == "bi-invariant":
            #TODO: fix
            raise NotImplementedError
        elif positional_information_type == "none":
            # TODO: fix
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown positional_information_type: {positional_information_type}")

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        *,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        context: torch.Tensor,
        context_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project query tokens onto the k nearest context tokens."""

        if self.positional_information_type == "rff":
            query = self.pos_encoder(query_pos)
            context = self.pos_encoder(context_pos)
        elif self.positional_information_type == "rope":
            # query_cos, query_sin = self.pos_encoder(query_pos)
            # context_cos, context_sin = self.pos_encoder(context_pos)
            # query = apply_rotary_pos_emb(query, query_cos, query_sin)
            # context = apply_rotary_pos_emb(context, context_cos, context_sin)
            raise NotImplementedError
        elif self.positional_information_type == "bi-invariant":
            raise NotImplementedError
        elif self.positional_information_type == "none":
            raise NotImplementedError

        batch_size, num_query, _ = query.shape
        _, num_context, _ = context.shape
        k = min(self.k_nearest, num_context)

        # (batch, nq, k)
        knn_idx = self._knn_indices(query_pos, context_pos, k)
        gathered_context = self._gather_with_indices(context, knn_idx)

        query_flat = query.reshape(batch_size * num_query, 1, self.embed_dim)
        context_flat = gathered_context.reshape(
            batch_size * num_query, k, self.embed_dim)

        attn_out, _ = self.attn(query_flat, context_flat, context_flat)
        attn_out = attn_out.reshape(batch_size, num_query, self.embed_dim)

        return attn_out, query_pos

    def _knn_indices(
        self, query_pos: torch.Tensor, context_pos: torch.Tensor, k: int
    ) -> torch.Tensor:
        # TODO: check gridded tnp for optimized code when grid is fixed
        distances = torch.cdist(query_pos, context_pos)
        knn_idx = torch.topk(distances, k=k, dim=-1, largest=False).indices
        return knn_idx

    def _gather_with_indices(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # TODO: check this working later (with gridded tnp)
        batch, seq_len, dim = tensor.shape
        _, num_query, k = indices.shape

        tensor_expanded = tensor.unsqueeze(1).expand(-1, num_query, -1, -1)
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, dim)

        gathered = torch.gather(tensor_expanded, 2, idx_expanded)
        return gathered
