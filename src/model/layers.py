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
        coord_dim: int = 2,  # Default to 2D (lat/lon)
        coordinate_system: str = "cartesian",
        k_nearest: int = 16,
        dropout: float = 0.0,
        positional_information_type: str = "rff", # options: rff, rope, bi-invariant, none
        pos_hidden_dim: Optional[int] = None, #TODO: needed?
        pos_mlp_layers: int = 2, #TODO: needed?
        pos_learnable_coefficients: bool = True,
        pos_init_std: float = 1.0,
        use_gridded_knn: bool = False,  # Use analytical KNN for regular grids
        roll_lon: bool = False,  # Longitude wraparound for global models
    ) -> None:
        super().__init__()
        if k_nearest <= 0:
            raise ValueError("k_nearest must be positive")

        self.embed_dim = embed_dim
        self.k_nearest = k_nearest
        self.positional_information_type = positional_information_type
        self.coordinate_system = coordinate_system
        self.use_gridded_knn = use_gridded_knn
        self.roll_lon = roll_lon

        if positional_information_type == "rff":
            self.pos_encoder = CoordinateEncoder( #TODO: check this logic
                coord_dim=coord_dim,
                embed_dim=embed_dim,
                coordinate_system=coordinate_system,
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
        context_grid_shape: Optional[torch.Tensor] = None,  # for analytical KNN
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project query tokens onto the k nearest context tokens.
        
        Args:
            query: [B, Q, D] query features
            query_pos: [B, Q, 2] query positions (normalized [-1, 1])
            context: [B, Z, D] context features (latents)
            context_pos: [B, Z, 2] context positions (not used if gridded KNN)
            context_grid_shape: [2] tensor (num_lat, num_lon) for analytical KNN
        """

        # Add position encoding to query and context features
        if self.positional_information_type == "rff":
            query_pos_enc = self.pos_encoder(query_pos)
            context_pos_enc = self.pos_encoder(context_pos)
            query = query + query_pos_enc
            context = context + context_pos_enc
        elif self.positional_information_type == "rope":
            # query_cos, query_sin = self.pos_encoder(query_pos)
            # context_cos, context_sin = self.pos_encoder(context_pos)
            # query = apply_rotary_pos_emb(query, query_cos, query_sin)
            # context = apply_rotary_pos_emb(context, context_cos, context_sin)
            raise NotImplementedError
        elif self.positional_information_type == "bi-invariant":
            raise NotImplementedError
        elif self.positional_information_type == "none":
            pass  # No positional encoding

        batch_size, num_query, _ = query.shape
        _, num_context, _ = context.shape
        k = min(self.k_nearest, num_context)

        # Choose KNN method: analytical (fast) or distance-based (general)
        if self.use_gridded_knn and context_grid_shape is not None:
            # Fast analytical KNN for regular grids (returns mask for duplicates)
            knn_idx, knn_mask = self._knn_indices_gridded(
                query_pos, context_grid_shape, k, roll_lon=self.roll_lon
            )
        else:
            # Fallback to distance-based KNN (no duplicates possible)
            knn_idx, knn_mask = self._knn_indices(query_pos, context_pos, k)

        # Use optimized gather to avoid OOM (chat)
        gathered_context = self._gather_with_indices_optimized(context, knn_idx)

        query_flat = query.reshape(batch_size * num_query, 1, self.embed_dim)
        context_flat = gathered_context.reshape(
            batch_size * num_query, k, self.embed_dim)

        # Apply duplicate mask if present (masks out duplicate neighbors)
        if knn_mask is not None:
            # knn_mask: [B, Q, k] -> [B*Q, 1, k] for attention
            attn_mask = ~knn_mask.reshape(batch_size * num_query, k)
            # Expand for num_heads: [B*Q, num_heads, 1, k]
            # But nn.MultiheadAttention expects [B*Q, 1, k] or [B*Q*num_heads, 1, k]
            # Use key_padding_mask which is [B*Q, k] where True = ignore
            attn_out, _ = self.attn(
                query_flat, context_flat, context_flat,
                key_padding_mask=attn_mask
            )
        else:
            attn_out, _ = self.attn(query_flat, context_flat, context_flat)
        
        attn_out = attn_out.reshape(batch_size, num_query, self.embed_dim)

        return attn_out, query_pos

    def _knn_indices(
        self, query_pos: torch.Tensor, context_pos: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute KNN indices using pairwise distances (expensive but general).
        
        Returns:
            knn_idx: [B, Q, k] indices
            mask: None (no duplicates possible with distance-based KNN)
        """
        distances = torch.cdist(query_pos, context_pos)
        knn_idx = torch.topk(distances, k=k, dim=-1, largest=False).indices
        return knn_idx, None  # No duplicates with distance-based KNN

    def _knn_indices_gridded(
        self,
        query_pos: torch.Tensor,
        grid_shape: torch.Tensor,
        k: int,
        roll_lon: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute KNN indices analytically for regular grids (fast!).
        
        Instead of computing O(Q × Z) distances, this directly computes which
        grid cells are neighbors based on the query position in normalized space.
        
        Args:
            query_pos: [B, Q, 2] normalized query positions in [-1, 1] (lat, lon)
            grid_shape: [2] tensor with (num_lat, num_lon)
            k: Number of neighbors (should be a perfect square like 4, 9, 16, 25)
            roll_lon: If True, longitude wraps around (for global models)
            
        Returns:
            knn_idx: [B, Q, k] indices into flattened context
            mask: [B, Q, k] boolean mask (True = valid, False = duplicate to ignore)
        """
        batch_size, num_query, _ = query_pos.shape
        num_lat, num_lon = grid_shape[0].item(), grid_shape[1].item()
        device = query_pos.device
        
        # Compute neighbor grid size (e.g., k=9 -> 3x3, k=16 -> 4x4)
        neighbor_size = int(k ** 0.5)
        if neighbor_size ** 2 != k:
            # Fall back to nearest perfect square
            neighbor_size = int((k ** 0.5) + 0.5)
        
        # Convert normalized coords [-1, 1] to grid indices [0, num-1]
        # query_pos[:,:,0] = lat, query_pos[:,:,1] = lon
        lat_idx = ((query_pos[..., 0] + 1) * (num_lat - 1) / 2).round().long()
        lon_idx = ((query_pos[..., 1] + 1) * (num_lon - 1) / 2).round().long()
        
        # Generate neighbor offsets (e.g., for 3x3: -1, 0, 1)
        half = neighbor_size // 2
        offsets = torch.arange(-half, neighbor_size - half, device=device)
        lat_offsets, lon_offsets = torch.meshgrid(offsets, offsets, indexing='ij')
        lat_offsets = lat_offsets.flatten()  # [k]
        lon_offsets = lon_offsets.flatten()  # [k]
        
        # Add offsets to get neighbor indices: [B, Q, k]
        neighbor_lat = lat_idx.unsqueeze(-1) + lat_offsets.view(1, 1, -1)
        neighbor_lon = lon_idx.unsqueeze(-1) + lon_offsets.view(1, 1, -1)
        
        # Handle boundaries
        # Latitude: CLAMP (poles are boundaries)
        neighbor_lat = neighbor_lat.clamp(0, num_lat - 1)
        
        # Longitude: MODULO (wraparound) or CLAMP
        if roll_lon:
            neighbor_lon = neighbor_lon % num_lon
        else:
            neighbor_lon = neighbor_lon.clamp(0, num_lon - 1)
        
        # Convert 2D grid indices to flat indices
        # Flattening order is (lat, lon), so: flat_idx = lat_idx * num_lon + lon_idx
        flat_indices = neighbor_lat * num_lon + neighbor_lon
        
        # Compute duplicate mask: True = valid (first occurrence), False = duplicate
        # This handles edge cases where clamping creates duplicates
        sorted_idx, argsort = torch.sort(flat_indices, dim=-1, stable=True)
        mask = torch.ones_like(sorted_idx, dtype=torch.bool)
        mask[..., 1:] = sorted_idx[..., 1:] != sorted_idx[..., :-1]
        # Unsort back to original order
        unsort_indices = torch.argsort(argsort, dim=-1)
        mask = torch.gather(mask, dim=-1, index=unsort_indices)
        
        return flat_indices, mask

    def _gather_with_indices(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather k-nearest context tokens for each query using expand + gather.
        
        SIMPLE BUT MEMORY-HEAVY IMPLEMENTATION
        
        How it works:
        1. Expand tensor from [B, seq_len, dim] to [B, Q, seq_len, dim]
           - Creates a "virtual" copy of context for each query position
        2. Expand indices from [B, Q, k] to [B, Q, k, dim]
           - Replicates each index across the dim dimension
        3. Use torch.gather to select k neighbors along dim=2 (seq_len axis)
        
        Memory issue:
        - tensor_expanded has shape [B, Q, seq_len, dim]
        - With B=32, Q=8192, seq_len=3200, dim=512: ~1.7 TB!
        - expand() creates a view (no immediate allocation), BUT:
          - torch.gather may materialize it
          - Backward pass WILL materialize gradients for full tensor
        
        When to use:
        - Small models or debugging
        - When you need to verify correctness of other implementations
        """
        batch, seq_len, dim = tensor.shape
        _, num_query, k = indices.shape

        # Step 1: Add query dimension and broadcast to [B, Q, seq_len, dim]
        # WARNING: This is a view, but gather/backward will materialize it!
        tensor_expanded = tensor.unsqueeze(1).expand(-1, num_query, -1, -1)
        
        # Step 2: Add dim dimension to indices: [B, Q, k] -> [B, Q, k, dim]
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, dim)

        # Step 3: Gather along seq_len axis (dim=2), selecting k entries per query
        # Output shape: [B, Q, k, dim]
        gathered = torch.gather(tensor_expanded, 2, idx_expanded)
        return gathered

    def _gather_with_indices_optimized(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Memory-efficient gather using flat indexing.
        
        OPTIMIZED IMPLEMENTATION - Avoids massive intermediate tensor
        NOTE: Opus' work
        
        How it works:
        1. Flatten tensor from [B, seq_len, dim] to [B*seq_len, dim]
           - Treat all batches as one long sequence
        2. Convert per-batch indices to global flat indices
           - Add batch_offset = batch_idx * seq_len to each index
        3. Use direct Python indexing (not torch.gather) to select neighbors
        
        Memory comparison (B=32, Q=8192, seq_len=3200, dim=512, k=16):
        - Original: [B, Q, seq_len, dim] = 1.7 TB (!)
        - Optimized: [B*seq_len, dim] + [B*Q*k, dim] = 52 MB + 34 MB = 86 MB
        
        Why it's faster:
        - No massive tensor expansion
        - Direct memory access pattern
        - Much better cache locality
        
        When to use:
        - Production training with large batches
        - When OOM errors occur with the simple version
        """
        batch, seq_len, dim = tensor.shape
        _, num_query, k = indices.shape

        # Step 1: Flatten all batches into one sequence: [B*seq_len, dim]
        # This is just a reshape, no memory copy
        flat_tensor = tensor.reshape(batch * seq_len, dim)

        # Step 2: Compute global flat indices
        # Each batch's indices need an offset: batch_0 uses [0, seq_len), batch_1 uses [seq_len, 2*seq_len), etc.
        batch_offsets = torch.arange(batch, device=tensor.device).view(batch, 1, 1) * seq_len
        flat_indices = (indices + batch_offsets).reshape(-1)  # Shape: [B*Q*k]

        # Step 3: Direct indexing - much more memory efficient than gather!
        # This only allocates [B*Q*k, dim] for the result
        gathered = flat_tensor[flat_indices].view(batch, num_query, k, dim)
        return gathered