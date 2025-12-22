"""
Profile the model forward pass to identify bottlenecks.

Usage:
    python scripts/profile_model.py
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.model import NeuralFieldSuperRes


def create_dummy_batch(batch_size=32, num_queries=4096, num_latents=3200, 
                       latent_dim=512, num_vars=2, num_aux=3, device="cuda"):
    """Create a dummy batch matching your training data."""
    return {
        "latents": torch.randn(batch_size, num_latents, latent_dim, device=device),
        "latent_pos": torch.rand(batch_size, num_latents, 2, device=device) * 2 - 1,
        "query_pos": torch.rand(batch_size, num_queries, 2, device=device) * 2 - 1,
        "query_fields": torch.randn(batch_size, num_queries, num_vars, device=device),
        "query_auxiliary_features": torch.randn(batch_size, num_queries, num_aux, device=device),
        "latent_grid_shape": torch.tensor([40, 80], device=device),  # 40x80 = 3200 latents
    }


def profile_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model config matching your exp_static_nll_fast.yaml
    model = NeuralFieldSuperRes(
        num_output_features=2,  # 2t and msl
        coord_dim=2,
        num_hidden_features=512,
        num_heads=8,
        num_decoder_layers=2,
        k_nearest=9,
        use_gridded_knn=True,
        lat_ascending=False,
        roll_lon=False,
        pos_init_std=1.0,
        num_auxiliary_features=3,
        predict_variance=True,  # heteroscedastic NLL - doubles output to 4
    ).to(device)
    
    model.eval()
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create batch
    batch = create_dummy_batch(device=device)
    
    print(f"\nBatch shapes:")
    print(f"  latents: {batch['latents'].shape}")
    print(f"  latent_pos: {batch['latent_pos'].shape}")
    print(f"  query_pos: {batch['query_pos'].shape}")
    print(f"  query_auxiliary_features: {batch['query_auxiliary_features'].shape}")
    print(f"  latent_grid_shape: {batch['latent_grid_shape']}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(
                query_pos=batch["query_pos"],
                latents=batch["latents"],
                latent_pos=batch["latent_pos"],
                query_auxiliary_features=batch["query_auxiliary_features"],
                latent_grid_shape=batch["latent_grid_shape"],
            )
    torch.cuda.synchronize()
    
    # Profile forward pass
    print("\n" + "=" * 60)
    print("PROFILING FORWARD PASS (10 iterations)")
    print("=" * 60)
    
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            output = model(
                query_pos=batch["query_pos"],
                latents=batch["latents"],
                latent_pos=batch["latent_pos"],
                query_auxiliary_features=batch["query_auxiliary_features"],
                latent_grid_shape=batch["latent_grid_shape"],
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iter {i+1}: {elapsed*1000:.2f}ms")
    
    print(f"\n📊 Forward pass stats:")
    print(f"   Mean: {np.mean(times)*1000:.2f}ms")
    print(f"   Std:  {np.std(times)*1000:.2f}ms")
    print(f"   Min:  {np.min(times)*1000:.2f}ms")
    print(f"   Max:  {np.max(times)*1000:.2f}ms")
    
    # Profile forward + backward
    print("\n" + "=" * 60)
    print("PROFILING FORWARD + BACKWARD (10 iterations)")
    print("=" * 60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    times_fwd_bwd = []
    for i in range(10):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = model(
            query_pos=batch["query_pos"],
            latents=batch["latents"],
            latent_pos=batch["latent_pos"],
            query_auxiliary_features=batch["query_auxiliary_features"],
            latent_grid_shape=batch["latent_grid_shape"],
        )
        
        # Simple MSE loss for profiling
        loss = loss_fn(output[:, :, :2], batch["query_fields"])  # Only use mean predictions
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times_fwd_bwd.append(elapsed)
        print(f"  Iter {i+1}: {elapsed*1000:.2f}ms")
    
    print(f"\n📊 Forward + Backward stats:")
    print(f"   Mean: {np.mean(times_fwd_bwd)*1000:.2f}ms")
    print(f"   Std:  {np.std(times_fwd_bwd)*1000:.2f}ms")
    print(f"   Min:  {np.min(times_fwd_bwd)*1000:.2f}ms")
    
    # Estimate training throughput
    samples_per_sec = 32 / np.mean(times_fwd_bwd)  # batch_size / time_per_batch
    batches_per_sec = 1 / np.mean(times_fwd_bwd)
    
    print(f"\n" + "=" * 60)
    print("ESTIMATED TRAINING THROUGHPUT")
    print("=" * 60)
    print(f"   {batches_per_sec:.2f} batches/sec")
    print(f"   {samples_per_sec:.1f} samples/sec")
    print(f"   {np.mean(times_fwd_bwd)*1000:.1f}ms per training step")
    
    # Compare forward pass times
    print(f"\n" + "=" * 60)
    print("BREAKDOWN")
    print("=" * 60)
    backward_time = np.mean(times_fwd_bwd) - np.mean(times)
    print(f"   Forward:  {np.mean(times)*1000:.1f}ms ({np.mean(times)/np.mean(times_fwd_bwd)*100:.0f}%)")
    print(f"   Backward: {backward_time*1000:.1f}ms ({backward_time/np.mean(times_fwd_bwd)*100:.0f}%)")


if __name__ == "__main__":
    profile_model()
