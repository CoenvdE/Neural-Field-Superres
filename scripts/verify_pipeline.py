#!/usr/bin/env python
"""
Quick verification script for the Neural Field Super-Resolution pipeline.

Tests data loading, alignment, and a single forward pass.

Usage:
    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --data-dir /projects/prjs1858
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import xarray as xr


def check_zarr_files(data_dir: str) -> dict:
    """Check that required Zarr files exist and are accessible."""
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print(" Step 1: Checking Zarr files")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    
    files = {
        "latents": data_dir / "latents_europe_2018_2020.zarr",
        "hres": data_dir / "hres_europe_2018.zarr",  # Using available data
        "static": data_dir / "static_hres.zarr",
    }
    
    results = {}
    for name, path in files.items():
        exists = path.exists()
        results[name] = {"path": str(path), "exists": exists}
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
    
    return results


def check_time_alignment(data_dir: str) -> dict:
    """Check time alignment between latents and HRES datasets."""
    from pathlib import Path
    import pandas as pd
    
    print("\n" + "=" * 60)
    print(" Step 2: Checking time alignment")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    
    latent_ds = xr.open_zarr(data_dir / "latents_europe_2018_2020.zarr", consolidated=True)
    hres_ds = xr.open_zarr(data_dir / "hres_europe_2018.zarr", consolidated=True)
    
    latent_times = pd.to_datetime(latent_ds['time'].values)
    hres_times = pd.to_datetime(hres_ds['time'].values)
    
    print(f"\n  Latent times:")
    print(f"    Range: {latent_times[0]} to {latent_times[-1]}")
    print(f"    Count: {len(latent_times)}")
    
    print(f"\n  HRES times:")
    print(f"    Range: {hres_times[0]} to {hres_times[-1]}")
    print(f"    Count: {len(hres_times)}")
    
    # Find overlapping times
    common_times = latent_times.intersection(hres_times)
    print(f"\n  Common timesteps: {len(common_times)}")
    if len(common_times) > 0:
        print(f"    Range: {common_times[0]} to {common_times[-1]}")
    
    return {
        "latent_count": len(latent_times),
        "hres_count": len(hres_times),
        "common_count": len(common_times),
    }


def check_spatial_alignment(data_dir: str):
    """Check spatial grids of latents and HRES."""
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print(" Step 3: Checking spatial grids")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    
    latent_ds = xr.open_zarr(data_dir / "latents_europe_2018_2020.zarr", consolidated=True)
    hres_ds = xr.open_zarr(data_dir / "hres_europe_2018.zarr", consolidated=True)
    static_ds = xr.open_zarr(data_dir / "static_hres.zarr", consolidated=True)
    
    print("\n  Latent grid:")
    print(f"    lat: {len(latent_ds['lat'])} points, range [{latent_ds['lat'].values.min():.2f}, {latent_ds['lat'].values.max():.2f}]")
    print(f"    lon: {len(latent_ds['lon'])} points, range [{latent_ds['lon'].values.min():.2f}, {latent_ds['lon'].values.max():.2f}]")
    
    print("\n  HRES grid:")
    print(f"    lat: {len(hres_ds['latitude'])} points, range [{hres_ds['latitude'].values.min():.2f}, {hres_ds['latitude'].values.max():.2f}]")
    print(f"    lon: {len(hres_ds['longitude'])} points, range [{hres_ds['longitude'].values.min():.2f}, {hres_ds['longitude'].values.max():.2f}]")
    
    print("\n  Static grid (global):")
    print(f"    lat: {len(static_ds['latitude'])} points")
    print(f"    lon: {len(static_ds['longitude'])} points")
    
    # Compute super-resolution factor
    lat_factor = len(hres_ds['latitude']) / len(latent_ds['lat'])
    lon_factor = len(hres_ds['longitude']) / len(latent_ds['lon'])
    print(f"\n  Super-resolution factor:")
    print(f"    lat: {lat_factor:.1f}x")
    print(f"    lon: {lon_factor:.1f}x")


def test_dataset_loading(data_dir: str):
    """Test loading a sample from the dataset."""
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print(" Step 4: Testing EraLatentHresDataset")
    print("=" * 60)
    
    try:
        from src.data.era_latent_hres_dataset import EraLatentHresDataset
    except ImportError as e:
        print(f"  ✗ Could not import dataset: {e}")
        return
    
    data_dir = Path(data_dir)
    
    try:
        dataset = EraLatentHresDataset(
            latent_zarr_path=str(data_dir / "latents_europe_2018_2020.zarr"),
            hres_zarr_path=str(data_dir / "hres_europe_2018.zarr"),
            variables=["2t", "msl"],
            num_query_samples=1000,  # Sample subset for speed
            normalize_coords=True,
            split=None,
        )
        print(f"  ✓ Dataset created with {len(dataset)} samples")
        
        # Load one sample
        sample = dataset[0]
        print(f"\n  Sample contents:")
        for key, val in sample.items():
            print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
        
        # Check for NaNs
        print(f"\n  NaN check:")
        for key, val in sample.items():
            nan_count = torch.isnan(val).sum().item()
            nan_pct = 100 * nan_count / val.numel()
            status = "✓" if nan_count == 0 else "⚠"
            print(f"    {status} {key}: {nan_count} NaNs ({nan_pct:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datamodule(data_dir: str):
    """Test the DataModule and DataLoader."""
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print(" Step 5: Testing NeuralFieldDataModule")
    print("=" * 60)
    
    try:
        from src.data import NeuralFieldDataModule
    except ImportError as e:
        print(f"  ✗ Could not import datamodule: {e}")
        return
    
    data_dir = Path(data_dir)
    
    try:
        dm = NeuralFieldDataModule(
            latent_zarr_path=str(data_dir / "latents_europe_2018_2020.zarr"),
            hres_zarr_path=str(data_dir / "hres_europe_2018.zarr"),
            variables=["2t", "msl"],
            num_query_samples=1000,
            batch_size=2,
            num_workers=0,  # Single-threaded for testing
            normalize_coords=True,
            val_months=1,
        )
        dm.setup("fit")
        
        print(f"  ✓ DataModule created")
        print(f"    Train samples: {len(dm.train_dataset)}")
        print(f"    Val samples: {len(dm.val_dataset)}")
        
        # Get a batch
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"\n  Batch contents:")
        for key, val in batch.items():
            print(f"    {key}: shape={val.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward(data_dir: str):
    """Test a single forward pass through the model."""
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print(" Step 6: Testing model forward pass")
    print("=" * 60)
    
    try:
        from src.model import NeuralFieldSuperResModule
        from src.data import NeuralFieldDataModule
    except ImportError as e:
        print(f"  ✗ Could not import: {e}")
        return
    
    data_dir = Path(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    try:
        # Create datamodule
        dm = NeuralFieldDataModule(
            latent_zarr_path=str(data_dir / "latents_europe_2018_2020.zarr"),
            hres_zarr_path=str(data_dir / "hres_europe_2018.zarr"),
            variables=["2t", "msl"],
            num_query_samples=500,  # Small for testing
            batch_size=2,
            num_workers=0,
            normalize_coords=True,
            val_months=1,
        )
        dm.setup("fit")
        
        # Get latent dimension from data
        sample = dm.train_dataset[0]
        latent_dim = sample['latents'].shape[-1]
        print(f"  Latent dimension: {latent_dim}")
        
        # Create model
        model = NeuralFieldSuperResModule(
            num_output_features=2,
            num_input_features=latent_dim,
            num_query_features=2,
            num_hidden_features=256,
            num_heads=8,
            coord_dim=2,
            num_decoder_layers=2,
            learning_rate=1e-4,
        )
        model = model.to(device)
        model.train()
        
        # Get batch
        batch = next(iter(dm.train_dataloader()))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        print(f"\n  Forward pass...")
        
        # Forward
        with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(batch)
            loss = model.training_step(batch, 0)
        
        print(f"  ✓ Forward pass successful!")
        print(f"    Output shape: {outputs.shape}")
        print(f"    Loss: {loss:.4f}")
        
        # Backward
        loss.backward()
        print(f"  ✓ Backward pass successful!")
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append((name.split('.')[-1], param.grad.norm().item()))
        
        print(f"\n  Gradient check (sample):")
        for name, norm in grad_norms[:5]:
            print(f"    {name}: {norm:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify Neural Field Super-Resolution pipeline")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="/projects/prjs1858",
        help="Path to data directory (default: /projects/prjs1858)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick checks (skip model forward pass)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print(" Neural Field Super-Resolution Pipeline Verification")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    
    # Run checks
    files = check_zarr_files(args.data_dir)
    
    # Only continue if required files exist
    if not files.get("latents", {}).get("exists") or not files.get("hres", {}).get("exists"):
        print("\n✗ Required files missing! Cannot continue.")
        return 1
    
    check_time_alignment(args.data_dir)
    check_spatial_alignment(args.data_dir)
    test_dataset_loading(args.data_dir)
    test_datamodule(args.data_dir)
    
    if not args.quick:
        test_model_forward(args.data_dir)
    
    print("\n" + "=" * 60)
    print(" Verification Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
