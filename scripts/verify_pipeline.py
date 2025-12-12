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


def test_dataset_loading(data_dir: str, region_bounds: dict = None):
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
    
    if region_bounds:
        print(f"\n  Testing with region: {region_bounds}")
    
    try:
        dataset = EraLatentHresDataset(
            latent_zarr_path=str(data_dir / "latents_europe_2018_2020.zarr"),
            hres_zarr_path=str(data_dir / "hres_europe_2018.zarr"),
            variables=["2t", "msl"],
            num_query_samples=1000,  # Sample subset for speed
            normalize_coords=True,
            split=None,
            region_bounds=region_bounds,
        )
        print(f"  ✓ Dataset created with {len(dataset)} samples")
        
        # Show clipped grid info
        print(f"\n  Grid info (after clipping):")
        print(f"    Latent grid: ({len(dataset.latent_lat)}, {len(dataset.latent_lon)})")
        print(f"    HRES grid: ({len(dataset.hres_lat)}, {len(dataset.hres_lon)})")
        
        geo = dataset.geo_bounds
        print(f"\n  Geographic bounds:")
        print(f"    lat: [{geo['lat_min']:.2f}, {geo['lat_max']:.2f}]")
        print(f"    lon: [{geo['lon_min']:.2f}, {geo['lon_max']:.2f}]")
        
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
        
        # Create model (decoder-only, hidden_dim should match latent_dim)
        model = NeuralFieldSuperResModule(
            num_output_features=2,          # 2t and msl
            num_hidden_features=latent_dim, # Must match latent dimension (512)
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
        
        # Forward - unpack batch into explicit arguments
        with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(
                query_pos=batch["query_pos"],
                latents=batch["latents"],
                latent_pos=batch["latent_pos"],
                query_auxiliary_features=batch.get("query_auxiliary_features"),
            )
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


def test_visualization(data_dir: str, region_bounds: dict = None):
    """Test the visualization callback by generating a sample plot."""
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print(" Step 7: Testing Visualization Callback")
    print("=" * 60)
    
    try:
        from src.model import NeuralFieldSuperResModule
        from src.data import NeuralFieldDataModule
        from src.callbacks.visualization import HRESVisualizationCallback
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"  ✗ Could not import: {e}")
        return False
    
    data_dir = Path(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if region_bounds:
        print(f"\n  Testing with region: {region_bounds}")
    
    try:
        # Create datamodule with FULL grid (no subsampling) for visualization
        dm = NeuralFieldDataModule(
            latent_zarr_path=str(data_dir / "latents_europe_2018_2020.zarr"),
            hres_zarr_path=str(data_dir / "hres_europe_2018.zarr"),
            variables=["2t", "msl"],
            num_query_samples=None,  # None = use full grid
            batch_size=1,
            num_workers=0,
            normalize_coords=True,
            val_months=1,
            region_bounds=region_bounds,
        )
        dm.setup("fit")
        
        # Check if datamodule has the required attributes
        print(f"  DataModule attributes:")
        print(f"    has hres_shape: {hasattr(dm, 'hres_shape')}")
        print(f"    has geo_bounds: {hasattr(dm, 'geo_bounds')}")
        
        if hasattr(dm, 'hres_shape'):
            print(f"    hres_shape: {dm.hres_shape}")
        if hasattr(dm, 'geo_bounds'):
            print(f"    geo_bounds: {dm.geo_bounds}")
        
        # Get one batch
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        print(f"\n  Batch shapes:")
        for k, v in batch.items():
            print(f"    {k}: {v.shape}")
        
        # Create model
        sample = dm.train_dataset[0]
        latent_dim = sample['latents'].shape[-1]
        
        model = NeuralFieldSuperResModule(
            num_output_features=2,
            num_hidden_features=latent_dim,
            num_heads=8,
            coord_dim=2,
            num_decoder_layers=2,
            learning_rate=1e-4,
        )
        model = model.to(device)
        model.eval()
        
        # Forward pass in chunks to avoid OOM
        query_pos = batch["query_pos"]  # [1, Q, 2]
        latents = batch["latents"]      # [1, Z, 512]
        latent_pos = batch["latent_pos"]  # [1, Z, 2]
        
        num_queries = query_pos.shape[1]
        chunk_size = 10000  # Process 10K points at a time
        predictions_list = []
        
        print(f"\n  Processing {num_queries} query points in chunks of {chunk_size}...")
        
        with torch.no_grad():
            for start_idx in range(0, num_queries, chunk_size):
                end_idx = min(start_idx + chunk_size, num_queries)
                chunk_pos = query_pos[:, start_idx:end_idx, :]
                
                chunk_pred = model(
                    query_pos=chunk_pos,
                    latents=latents,
                    latent_pos=latent_pos,
                )
                predictions_list.append(chunk_pred.cpu())
                
                if (start_idx // chunk_size) % 10 == 0:
                    print(f"    Processed {end_idx}/{num_queries} points...")
        
        predictions = torch.cat(predictions_list, dim=1)
        print(f"\n  Predictions shape: {predictions.shape}")
        
        # Manual visualization test (mirroring callback logic)
        pred = predictions[0, :, 0].cpu().numpy()  # First variable (2t)
        target = batch["query_fields"][0, :, 0].cpu().numpy()
        
        # Try to get HRES shape for reshaping
        if hasattr(dm, 'hres_shape') and dm.hres_shape is not None:
            hres_shape = dm.hres_shape
            geo = dm.geo_bounds
            print(f"\n  Attempting to reshape to {hres_shape}...")
            try:
                pred_2d = pred.reshape(hres_shape)
                target_2d = target.reshape(hres_shape)
                diff_2d = pred_2d - target_2d
                
                # Compute extent for geographic coordinates [lon_min, lon_max, lat_min, lat_max]
                extent = [geo['lon_min'], geo['lon_max'], geo['lat_min'], geo['lat_max']]
                
                # Create visualization with individual min/max scaling
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                err_max = max(abs(diff_2d.min()), abs(diff_2d.max()))
                
                # Ground truth with its own min/max
                im0 = axes[0].imshow(target_2d, origin='upper', cmap='RdYlBu_r', 
                                      extent=extent, aspect='auto')
                axes[0].set_title(f"Ground Truth (2t)\nRange: [{target_2d.min():.1f}, {target_2d.max():.1f}] K")
                axes[0].set_xlabel("Longitude")
                axes[0].set_ylabel("Latitude")
                plt.colorbar(im0, ax=axes[0], label="K")
                
                # Prediction with its own min/max
                im1 = axes[1].imshow(pred_2d, origin='upper', cmap='RdYlBu_r', 
                                      extent=extent, aspect='auto')
                axes[1].set_title(f"Prediction (2t) - UNTRAINED\nRange: [{pred_2d.min():.1f}, {pred_2d.max():.1f}] K")
                axes[1].set_xlabel("Longitude")
                axes[1].set_ylabel("Latitude")
                plt.colorbar(im1, ax=axes[1], label="K")
                
                im2 = axes[2].imshow(diff_2d, origin='upper', cmap='RdBu_r', 
                                      vmin=-err_max, vmax=err_max, extent=extent, aspect='auto')
                axes[2].set_title("Error (Pred - GT)")
                axes[2].set_xlabel("Longitude")
                axes[2].set_ylabel("Latitude")
                plt.colorbar(im2, ax=axes[2], label="K")
                
                rmse = np.sqrt(np.mean(diff_2d ** 2))
                mae = np.mean(np.abs(diff_2d))
                fig.suptitle(f"RMSE: {rmse:.4f} K | MAE: {mae:.4f} K (Note: Model is untrained)")
                
                plt.tight_layout()
                
                # Save to file
                output_path = Path("test_visualization.png")
                fig.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  ✓ Visualization saved to: {output_path.absolute()}")
                
            except ValueError as e:
                print(f"  ⚠ Could not reshape to 2D grid: {e}")
                print(f"    Query points: {len(pred)}, HRES shape: {hres_shape}")
                print(f"    Expected: {hres_shape[0] * hres_shape[1]} points")
        else:
            print("  ⚠ DataModule doesn't have hres_shape attribute")
            print("    Visualization callback won't work without it.")
            print("    Consider adding hres_shape and geo_bounds to NeuralFieldDataModule")
        
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
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Also run visualization test (requires full grid loading)"
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["alps", "iberian", "scandinavia"],
        help="Test with a specific sub-region"
    )
    args = parser.parse_args()
    
    # Predefined regions
    regions = {
        "alps": {"lat_min": 45.0, "lat_max": 48.0, "lon_min": 5.0, "lon_max": 16.0},
        "iberian": {"lat_min": 36.0, "lat_max": 44.0, "lon_min": -10.0, "lon_max": 4.0},
        "scandinavia": {"lat_min": 55.0, "lat_max": 70.0, "lon_min": 5.0, "lon_max": 30.0},
    }
    region_bounds = regions.get(args.region) if args.region else None
    
    print("=" * 60)
    print(" Neural Field Super-Resolution Pipeline Verification")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    if region_bounds:
        print(f"Region: {args.region} {region_bounds}")
    
    # Run checks
    files = check_zarr_files(args.data_dir)
    
    # Only continue if required files exist
    if not files.get("latents", {}).get("exists") or not files.get("hres", {}).get("exists"):
        print("\n✗ Required files missing! Cannot continue.")
        return 1
    
    check_time_alignment(args.data_dir)
    check_spatial_alignment(args.data_dir)
    test_dataset_loading(args.data_dir, region_bounds)
    test_datamodule(args.data_dir)
    
    if not args.quick:
        test_model_forward(args.data_dir)
    
    if args.viz:
        test_visualization(args.data_dir, region_bounds)
    
    print("\n" + "=" * 60)
    print(" Verification Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
