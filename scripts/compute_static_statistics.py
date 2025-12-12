#!/usr/bin/env python
"""
Compute statistics (mean, std) for static HRES variables.

Usage:
    python scripts/compute_static_statistics.py \
        --zarr-path /projects/prjs1858/static_hres.zarr \
        --output /projects/prjs1858/static_hres_statistics.json \
        --variables z lsm slt
"""

import argparse
import json
import numpy as np
import xarray as xr
from pathlib import Path


def compute_statistics(zarr_path: str, variables: list, output_path: str = None):
    """Compute mean and std for static variables."""
    print(f"Loading static data from: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)
    
    print(f"Available variables: {list(ds.data_vars.keys())}")
    print(f"Available coordinates: {list(ds.coords.keys())}")
    
    stats = {}
    
    for var in variables:
        # Check both data_vars and coords
        if var in ds.data_vars:
            data = ds[var].values
        elif var in ds.coords:
            data = ds[var].values
        else:
            print(f"  Warning: Variable '{var}' not found, skipping")
            continue
        
        # Flatten and remove NaNs
        data_flat = data.flatten()
        data_valid = data_flat[~np.isnan(data_flat)]
        
        if len(data_valid) == 0:
            print(f"  Warning: Variable '{var}' has no valid data, skipping")
            continue
        
        mean = float(np.mean(data_valid))
        std = float(np.std(data_valid))
        
        # Ensure std is not zero (would cause division issues)
        if std < 1e-8:
            print(f"  Warning: Variable '{var}' has near-zero std ({std}), using 1.0")
            std = 1.0
        
        stats[var] = {
            "mean": mean,
            "std": std,
            "min": float(np.min(data_valid)),
            "max": float(np.max(data_valid)),
            "count": int(len(data_valid)),
        }
        
        print(f"  {var}: mean={mean:.4f}, std={std:.4f}, range=[{stats[var]['min']:.4f}, {stats[var]['max']:.4f}]")
    
    # Determine output path
    if output_path is None:
        zarr_stem = Path(zarr_path).stem
        output_path = Path(zarr_path).parent / f"{zarr_stem}_statistics.json"
    
    # Save statistics
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {output_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute statistics for static HRES variables")
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to static zarr store")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: derived from zarr path)")
    parser.add_argument("--variables", nargs="+", default=["z", "lsm", "slt"], help="Variables to compute stats for")
    
    args = parser.parse_args()
    
    compute_statistics(args.zarr_path, args.variables, args.output)


if __name__ == "__main__":
    main()

