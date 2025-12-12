"""Compute statistics for static features and save to JSON."""
import json
import numpy as np
import xarray as xr
import argparse

def compute_statistics(zarr_path: str, output_path: str, variables: list):
    """Compute mean and std for static variables."""
    print(f"Opening: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)
    
    stats = {}
    print("\nStatic variable statistics:")
    print("-" * 50)
    
    for var in variables:
        if var in ds.data_vars:
            data = ds[var].values.flatten()
            # Remove NaN values
            data = data[~np.isnan(data)]
            
            mean = float(np.mean(data))
            std = float(np.std(data))
            
            stats[var] = {"mean": mean, "std": std}
            print(f"{var:10s}: mean={mean:12.4f}, std={std:12.4f}")
        else:
            print(f"{var:10s}: NOT FOUND in dataset")
    
    print("-" * 50)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved statistics to: {output_path}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="/projects/prjs1858/static_hres_europe.zarr",
                        help="Path to static zarr file")
    parser.add_argument("--output", default="/projects/prjs1858/static_hres_statistics.json",
                        help="Output JSON path")
    parser.add_argument("--variables", nargs="+", default=["z", "lsm", "slt"],
                        help="Variables to compute statistics for")
    args = parser.parse_args()
    
    compute_statistics(args.zarr, args.output, args.variables)
