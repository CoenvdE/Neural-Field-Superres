"""
Debug script to verify gridded KNN correctness.

The bug: The gridded KNN assumes latitude is in ASCENDING order (south to north),
but ERA5/Aurora data typically has latitude in DESCENDING order (north to south).

This causes the index mapping to be inverted!
"""

import numpy as np
import torch


def test_coordinate_mapping():
    """Test if the coordinate mapping is correct."""
    
    # Simulate typical ERA5 latent grid (DESCENDING latitude order!)
    latent_lat = np.array([70, 65, 60, 55, 50, 45, 40, 35, 30])  # North to South
    latent_lon = np.array([-30, -20, -10, 0, 10, 20, 30, 40, 50])  # West to East
    
    # The normalization bounds
    lat_min, lat_max = latent_lat.min(), latent_lat.max()  # 30, 70
    lon_min, lon_max = latent_lon.min(), latent_lon.max()  # -30, 50
    
    print("=" * 60)
    print("LATITUDE ORDER ANALYSIS")
    print("=" * 60)
    print(f"latent_lat: {latent_lat}")
    print(f"lat_min={lat_min}, lat_max={lat_max}")
    print(f"\nIs latitude DESCENDING? {latent_lat[0] > latent_lat[-1]}")
    
    # Normalize latitude values
    def normalize(arr, vmin, vmax):
        return 2 * (arr - vmin) / (vmax - vmin) - 1
    
    lat_norm = normalize(latent_lat, lat_min, lat_max)
    print(f"\nNormalized latitudes: {lat_norm}")
    
    # The gridded KNN assumes this mapping:
    # lat_idx = ((norm + 1) * (num_lat - 1) / 2).round()
    num_lat = len(latent_lat)
    mapped_indices = ((lat_norm + 1) * (num_lat - 1) / 2).round().astype(int)
    
    print("\n" + "=" * 60)
    print("INDEX MAPPING (what gridded KNN computes)")
    print("=" * 60)
    print(f"{'Actual Lat':<12} {'Norm Value':<12} {'Mapped Index':<15} {'Expected Index'}")
    for i, (lat, norm, mapped) in enumerate(zip(latent_lat, lat_norm, mapped_indices)):
        expected = i  # The actual position in the array
        status = "✓" if mapped == expected else "✗ BUG!"
        print(f"{lat:<12} {norm:<12.2f} {mapped:<15} {expected} {status}")
    
    # Show the problem
    print("\n" + "=" * 60)
    print("THE BUG EXPLAINED")
    print("=" * 60)
    print("""
The gridded KNN mapping assumes:
  - normalized coord -1 → grid index 0
  - normalized coord +1 → grid index (num_lat - 1)

But with DESCENDING latitudes:
  - lat=70 (northernmost) is at array index 0, but normalizes to +1
  - lat=30 (southernmost) is at array index 8, but normalizes to -1

This means the KNN is looking at the WRONG latent neighbors!
For a query at lat=70, it looks at neighbors near index 8 instead of index 0.
""")
    
    # Show the fix
    print("=" * 60)
    print("THE FIX")
    print("=" * 60)
    print("""
Option 1: Flip the index for descending coordinates:
    lat_idx = (num_lat - 1) - ((query_pos[..., 0] + 1) * (num_lat - 1) / 2).round()

Option 2: Sort latitudes ascending before building the grid
    
Option 3: Store is_descending flag and handle in KNN function
""")


def test_with_actual_shapes():
    """Simulate actual grid shapes and show the mismatch."""
    print("\n" + "=" * 60)
    print("SIMULATION WITH ACTUAL GRID SHAPES")
    print("=" * 60)
    
    # Typical Aurora latent grid
    num_lat, num_lon = 40, 80
    
    # Create descending latitudes (typical for ERA5/Aurora)
    latent_lat = np.linspace(70, 30, num_lat)  # DESCENDING
    latent_lon = np.linspace(-30, 50, num_lon)
    
    print(f"Grid shape: ({num_lat}, {num_lon})")
    print(f"Lat range: [{latent_lat[0]}° (north, idx=0) → {latent_lat[-1]}° (south, idx={num_lat-1})]")
    
    # Create flattened latent positions (this is what the model sees)
    lat_min, lat_max = latent_lat.min(), latent_lat.max()
    lon_min, lon_max = latent_lon.min(), latent_lon.max()
    
    lat_norm = 2 * (latent_lat - lat_min) / (lat_max - lat_min) - 1
    lon_norm = 2 * (latent_lon - lon_min) / (lon_max - lon_min) - 1
    
    lon_grid, lat_grid = np.meshgrid(lon_norm, lat_norm)
    latent_positions = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=-1)
    
    # Test: Where is the latent at position (0, 0) in the flattened array?
    flat_idx_00 = 0 * num_lon + 0  # = 0
    pos_at_00 = latent_positions[flat_idx_00]
    print(f"\nLatent at flat index 0 (array position [0,0]):")
    print(f"  Position: lat_norm={pos_at_00[0]:.2f}, lon_norm={pos_at_00[1]:.2f}")
    print(f"  This corresponds to: lat=70° (NORTH), lon=-30° (WEST)")
    
    # Now test what the gridded KNN would compute for a query at this position
    query_lat_norm, query_lon_norm = pos_at_00[0], pos_at_00[1]
    
    # This is what the CURRENT gridded KNN does:
    computed_lat_idx = round((query_lat_norm + 1) * (num_lat - 1) / 2)
    computed_lon_idx = round((query_lon_norm + 1) * (num_lon - 1) / 2)
    computed_flat_idx = computed_lat_idx * num_lon + computed_lon_idx
    
    print(f"\nGridded KNN computation for query at (lat_norm={query_lat_norm:.2f}, lon_norm={query_lon_norm:.2f}):")
    print(f"  Computed lat_idx: {computed_lat_idx}")
    print(f"  Computed lon_idx: {computed_lon_idx}")
    print(f"  Computed flat_idx: {computed_flat_idx}")
    print(f"  Expected flat_idx: 0")
    print(f"  MATCH: {computed_flat_idx == 0}")
    
    if computed_flat_idx != 0:
        print(f"\n  ❌ BUG: KNN is looking at the WRONG latent neighbors!")
        print(f"  For a query in the NORTH, it's looking at latents in the SOUTH!")
    
    # Now test THE FIX
    print("\n" + "=" * 60)
    print("AFTER FIX (lat_ascending=False)")
    print("=" * 60)
    
    # Fixed calculation: flip for descending lat
    fixed_lat_idx = round((1 - query_lat_norm) * (num_lat - 1) / 2)  # Note: (1 - x) instead of (x + 1)
    fixed_lon_idx = round((query_lon_norm + 1) * (num_lon - 1) / 2)
    fixed_flat_idx = fixed_lat_idx * num_lon + fixed_lon_idx
    
    print(f"Fixed computation for query at (lat_norm={query_lat_norm:.2f}, lon_norm={query_lon_norm:.2f}):")
    print(f"  Fixed lat_idx: {fixed_lat_idx}")
    print(f"  Fixed lon_idx: {fixed_lon_idx}")
    print(f"  Fixed flat_idx: {fixed_flat_idx}")
    print(f"  Expected flat_idx: 0")
    print(f"  MATCH: {fixed_flat_idx == 0}")
    
    if fixed_flat_idx == 0:
        print(f"\n  ✓ FIX WORKS! KNN now looks at correct neighbors.")


if __name__ == "__main__":
    test_coordinate_mapping()
    test_with_actual_shapes()
