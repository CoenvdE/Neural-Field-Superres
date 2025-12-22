"""
Minimal script for benchmarking dataloading speed.

Tests:
1. Raw Zarr access times (single variable loads)
2. Dataset __getitem__ times
3. DataLoader throughput with different num_workers
4. Comparison between normal vs squashfs paths (configure below)

Usage:
    python scripts/benchmark_dataload.py --mode zarr  # Test raw zarr access
    python scripts/benchmark_dataload.py --mode dataset  # Test dataset __getitem__
    python scripts/benchmark_dataload.py --mode dataloader  # Test full dataloader
    python scripts/benchmark_dataload.py --mode all  # Run all tests
"""

import argparse
import time
import numpy as np
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONFIGURATION - Modify these paths for your setup
# =============================================================================

# Normal storage (Zarr v2)
NORMAL_CONFIG = {
    "latent_zarr_path": "/projects/prjs1858/latents_europe_2018_2020.zarr",
    "hres_zarr_path": "/projects/prjs1858/hres_europe_2018.zarr",
    "static_zarr_path": "/projects/prjs1858/static_hres_europe.zarr",
    "statistics_path": "/projects/prjs1858/hres_europe_2018_statistics.json",
    "static_statistics_path": "/projects/prjs1858/static_hres_europe_statistics.json",
    "zarr_format": None,  # auto-detect (v2)
}

# SquashFS storage (mounted)
SQUASHFS_CONFIG = {
    "latent_zarr_path": "/dev/shm/latents",  # Mounted squashfs
    "hres_zarr_path": "/dev/shm/hres",
    "static_zarr_path": "/dev/shm/static",
    "statistics_path": "/projects/prjs1858/hres_europe_2018_statistics.json",
    "static_statistics_path": "/projects/prjs1858/static_hres_europe_statistics.json",
    "zarr_format": 3,  # v3 format
}

# Dataset common config
DATASET_CONFIG = {
    "variables": ["2t", "msl"],
    "num_query_samples": 4096,
    "normalize_coords": True,
    "use_static_features": True,
    "static_variables": ["z", "lsm", "slt"],
    "normalize_static_features": True,
    "region_bounds": {
        "lat_min": 30,
        "lat_max": 70,
        "lon_min": -30,
        "lon_max": 50,
    },
}


def benchmark_raw_zarr(config: dict, n_samples: int = 50):
    """Benchmark raw Zarr read times for each component."""
    import xarray as xr
    
    print("\n" + "=" * 60)
    print("RAW ZARR ACCESS BENCHMARK")
    print("=" * 60)
    
    latent_path = config["latent_zarr_path"]
    hres_path = config["hres_zarr_path"]
    static_path = config["static_zarr_path"]
    zarr_format = config.get("zarr_format")
    
    # Check paths exist
    for name, path in [("latent", latent_path), ("hres", hres_path), ("static", static_path)]:
        if not Path(path).exists():
            print(f"⚠️  {name} path not found: {path}")
            return
    
    # Open datasets
    print(f"\nOpening Zarr stores (format={zarr_format or 'auto'})...")
    open_start = time.perf_counter()
    
    if zarr_format == 3:
        latent_ds = xr.open_zarr(latent_path, consolidated=False, zarr_format=3)
        hres_ds = xr.open_zarr(hres_path, consolidated=False, zarr_format=3)
        static_ds = xr.open_zarr(static_path, consolidated=False, zarr_format=3)
    else:
        latent_ds = xr.open_zarr(latent_path, consolidated=True)
        hres_ds = xr.open_zarr(hres_path, consolidated=True)
        static_ds = xr.open_zarr(static_path, consolidated=True)
    
    open_time = time.perf_counter() - open_start
    print(f"  Open time: {open_time:.3f}s")
    
    # Print dataset info
    print(f"\nDataset shapes:")
    print(f"  Latent 'surface_latents': {latent_ds['surface_latents'].shape}")
    print(f"  HRES '2t': {hres_ds['2t'].shape}")
    print(f"  Static 'z': {static_ds['z'].shape}")
    
    # Get random time indices
    n_times = min(len(latent_ds['time']), len(hres_ds['time']))
    time_indices = np.random.choice(n_times, min(n_samples, n_times), replace=False)
    
    # Benchmark: Latent reads
    print(f"\nBenchmarking {n_samples} random samples...")
    
    latent_times = []
    for t in time_indices:
        start = time.perf_counter()
        _ = latent_ds['surface_latents'].isel(time=int(t)).values
        latent_times.append(time.perf_counter() - start)
    
    print(f"\n📊 Latent reads (surface_latents):")
    print(f"   Mean: {np.mean(latent_times)*1000:.2f}ms")
    print(f"   Std:  {np.std(latent_times)*1000:.2f}ms")
    print(f"   Min:  {np.min(latent_times)*1000:.2f}ms")
    print(f"   Max:  {np.max(latent_times)*1000:.2f}ms")
    
    # Benchmark: HRES reads (2t)
    hres_2t_times = []
    for t in time_indices:
        start = time.perf_counter()
        _ = hres_ds['2t'].isel(time=int(t)).values
        hres_2t_times.append(time.perf_counter() - start)
    
    print(f"\n📊 HRES reads (2t):")
    print(f"   Mean: {np.mean(hres_2t_times)*1000:.2f}ms")
    print(f"   Std:  {np.std(hres_2t_times)*1000:.2f}ms")
    print(f"   Min:  {np.min(hres_2t_times)*1000:.2f}ms")
    print(f"   Max:  {np.max(hres_2t_times)*1000:.2f}ms")
    
    # Benchmark: HRES reads (msl)
    hres_msl_times = []
    for t in time_indices:
        start = time.perf_counter()
        _ = hres_ds['msl'].isel(time=int(t)).values
        hres_msl_times.append(time.perf_counter() - start)
    
    print(f"\n📊 HRES reads (msl):")
    print(f"   Mean: {np.mean(hres_msl_times)*1000:.2f}ms")
    print(f"   Std:  {np.std(hres_msl_times)*1000:.2f}ms")
    print(f"   Min:  {np.min(hres_msl_times)*1000:.2f}ms")
    print(f"   Max:  {np.max(hres_msl_times)*1000:.2f}ms")
    
    # Static (once)
    start = time.perf_counter()
    _ = static_ds['z'].values
    static_time = time.perf_counter() - start
    print(f"\n📊 Static read (z, once): {static_time*1000:.2f}ms")
    
    # Total per-sample estimate
    total_per_sample = np.mean(latent_times) + np.mean(hres_2t_times) + np.mean(hres_msl_times)
    print(f"\n=== ESTIMATED TOTAL PER SAMPLE: {total_per_sample*1000:.2f}ms ===")
    print(f"=== MAX THROUGHPUT: {1/total_per_sample:.1f} samples/sec ===")
    
    return {
        "latent_ms": np.mean(latent_times) * 1000,
        "hres_2t_ms": np.mean(hres_2t_times) * 1000,
        "hres_msl_ms": np.mean(hres_msl_times) * 1000,
        "total_per_sample_ms": total_per_sample * 1000,
    }


def benchmark_dataset_getitem(config: dict, n_samples: int = 50):
    """Benchmark Dataset.__getitem__ times."""
    from src.data.era_latent_hres_dataset import EraLatentHresDataset
    
    print("\n" + "=" * 60)
    print("DATASET __getitem__ BENCHMARK")
    print("=" * 60)
    
    # Check paths exist
    for name, path in [("latent", config["latent_zarr_path"]), 
                       ("hres", config["hres_zarr_path"])]:
        if not Path(path).exists():
            print(f"⚠️  {name} path not found: {path}")
            return
    
    print(f"\nInitializing dataset...")
    init_start = time.perf_counter()
    
    dataset = EraLatentHresDataset(
        latent_zarr_path=config["latent_zarr_path"],
        hres_zarr_path=config["hres_zarr_path"],
        static_zarr_path=config["static_zarr_path"],
        statistics_path=config["statistics_path"],
        static_statistics_path=config["static_statistics_path"],
        zarr_format=config.get("zarr_format"),
        **DATASET_CONFIG,
    )
    
    init_time = time.perf_counter() - init_start
    print(f"  Init time: {init_time:.3f}s")
    print(f"  Dataset length: {len(dataset)}")
    
    # Warmup
    print("\nWarmup (2 samples)...")
    _ = dataset[0]
    _ = dataset[1]
    
    # Random sample indices
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    print(f"\nBenchmarking {n_samples} random __getitem__ calls...")
    getitem_times = []
    for idx in indices:
        start = time.perf_counter()
        _ = dataset[int(idx)]
        getitem_times.append(time.perf_counter() - start)
    
    print(f"\n📊 __getitem__ timing:")
    print(f"   Mean: {np.mean(getitem_times)*1000:.2f}ms")
    print(f"   Std:  {np.std(getitem_times)*1000:.2f}ms")
    print(f"   Min:  {np.min(getitem_times)*1000:.2f}ms")
    print(f"   Max:  {np.max(getitem_times)*1000:.2f}ms")
    print(f"   P50:  {np.percentile(getitem_times, 50)*1000:.2f}ms")
    print(f"   P95:  {np.percentile(getitem_times, 95)*1000:.2f}ms")
    
    throughput = 1 / np.mean(getitem_times)
    print(f"\n=== MAX SINGLE-WORKER THROUGHPUT: {throughput:.1f} samples/sec ===")
    
    return {
        "mean_ms": np.mean(getitem_times) * 1000,
        "std_ms": np.std(getitem_times) * 1000,
        "p95_ms": np.percentile(getitem_times, 95) * 1000,
        "throughput_samples_sec": throughput,
    }


def benchmark_dataloader(config: dict, n_batches: int = 20, batch_size: int = 32):
    """Benchmark full DataLoader throughput with different num_workers."""
    from src.data.era_latent_hres_dataset import EraLatentHresDataset
    from torch.utils.data import DataLoader
    
    print("\n" + "=" * 60)
    print("DATALOADER THROUGHPUT BENCHMARK")
    print("=" * 60)
    
    # Check paths exist
    for name, path in [("latent", config["latent_zarr_path"]), 
                       ("hres", config["hres_zarr_path"])]:
        if not Path(path).exists():
            print(f"⚠️  {name} path not found: {path}")
            return
    
    print(f"\nInitializing dataset...")
    dataset = EraLatentHresDataset(
        latent_zarr_path=config["latent_zarr_path"],
        hres_zarr_path=config["hres_zarr_path"],
        static_zarr_path=config["static_zarr_path"],
        statistics_path=config["statistics_path"],
        static_statistics_path=config["static_statistics_path"],
        zarr_format=config.get("zarr_format"),
        **DATASET_CONFIG,
    )
    print(f"  Dataset length: {len(dataset)}")
    
    worker_configs = [0, 2, 4, 8, 10, 12]
    results = {}
    
    for num_workers in worker_configs:
        print(f"\n--- Testing num_workers={num_workers} ---")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            multiprocessing_context="forkserver" if num_workers > 0 else None,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )
        
        # Warmup
        warmup_iter = iter(dataloader)
        try:
            for _ in range(2):
                _ = next(warmup_iter)
        except StopIteration:
            pass
        del warmup_iter
        
        # Benchmark
        batch_times = []
        loader_iter = iter(dataloader)
        
        start_total = time.perf_counter()
        for i in range(n_batches):
            start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch_times.append(time.perf_counter() - start)
        
        total_time = time.perf_counter() - start_total
        samples_per_sec = (len(batch_times) * batch_size) / total_time
        
        print(f"   Batches loaded: {len(batch_times)}")
        print(f"   Mean batch time: {np.mean(batch_times)*1000:.2f}ms")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Throughput: {samples_per_sec:.1f} samples/sec")
        
        results[num_workers] = {
            "mean_batch_ms": np.mean(batch_times) * 1000,
            "samples_per_sec": samples_per_sec,
        }
        
        # Cleanup
        del dataloader
    
    print("\n" + "=" * 60)
    print("SUMMARY: Throughput by num_workers")
    print("=" * 60)
    for nw, res in results.items():
        print(f"  num_workers={nw:2d}: {res['samples_per_sec']:7.1f} samples/sec ({res['mean_batch_ms']:.1f}ms/batch)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark dataloading speed")
    parser.add_argument("--mode", choices=["zarr", "dataset", "dataloader", "all"], 
                        default="all", help="Benchmark mode")
    parser.add_argument("--storage", choices=["normal", "squashfs", "both"],
                        default="normal", help="Storage type to benchmark")
    parser.add_argument("--n-samples", type=int, default=50, 
                        help="Number of samples to benchmark")
    parser.add_argument("--n-batches", type=int, default=20,
                        help="Number of batches for dataloader benchmark")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dataloader benchmark")
    args = parser.parse_args()
    
    storage_configs = []
    if args.storage in ["normal", "both"]:
        storage_configs.append(("NORMAL", NORMAL_CONFIG))
    if args.storage in ["squashfs", "both"]:
        storage_configs.append(("SQUASHFS", SQUASHFS_CONFIG))
    
    all_results = {}
    
    for storage_name, config in storage_configs:
        print("\n" + "#" * 70)
        print(f"# STORAGE: {storage_name}")
        print("#" * 70)
        
        results = {}
        
        if args.mode in ["zarr", "all"]:
            results["zarr"] = benchmark_raw_zarr(config, args.n_samples)
        
        if args.mode in ["dataset", "all"]:
            results["dataset"] = benchmark_dataset_getitem(config, args.n_samples)
        
        if args.mode in ["dataloader", "all"]:
            results["dataloader"] = benchmark_dataloader(
                config, args.n_batches, args.batch_size
            )
        
        all_results[storage_name] = results
    
    # Comparison if both storages tested
    if len(storage_configs) == 2:
        print("\n" + "#" * 70)
        print("# COMPARISON: NORMAL vs SQUASHFS")
        print("#" * 70)
        
        if "dataset" in all_results.get("NORMAL", {}) and "dataset" in all_results.get("SQUASHFS", {}):
            normal_ms = all_results["NORMAL"]["dataset"]["mean_ms"]
            sqsh_ms = all_results["SQUASHFS"]["dataset"]["mean_ms"]
            speedup = normal_ms / sqsh_ms
            print(f"\n__getitem__ comparison:")
            print(f"  NORMAL:   {normal_ms:.2f}ms")
            print(f"  SQUASHFS: {sqsh_ms:.2f}ms")
            print(f"  Speedup:  {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")


if __name__ == "__main__":
    main()
