#!/usr/bin/env python3
"""
Evaluate Interpolation Baselines for Super-Resolution.

Compares bilinear and bicubic interpolation against ground truth HRES data.
These are non-learned baselines that require no training.

Usage:
    python scripts/evaluate_interpolation.py --num-samples 100
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models_baselines import LinearInterpolationBaseline, CubicInterpolationBaseline
from src.data import EraPredictionHresDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate interpolation baselines")
    
    parser.add_argument("--prediction-zarr", type=str,
        default="/projects/prjs1858/predictions_europe_2018_2020.zarr")
    parser.add_argument("--hres-zarr", type=str,
        default="/projects/prjs1858/hres_europe_2018.zarr")
    parser.add_argument("--statistics-path", type=str,
        default="/projects/prjs1858/hres_europe_2018_statistics.json")
    parser.add_argument("--prediction-statistics-path", type=str,
        default="/projects/prjs1858/predictions_europe_2018_2020_statistics.json")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output", type=str, default="results/interpolation_metrics.json")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("INTERPOLATION BASELINE EVALUATION")
    print("=" * 60)
    
    dataset = EraPredictionHresDataset(
        prediction_zarr_path=args.prediction_zarr,
        hres_zarr_path=args.hres_zarr,
        variables=["2t", "msl"],
        statistics_path=args.statistics_path,
        prediction_statistics_path=args.prediction_statistics_path,
        split=args.split,
        num_query_samples=None,
    )
    
    print(f"Dataset: {len(dataset)} samples, split={args.split}")
    
    results = {}
    
    # Linear interpolation
    print("\n1. LINEAR (Bilinear)")
    linear = LinearInterpolationBaseline()
    results["linear"] = linear.evaluate_on_dataset(dataset, num_samples=args.num_samples)
    print(f"   MSE: {results['linear']['mse']:.6f}, RMSE: {results['linear']['rmse']:.6f}")
    
    # Cubic interpolation
    print("\n2. CUBIC (Bicubic)")
    cubic = CubicInterpolationBaseline()
    results["cubic"] = cubic.evaluate_on_dataset(dataset, num_samples=args.num_samples)
    print(f"   MSE: {results['cubic']['mse']:.6f}, RMSE: {results['cubic']['rmse']:.6f}")
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

