#!/usr/bin/env python3
"""
Train and Evaluate Random Forest Baseline for Super-Resolution.

This is a learned baseline that requires training on data.
Separate from interpolation baselines which require no training.

Usage:
    python scripts/train_random_forest.py --train-samples 50 --eval-samples 100
    python scripts/train_random_forest.py --load checkpoints/rf_baseline.joblib --eval-samples 100
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models_baselines import RandomForestBaseline
from src.data import EraPredictionHresDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Random Forest")
    
    parser.add_argument("--prediction-zarr", type=str,
        default="/projects/prjs1858/predictions_europe_2018_2020.zarr")
    parser.add_argument("--hres-zarr", type=str,
        default="/projects/prjs1858/hres_europe_2018.zarr")
    parser.add_argument("--statistics-path", type=str,
        default="/projects/prjs1858/hres_europe_2018_statistics.json")
    parser.add_argument("--prediction-statistics-path", type=str,
        default="/projects/prjs1858/predictions_europe_2018_2020_statistics.json")
    
    # Training
    parser.add_argument("--train-samples", type=int, default=50,
        help="Number of timesteps for training")
    parser.add_argument("--samples-per-timestep", type=int, default=1000,
        help="Query points per timestep for training")
    
    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    
    # Evaluation
    parser.add_argument("--eval-samples", type=int, default=100,
        help="Number of timesteps for evaluation")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    
    # Checkpoint
    parser.add_argument("--save", type=str, default="checkpoints/rf_baseline.joblib",
        help="Path to save trained model")
    parser.add_argument("--load", type=str, default=None,
        help="Path to load pre-trained model (skip training)")
    
    parser.add_argument("--output", type=str, default="results/random_forest_metrics.json")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("RANDOM FOREST BASELINE")
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
    
    # Initialize model
    rf = RandomForestBaseline(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
    )
    
    # Train or load
    if args.load and Path(args.load).exists():
        print(f"\nLoading from {args.load}...")
        rf.load(args.load)
    else:
        print(f"\nTraining on {args.train_samples} samples...")
        rf.fit(dataset, num_samples=args.train_samples, 
               samples_per_timestep=args.samples_per_timestep)
        
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        rf.save(args.save)
    
    # Evaluate
    print(f"\nEvaluating on {args.eval_samples} samples...")
    results = rf.evaluate_on_dataset(dataset, num_samples=args.eval_samples)
    
    print(f"\nResults:")
    print(f"  MSE:  {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
