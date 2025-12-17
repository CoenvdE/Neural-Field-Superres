#!/usr/bin/env python3
"""
Visualize Interpolation Baseline Predictions.

Shows bilinear and bicubic interpolation for both 2t and msl variables.

Usage:
    python scripts/visualize_interpolation.py --sample-idx 0
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models_baselines import LinearInterpolationBaseline, CubicInterpolationBaseline
from src.data import EraPredictionHresDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize interpolation baselines")
    
    parser.add_argument("--prediction-zarr", type=str,
        default="/projects/prjs1858/predictions_europe_2018_2020.zarr")
    parser.add_argument("--hres-zarr", type=str,
        default="/projects/prjs1858/hres_europe_2018.zarr")
    parser.add_argument("--statistics-path", type=str,
        default="/projects/prjs1858/hres_europe_2018_statistics.json")
    parser.add_argument("--prediction-statistics-path", type=str,
        default="/projects/prjs1858/predictions_europe_2018_2020_statistics.json")
    
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    
    return parser.parse_args()


def plot_variable(axes_row, var_name, var_idx, low_res_data, linear_pred, cubic_pred, 
                  ground_truth, pred_lat, pred_lon, hres_lat, hres_lon):
    """Plot one variable across a row of axes."""
    vmin = min(ground_truth.min(), linear_pred.min(), cubic_pred.min())
    vmax = max(ground_truth.max(), linear_pred.max(), cubic_pred.max())
    
    linear_diff = linear_pred - ground_truth
    cubic_diff = cubic_pred - ground_truth
    
    # Low-res
    im0 = axes_row[0].imshow(low_res_data, origin='upper',
                              extent=[pred_lon.min(), pred_lon.max(), pred_lat.min(), pred_lat.max()],
                              cmap='RdBu_r')
    axes_row[0].set_title(f'{var_name}\nLow-Res Input')
    plt.colorbar(im0, ax=axes_row[0], shrink=0.7)
    
    # Bilinear
    im1 = axes_row[1].imshow(linear_pred, origin='upper',
                              extent=[hres_lon.min(), hres_lon.max(), hres_lat.min(), hres_lat.max()],
                              cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes_row[1].set_title(f'Bilinear\nRMSE: {np.sqrt(np.mean(linear_diff**2)):.4f}')
    plt.colorbar(im1, ax=axes_row[1], shrink=0.7)
    
    # Bicubic
    im2 = axes_row[2].imshow(cubic_pred, origin='upper',
                              extent=[hres_lon.min(), hres_lon.max(), hres_lat.min(), hres_lat.max()],
                              cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes_row[2].set_title(f'Bicubic\nRMSE: {np.sqrt(np.mean(cubic_diff**2)):.4f}')
    plt.colorbar(im2, ax=axes_row[2], shrink=0.7)
    
    # Ground truth
    im3 = axes_row[3].imshow(ground_truth, origin='upper',
                              extent=[hres_lon.min(), hres_lon.max(), hres_lat.min(), hres_lat.max()],
                              cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes_row[3].set_title(f'Ground Truth (HRES)')
    plt.colorbar(im3, ax=axes_row[3], shrink=0.7)


def main():
    args = parse_args()
    
    dataset = EraPredictionHresDataset(
        prediction_zarr_path=args.prediction_zarr,
        hres_zarr_path=args.hres_zarr,
        variables=["2t", "msl"],
        statistics_path=args.statistics_path,
        prediction_statistics_path=args.prediction_statistics_path,
        split=args.split,
        num_query_samples=None,
    )
    
    sample = dataset[args.sample_idx]
    timestamp = dataset.get_timestamp(args.sample_idx)
    
    pred_lat, pred_lon = dataset.pred_lat, dataset.pred_lon
    hres_lat, hres_lon = dataset.hres_lat, dataset.hres_lon
    hres_shape = (len(hres_lat), len(hres_lon))
    
    low_res_data = sample['low_res_features'].numpy().reshape(len(pred_lat), len(pred_lon), -1)
    query_fields = sample['query_fields'].numpy()
    
    linear = LinearInterpolationBaseline()
    cubic = CubicInterpolationBaseline()
    
    # Create figure: 2 rows (2t, msl) x 4 cols (input, linear, cubic, truth)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    for var_idx, var_name in enumerate(["2t", "msl"]):
        ground_truth = query_fields[:, var_idx].reshape(hres_shape)
        
        linear_pred = linear.interpolate(
            low_res_data[:, :, var_idx], pred_lat, pred_lon, hres_lat, hres_lon
        )
        cubic_pred = cubic.interpolate(
            low_res_data[:, :, var_idx], pred_lat, pred_lon, hres_lat, hres_lon
        )
        
        plot_variable(axes[var_idx], var_name, var_idx, 
                     low_res_data[:, :, var_idx], linear_pred, cubic_pred, ground_truth,
                     pred_lat, pred_lon, hres_lat, hres_lon)
    
    plt.suptitle(f'Interpolation Baselines - {timestamp}', fontsize=14)
    plt.tight_layout()
    
    output = args.output or f"results/interpolation_viz_{args.sample_idx}.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output}")
    
    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
