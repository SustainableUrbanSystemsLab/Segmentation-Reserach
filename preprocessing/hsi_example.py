"""Example: read Sentinel-2 product folder, stack bands, compute NDVI and PCA, and align to an RGB tile.

Usage:
    python hsi_example.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure top-level packages are importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from preprocessing.hyperspectral_import import (
    compute_ndvi,
    export_native_paired_tiles,
    find_sentinel_band_files,
    pca_reduce,
    reproject_match,
    save_quicklook_png,
    resample_hsi_to_rgb_grid,
    stack_band_files,
    upsample_hsi_to_rgb,
    validate_upsample_roundtrip,
)
from models.image_processing import normalize_to_uint8_robust, report_geotiff_spatial_info


S2_FOLDER = r"Maps/Atlanta(Tiles)/Atlanta_Hyperspectral"
OUT_DIR = r"Maps/Atlanta(Tiles)/HSI_products"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    bands = find_sentinel_band_files(S2_FOLDER)
    print("Found bands:", sorted(bands.keys()))

    # Choose 10m bands if present: B02 (blue), B03 (green), B04 (red), B08 (nir)
    order = []
    for key in ('B02', 'B03', 'B04', 'B08'):
        if key in bands:
            order.append(bands[key])

    if not order:
        raise RuntimeError('No sentinel JP2 bands found in folder')

    stacked_path = os.path.join(OUT_DIR, 's2_stacked.tif')
    print('Stacking bands to', stacked_path)
    stack_band_files(order, stacked_path)

    report_geotiff_spatial_info(stacked_path)

    # Read stacked for processing
    import rasterio

    with rasterio.open(stacked_path) as src:
        stacked = src.read()  # (bands, h, w)

    # Compute NDVI (red index 2 -> B04 is index 2 as we stacked B02,B03,B04,B08)
    ndvi = compute_ndvi(stacked, red_idx=2, nir_idx=3)
    ndvi_vis = normalize_to_uint8_robust(ndvi, low_percentile=2, high_percentile=98)
    plt.imsave(os.path.join(OUT_DIR, 'ndvi_vis.png'), ndvi_vis, cmap='RdYlGn')
    print('Saved NDVI visualization')

    # PCA reduce to 3 components for RGB-like visualization
    pca = pca_reduce(stacked, n_components=3)
    pca_vis = np.stack([normalize_to_uint8_robust(pca[i]) for i in range(pca.shape[0])], axis=-1)
    plt.imsave(os.path.join(OUT_DIR, 'pca_vis.png'), pca_vis)
    print('Saved PCA visualization')

    # Optional: align to an RGB tile if you have one (project-local Maps folder)
    rgb_tile = os.path.join(os.getcwd(), "Maps", "Atlanta(Tiles)", "Atlanta_base.tif")
    if os.path.exists(rgb_tile):
        baseline_out = os.path.join(OUT_DIR, 's2_resampled_to_rgb_grid.tif')
        print('Resampling S2 stack to the RGB grid without guided refinement...')
        resample_hsi_to_rgb_grid(stacked_path, rgb_tile, baseline_out)
        print('Wrote', baseline_out)

        aligned_out = os.path.join(OUT_DIR, 's2_upsampled_to_rgb.tif')
        print('Upsampling S2 stack to match RGB tile with RGB guidance...')
        upsample_hsi_to_rgb(stacked_path, rgb_tile, aligned_out)
        print('Wrote', aligned_out)

        quicklook_path = os.path.join(OUT_DIR, 's2_upsampled_quicklook.png')
        save_quicklook_png(aligned_out, quicklook_path, band_indices=(2, 1, 0), max_dim=8192)
        print('Saved quicklook to', quicklook_path)

        training_quicklook_path = os.path.join(OUT_DIR, 's2_resampled_quicklook.png')
        save_quicklook_png(baseline_out, training_quicklook_path, band_indices=(2, 1, 0), max_dim=8192)
        print('Saved training quicklook to', training_quicklook_path)

        report_path = os.path.join(OUT_DIR, 'upsample_validation_report.json')
        report = validate_upsample_roundtrip(stacked_path, aligned_out, report_path=report_path)
        print('Validation report saved to', report_path)
        print(json.dumps(report, indent=2))

        if os.environ.get('EXPORT_TRAINING_TILES', '0') == '1':
            training_dir = os.path.join(OUT_DIR, 'training_tiles')
            manifest_path = export_native_paired_tiles(
                stacked_path,
                rgb_tile,
                training_dir,
                rgb_tile_size=1024,
                rgb_overlap=128,
                prefix='atlanta',
            )
            print('Exported paired training tiles to', training_dir)
            print('Training manifest:', manifest_path)
    else:
        print('No RGB tile found at', rgb_tile)


if __name__ == '__main__':
    main()
