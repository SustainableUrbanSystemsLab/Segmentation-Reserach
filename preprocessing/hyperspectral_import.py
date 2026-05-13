"""Helpers to load, stack, resample hyperspectral / multispectral tiles.

Supports:
- Sentinel-2 L2A folder structure (JP2 bands)
- ENVI (.hdr + .dat) via rasterio or spectral if available

Functions:
- stack_band_files(dirpath, pattern_map, out_path)
- reproject_match(src_path, target_path, out_path)
- reproject_match(src_path, target_path, out_path)
- resample_hsi_to_rgb_grid(hsi_path, rgb_path, out_path)
- upsample_hsi_to_rgb(hsi_path, rgb_path, out_path)
- validate_upsample_roundtrip(original_path, upsampled_path)
- compute_ndvi(stacked_array, red_index, nir_index)
"""
from __future__ import annotations

import json
import os
import csv
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import reproject


def find_sentinel_band_files(s2_folder: str) -> Dict[str, str]:
    """Return mapping of band keys to file paths for a Sentinel-2 product folder.

    Looks under GRANULE/*/IMG_DATA/R10m and R20m if present. Keys are 'B02','B03','B04','B08','TCI', etc.
    """
    bands = {}
    for root, dirs, files in os.walk(s2_folder):
        for f in files:
            if f.endswith('.jp2'):
                name = f.split('_')
                # pattern: T16SGC_20260510T162701_B02_10m.jp2
                for part in name:
                    if part.startswith('B') or part == 'TCI' or part.endswith('10m.jp2'):
                        pass
                # easier: parse token that starts with 'B' and is 3 chars like B02
                parts = f.split('_')
                for p in parts:
                    if p.startswith('B') and len(p) == 3:
                        bands[p] = os.path.join(root, f)
                        break
                    if p == 'TCI':
                        bands['TCI'] = os.path.join(root, f)
                        break
    return bands


def stack_band_files(band_files: List[str], out_path: str) -> None:
    """Stack a list of single-band files into a multi-band GeoTIFF using the first band as profile.
    band_files: list of file paths in desired band order
    """
    if len(band_files) == 0:
        raise ValueError('No band files provided')

    with rasterio.open(band_files[0]) as src0:
        profile = src0.profile.copy()
        profile.update(driver="GTiff", count=len(band_files), dtype=src0.dtypes[0])

    with rasterio.open(out_path, 'w', **profile) as dst:
        for i, bf in enumerate(band_files, start=1):
            with rasterio.open(bf) as bsrc:
                data = bsrc.read(1)
                dst.write(data, i)


def reproject_match(src_path: str, target_path: str, out_path: str, resampling: Resampling = Resampling.bilinear) -> None:
    """Reproject and resample src to match target (CRS, transform, width/height).

    Writes out_path GeoTIFF.
    """
    with rasterio.open(target_path) as tgt:
        dst_crs = tgt.crs
        dst_transform = tgt.transform
        dst_width = tgt.width
        dst_height = tgt.height

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", crs=dst_crs, transform=dst_transform, width=dst_width, height=dst_height)

        with rasterio.open(out_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )


def resample_hsi_to_rgb_grid(
    hsi_path: str,
    rgb_path: str,
    out_path: str,
    resampling: Resampling = Resampling.cubic,
) -> str:
    """Resample HSI onto the RGB grid without RGB-guided refinement.

    This is the safer choice for training because it preserves the HSI spectrum more faithfully
    than the guided upsampling path.
    """
    with rasterio.open(rgb_path) as rgb_src, rasterio.open(hsi_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=rgb_src.crs,
            transform=rgb_src.transform,
            width=rgb_src.width,
            height=rgb_src.height,
            dtype="float32",
            count=src.count,
        )
        if src.nodata is not None:
            profile.update(nodata=float(src.nodata))

        with rasterio.open(out_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=rgb_src.transform,
                    dst_crs=rgb_src.crs,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                    resampling=resampling,
                )

    return out_path


def compute_ndvi(stacked: np.ndarray, red_idx: int, nir_idx: int) -> np.ndarray:
    """Compute NDVI from a stacked array (bands, h, w). red_idx/nir_idx are 0-based indices."""
    red = stacked[red_idx].astype('float32')
    nir = stacked[nir_idx].astype('float32')
    denom = (nir + red)
    denom[denom == 0] = 1e-8
    ndvi = (nir - red) / denom
    return ndvi


def pca_reduce(stacked: np.ndarray, n_components: int = 6) -> np.ndarray:
    """Simple PCA reduction across bands. Input: (bands, h, w). Returns (n_components, h, w)."""
    from sklearn.decomposition import PCA

    b, h, w = stacked.shape
    X = stacked.reshape(b, -1).T  # (pixels, bands)
    pca = PCA(n_components=min(n_components, b))
    Xr = pca.fit_transform(X)
    Xr = Xr.T.reshape(-1, h, w)
    return Xr


def _band_to_float01(band: np.ndarray) -> tuple[np.ndarray, float, float]:
    band = band.astype(np.float32, copy=False)
    finite = np.isfinite(band)
    if not np.any(finite):
        return np.zeros_like(band, dtype=np.float32), 0.0, 0.0

    lo = float(np.nanpercentile(band[finite], 1.0))
    hi = float(np.nanpercentile(band[finite], 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(band[finite]))
        hi = float(np.nanmax(band[finite]))
    if hi <= lo:
        return np.zeros_like(band, dtype=np.float32), lo, hi

    scaled = (band - lo) / (hi - lo)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    return scaled.astype(np.float32, copy=False), lo, hi


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB guidance image must have shape (H, W, 3+)")
    rgb = rgb.astype(np.float32, copy=False)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    luma = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    np.clip(luma, 0.0, 1.0, out=luma)
    return luma.astype(np.float32, copy=False)


def _guided_refine_band(
    band: np.ndarray,
    guide: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    finite = np.isfinite(band)
    if not np.any(finite):
        return np.zeros_like(band, dtype=np.float32)

    band01, lo, hi = _band_to_float01(band)
    if hi <= lo:
        return band.astype(np.float32, copy=True)

    try:
        import cv2

        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            refined = cv2.ximgproc.guidedFilter(
                guide.astype(np.float32, copy=False),
                band01.astype(np.float32, copy=False),
                radius=int(radius),
                eps=float(eps),
            )
            refined = np.asarray(refined, dtype=np.float32)
            refined = refined * (hi - lo) + lo
            return refined
    except Exception:
        pass

    try:
        from skimage.restoration import denoise_bilateral

        refined = denoise_bilateral(
            band01.astype(np.float32, copy=False),
            sigma_color=0.08,
            sigma_spatial=max(1, int(radius)),
            channel_axis=None,
        ).astype(np.float32)
        return refined * (hi - lo) + lo
    except Exception:
        return band.astype(np.float32, copy=True)


def upsample_hsi_to_rgb(
    hsi_path: str,
    rgb_path: str,
    out_path: str,
    initial_resampling: Resampling = Resampling.cubic,
    guided_radius: int = 8,
    guided_eps: float = 1e-3,
    window_size: int = 1024,
) -> str:
    """Reproject HSI to the RGB grid and refine it with RGB-guided edge preservation.

    This is a practical, no-training fusion method: first resample HSI to the RGB grid,
    then use the RGB luminance as a guide to preserve edges while smoothing within regions.
    """
    with rasterio.open(rgb_path) as rgb_src, rasterio.open(hsi_path) as src:
        dst_crs = rgb_src.crs
        dst_transform = rgb_src.transform
        dst_width = rgb_src.width
        dst_height = rgb_src.height

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            dtype="float32",
            count=src.count,
        )
        if src.nodata is not None:
            profile.update(nodata=float(src.nodata))

        profile.update(dtype="float32")
        with rasterio.open(out_path, "w", **profile) as dst:
            step = max(128, int(window_size))
            for row_off in range(0, dst_height, step):
                for col_off in range(0, dst_width, step):
                    win_width = min(step, dst_width - col_off)
                    win_height = min(step, dst_height - row_off)
                    win = Window(col_off, row_off, win_width, win_height)
                    win_transform = window_transform(win, dst_transform)

                    rgb_window = rgb_src.read([1, 2, 3], window=win, out_dtype="float32").transpose(1, 2, 0)
                    guide = _rgb_to_luma(rgb_window)

                    chunk = np.empty((src.count, win_height, win_width), dtype=np.float32)
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=chunk[i - 1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=win_transform,
                            dst_crs=dst_crs,
                            src_nodata=src.nodata,
                            dst_nodata=np.nan,
                            resampling=initial_resampling,
                        )

                    for i in range(chunk.shape[0]):
                        chunk[i] = _guided_refine_band(chunk[i], guide, guided_radius, guided_eps)

                    dst.write(chunk.astype(np.float32, copy=False), window=win)

    return out_path


def _spectral_angle_map(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Compute per-pixel spectral angle in degrees."""
    ref = reference.reshape(reference.shape[0], -1).T.astype(np.float32, copy=False)
    tst = test.reshape(test.shape[0], -1).T.astype(np.float32, copy=False)

    valid = np.all(np.isfinite(ref), axis=1) & np.all(np.isfinite(tst), axis=1)
    sam = np.full(ref.shape[0], np.nan, dtype=np.float32)
    if not np.any(valid):
        return sam

    ref_v = ref[valid]
    tst_v = tst[valid]
    dot = np.sum(ref_v * tst_v, axis=1)
    denom = np.linalg.norm(ref_v, axis=1) * np.linalg.norm(tst_v, axis=1)
    denom = np.maximum(denom, 1e-8)
    cosang = np.clip(dot / denom, -1.0, 1.0)
    sam[valid] = np.degrees(np.arccos(cosang)).astype(np.float32)
    return sam


def validate_upsample_roundtrip(
    original_path: str,
    upsampled_path: str,
    report_path: str | None = None,
) -> dict:
    """Validate an upsampled raster by downsampling it back to the original grid.

    Returns a dictionary with per-band RMSE and mean/median spectral angle in degrees.
    """
    with rasterio.open(original_path) as original, rasterio.open(upsampled_path) as upsampled:
        original_data = original.read().astype(np.float32, copy=False)
        recon = np.empty_like(original_data, dtype=np.float32)

        for i in range(1, original.count + 1):
            reproject(
                source=rasterio.band(upsampled, i),
                destination=recon[i - 1],
                src_transform=upsampled.transform,
                src_crs=upsampled.crs,
                dst_transform=original.transform,
                dst_crs=original.crs,
                resampling=Resampling.bilinear,
            )

    valid = np.isfinite(original_data) & np.isfinite(recon)
    if not np.any(valid):
        raise RuntimeError("No valid pixels available to validate the upsampled result.")

    rmse_per_band = []
    for i in range(original_data.shape[0]):
        band_valid = valid[i]
        if not np.any(band_valid):
            rmse_per_band.append(np.nan)
            continue
        diff = original_data[i][band_valid] - recon[i][band_valid]
        rmse_per_band.append(float(np.sqrt(np.mean(diff * diff))))

    sam_map = _spectral_angle_map(original_data, recon)
    sam_valid = np.isfinite(sam_map)
    sam_mean = float(np.mean(sam_map[sam_valid])) if np.any(sam_valid) else float("nan")
    sam_median = float(np.median(sam_map[sam_valid])) if np.any(sam_valid) else float("nan")

    report = {
        "original_path": original_path,
        "upsampled_path": upsampled_path,
        "original_shape": list(original_data.shape),
        "upsampled_shape": list(recon.shape),
        "rmse_per_band": rmse_per_band,
        "rmse_mean": float(np.nanmean(rmse_per_band)),
        "sam_mean_deg": sam_mean,
        "sam_median_deg": sam_median,
    }

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def save_quicklook_png(
    raster_path: str,
    output_png: str,
    band_indices: tuple[int, int, int] | None = None,
    max_dim: int = 2048,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> str:
    """Save a downscaled PNG quicklook from a raster.

    band_indices are 0-based band indices. If omitted, uses the first three bands.
    """
    with rasterio.open(raster_path) as src:
        if band_indices is None:
            if src.count < 3:
                raise ValueError("Raster has fewer than 3 bands and no band_indices were provided.")
            band_indices = (0, 1, 2)

        bands_1based = [idx + 1 for idx in band_indices]
        scale = min(1.0, float(max_dim) / float(max(src.width, src.height)))
        out_width = max(1, int(round(src.width * scale)))
        out_height = max(1, int(round(src.height * scale)))

        rgb = src.read(
            bands_1based,
            out_shape=(len(bands_1based), out_height, out_width),
            resampling=Resampling.bilinear,
        ).transpose(1, 2, 0)

    rgb = rgb.astype(np.float32, copy=False)
    out = np.empty_like(rgb, dtype=np.uint8)
    for i in range(rgb.shape[2]):
        channel = rgb[..., i]
        lo = float(np.nanpercentile(channel, low_percentile))
        hi = float(np.nanpercentile(channel, high_percentile))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(channel))
            hi = float(np.nanmax(channel))
        if hi <= lo:
            out[..., i] = 0
            continue
        scaled = (channel - lo) * (255.0 / (hi - lo))
        np.clip(scaled, 0.0, 255.0, out=scaled)
        out[..., i] = scaled.astype(np.uint8)

    plt.imsave(output_png, out)
    return output_png


def export_paired_tiles(
    hsi_path: str,
    rgb_path: str,
    out_dir: str,
    tile_size: int = 1024,
    overlap: int = 0,
    prefix: str = "tile",
) -> str:
    """Export aligned HSI and RGB tiles plus a CSV manifest for manual labeling.

    Assumes hsi_path and rgb_path are already on the same grid.
    Writes:
      - out_dir/hsi/*.tif
      - out_dir/rgb/*.tif
      - out_dir/manifest.csv
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and < tile_size")

    hsi_out = os.path.join(out_dir, "hsi")
    rgb_out = os.path.join(out_dir, "rgb")
    os.makedirs(hsi_out, exist_ok=True)
    os.makedirs(rgb_out, exist_ok=True)

    step = tile_size - overlap
    rows = []

    with rasterio.open(hsi_path) as hsi_src, rasterio.open(rgb_path) as rgb_src:
        if (hsi_src.crs != rgb_src.crs or hsi_src.transform != rgb_src.transform or
                hsi_src.width != rgb_src.width or hsi_src.height != rgb_src.height):
            raise ValueError("HSI and RGB rasters must already be aligned before tiling.")

        tile_index = 0
        for row_off in range(0, hsi_src.height, step):
            for col_off in range(0, hsi_src.width, step):
                win_width = min(tile_size, hsi_src.width - col_off)
                win_height = min(tile_size, hsi_src.height - row_off)
                if win_width <= 0 or win_height <= 0:
                    continue

                win = Window(col_off, row_off, win_width, win_height)
                hsi_tile = hsi_src.read(window=win)
                rgb_tile = rgb_src.read([1, 2, 3], window=win)
                tile_name = f"{prefix}_{tile_index:05d}_{row_off:06d}_{col_off:06d}"

                hsi_path_out = os.path.join(hsi_out, f"{tile_name}_hsi.tif")
                rgb_path_out = os.path.join(rgb_out, f"{tile_name}_rgb.tif")

                hsi_profile = hsi_src.profile.copy()
                hsi_profile.update(driver="GTiff", height=win_height, width=win_width, transform=window_transform(win, hsi_src.transform))
                with rasterio.open(hsi_path_out, "w", **hsi_profile) as dst:
                    dst.write(hsi_tile)

                rgb_profile = rgb_src.profile.copy()
                rgb_profile.update(driver="GTiff", count=3, height=win_height, width=win_width, transform=window_transform(win, rgb_src.transform))
                with rasterio.open(rgb_path_out, "w", **rgb_profile) as dst:
                    dst.write(rgb_tile)

                rows.append({
                    "tile_name": tile_name,
                    "row_off": row_off,
                    "col_off": col_off,
                    "width": win_width,
                    "height": win_height,
                    "hsi_path": os.path.relpath(hsi_path_out, out_dir),
                    "rgb_path": os.path.relpath(rgb_path_out, out_dir),
                    "label": "",
                })
                tile_index += 1

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tile_name", "row_off", "col_off", "width", "height", "hsi_path", "rgb_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def export_native_paired_tiles(
    hsi_path: str,
    rgb_path: str,
    out_dir: str,
    rgb_tile_size: int = 1024,
    rgb_overlap: int = 0,
    prefix: str = "tile",
) -> str:
    """Export paired RGB and HSI chips for the same world extent at their native resolutions.

    RGB defines the chip footprint. HSI is cropped to the same geographic bounds but keeps
    its own native pixel size and dimensions. This is the recommended export for a dual-stream
    model because it avoids resampling the RGB image and avoids forcing the HSI into the RGB grid.
    """
    if rgb_tile_size <= 0:
        raise ValueError("rgb_tile_size must be > 0")
    if rgb_overlap < 0 or rgb_overlap >= rgb_tile_size:
        raise ValueError("rgb_overlap must be >= 0 and < rgb_tile_size")

    rgb_out = os.path.join(out_dir, "rgb")
    hsi_out = os.path.join(out_dir, "hsi")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(hsi_out, exist_ok=True)

    step = rgb_tile_size - rgb_overlap
    rows = []

    with rasterio.open(rgb_path) as rgb_src, rasterio.open(hsi_path) as hsi_src:
        tile_index = 0
        for row_off in range(0, rgb_src.height, step):
            for col_off in range(0, rgb_src.width, step):
                win_width = min(rgb_tile_size, rgb_src.width - col_off)
                win_height = min(rgb_tile_size, rgb_src.height - row_off)
                if win_width <= 0 or win_height <= 0:
                    continue

                rgb_win = Window(col_off, row_off, win_width, win_height)
                bounds = rasterio.windows.bounds(rgb_win, rgb_src.transform)

                rgb_chip = rgb_src.read([1, 2, 3], window=rgb_win)
                hsi_win = from_bounds(*bounds, transform=hsi_src.transform)
                hsi_chip = hsi_src.read(window=hsi_win, boundless=True, fill_value=np.nan)

                tile_name = f"{prefix}_{tile_index:05d}_{row_off:06d}_{col_off:06d}"
                rgb_path_out = os.path.join(rgb_out, f"{tile_name}_rgb.tif")
                hsi_path_out = os.path.join(hsi_out, f"{tile_name}_hsi.tif")

                rgb_profile = rgb_src.profile.copy()
                rgb_profile.update(
                    driver="GTiff",
                    count=3,
                    height=win_height,
                    width=win_width,
                    transform=window_transform(rgb_win, rgb_src.transform),
                )
                with rasterio.open(rgb_path_out, "w", **rgb_profile) as dst:
                    dst.write(rgb_chip)

                hsi_profile = hsi_src.profile.copy()
                hsi_profile.update(
                    driver="GTiff",
                    height=hsi_chip.shape[1],
                    width=hsi_chip.shape[2],
                    transform=window_transform(hsi_win, hsi_src.transform),
                )
                with rasterio.open(hsi_path_out, "w", **hsi_profile) as dst:
                    dst.write(hsi_chip)

                rows.append({
                    "tile_name": tile_name,
                    "row_off": row_off,
                    "col_off": col_off,
                    "rgb_width": win_width,
                    "rgb_height": win_height,
                    "hsi_width": hsi_chip.shape[2],
                    "hsi_height": hsi_chip.shape[1],
                    "bounds_left": bounds[0],
                    "bounds_bottom": bounds[1],
                    "bounds_right": bounds[2],
                    "bounds_top": bounds[3],
                    "rgb_path": os.path.relpath(rgb_path_out, out_dir),
                    "hsi_path": os.path.relpath(hsi_path_out, out_dir),
                    "label": "",
                })
                tile_index += 1

    manifest_path = os.path.join(out_dir, "native_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_name",
                "row_off",
                "col_off",
                "rgb_width",
                "rgb_height",
                "hsi_width",
                "hsi_height",
                "bounds_left",
                "bounds_bottom",
                "bounds_right",
                "bounds_top",
                "rgb_path",
                "hsi_path",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path
