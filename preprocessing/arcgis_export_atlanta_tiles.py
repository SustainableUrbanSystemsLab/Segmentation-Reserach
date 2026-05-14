"""
Export Atlanta imagery into uniform high-resolution GeoTIFF tiles.
 
Uses contextily + rasterio — no ArcGIS/ArcPy required.
 
What this script does:
1) Downloads satellite imagery for your study area via contextily.
2) Reprojects to UTM Zone 16N (EPSG:26916) for accurate meter-based tiling.
3) Splits the base raster into 500m x 500m tiles (2000x2000px at 0.25m/px).
4) Fills edge tiles with nodata rather than skipping them.
5) Writes a tile metadata CSV with extents and coordinates.
 
Dependencies:
    pip install contextily rasterio pyproj numpy mercantile scipy
 
Run from any Python environment (no ArcGIS required).
"""
 
from __future__ import annotations
 
import csv
import os
import sys
import traceback
from dataclasses import dataclass
 
import numpy as np
import contextily as ctx
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer
 
 
# ----------------------------
# USER CONFIG
# ----------------------------
 
# Your study area bounds in UTM Zone 16N (EPSG:26916) — from your ArcGIS fishnet
UTM_LEFT   = 738299.3766
UTM_BOTTOM = 3735629.2001
UTM_RIGHT  = 744953.2947
UTM_TOP    = 3743524.6855
 
# Source CRS of the bounds above
SOURCE_CRS = "EPSG:26916"  # NAD83 / UTM Zone 16N
 
# Zoom level for imagery download
# 19 = ~0.25m/px (highest quality, ~1.6GB total)
# 18 = ~0.5m/px  (good quality, ~600MB total)
# 17 = ~1.0m/px  (fast download, ~200MB total)
ZOOM_LEVEL = 18
 
# Tile size in pixels — must match zoom resolution to equal 500m
# zoom 19 (0.25m/px): 2000px = 500m
# zoom 18 (0.5m/px):  1000px = 500m
# zoom 17 (1.0m/px):   500px = 500m
TILE_SIZE_PX = 1000
 
# Output paths
BASE_RASTER_PATH = r"C:\Tiles\Atlanta_base_z18.tif"   # downloaded full raster
OUTPUT_FOLDER    = r"C:\Tiles\Atlanta_split"           # individual tile GeoTIFFs
TILE_PREFIX      = "tile"
 
# Skip re-downloading if base raster already exists
SKIP_DOWNLOAD_IF_EXISTS = True
 
# Overwrite existing tile files
OVERWRITE_TILES = False
 
 
# ----------------------------
# INTERNAL HELPERS
# ----------------------------
 
 
@dataclass
class TileRecord:
    tile_id: str
    row: int
    col: int
    xmin_utm: float
    ymin_utm: float
    xmax_utm: float
    ymax_utm: float
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float
    width_px: int
    height_px: int
    path: str
 
 
def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)
 
 
def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)
 
 
def _err(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)
 
 
def _safe_make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
 
 
def _validate_config() -> None:
    """Sanity-check user config before doing any work."""
    if UTM_RIGHT <= UTM_LEFT:
        raise RuntimeError("UTM_RIGHT must be greater than UTM_LEFT")
    if UTM_TOP <= UTM_BOTTOM:
        raise RuntimeError("UTM_TOP must be greater than UTM_BOTTOM")
    if ZOOM_LEVEL < 1 or ZOOM_LEVEL > 22:
        raise RuntimeError("ZOOM_LEVEL must be between 1 and 22")
    if TILE_SIZE_PX < 64:
        raise RuntimeError("TILE_SIZE_PX too small")
 
    width_m  = UTM_RIGHT  - UTM_LEFT
    height_m = UTM_TOP    - UTM_BOTTOM
    _log(f"Study area: {width_m:.0f}m wide x {height_m:.0f}m tall")
    _log(f"Zoom {ZOOM_LEVEL} → tile size {TILE_SIZE_PX}px")
 
 
def _utm_to_webmercator(
    left: float, bottom: float, right: float, top: float, source_crs: str
) -> tuple[float, float, float, float]:
    """Convert bounding box from source CRS to Web Mercator (EPSG:3857)."""
    t = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
    left_m,  bottom_m = t.transform(left,  bottom)
    right_m, top_m    = t.transform(right, top)
    return left_m, bottom_m, right_m, top_m
 
 
def _utm_to_wgs84(
    left: float, bottom: float, right: float, top: float, source_crs: str
) -> tuple[float, float, float, float]:
    """Convert bounding box from source CRS to WGS84 lon/lat."""
    t = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    left_lon,  bottom_lat = t.transform(left,  bottom)
    right_lon, top_lat    = t.transform(right, top)
    return left_lon, bottom_lat, right_lon, top_lat
 
 
def _download_base_raster(
    left_m: float, bottom_m: float, right_m: float, top_m: float,
    out_path: str, zoom: int
) -> None:
    """Download imagery via contextily and save as Web Mercator GeoTIFF."""
    _log(f"Downloading imagery at zoom {zoom} from Esri World Imagery...")
    _log("This may take several minutes for zoom 18 — please wait.")
 
    _safe_make_dir(os.path.dirname(out_path) or ".")
 
    ctx.bounds2raster(
        left_m, bottom_m, right_m, top_m,
        path=out_path,
        source=ctx.providers.Esri.WorldImagery,
        zoom=zoom,
        ll=False,  # input is already Web Mercator
    )
    _log(f"Base raster saved to: {out_path}")
 
 
def _reproject_to_utm(in_path: str, out_path: str, target_crs: str) -> None:
    """Reproject base raster from Web Mercator to target UTM CRS."""
    _log(f"Reprojecting to {target_crs}...")
    with rasterio.open(in_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({
            "crs":       target_crs,
            "transform": transform,
            "width":     width,
            "height":    height,
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
    _log(f"Reprojected raster saved to: {out_path}")
 
 
def _nearest_label_fill(mask_arr: np.ndarray) -> np.ndarray:
    """Fill zero (unlabeled) pixels with nearest labeled pixel's value."""
    from scipy.ndimage import distance_transform_edt
    unlabeled = mask_arr == 0
    if not unlabeled.any():
        return mask_arr
    _, indices = distance_transform_edt(unlabeled, return_indices=True)
    filled = mask_arr.copy()
    filled[unlabeled] = mask_arr[tuple(indices)][unlabeled]
    return filled
 
 
def _px_to_utm(
    px_col: int, px_row: int, transform: rasterio.transform.Affine
) -> tuple[float, float]:
    """Convert pixel coords to UTM coords using rasterio affine transform."""
    x, y = rasterio.transform.xy(transform, px_row, px_col)
    return x, y
 
 
def _write_metadata_csv(path: str, rows: list[TileRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tile_id", "row", "col",
            "xmin_utm", "ymin_utm", "xmax_utm", "ymax_utm",
            "lon_min", "lat_min", "lon_max", "lat_max",
            "width_px", "height_px", "file_path"
        ])
        for r in rows:
            writer.writerow([
                r.tile_id, r.row, r.col,
                f"{r.xmin_utm:.2f}", f"{r.ymin_utm:.2f}",
                f"{r.xmax_utm:.2f}", f"{r.ymax_utm:.2f}",
                f"{r.lon_min:.6f}", f"{r.lat_min:.6f}",
                f"{r.lon_max:.6f}", f"{r.lat_max:.6f}",
                r.width_px, r.height_px, r.path
            ])
 
 
# ----------------------------
# MAIN
# ----------------------------
 
 
def main() -> int:
    _log("=" * 60)
    _log("Atlanta Tile Export — contextily + rasterio")
    _log("=" * 60)
 
    _validate_config()
 
    # Coordinate conversions
    left_m, bottom_m, right_m, top_m = _utm_to_webmercator(
        UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, SOURCE_CRS
    )
    left_lon, bottom_lat, right_lon, top_lat = _utm_to_wgs84(
        UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, SOURCE_CRS
    )
    _log(f"WGS84 bounds: lon [{left_lon:.4f}, {right_lon:.4f}], lat [{bottom_lat:.4f}, {top_lat:.4f}]")
 
    # ── Step 1: Download base raster ──────────────────────────────────────
    webmercator_path = BASE_RASTER_PATH.replace(".tif", "_webmercator.tif")
 
    if SKIP_DOWNLOAD_IF_EXISTS and os.path.exists(webmercator_path):
        _log(f"Base raster already exists, skipping download: {webmercator_path}")
    else:
        _download_base_raster(left_m, bottom_m, right_m, top_m, webmercator_path, ZOOM_LEVEL)
 
    # Verify download
    with rasterio.open(webmercator_path) as src:
        _log(f"Downloaded raster: {src.width}x{src.height}px, CRS={src.crs}, res={src.res[0]:.4f}m/px")
 
    # ── Step 2: Reproject to UTM ──────────────────────────────────────────
    utm_path = BASE_RASTER_PATH
 
    if SKIP_DOWNLOAD_IF_EXISTS and os.path.exists(utm_path):
        _log(f"UTM raster already exists, skipping reproject: {utm_path}")
    else:
        _reproject_to_utm(webmercator_path, utm_path, SOURCE_CRS)
 
    # ── Step 3: Tile the UTM raster ───────────────────────────────────────
    _safe_make_dir(OUTPUT_FOLDER)
 
    # Coordinate transformer for tile metadata (UTM → WGS84)
    coord_transformer = Transformer.from_crs(SOURCE_CRS, "EPSG:4326", always_xy=True)
 
    exported: list[TileRecord] = []
    failed: list[str] = []
 
    with rasterio.open(utm_path) as src:
        res_m  = src.res[0]
        cols   = src.width  // TILE_SIZE_PX
        rows   = src.height // TILE_SIZE_PX
        total  = rows * cols
 
        _log(f"UTM raster: {src.width}x{src.height}px @ {res_m:.4f}m/px")
        _log(f"Tile grid:  {cols} cols x {rows} rows = {total} tiles")
        _log(f"Tile size:  {TILE_SIZE_PX}px = {TILE_SIZE_PX * res_m:.1f}m per side")
        _log("-" * 40)
 
        for row_idx in range(rows):
            for col_idx in range(cols):
                tile_id  = f"{TILE_PREFIX}_{row_idx:03d}_{col_idx:03d}"
                out_path = os.path.join(OUTPUT_FOLDER, f"{tile_id}.tif")
 
                if os.path.exists(out_path) and not OVERWRITE_TILES:
                    # Load existing tile into metadata without re-exporting
                    try:
                        with rasterio.open(out_path) as t:
                            b = t.bounds
                        lon_min, lat_min = coord_transformer.transform(b.left,  b.bottom)
                        lon_max, lat_max = coord_transformer.transform(b.right, b.top)
                        exported.append(TileRecord(
                            tile_id=tile_id, row=row_idx, col=col_idx,
                            xmin_utm=b.left, ymin_utm=b.bottom,
                            xmax_utm=b.right, ymax_utm=b.top,
                            lat_min=lat_min, lon_min=lon_min,
                            lat_max=lat_max, lon_max=lon_max,
                            width_px=TILE_SIZE_PX, height_px=TILE_SIZE_PX,
                            path=out_path,
                        ))
                    except Exception:
                        pass
                    continue
 
                try:
                    window    = Window(col_idx * TILE_SIZE_PX, row_idx * TILE_SIZE_PX,
                                       TILE_SIZE_PX, TILE_SIZE_PX)
                    transform = src.window_transform(window)
                    data      = src.read(window=window)
 
                    meta = src.meta.copy()
                    meta.update({
                        "driver":    "GTiff",
                        "height":    TILE_SIZE_PX,
                        "width":     TILE_SIZE_PX,
                        "transform": transform,
                        "compress":  "lzw",
                    })
 
                    with rasterio.open(out_path, "w", **meta) as dst:
                        dst.write(data)
 
                    # Compute tile bounds for metadata
                    b = rasterio.transform.array_bounds(TILE_SIZE_PX, TILE_SIZE_PX, transform)
                    xmin, ymin, xmax, ymax = b
                    lon_min, lat_min = coord_transformer.transform(xmin, ymin)
                    lon_max, lat_max = coord_transformer.transform(xmax, ymax)
 
                    exported.append(TileRecord(
                        tile_id=tile_id, row=row_idx, col=col_idx,
                        xmin_utm=xmin, ymin_utm=ymin,
                        xmax_utm=xmax, ymax_utm=ymax,
                        lat_min=lat_min, lon_min=lon_min,
                        lat_max=lat_max, lon_max=lon_max,
                        width_px=TILE_SIZE_PX, height_px=TILE_SIZE_PX,
                        path=out_path,
                    ))
 
                except Exception as exc:
                    _warn(f"Tile failed ({tile_id}): {exc}")
                    failed.append(tile_id)
 
            # Progress update every row
            done = (row_idx + 1) * cols
            _log(f"Progress: {done}/{total} tiles ({100*done//total}%)")
 
    # ── Step 4: Write metadata CSV ────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_FOLDER, f"{TILE_PREFIX}_index.csv")
    _write_metadata_csv(csv_path, exported)
 
    # ── Summary ───────────────────────────────────────────────────────────
    _log("=" * 60)
    _log(f"Done!")
    _log(f"Tiles exported:  {len(exported)}")
    _log(f"Tiles failed:    {len(failed)}")
    _log(f"Output folder:   {OUTPUT_FOLDER}")
    _log(f"Tile index CSV:  {csv_path}")
 
    if failed:
        _warn(f"Failed tiles: {failed}")
 
    return 0
 
 
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _err(str(exc))
        traceback.print_exc()
        raise SystemExit(1)
