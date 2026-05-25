import io
import os
import math
import time
from pathlib import Path
import numpy as np
import requests
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

# ----------------------------
# USER CONFIG
# ----------------------------
# DO NOT CHANGE - Fixed Study Area Bounds (UTM 16N)
UTM_LEFT   = 738299.3766
UTM_BOTTOM = 3735629.2001
UTM_RIGHT  = 744953.2947
UTM_TOP    = 3743524.6855

SR_UTM = CRS.from_epsg(26916)
SR_WEB_MERCATOR = CRS.from_epsg(3857)
SR_WGS84 = CRS.from_epsg(4326)

OUTPUT_FOLDER = Path(r"C:\Tiles\Atlanta_split")
TILE_PREFIX   = "tile"
TILE_CACHE_FOLDER = OUTPUT_FOLDER / "_source_tile_cache"

# RESOLUTION & GRID CONFIG
ZOOM_LEVEL = 19         # Zoom 19 provides crisp ~0.25m-0.29m native engineering-grade pixels
FINAL_GRID_ROWS = 8     # 8 Rows
FINAL_GRID_COLS = 7     # 7 Columns

# High-Quality USGS Orthoimagery XYZ Tile endpoint
TILE_URL = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
headers = {"User-Agent": "Mozilla/5.0"}

def log(msg):
    print(f"[INFO] {msg}")

def project_bounds(left, bottom, right, top, src_crs, dst_crs):
    return transform_bounds(src_crs, dst_crs, left, bottom, right, top)

def lonlat_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_to_lonlat(x, y, zoom):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lon_deg, lat_deg

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    TILE_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # 1. Convert UTM bounds to Lat/Lon for download grid calculations
    lon_min, lat_min, lon_max, lat_max = project_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, SR_UTM, SR_WGS84)
    
    # 2. Identify web map tiles required to completely cover the study area
    x_start, y_start = lonlat_to_tile(lon_min, lat_max, ZOOM_LEVEL)
    x_end, y_end = lonlat_to_tile(lon_max, lat_min, ZOOM_LEVEL)
    
    total_x = x_end - x_start + 1
    total_y = y_end - y_start + 1
    total_tiles = total_x * total_y
    
    log(f"Downloading {total_tiles} high-res source chunks to assemble canvas...")
    
    # 3. Stitch source chunks into a temporary high-res master canvas
    canvas_cache_path = OUTPUT_FOLDER / f"stitched_z{ZOOM_LEVEL}_x{x_start}_y{y_start}_to_x{x_end}_y{y_end}.png"
    if canvas_cache_path.exists():
        canvas = Image.open(canvas_cache_path).convert("RGB")
        log(f"Loaded stitched source canvas cache: {canvas_cache_path}")
    else:
        canvas = Image.new("RGB", (total_x * 256, total_y * 256))

        def load_or_fetch_tile(x: int, y: int) -> Image.Image:
            cache_path = TILE_CACHE_FOLDER / f"z{ZOOM_LEVEL}_x{x}_y{y}.png"
            if cache_path.exists():
                try:
                    return Image.open(cache_path).convert("RGB")
                except Exception:
                    try:
                        cache_path.unlink()
                    except OSError:
                        pass

            url = TILE_URL.format(z=ZOOM_LEVEL, x=x, y=y)
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                tile_img = Image.open(io.BytesIO(r.content)).convert("RGB")
                tile_img.save(cache_path, format="PNG")
                return tile_img

            return Image.new("RGB", (256, 256), (128, 128, 128))
    
        count = 0
        for i, x in enumerate(range(x_start, x_end + 1)):
            for j, y in enumerate(range(y_start, y_end + 1)):
                count += 1
                if count % 20 == 0 or count == total_tiles:
                    log(f"Fetching chunk {count}/{total_tiles}...")

                try:
                    tile_img = load_or_fetch_tile(x, y)
                except Exception:
                    tile_img = Image.new("RGB", (256, 256), (128, 128, 128))
                canvas.paste(tile_img, (i * 256, j * 256))

        canvas.save(canvas_cache_path)
        log(f"Saved stitched source canvas cache: {canvas_cache_path}")

    # Calculate exact Web Mercator geographic bounds of our master stitched canvas
    lon_west, lat_north = tile_to_lonlat(x_start, y_start, ZOOM_LEVEL)
    lon_east, lat_south = tile_to_lonlat(x_end + 1, y_end + 1, ZOOM_LEVEL)
    canvas_wm_left, canvas_wm_bottom, canvas_wm_right, canvas_wm_top = project_bounds(
        lon_west, lat_south, lon_east, lat_north, SR_WGS84, SR_WEB_MERCATOR
    )
    
    # Calculate Web Mercator bounds of your target study area
    target_wm_left, target_wm_bottom, target_wm_right, target_wm_top = project_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, SR_UTM, SR_WEB_MERCATOR)
    
    # Map geographical bounds directly to pixel locations on our canvas
    canvas_px_w = canvas.width
    canvas_px_h = canvas.height
    wm_width = canvas_wm_right - canvas_wm_left
    wm_height = canvas_wm_top - canvas_wm_bottom
    
    crop_x0 = int(round((target_wm_left - canvas_wm_left) / wm_width * canvas_px_w))
    crop_y0 = int(round((canvas_wm_top - target_wm_top) / wm_height * canvas_px_h))
    crop_x1 = int(round((target_wm_right - canvas_wm_left) / wm_width * canvas_px_w))
    crop_y1 = int(round((canvas_wm_top - target_wm_bottom) / wm_height * canvas_px_h))
    
    # Crop down precisely to the requested study area boundary
    study_area_image = canvas.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    
    # 4. Slice the clean study area into the required 8x7 grid
    log(f"Slicing crisp study area into an {FINAL_GRID_ROWS}x{FINAL_GRID_COLS} output tile grid...")
    
    sa_w, sa_h = study_area_image.width, study_area_image.height
    tile_w_px = sa_w / FINAL_GRID_COLS
    tile_h_px = sa_h / FINAL_GRID_ROWS
    
    wm_step_x = (target_wm_right - target_wm_left) / FINAL_GRID_COLS
    wm_step_y = (target_wm_top - target_wm_bottom) / FINAL_GRID_ROWS
    
    for r in range(FINAL_GRID_ROWS):
        for c in range(FINAL_GRID_COLS):
            tile_id = f"{TILE_PREFIX}_{r:02d}_{c:02d}.tif"
            tile_path = OUTPUT_FOLDER / tile_id
            
            # Pixel cropping boundaries
            px_l = int(round(c * tile_w_px))
            px_t = int(round(r * tile_h_px))
            px_r = int(round((c + 1) * tile_w_px))
            px_b = int(round((r + 1) * tile_h_px))
            
            tile_img = study_area_image.crop((px_l, px_t, px_r, px_b))
            
            # Geographic positioning for this specific grid piece
            t_wm_left   = target_wm_left + (c * wm_step_x)
            t_wm_top    = target_wm_top - (r * wm_step_y)
            t_wm_right  = target_wm_left + ((c + 1) * wm_step_x)
            t_wm_bottom = target_wm_top - ((r + 1) * wm_step_y)
            
            # Convert to numpy and export as georeferenced GeoTIFF
            arr = np.asarray(tile_img, dtype=np.uint8)
            transform = from_bounds(t_wm_left, t_wm_bottom, t_wm_right, t_wm_top, tile_img.width, tile_img.height)
            
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                height=tile_img.height,
                width=tile_img.width,
                count=3,
                dtype=arr.dtype,
                crs=SR_WEB_MERCATOR,
                transform=transform,
                compress="lzw",
                tiled=True,
            ) as dst:
                dst.write(arr.transpose(2, 0, 1))
                
    print(f"[INFO] Process Complete! 56 perfectly crisp high-res tiles generated in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()