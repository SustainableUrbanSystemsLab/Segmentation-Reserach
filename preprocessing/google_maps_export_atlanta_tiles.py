import io
import math
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ============================================================
# USER CONFIG
# ============================================================

UTM_LEFT   = 738299.3766
UTM_BOTTOM = 3735629.2001
UTM_RIGHT  = 744953.2947
UTM_TOP    = 3743524.6855

FINAL_GRID_COLS = 7
FINAL_GRID_ROWS = 8

ZOOM_LEVEL  = 20
MAP_SIZE_PX = 640  # 640x640 base size
MAP_SCALE   = 2    # scale=2 gives 1280x1280 raw pixels

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = PROJECT_ROOT / "Maps" / "Tiles" / "Atlanta_split_google"
CHUNK_CACHE   = OUTPUT_FOLDER / f"_native_pixel_cache_z{ZOOM_LEVEL}"

OVERWRITE = False

REQUEST_TIMEOUT_S = 60
RETRY_COUNT = 3
RETRY_SLEEP_S = 5
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()

if not GOOGLE_MAPS_API_KEY:
    raise RuntimeError(
        "GOOGLE_MAPS_API_KEY is not set. Export it in your shell or place it in a local .env file before running this script."
    )

# ============================================================
# GOOGLE PILE PROJECTION UNIT SYSTEM MATH
# ============================================================

SR_WEB_MERCATOR = CRS.from_epsg(3857)

RAW_CHUNK_PX = MAP_SIZE_PX * MAP_SCALE  # 1280 px

# Google coverage is based on the unscaled size; scale only increases returned pixels.
RAW_CHUNK_WORLD_PX = MAP_SIZE_PX

# Fixed clean crop size. This is the stitched pixel footprint for every chunk.
CROP_SIZE_PX = 1000

CROP_SIZE_WORLD_PX = CROP_SIZE_PX / MAP_SCALE

# Google requests are still 1280x1280, but the stitched grid advances by the
# cropped footprint so every cleaned chunk abuts exactly with no overlap.
CHUNK_STEP_WORLD_PX = CROP_SIZE_WORLD_PX

# Keep the crop anchored at the top-left so the bottom/right logo margin is removed.
CROP_X0 = 0
CROP_Y0 = 0
CROP_X1 = CROP_X0 + CROP_SIZE_PX
CROP_Y1 = CROP_Y0 + CROP_SIZE_PX

# Raw request centers use the unscaled coverage footprint; output placement uses scaled pixels.
REQUEST_CENTER_OFFSET_WORLD_PX = RAW_CHUNK_WORLD_PX / 2.0
REQUEST_CENTER_OFFSET_PX = RAW_CHUNK_PX // 2


def world_to_output_px(value: float) -> float:
    return value * MAP_SCALE


def output_to_world_px(value: float) -> float:
    return value / MAP_SCALE

def utm_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """Converts local UTM coordinates to WGS84 Latitude/Longitude."""
    t = Transformer.from_crs("EPSG:26916", "EPSG:4326", always_xy=True)
    return t.transform(x, y)

def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Converts WGS84 Latitude/Longitude to Web Mercator meters."""
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return t.transform(lon, lat)

def lonlat_to_google_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Maps a global coordinate directly to Google's continuous worldwide pixel canvas."""
    x = (lon + 180.0) / 360.0 * (256 << zoom)
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * (256 << zoom)
    return x, y

def google_pixel_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Maps a position on Google's worldwide pixel canvas back to Latitude/Longitude."""
    lon = x / (256 << zoom) * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / (256 << zoom)
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat

# ============================================================
# RESILIENT API NETWORK MANAGER
# ============================================================

def request_static_map(center_lat: float, center_lng: float) -> Image.Image:
    params = {
        "center": f"{center_lat},{center_lng}",
        "zoom": str(ZOOM_LEVEL),
        "size": f"{MAP_SIZE_PX}x{MAP_SIZE_PX}",
        "scale": str(MAP_SCALE),
        "maptype": "satellite",
        "format": "png",
        "key": GOOGLE_MAPS_API_KEY,
    }
    url = f"https://maps.googleapis.com/maps/api/staticmap?{urlencode(params)}"
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=REQUEST_TIMEOUT_S) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        except (HTTPError, URLError, TimeoutError, OSError):
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(f"Google Maps API failed at location: {center_lat}, {center_lng}")

def get_clean_cropped_chunk(cx_px: float, cy_px: float) -> Image.Image:
    """Fetches a raw chunk and returns the fixed stitched crop."""
    cache_path = CHUNK_CACHE / f"gpx_{cx_px:.1f}_{cy_px:.1f}.png"
    
    if cache_path.exists():
        try:
            raw_img = Image.open(cache_path).convert("RGB")
        except Exception:
            try: cache_path.unlink()
            except OSError: pass
            raw_img = None
    else:
        raw_img = None

    if raw_img is None:
        # Resolve the absolute Google canvas pixel center to Lat/Lon coordinates
        lon, lat = google_pixel_to_lonlat(cx_px, cy_px, ZOOM_LEVEL)
        raw_img = request_static_map(lat, lon)
        CHUNK_CACHE.mkdir(parents=True, exist_ok=True)
        raw_img.save(cache_path)

    # Crop the same square footprint from every raw request so chunk boundaries stay aligned.
    return raw_img.crop((CROP_X0, CROP_Y0, CROP_X1, CROP_Y1))

def paste_chunk_overlap(
    tile_canvas: Image.Image,
    chunk_image: Image.Image,
    chunk_x0: float,
    chunk_y0: float,
    tile_x0: float,
    tile_y0: float,
    tile_x1: float,
    tile_y1: float,
) -> None:
    """Pastes only the overlap between a chunk footprint and the target tile."""
    overlap_x0 = max(tile_x0, chunk_x0)
    overlap_y0 = max(tile_y0, chunk_y0)
    overlap_x1 = min(tile_x1, chunk_x0 + chunk_image.width)
    overlap_y1 = min(tile_y1, chunk_y0 + chunk_image.height)

    if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
        return

    src_x0 = int(overlap_x0 - chunk_x0)
    src_y0 = int(overlap_y0 - chunk_y0)
    src_x1 = int(overlap_x1 - chunk_x0)
    src_y1 = int(overlap_y1 - chunk_y0)

    dst_x0 = int(overlap_x0 - tile_x0)
    dst_y0 = int(overlap_y0 - tile_y0)

    tile_canvas.paste(chunk_image.crop((src_x0, src_y0, src_x1, src_y1)), (dst_x0, dst_y0))

# ============================================================
# PROCESSING PIPELINE
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CHUNK_CACHE, exist_ok=True)

    # Convert extreme study area bounds into global coordinates
    lon_min, lat_min = utm_to_lonlat(UTM_LEFT, UTM_BOTTOM)
    lon_max, lat_max = utm_to_lonlat(UTM_RIGHT, UTM_TOP)

    # Pin down your entire operational area bounds onto Google's worldwide pixel grid
    p_x_min, p_y_max = lonlat_to_google_pixel(lon_min, lat_min, ZOOM_LEVEL)
    p_x_max, p_y_min = lonlat_to_google_pixel(lon_max, lat_max, ZOOM_LEVEL)

    print("=" * 80)
    print(" GOOGLE UNIT SYSTEM ENGINE: BALANCED SEAMLESS CHUNK MAPPING ")
    print("=" * 80)
    print(f"Global Pixel Window: X({int(p_x_min)} to {int(p_x_max)}), Y({int(p_y_min)} to {int(p_y_max)})")
    print(f"Every chunk is cropped down to an identical, clean: {CROP_SIZE_PX}x{CROP_SIZE_PX} px frame.")
    print("-" * 80)

    total_tiles = FINAL_GRID_COLS * FINAL_GRID_ROWS
    done_tiles = 0
    start_time = time.time()

    # Step through every targeted macro-tile grid segment independently
    for r in range(FINAL_GRID_ROWS):
        for c in range(FINAL_GRID_COLS):
            tile_id = f"tile_{r:03d}_{c:03d}"
            out_path = OUTPUT_FOLDER / f"{tile_id}.tif"

            # Compute the clean bounding window boundaries for this specific output tile in Google pixels
            t_px_x0 = p_x_min + (p_x_max - p_x_min) * c / FINAL_GRID_COLS
            t_px_x1 = p_x_min + (p_x_max - p_x_min) * (c + 1) / FINAL_GRID_COLS
            t_px_y0 = p_y_min + (p_y_max - p_y_min) * r / FINAL_GRID_ROWS
            t_px_y1 = p_y_min + (p_y_max - p_y_min) * (r + 1) / FINAL_GRID_ROWS

            if not OVERWRITE and out_path.exists():
                done_tiles += 1
                continue

            print(f"[PROCESSING TILE] Building independent frame for {tile_id}...")

            # Calculate where the first raw request center must align so the cropped blocks
            # land on the fixed stitched grid.
            start_cx = p_x_min + (int((t_px_x0 - p_x_min) // CHUNK_STEP_WORLD_PX) * CHUNK_STEP_WORLD_PX) + REQUEST_CENTER_OFFSET_WORLD_PX
            start_cy = p_y_min + (int((t_px_y0 - p_y_min) // CHUNK_STEP_WORLD_PX) * CHUNK_STEP_WORLD_PX) + REQUEST_CENTER_OFFSET_WORLD_PX

            cx_list = []
            curr_x = start_cx
            while curr_x < t_px_x1 + REQUEST_CENTER_OFFSET_WORLD_PX:
                cx_list.append(curr_x)
                curr_x += CHUNK_STEP_WORLD_PX

            cy_list = []
            curr_y = start_cy
            while curr_y < t_px_y1 + REQUEST_CENTER_OFFSET_WORLD_PX:
                cy_list.append(curr_y)
                curr_y += CHUNK_STEP_WORLD_PX

            tile_x0_px = world_to_output_px(t_px_x0)
            tile_y0_px = world_to_output_px(t_px_y0)
            tile_x1_px = world_to_output_px(t_px_x1)
            tile_y1_px = world_to_output_px(t_px_y1)

            tile_width_px = max(1, int(math.ceil(tile_x1_px - tile_x0_px)))
            tile_height_px = max(1, int(math.ceil(tile_y1_px - tile_y0_px)))

            # Build the tile directly at its final pixel size.
            local_canvas = Image.new("RGB", (tile_width_px, tile_height_px))

            for chk_r, cy in enumerate(cy_list):
                for chk_c, cx in enumerate(cx_list):
                    clean_block = get_clean_cropped_chunk(cx, cy)
                    chunk_x0 = world_to_output_px(cx - REQUEST_CENTER_OFFSET_WORLD_PX)
                    chunk_y0 = world_to_output_px(cy - REQUEST_CENTER_OFFSET_WORLD_PX)
                    paste_chunk_overlap(
                        local_canvas,
                        clean_block,
                        chunk_x0,
                        chunk_y0,
                        tile_x0_px,
                        tile_y0_px,
                        tile_x1_px,
                        tile_y1_px,
                    )

            final_tile_image = local_canvas

            # Convert the tile's pixel coordinate boundaries back to spatial Web Mercator meters
            t_lon_0, t_lat_0 = google_pixel_to_lonlat(t_px_x0, t_px_y0, ZOOM_LEVEL)
            t_lon_1, t_lat_1 = google_pixel_to_lonlat(t_px_x1, t_px_y1, ZOOM_LEVEL)
            
            wm_x0, wm_y1 = lonlat_to_mercator(t_lon_0, t_lat_0)
            wm_x1, wm_y0 = lonlat_to_mercator(t_lon_1, t_lat_1)

            # Commit the final tile directly to a georeferenced GeoTIFF
            arr = np.asarray(final_tile_image, dtype=np.uint8)
            transform = from_bounds(wm_x0, wm_y0, wm_x1, wm_y1, final_tile_image.width, final_tile_image.height)

            with rasterio.open(
                out_path, "w", driver="GTiff",
                height=final_tile_image.height, width=final_tile_image.width,
                count=3, dtype=arr.dtype, crs=SR_WEB_MERCATOR,
                transform=transform, compress="deflate", tiled=True
            ) as dst:
                dst.write(arr.transpose(2, 0, 1))

            done_tiles += 1
            elapsed = time.time() - start_time
            remaining = (elapsed / done_tiles) * (total_tiles - done_tiles)
            print(f"--- [COMMITTED] {tile_id}.tif Size: {final_tile_image.width}x{final_tile_image.height} px | Remaining ETA: {time.strftime('%H:%M:%S', time.gmtime(remaining))}")

    print(f"\nProcessing complete! All tiles saved inside: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()