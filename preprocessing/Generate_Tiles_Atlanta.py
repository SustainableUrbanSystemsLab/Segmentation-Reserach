import os

import contextily as ctx
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

# Your study area bounds (from your fishnet extent in UTM Zone 16N)
left, bottom, right, top = 738299.3766, 3735629.2001, 744953.2947, 3743524.6855

# Convert UTM to Web Mercator (EPSG:3857) for contextily
transformer = Transformer.from_crs("EPSG:26916", "EPSG:3857", always_xy=True)
left_m, bottom_m = transformer.transform(left, bottom)
right_m, top_m = transformer.transform(right, top)

def download_base_raster(bounds_m, output_base, zoom_levels):
    left_m, bottom_m, right_m, top_m = bounds_m
    last_error = None

    for zoom in zoom_levels:
        try:
            print(f"Downloading base imagery at zoom {zoom}...")
            ctx.bounds2raster(
                left_m,
                bottom_m,
                right_m,
                top_m,
                path=output_base,
                source=ctx.providers.Esri.WorldImagery,
                zoom=zoom,
            )
            return zoom
        except Exception as exc:
            last_error = exc
            print(f"Zoom {zoom} failed: {exc}")

    raise RuntimeError(f"Unable to download imagery at any zoom level: {last_error}")


# Download imagery (project-local Maps folder)
maps_dir = os.path.join(os.getcwd(), "Maps", "Atlanta(Tiles)")
os.makedirs(maps_dir, exist_ok=True)
output_base = os.path.join(maps_dir, "Atlanta_base.tif")

used_zoom = download_base_raster(
    (left_m, bottom_m, right_m, top_m),
    output_base,
    zoom_levels=(20, 19, 18, 17),
)

print(f"Base raster saved at zoom {used_zoom}. Now tiling...")

# Tile into 1000x1000px chunks (project-local)
output_dir = os.path.join(maps_dir, "Atlanta_split")
os.makedirs(output_dir, exist_ok=True)

with rasterio.open(output_base) as src:
    tile_px = 2000
    cols = src.width // tile_px
    rows = src.height // tile_px
    print(f"Generating {rows * cols} tiles ({rows} rows x {cols} cols)...")

    for row in range(rows):
        for col in range(cols):
            window = Window(col * tile_px, row * tile_px, tile_px, tile_px)
            data = src.read(window=window)
            transform = src.window_transform(window)

            out_path = os.path.join(output_dir, f"tile_{row:03d}_{col:03d}.tif")
            with rasterio.open(
                out_path, 'w',
                driver='GTiff',
                height=tile_px, width=tile_px,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=transform
            ) as dst:
                dst.write(data)

print(f"Done! {rows * cols} tiles saved to {output_dir}")