from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


def log_stage(message: str, start_time: float | None = None) -> None:
    if start_time is None:
        print(f"[INFO] {message}")
        return
    elapsed = perf_counter() - start_time
    print(f"[INFO] {message} ({elapsed:.2f}s)")


def save_figure_high_resolution(
    output_path,
    dpi: int = 100,
    close_figure: bool = True,
) -> None:
    fig = plt.gcf()
    ax = fig.axes[0] if fig.axes else None

    effective_dpi = max(1, int(dpi))
    # Set lower DPI for rendering to reduce memory usage on large images
    plt.rcParams["figure.dpi"] = effective_dpi

    # If a plotted image is present, match figure size to source pixel dimensions.
    if ax is not None and ax.images:
        img_arr = ax.images[0].get_array()
        if hasattr(img_arr, "shape") and len(img_arr.shape) >= 2:
            h, w = int(img_arr.shape[0]), int(img_arr.shape[1])
            if h > 0 and w > 0:
                fig.set_size_inches(w / effective_dpi, h / effective_dpi, forward=True)

    fig.savefig(
        output_path,
        dpi=effective_dpi,
        facecolor="white",
        edgecolor="none",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    if close_figure:
        plt.close(fig)


def report_geotiff_spatial_info(
    geotiff_path: str,
    use_crop: bool = False,
    lonlat_bbox: tuple[float, float, float, float] | None = None,
) -> None:
    print("[INFO] Reading GeoTIFF spatial metadata")
    with rasterio.open(geotiff_path) as src:
        crs = src.crs
        print(f"[INFO] CRS: {crs}")
        print(f"[INFO] Raster size (pixels): {src.width} x {src.height}")
        print(f"[INFO] Raster bounds (native CRS units): {src.bounds}")

        if crs is None:
            print("[WARN] No CRS found; cannot compute real-world width/height from this TIFF.")
            return

        if crs.is_projected:
            full_left, full_bottom, full_right, full_top = src.bounds
            full_width_m = abs(full_right - full_left)
            full_height_m = abs(full_top - full_bottom)
        else:
            full_left, full_bottom, full_right, full_top = transform_bounds(
                crs, "EPSG:3857", *src.bounds, densify_pts=21
            )
            full_width_m = abs(full_right - full_left)
            full_height_m = abs(full_top - full_bottom)

        full_pixel_width_m = full_width_m / max(src.width, 1)
        full_pixel_height_m = full_height_m / max(src.height, 1)

        print(
            "[INFO] Full image real size (approx meters): "
            f"{full_width_m:.2f} m x {full_height_m:.2f} m"
        )
        print(
            "[INFO] Full image pixel size (approx meters/pixel): "
            f"{full_pixel_width_m:.4f} x {full_pixel_height_m:.4f}"
        )

        if use_crop:
            if lonlat_bbox is None:
                raise ValueError("lonlat_bbox must be provided when use_crop=True")

            lon_min, lat_min, lon_max, lat_max = lonlat_bbox
            if src.crs and src.crs.to_string() != "EPSG:4326":
                req_bounds = transform_bounds("EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max)
            else:
                req_bounds = (lon_min, lat_min, lon_max, lat_max)

            ds_left, ds_bottom, ds_right, ds_top = src.bounds
            req_left, req_bottom, req_right, req_top = req_bounds
            left = max(ds_left, req_left)
            bottom = max(ds_bottom, req_bottom)
            right = min(ds_right, req_right)
            top = min(ds_top, req_top)

            if left >= right or bottom >= top:
                print("[WARN] Requested crop bbox does not overlap raster; crop size cannot be estimated.")
                return

            if crs.is_projected:
                crop_width_m = abs(right - left)
                crop_height_m = abs(top - bottom)
            else:
                c_left, c_bottom, c_right, c_top = transform_bounds(
                    crs, "EPSG:3857", left, bottom, right, top, densify_pts=21
                )
                crop_width_m = abs(c_right - c_left)
                crop_height_m = abs(c_top - c_bottom)

            print(
                "[INFO] Crop real size (approx meters): "
                f"{crop_width_m:.2f} m x {crop_height_m:.2f} m"
            )


def get_loaded_extent_meters(
    geotiff_path: str,
    use_crop: bool = False,
    lonlat_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[float, float] | None:
    with rasterio.open(geotiff_path) as src:
        crs = src.crs
        if crs is None:
            return None

        if use_crop:
            if lonlat_bbox is None:
                raise ValueError("lonlat_bbox must be provided when use_crop=True")

            lon_min, lat_min, lon_max, lat_max = lonlat_bbox
            if src.crs and src.crs.to_string() != "EPSG:4326":
                req_bounds = transform_bounds("EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max)
            else:
                req_bounds = (lon_min, lat_min, lon_max, lat_max)

            ds_left, ds_bottom, ds_right, ds_top = src.bounds
            req_left, req_bottom, req_right, req_top = req_bounds
            left = max(ds_left, req_left)
            bottom = max(ds_bottom, req_bottom)
            right = min(ds_right, req_right)
            top = min(ds_top, req_top)

            if left >= right or bottom >= top:
                return None
        else:
            left, bottom, right, top = src.bounds

        if crs.is_projected:
            width_m = abs(right - left)
            height_m = abs(top - bottom)
        else:
            m_left, m_bottom, m_right, m_top = transform_bounds(
                crs, "EPSG:3857", left, bottom, right, top, densify_pts=21
            )
            width_m = abs(m_right - m_left)
            height_m = abs(m_top - m_bottom)

        return width_m, height_m


def load_rgb_image(
    geotiff_path: str,
    use_crop: bool = False,
    lonlat_bbox: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    stage_start = perf_counter()
    print(f"[INFO] Reading GeoTIFF: {geotiff_path}")

    with rasterio.open(geotiff_path) as src:
        if use_crop:
            if lonlat_bbox is None:
                raise ValueError("lonlat_bbox must be provided when use_crop=True")

            lon_min, lat_min, lon_max, lat_max = lonlat_bbox
            if src.crs and src.crs.to_string() != "EPSG:4326":
                data_bounds = transform_bounds("EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max)
            else:
                data_bounds = (lon_min, lat_min, lon_max, lat_max)

            ds_left, ds_bottom, ds_right, ds_top = src.bounds
            req_left, req_bottom, req_right, req_top = data_bounds
            left = max(ds_left, req_left)
            bottom = max(ds_bottom, req_bottom)
            right = min(ds_right, req_right)
            top = min(ds_top, req_top)

            if left >= right or bottom >= top:
                raise RuntimeError(
                    "Requested bbox does not overlap this GeoTIFF after CRS conversion. "
                    "Use bounds within the raster extent."
                )

            window = from_bounds(left, bottom, right, top, transform=src.transform)
            rgb = src.read([1, 2, 3], window=window).transpose(1, 2, 0)
            log_stage("Finished reading cropped raster window", stage_start)
            return rgb

        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        log_stage("Finished reading full raster image", stage_start)
        return rgb


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max == img_min:
        raise RuntimeError("Cropped image has constant pixel values; cannot normalize display range.")
    img -= img_min
    img *= 255.0 / (img_max - img_min)
    np.clip(img, 0.0, 255.0, out=img)
    return img.astype(np.uint8)


def normalize_to_uint8_robust(
    img: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    if high_percentile <= low_percentile:
        raise ValueError("high_percentile must be greater than low_percentile")

    arr = img.astype(np.float32, copy=False)
    if arr.ndim == 2:
        arr = arr[..., None]

    out = np.empty(arr.shape, dtype=np.uint8)
    for c in range(arr.shape[2]):
        channel = arr[..., c]
        lo = float(np.percentile(channel, low_percentile))
        hi = float(np.percentile(channel, high_percentile))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            cmin = float(channel.min())
            cmax = float(channel.max())
            lo, hi = cmin, cmax

        if hi <= lo:
            out[..., c] = 0
            continue

        scaled = (channel - lo) * (255.0 / (hi - lo))
        np.clip(scaled, 0.0, 255.0, out=scaled)
        out[..., c] = scaled.astype(np.uint8)

    return out[..., 0] if img.ndim == 2 else out


def build_amenity_heatmap(
    amenity_mask: np.ndarray,
    image_shape: tuple[int, int, int],
    extent_meters: tuple[float, float],
    cell_area_m2: float,
    taper_sigma_cells: float = 0.90,
    taper_blend: float = 0.75,
) -> tuple[np.ndarray, int, int, float]:
    if cell_area_m2 <= 0:
        raise ValueError("cell_area_m2 must be > 0")

    h, w = image_shape[:2]
    width_m, height_m = extent_meters

    meters_per_pixel_x = width_m / max(w, 1)
    meters_per_pixel_y = height_m / max(h, 1)
    side_m = float(np.sqrt(cell_area_m2))

    cell_px_w = max(1, int(round(side_m / max(meters_per_pixel_x, 1e-9))))
    cell_px_h = max(1, int(round(side_m / max(meters_per_pixel_y, 1e-9))))

    heatmap = np.zeros((h, w), dtype=np.float32)
    for y0 in range(0, h, cell_px_h):
        y1 = min(h, y0 + cell_px_h)
        for x0 in range(0, w, cell_px_w):
            x1 = min(w, x0 + cell_px_w)
            cell = amenity_mask[y0:y1, x0:x1]
            density = float(cell.mean()) if cell.size > 0 else 0.0
            heatmap[y0:y1, x0:x1] = density

    # Add spatial taper so high-amenity areas softly influence nearby cells.
    if gaussian_filter is not None and float(taper_sigma_cells) > 0.0:
        sigma_y = max(0.5, float(cell_px_h) * float(taper_sigma_cells))
        sigma_x = max(0.5, float(cell_px_w) * float(taper_sigma_cells))
        smoothed = gaussian_filter(heatmap.astype(np.float32), sigma=(sigma_y, sigma_x), mode="nearest")
        max_smoothed = float(smoothed.max())
        if max_smoothed > 0.0:
            smoothed = smoothed / max_smoothed
        blend = float(np.clip(taper_blend, 0.0, 1.0))
        heatmap = (1.0 - blend) * heatmap + blend * smoothed

    np.clip(heatmap, 0.0, 1.0, out=heatmap)

    return heatmap, cell_px_w, cell_px_h, side_m
