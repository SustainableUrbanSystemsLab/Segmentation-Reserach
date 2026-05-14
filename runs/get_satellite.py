from datetime import datetime
import faulthandler
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use non-interactive Agg backend to avoid pixmap allocation errors on large images
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from models import config as cfg

faulthandler.enable(all_threads=True)

# Optional, targeted suppression for known low-risk third-party warnings.
if getattr(cfg, "dino_suppress_low_risk_warnings", False):
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.layers is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`resume_download` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The `device` argument is deprecated and will be removed in v5 of Transformers\..*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"None of the inputs have requires_grad=True\. Gradients will be None",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
        category=FutureWarning,
    )
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

from models.dino_processing import (
    build_dino_model_and_transform,
    iter_tile_coords,
    run_dino_prompts,
    save_dino_detection_viz,
    save_dino_detection_viz_pil,
)
from visualizations.dino_visualizations import (
    save_per_prompt_breakdown,
    save_filtering_stage_comparison,
    save_detection_heatmap,
    save_box_size_distribution,
)
from models.image_processing import (
    build_amenity_heatmap,
    get_loaded_extent_meters,
    load_rgb_image,
    log_stage,
    normalize_to_uint8,
    normalize_to_uint8_robust,
    report_geotiff_spatial_info,
    save_figure_high_resolution,
)
from models.sam_processing import (
    build_sam_predictor,
    generate_sam_masks_from_detections,
    generate_sam_masks_automatic,
    generate_sam_masks_automatic_tiled,
    load_sam_model,
    ensure_checkpoint,
)


script_start = perf_counter()
log_stage("Starting satellite sidewalk segmentation pipeline")


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    total_safe = max(1, int(total))
    current_clamped = min(max(0, int(current)), total_safe)
    filled = int(round((current_clamped / total_safe) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


batch_index_env = os.environ.get("SEGMENTATION_BATCH_INDEX")
batch_total_env = os.environ.get("SEGMENTATION_BATCH_TOTAL")
batch_index = int(batch_index_env) if batch_index_env and batch_index_env.isdigit() else 1
batch_total = int(batch_total_env) if batch_total_env and batch_total_env.isdigit() else 1


def log_image_stage(stage_name: str, stage_idx: int | None = None, stage_total: int | None = None) -> None:
    if stage_idx is None or stage_total is None:
        print(
            f"[PROGRESS] [Image {batch_index}/{batch_total}] "
            f"{_progress_bar(batch_index, batch_total)} {stage_name}"
        )
        return

    print(
        f"[PROGRESS] [Image {batch_index}/{batch_total}] "
        f"{_progress_bar(batch_index, batch_total)} "
        f"[Stage {stage_idx}/{stage_total}] {stage_name}"
    )

override_tif_file = os.environ.get("SEGMENTATION_TIF_FILE")
if override_tif_file:
    active_tif_file = override_tif_file
else:
    configured_tif_files = getattr(cfg, "tif_files", None)
    if configured_tif_files is None:
        configured_tif_files = cfg.tif_file

    if isinstance(configured_tif_files, (str, Path)):
        tif_files = [str(configured_tif_files)]
    else:
        tif_files = [str(p) for p in configured_tif_files]

    if not tif_files:
        raise RuntimeError("No tif files configured. Set cfg.tif_files or cfg.tif_file.")

    tif_run_mode = str(getattr(cfg, "tif_run_mode", "all")).strip().lower()
    if tif_run_mode not in {"all", "single"}:
        raise RuntimeError("Invalid cfg.tif_run_mode. Expected 'all' or 'single'.")

    if tif_run_mode == "single":
        selected_tif_file = getattr(cfg, "tif_single_file", None)
        if not selected_tif_file:
            legacy_tif_file = getattr(cfg, "tif_file", None)
            if isinstance(legacy_tif_file, (str, Path)):
                selected_tif_file = str(legacy_tif_file)

        if selected_tif_file:
            tif_files = [str(selected_tif_file)]
        else:
            tif_single_index = int(getattr(cfg, "tif_single_index", 0))
            if tif_single_index < 0 or tif_single_index >= len(tif_files):
                raise RuntimeError(
                    f"cfg.tif_single_index={tif_single_index} is out of range for "
                    f"{len(tif_files)} configured files"
                )
            tif_files = [tif_files[tif_single_index]]

        print(f"[INFO] Single-image mode enabled: {tif_files[0]}")

    if len(tif_files) > 1:
        print(f"[INFO] Running batch pipeline on {len(tif_files)} images")
        print(f"[INFO] Batch progress: {_progress_bar(0, len(tif_files))} 0/{len(tif_files)} complete")
        failures: list[tuple[str, int]] = []
        successes = 0
        for idx, tif_path in enumerate(tif_files, start=1):
            print(f"[INFO] [{idx}/{len(tif_files)}] Starting image: {tif_path}")
            env = os.environ.copy()
            env["SEGMENTATION_TIF_FILE"] = tif_path
            env["SEGMENTATION_BATCH_INDEX"] = str(idx)
            env["SEGMENTATION_BATCH_TOTAL"] = str(len(tif_files))
            result = subprocess.run([sys.executable, __file__], env=env)
            if result.returncode != 0:
                failures.append((tif_path, int(result.returncode)))
                print(
                    f"[WARN] [{idx}/{len(tif_files)}] Failed image: {tif_path} "
                    f"(exit code {result.returncode}); skipping and continuing"
                )
            else:
                successes += 1
                print(f"[INFO] [{idx}/{len(tif_files)}] Completed image: {tif_path}")

            remaining = len(tif_files) - idx
            print(
                f"[INFO] Batch progress: {_progress_bar(idx, len(tif_files))} "
                f"{idx}/{len(tif_files)} complete, {remaining} remaining"
            )

        if failures:
            print("[WARN] Batch run finished with failures:")
            for tif_path, code in failures:
                print(f"[WARN] - {tif_path} (exit code {code})")

        print(
            f"[INFO] Batch run summary: {successes} succeeded, "
            f"{len(failures)} failed, {len(tif_files)} total"
        )
        # Return success if at least one image completed; this preserves successful outputs.
        sys.exit(0 if successes > 0 else 1)

    active_tif_file = tif_files[0]

run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(active_tif_file).stem}"
cfg.results_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Saving output figures to: {cfg.results_dir.resolve()}")
saved_figure_paths: list[Path] = []

# Save figures at native raster pixel resolution when possible.
DEFAULT_SAVE_DPI = int(getattr(cfg, "output_dpi", 100))
DINO_SAVE_DPI = int(getattr(cfg, "dino_visualization_dpi", 75))

# Function for consistent figure saving with logging and tracking of saved paths.
def save_current_figure(filename: str, category: str, dpi: int | None = None, pil_img=None) -> None:
    category_dir = cfg.results_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    output_path = category_dir / filename
   
    if pil_img is not None:
        # Save PIL image directly
        pil_img.save(str(output_path), "PNG", compress_level=6)
        saved_figure_paths.append(output_path)
        print(f"[INFO] Saved figure (PIL): {output_path}")
    else:
        # Save matplotlib figure
        use_dpi = dpi if dpi is not None else DEFAULT_SAVE_DPI
        save_figure_high_resolution(output_path, dpi=use_dpi, close_figure=True)
        saved_figure_paths.append(output_path)
        print(f"[INFO] Saved figure: {output_path}")


def log_pipeline_error(context: str, exc: Exception) -> None:
    print(f"[ERROR] {context}: {exc}", flush=True)
    import traceback
    traceback.print_exc()


import pickle

def get_cache_dir() -> Path:
    """Return cache directory, creating it if needed."""
    cache_dir = cfg.results_dir / ".segmentation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key_for_image_dino(image_path: str, image_hash: str) -> str:
    """Generate cache key for full-image DINO results."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_dino_full.pkl"


def get_cache_key_for_image_masks(image_path: str, image_hash: str) -> str:
    """Generate cache key for full-image SAM masks."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_masks_full.pkl"


def get_cache_key_for_tile_dino(image_path: str, image_hash: str, tile_idx: int, tile_count: int) -> str:
    """Generate cache key for tiled DINO results."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_tile{tile_idx}-of-{tile_count}_dino.pkl"


def get_cache_key_for_tile_masks(image_path: str, image_hash: str, tile_idx: int, tile_count: int) -> str:
    """Generate cache key for tiled SAM masks."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_tile{tile_idx}-of-{tile_count}_masks.pkl"


def compute_image_hash(img_array: np.ndarray) -> str:
    """Compute a quick hash of image array to detect if image changed."""
    import hashlib
    # Only hash shape and first/last pixel values to keep it fast
    hash_input = f"{img_array.shape}_{img_array.dtype}_{img_array[0,0].tobytes()}_{img_array[-1,-1].tobytes()}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]


def save_dino_cache(image_path: str, image_hash: str, dino_records: list | None, file_context: str) -> bool:
    """Save DINO results to cache. Returns True if saved, False otherwise."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return False
    if dino_records is None:
        return False
    cache_empty_dino = bool(getattr(cfg, "cache_empty_dino_results", False))
    if len(dino_records) == 0 and not cache_empty_dino:
        print("[INFO] Skipping DINO cache write for empty result set")
        return False
    
    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        with open(cache_file, "wb") as f:
            pickle.dump({"dino_records": dino_records}, f)
        print(f"[INFO] Cached DINO results: {cache_file.name}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to save DINO cache: {e}")
        return False


def load_dino_cache(image_path: str, image_hash: str, file_context: str) -> list | None:
    """Load DINO results from cache. Returns None if not found or caching disabled."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return None
    if bool(getattr(cfg, "overwrite_pipeline_cache", False)):
        return None
    
    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        if not cache_file.exists():
            return None
        
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        cached_records = data.get("dino_records")
        cache_empty_dino = bool(getattr(cfg, "cache_empty_dino_results", False))
        if isinstance(cached_records, list) and len(cached_records) == 0 and not cache_empty_dino:
            print(f"[INFO] Ignoring empty DINO cache entry: {cache_file.name}")
            return None
        print(f"[INFO] Loaded DINO from cache: {cache_file.name}")
        return cached_records
    except Exception as e:
        print(f"[WARN] Failed to load DINO cache: {e}")
        return None


def save_masks_cache(image_path: str, image_hash: str, masks: list | None, file_context: str) -> bool:
    """Save SAM masks to cache. Returns True if saved, False otherwise."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return False
    if masks is None or not masks:
        return False
    
    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        with open(cache_file, "wb") as f:
            pickle.dump({"masks": masks}, f)
        print(f"[INFO] Cached {len(masks)} SAM masks: {cache_file.name}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to save masks cache: {e}")
        return False


def load_masks_cache(image_path: str, image_hash: str, file_context: str) -> list | None:
    """Load SAM masks from cache. Returns None if not found or caching disabled."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return None
    if bool(getattr(cfg, "overwrite_pipeline_cache", False)):
        return None
    
    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        if not cache_file.exists():
            return None
        
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        masks = data.get("masks")
        print(f"[INFO] Loaded {len(masks) if masks else 0} masks from cache: {cache_file.name}")
        return masks
    except Exception as e:
        print(f"[WARN] Failed to load masks cache: {e}")
        return None


def process_large_tile(
    tile_idx: int,
    tile_coords: list[tuple[int, int, int, int]],
    img_model: np.ndarray,
    image_path: str,
    image_hash: str,
    cfg,
    dino_model,
    dino_transform,
    dino_device: str,
    sam_model,
    sam_predictor,
    pixels_per_meter_sq: float | None,
) -> list[dict]:
    y0, y1, x0, x1 = tile_coords[tile_idx - 1]
    tile_img_model = img_model[y0:y1, x0:x1]
    if tile_img_model.size == 0:
        return []

    print(
        f"[INFO] [Tile {tile_idx}/{len(tile_coords)}] "
        f"Processing pixels y={y0}:{y1}, x={x0}:{x1}"
    )

    # Try loading DINO from cache first
    cache_key_dino = get_cache_key_for_tile_dino(image_path, image_hash, tile_idx, len(tile_coords))
    cached_dino = load_dino_cache(image_path, image_hash, cache_key_dino)
    
    tile_dino_records: list[dict] = []
    if cached_dino is not None:
        tile_dino_records = cached_dino
        print(f"[INFO] [Tile {tile_idx}/{len(tile_coords)}] Using cached DINO results ({len(tile_dino_records)} boxes)")
    else:
        try:
            tile_dino_records, _, _ = run_dino_prompts(
                tile_img_model,
                cfg,
                dino_model,
                dino_transform,
                dino_device,
                pixels_per_meter_sq,
                return_unfiltered=True,
                show_timing_summary=False,
            )
            save_dino_cache(image_path, image_hash, tile_dino_records, cache_key_dino)
        except Exception as exc:
            log_pipeline_error(f"[Tile {tile_idx}/{len(tile_coords)}] DINO failed", exc)
            print(f"[INFO] [Tile {tile_idx}/{len(tile_coords)}] Retrying with automatic SAM instead")

    if tile_dino_records:
        # Try loading SAM masks from cache
        cache_key_masks = get_cache_key_for_tile_masks(image_path, image_hash, tile_idx, len(tile_coords))
        cached_masks = load_masks_cache(image_path, image_hash, cache_key_masks)
        
        if cached_masks is not None:
            print(f"[INFO] [Tile {tile_idx}/{len(tile_coords)}] Using cached SAM masks ({len(cached_masks)} masks)")
            return cached_masks
        
        try:
            sam_predictor.set_image(tile_img_model)
            tile_masks = generate_sam_masks_from_detections(
                sam_predictor,
                tile_dino_records,
                cfg,
                tile_origin=(y0, x0),
                full_shape=img_model.shape[:2],
            )
            save_masks_cache(image_path, image_hash, tile_masks, cache_key_masks)
            return tile_masks
        except Exception as exc:
            log_pipeline_error(f"[Tile {tile_idx}/{len(tile_coords)}] SAM refinement failed", exc)
            print(f"[INFO] [Tile {tile_idx}/{len(tile_coords)}] Falling back to automatic SAM on the tile")

    try:
        tile_masks = generate_sam_masks_automatic_tiled(
            sam_model,
            cfg,
            tile_img_model,
            tile_origin=(y0, x0),
            full_shape=img_model.shape[:2],
        )
        save_masks_cache(image_path, image_hash, tile_masks, get_cache_key_for_tile_masks(image_path, image_hash, tile_idx, len(tile_coords)))
        return tile_masks
    except Exception as exc:
        log_pipeline_error(f"[Tile {tile_idx}/{len(tile_coords)}] Automatic SAM failed", exc)
        print(f"[INFO] [Tile {tile_idx}/{len(tile_coords)}] Final fallback: automatic SAM on whole tile")

    try:
        tile_masks = generate_sam_masks_automatic(
            sam_model,
            cfg,
            tile_img_model,
            tile_origin=(y0, x0),
            full_shape=img_model.shape[:2],
        )
        save_masks_cache(image_path, image_hash, tile_masks, get_cache_key_for_tile_masks(image_path, image_hash, tile_idx, len(tile_coords)))
        return tile_masks
    except Exception as exc:
        log_pipeline_error(f"[Tile {tile_idx}/{len(tile_coords)}] All tile methods failed", exc)
        print(f"[ERROR] [Tile {tile_idx}/{len(tile_coords)}] No method worked; skipping tile")
        return []


def finish_pipeline_early(reason: str) -> None:
    print(f"[INFO] {reason}")
    print(f"[INFO] Total figures saved this run: {len(saved_figure_paths)}")
    for p in saved_figure_paths:
        print(f"[INFO] -> {p}")
    log_stage("Pipeline complete", script_start)
    sys.exit(0)

# Function used for combining masks that may be from tiled SAM runs back into a full-size image mask for visualization and heatmap building.
def expand_mask_to_full_image(mask_dict: dict) -> np.ndarray:
    if "tile_bounds" not in mask_dict:
        return mask_dict["segmentation"]
    ty0, ty1, tx0, tx1 = mask_dict["tile_bounds"]
    full_h, full_w = mask_dict["full_shape"]
    full_seg = np.zeros((full_h, full_w), dtype=bool)
    full_seg[ty0:ty1, tx0:tx1] = mask_dict["segmentation"]
    return full_seg


PIPELINE_STAGE_TOTAL = 6
log_image_stage(f"Preparing image '{Path(active_tif_file).name}'", 1, PIPELINE_STAGE_TOTAL)

preprocessing_start = perf_counter()
report_geotiff_spatial_info(active_tif_file, use_crop=cfg.use_bbox_crop, lonlat_bbox=cfg.bbox_lonlat)

img_raw = load_rgb_image(active_tif_file, use_crop=cfg.use_bbox_crop, lonlat_bbox=cfg.bbox_lonlat)
loaded_extent_m = get_loaded_extent_meters(active_tif_file, use_crop=cfg.use_bbox_crop, lonlat_bbox=cfg.bbox_lonlat)

if loaded_extent_m is None:
    print("[WARN] Could not estimate loaded image extent in meters; amenity heatmap will be skipped.")
else:
    print(
        "[INFO] Loaded image real size (approx meters): "
        f"{loaded_extent_m[0]:.2f} m x {loaded_extent_m[1]:.2f} m"
    )

if img_raw.size == 0:
    raise RuntimeError("Cropped image is empty. Check bbox values and raster coverage.")

# Keep model input separate from display image so DINO/SAM can run at full resolution.
if img_raw.dtype == np.uint8:
    img_model = img_raw
else:
    if bool(getattr(cfg, "model_input_use_robust_uint8", True)):
        img_model = normalize_to_uint8_robust(
            img_raw,
            low_percentile=float(getattr(cfg, "model_input_percentile_low", 1.0)),
            high_percentile=float(getattr(cfg, "model_input_percentile_high", 99.0)),
        )
    else:
        img_model = normalize_to_uint8(img_raw)

img_display = normalize_to_uint8(img_raw.copy()) if img_raw.dtype != np.uint8 else img_raw.copy()
print(
    f"[INFO] Model image shape: {img_model.shape[0]}x{img_model.shape[1]} "
    f"({img_model.shape[2]} channels), dtype={img_model.dtype}"
)

# Save input image using PIL to avoid matplotlib memory issues with large arrays
from PIL import Image
input_image_filename = f"{Path(active_tif_file).stem}_input_image.png"
input_image_out = cfg.results_dir / "input_images" / input_image_filename
input_image_out.parent.mkdir(parents=True, exist_ok=True)
img_pil = Image.fromarray(img_display, mode="RGB")
img_pil.save(str(input_image_out))
saved_figure_paths.append(input_image_out)
print(f"[INFO] Saved figure: {input_image_out}")
log_stage("Image preprocessing complete", preprocessing_start)

# Compute image hash for cache keying
image_hash = compute_image_hash(img_model)
print(f"[INFO] Image hash for caching: {image_hash}")

# Conditional DINO + SAM or automatic SAM
stage_start = perf_counter()
if cfg.dino_only and not cfg.use_dino:
    raise RuntimeError("dino_only=True requires use_dino=True")

pixels_per_meter_sq = None
if loaded_extent_m is not None:
    img_h, img_w = img_model.shape[:2]
    extent_w_m, extent_h_m = loaded_extent_m
    pixels_per_meter_sq = (img_w / extent_w_m) * (img_h / extent_h_m)

image_pixels = int(img_model.shape[0] * img_model.shape[1])
large_image_tile_limit = int(getattr(cfg, "large_image_tile_max_pixels", 120_000_000))
large_image_tile_mode = bool(cfg.use_dino and image_pixels > large_image_tile_limit)

if large_image_tile_mode:
    if cfg.dino_only:
        raise RuntimeError("dino_only=True is not supported in large-image tiled mode.")

    print(
        "[INFO] Large image detected; running tiled DINO+SAM pipeline "
        f"({image_pixels:,} px > {large_image_tile_limit:,} limit)"
    )
    tile_size = int(getattr(cfg, "large_image_tile_size_px", 4096))
    tile_overlap = int(getattr(cfg, "large_image_tile_overlap_px", 384))
    tile_coords = iter_tile_coords(img_model.shape[0], img_model.shape[1], tile_size, tile_overlap)
    print(
        f"[INFO] Tiled pipeline using {len(tile_coords)} tiles "
        f"(tile={tile_size}, overlap={tile_overlap})"
    )

    dino_model, dino_transform, dino_device = build_dino_model_and_transform(cfg)
    sam_model, sam_device = load_sam_model(cfg)
    sam_predictor = SamPredictor(sam_model)
    masks = []

    for tile_idx, (y0, y1, x0, x1) in enumerate(tile_coords, start=1):
        try:
            tile_masks = process_large_tile(
                tile_idx,
                tile_coords,
                img_model,
                active_tif_file,
                image_hash,
                cfg,
                dino_model,
                dino_transform,
                dino_device,
                sam_model,
                sam_predictor,
                pixels_per_meter_sq,
            )
            masks.extend(tile_masks)
        except Exception as exc:
            log_pipeline_error(f"[Tile {tile_idx}/{len(tile_coords)}] Unhandled tile failure", exc)
            print(f"[ERROR] [Tile {tile_idx}/{len(tile_coords)}] Skipping tile after failure")

    if not masks:
        print("[ERROR] Tiled large-image pipeline produced no masks after all fallbacks.")

elif cfg.use_dino:
    log_image_stage("Running DINO detection", 2, PIPELINE_STAGE_TOTAL)
    print("[INFO] Running Grounding DINO detection")
    dino_model, dino_transform, dino_device = build_dino_model_and_transform(cfg)
    dino_run_id = f"{run_id}_dino_only" if cfg.dino_only else run_id
    dino_title_suffix = " (DINO-Only)" if cfg.dino_only else ""
    if cfg.dino_only:
        print("[INFO] DINO-only mode active: DINO outputs will be labeled with '_dino_only'.")
    
    # Calculate pixels-to-meters conversion for real-world area checking
    if pixels_per_meter_sq is not None:
        print(f"[INFO] Pixel-to-meters: {pixels_per_meter_sq:.6f} px²/m²")
    
    # Try loading DINO from cache first
    cache_key_dino_full = get_cache_key_for_image_dino(active_tif_file, image_hash)
    cached_dino_records = load_dino_cache(active_tif_file, image_hash, cache_key_dino_full)
    
    if cached_dino_records is not None:
        print(f"[INFO] Using cached DINO results ({len(cached_dino_records)} boxes)")
        dino_records = cached_dino_records
        dino_unfiltered_records = []
        dino_filtered_records = []
    else:
        print("[DEBUG] About to call run_dino_prompts", flush=True)
        try:
            dino_records, dino_unfiltered_records, dino_filtered_records = run_dino_prompts(
                img_model,
                cfg,
                dino_model,
                dino_transform,
                dino_device,
                pixels_per_meter_sq,
                return_unfiltered=True,
                show_timing_summary=True,
            )
            save_dino_cache(active_tif_file, image_hash, dino_records, cache_key_dino_full)
            print(f"[DEBUG] run_dino_prompts returned: {len(dino_records) if dino_records else 0} kept, {len(dino_unfiltered_records) if dino_unfiltered_records else 0} unfiltered", flush=True)
        except Exception as e:
            print(f"[ERROR] run_dino_prompts failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if dino_records:
        print(f"[INFO] Grounding DINO total kept detections: {len(dino_records)}")
        # Choose visualization backend based on config
        if cfg.dino_visualization_backend == "pil":
            save_dino_detection_viz_pil(
                img_display,
                dino_records,
                dino_run_id,
                save_current_figure,
                title_suffix=dino_title_suffix,
                filtered_records=dino_filtered_records,
            )
        else:
            save_dino_detection_viz(
                img_display,
                dino_records,
                dino_run_id,
                save_current_figure,
                title_suffix=dino_title_suffix,
                filtered_records=dino_filtered_records,
                dpi=DINO_SAVE_DPI,
            )
    else:
        print("[WARN] Grounding DINO produced no valid detections after filtering.")
        if dino_unfiltered_records:
            print(
                f"[INFO] DINO produced {len(dino_unfiltered_records)} unfiltered candidates; "
                "saving debug visualization before filtering."
            )
            if cfg.dino_visualization_backend == "pil":
                save_dino_detection_viz_pil(
                    img_display,
                    dino_unfiltered_records,
                    f"{dino_run_id}_prefilter",
                    save_current_figure,
                    title_suffix=f"{dino_title_suffix} (Pre-Filter)",
                )
            else:
                save_dino_detection_viz(
                    img_display,
                    dino_unfiltered_records,
                    f"{dino_run_id}_prefilter",
                    save_current_figure,
                    title_suffix=f"{dino_title_suffix} (Pre-Filter)",
                    dpi=DINO_SAVE_DPI,
                )
        else:
            if cfg.dino_visualization_backend == "pil":
                save_dino_detection_viz_pil(
                    img_display,
                    [],
                    f"{dino_run_id}_prefilter",
                    save_current_figure,
                    title_suffix=f"{dino_title_suffix} (Pre-Filter)",
                )
            else:
                save_dino_detection_viz(
                    img_display,
                    [],
                    f"{dino_run_id}_prefilter",
                    save_current_figure,
                    title_suffix=f"{dino_title_suffix} (Pre-Filter)",
                    dpi=DINO_SAVE_DPI,
                )

    # Generate diagnostic visualizations for DINO detections.
    # For very large rasters, skip heavy diagnostics to avoid memory exhaustion.
    if dino_records or dino_unfiltered_records:
        max_diag_pixels = int(getattr(cfg, "dino_diagnostic_max_pixels", 120_000_000))
        enable_diag = bool(getattr(cfg, "dino_enable_diagnostic_visualizations", True))
        image_pixels = int(img_display.shape[0] * img_display.shape[1])

        if enable_diag and image_pixels > max_diag_pixels:
            print(
                "[INFO] Skipping heavy DINO diagnostics for large image "
                f"({image_pixels:,} px > {max_diag_pixels:,} limit)"
            )
            enable_diag = False

        if enable_diag:
            diagnostics_start = perf_counter()
            print("[INFO] Generating DINO diagnostic visualizations...")

            # 1. Per-prompt breakdown - separate visualization for each detector
            if dino_records:
                try:
                    save_per_prompt_breakdown(img_display, dino_records, dino_run_id, config=cfg)
                except Exception as e:
                    print(f"[ERROR] Per-prompt breakdown visualization failed: {e}")
                    import traceback
                    traceback.print_exc()

            # 2. Filtering stage comparison - show stages from unfiltered to passed to filtered-out
            if dino_unfiltered_records:
                try:
                    save_filtering_stage_comparison(
                        img_display,
                        dino_unfiltered_records,
                        dino_records,
                        dino_filtered_records,
                        dino_run_id,
                        config=cfg
                    )
                except Exception as e:
                    print(f"[ERROR] Filtering stage comparison visualization failed: {e}")
                    import traceback
                    traceback.print_exc()

            # 3. Detection heatmap - confidence density map
            if dino_records:
                try:
                    save_detection_heatmap(img_display, dino_records, dino_run_id, config=cfg)
                except Exception as e:
                    print(f"[ERROR] Detection heatmap visualization failed: {e}")
                    import traceback
                    traceback.print_exc()

            # 4. Box size distribution - analysis of detected object sizes
            if dino_records:
                try:
                    save_box_size_distribution(img_display, dino_records, dino_run_id, config=cfg)
                except Exception as e:
                    print(f"[ERROR] Box size distribution visualization failed: {e}")
                    import traceback
                    traceback.print_exc()

            log_stage("DINO diagnostics complete", diagnostics_start)

    if cfg.dino_only:
        finish_pipeline_early("DINO-only mode enabled; skipping SAM and CLIP stages.")

    if not dino_records:
        print("[WARN] Grounding DINO produced no valid detections after filtering. Running tiled automatic SAM fallback.")
        from models.sam_processing import resolve_device
        log_image_stage("Running SAM fallback auto-generation", 3, PIPELINE_STAGE_TOTAL)

        device = resolve_device(cfg.sam_device)
        
        # Try loading masks from cache
        cache_key_masks_full = get_cache_key_for_image_masks(active_tif_file, image_hash)
        cached_masks = load_masks_cache(active_tif_file, image_hash, cache_key_masks_full)
        if cached_masks is not None:
            print(f"[INFO] Using cached SAM masks ({len(cached_masks)} masks)")
            masks = cached_masks
        else:
            ensure_checkpoint(cfg.sam_checkpoint, cfg.sam_checkpoint_url)
            print(f"[INFO] Loading SAM model '{cfg.sam_model_type}' on device '{device}'")
            sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
            sam.to(device=device)

            log_stage("SAM model ready", stage_start)
            try:
                masks = generate_sam_masks_automatic_tiled(sam, cfg, img_model)
                save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
            except Exception as exc:
                log_pipeline_error("Automatic SAM tiled fallback failed", exc)
                print("[INFO] Retrying with non-tiled automatic SAM")
                try:
                    masks = generate_sam_masks_automatic(sam, cfg, img_model)
                    save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
                except Exception as retry_exc:
                    log_pipeline_error("Automatic SAM retry failed", retry_exc)
                    print("[ERROR] No SAM fallback method worked for this image")
                    masks = []
        sam_device = device
    else:
        print(f"[INFO] Grounding DINO total kept detections for SAM: {len(dino_records)}")

        log_image_stage("Running SAM refinement", 3, PIPELINE_STAGE_TOTAL)
        
        # Try loading masks from cache
        cache_key_masks_full = get_cache_key_for_image_masks(active_tif_file, image_hash)
        cached_masks = load_masks_cache(active_tif_file, image_hash, cache_key_masks_full)
        if cached_masks is not None:
            print(f"[INFO] Using cached SAM masks ({len(cached_masks)} masks)")
            masks = cached_masks
            from models.sam_processing import resolve_device
            sam_device = resolve_device(cfg.sam_device)
        else:
            predictor, sam_device = build_sam_predictor(cfg, img_model)
            log_stage("SAM model ready", stage_start)
            try:
                masks = generate_sam_masks_from_detections(predictor, dino_records, cfg)
                save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
            except Exception as exc:
                log_pipeline_error("SAM refinement from DINO failed", exc)
                print("[INFO] Retrying with automatic SAM fallback")
                sam_fallback = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
                sam_fallback.to(device=sam_device)
                try:
                    masks = generate_sam_masks_automatic_tiled(sam_fallback, cfg, img_model)
                    save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
                except Exception as retry_exc:
                    log_pipeline_error("Automatic SAM fallback after DINO-SAM failure failed", retry_exc)
                    print("[INFO] Retrying with non-tiled automatic SAM")
                    try:
                        masks = generate_sam_masks_automatic(sam_fallback, cfg, img_model)
                        save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
                    except Exception as final_exc:
                        log_pipeline_error("Final automatic SAM fallback failed", final_exc)
                        print("[ERROR] No SAM method worked after DINO-SAM failure")
                        masks = []
else:
    log_image_stage("Running SAM auto-generation", 3, PIPELINE_STAGE_TOTAL)
    print("[INFO] Skipping DINO and using automatic SAM mask generation")
    from models.sam_processing import resolve_device
    
    device = resolve_device(cfg.sam_device)
    
    # Try loading masks from cache
    cache_key_masks_full = get_cache_key_for_image_masks(active_tif_file, image_hash)
    cached_masks = load_masks_cache(active_tif_file, image_hash, cache_key_masks_full)
    if cached_masks is not None:
        print(f"[INFO] Using cached SAM masks ({len(cached_masks)} masks)")
        masks = cached_masks
    else:
        ensure_checkpoint(cfg.sam_checkpoint, cfg.sam_checkpoint_url)
        print(f"[INFO] Loading SAM model '{cfg.sam_model_type}' on device '{device}'")
        sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
        sam.to(device=device)
        
        log_stage("SAM model ready", stage_start)
        try:
            masks = generate_sam_masks_automatic_tiled(sam, cfg, img_model)
            save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
        except Exception as exc:
            log_pipeline_error("Automatic SAM tiled generation failed", exc)
            print("[INFO] Retrying with non-tiled automatic SAM")
            try:
                masks = generate_sam_masks_automatic(sam, cfg, img_model)
                save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
            except Exception as retry_exc:
                log_pipeline_error("Automatic SAM retry failed", retry_exc)
                print("[ERROR] No SAM method worked for this image")
                masks = []
    sam_device = device

if not masks:
    print("[ERROR] No masks were produced for selected prompts after all fallbacks.")

log_image_stage("Running CLIP scoring", 4, PIPELINE_STAGE_TOTAL)
clip_device = sam_device if sam_device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")

stage_start = perf_counter()
print("[INFO] Loading CLIP model and preprocessing")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
)
clip_model = clip_model.to(clip_device)
clip_model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")
log_stage("CLIP model ready", stage_start)


def build_text_features(target_prompt: str, negative_prompts: list[str] | None = None):
    with torch.no_grad():
        pos_templates = [
            f"aerial image of {target_prompt}",
            f"overhead view of {target_prompt}",
            f"satellite photo of {target_prompt}",
            f"{target_prompt} in an urban area",
        ]

        pos_tokens = tokenizer(pos_templates).to(clip_device)
        pos_features = clip_model.encode_text(pos_tokens)
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        pos_text_feature = pos_features.mean(dim=0, keepdim=True)
        pos_text_feature = pos_text_feature / pos_text_feature.norm(dim=-1, keepdim=True)

        neg_text_feature = None
        if negative_prompts:
            neg_templates: list[str] = []
            for neg in negative_prompts:
                neg_templates.extend(
                    [
                        f"aerial image of {neg}",
                        f"overhead view of {neg}",
                        f"satellite photo of {neg}",
                    ]
                )
            neg_tokens = tokenizer(neg_templates).to(clip_device)
            neg_features = clip_model.encode_text(neg_tokens)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            neg_text_feature = neg_features.mean(dim=0, keepdim=True)
            neg_text_feature = neg_text_feature / neg_text_feature.norm(dim=-1, keepdim=True)

    return pos_text_feature, neg_text_feature


stage_start = perf_counter()
print("[INFO] Encoding CLIP prompts for active amenities")
# Build text features for all active prompts using their captions
clip_features = {}
for prompt_config in cfg.dino_prompt_configs:
    prompt_name = prompt_config["name"]
    caption = prompt_config["caption"]
    neg_captions = prompt_config.get("negative_captions", [])
    pos_feature, neg_feature = build_text_features(caption, neg_captions)
    clip_features[prompt_name] = {
        "pos": pos_feature,
        "neg": neg_feature,
        "text": caption,
        "config": prompt_config,
    }
log_stage("Text prompt encoding complete", stage_start)


def _extract_mask_geometry(mask_dict: dict) -> tuple | None:
    """Extract geometric coordinates and area safely from a mask dict. Returns None if invalid."""
    tile_seg = mask_dict["segmentation"]
    tile_bounds = mask_dict.get("tile_bounds")

    ys, xs = np.where(tile_seg)
    if ys.size == 0 or xs.size == 0:
        return None

    ly0, ly1 = int(ys.min()), int(ys.max()) + 1
    lx0, lx1 = int(xs.min()), int(xs.max()) + 1

    if tile_bounds is not None:
        ty0, _, tx0, _ = tile_bounds
        full_h, full_w = mask_dict["full_shape"]
        iy0, iy1 = ly0 + ty0, ly1 + ty0
        ix0, ix1 = lx0 + tx0, lx1 + tx0
    else:
        full_h, full_w = tile_seg.shape[:2]
        iy0, iy1, ix0, ix1 = ly0, ly1, lx0, lx1

    area_ratio = float(tile_seg.sum()) / float(full_h * full_w)
    h = max(1, iy1 - iy0)
    w = max(1, ix1 - ix0)
    bbox_fill = float(tile_seg.sum()) / float(h * w)

    return (iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, full_h, full_w)


def score_mask(
    mask_dict: dict,
    pos_text_feature: torch.Tensor,
    neg_text_feature: torch.Tensor | None,
    prompt_config: dict,
) -> float:
    """Score one mask with CLIP plus lightweight geometry and color heuristics."""
    geom = _extract_mask_geometry(mask_dict)
    if geom is None:
        return -1.0

    iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, _, _ = geom

    min_area = prompt_config.get("clip_min_area_ratio", 0.00015)
    max_area = prompt_config.get("clip_max_area_ratio", 0.25)
    if area_ratio < min_area or area_ratio > max_area:
        return -1.0

    tile_seg = mask_dict["segmentation"]
    crop = img_model[iy0:iy1, ix0:ix1]
    crop_mask = tile_seg[ly0:ly1, lx0:lx1][..., None]
    object_only = np.where(crop_mask, crop, 255).astype(np.uint8)
    pil_img = Image.fromarray(object_only)

    with torch.no_grad():
        image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(clip_device)
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        pos_sim = float((image_features @ pos_text_feature.T).item())
        score = pos_sim

        if neg_text_feature is not None:
            neg_sim = float((image_features @ neg_text_feature.T).item())
            neg_weight = float(prompt_config.get("clip_negative_weight", 0.35))
            score -= neg_weight * max(0.0, neg_sim)

        iou = float(mask_dict.get("predicted_iou", 0.0))
        stability = float(mask_dict.get("stability_score", 0.0))
        score += 0.05 * iou + 0.05 * stability

    aspect = max(h, w) / float(min(h, w))
    bbox_fill_score = np.clip((0.65 - bbox_fill) / 0.65, 0.0, 1.0)
    elongation_score = np.clip((aspect - 1.5) / 6.0, 0.0, 1.0)
    score += 0.12 * float(elongation_score)
    score += 0.06 * float(bbox_fill_score)

    mask_pixels = img_model[iy0:iy1, ix0:ix1][tile_seg[ly0:ly1, lx0:lx1]]
    if mask_pixels.size > 0:
        pixels_norm = mask_pixels.astype(np.float32) / 255.0
        hsv = rgb_to_hsv(pixels_norm.reshape(-1, 1, 3)).reshape(-1, 3)
        sat = float(hsv[:, 1].mean())
        val = float(hsv[:, 2].mean())

        max_sat = prompt_config.get("max_saturation", 0.28)
        min_val = prompt_config.get("min_value", 0.35)

        low_sat_score = np.clip((max_sat - sat) / max_sat, 0.0, 1.0)
        bright_enough_score = np.clip((val - min_val) / (1.0 - min_val), 0.0, 1.0)

        score += 0.05 * float(low_sat_score)
        score += 0.04 * float(bright_enough_score)

    return score


def batch_score_masks_clip(
    masks_input: list,
    pos_text_feature: torch.Tensor,
    neg_text_feature: torch.Tensor | None,
    prompt_config: dict,
) -> list[float]:
    """Batch-score masks using vectorized CLIP encoding. ~2-4x faster than serial."""
    scores = [0.0] * len(masks_input)
    
    # First pass: quick geometric filtering and crop preparation
    min_area = prompt_config.get("clip_min_area_ratio", 0.00015)
    max_area = prompt_config.get("clip_max_area_ratio", 0.25)
    
    candidates: list[tuple[int, np.ndarray]] = []  # (orig_idx, pil_img_tensor)
    
    for idx, m in enumerate(masks_input):
        geom = _extract_mask_geometry(m)
        if geom is None:
            scores[idx] = -1.0
            continue
        
        iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, full_h, full_w = geom
        
        # Area filtering
        if area_ratio < min_area or area_ratio > max_area:
            scores[idx] = -1.0
            continue
        
        # Prepare crop
        tile_seg = m["segmentation"]
        crop = img_model[iy0:iy1, ix0:ix1]
        crop_mask = tile_seg[ly0:ly1, lx0:lx1][..., None]
        object_only = np.where(crop_mask, crop, 255).astype(np.uint8)
        pil_img = Image.fromarray(object_only)
        img_tensor = clip_preprocess(pil_img)
        
        candidates.append((idx, img_tensor, iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, tile_seg, full_h, full_w, m))
    
    if not candidates:
        return scores
    
    # Batch CLIP encoding
    batch_indices = [c[0] for c in candidates]
    batch_tensors = torch.stack([c[1] for c in candidates]).to(clip_device)
    
    with torch.no_grad():
        batch_features = clip_model.encode_image(batch_tensors)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
    
    # Unbatch results and compute scores
    neg_weight = float(prompt_config.get("clip_negative_weight", 0.35))
    
    for batch_idx, (orig_idx, _, iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, tile_seg, full_h, full_w, m) in enumerate(candidates):
        image_features = batch_features[batch_idx:batch_idx+1]
        
        # CLIP scores
        pos_sim = float((image_features @ pos_text_feature.T).item())
        score = pos_sim
        
        if neg_text_feature is not None:
            neg_sim = float((image_features @ neg_text_feature.T).item())
            score -= neg_weight * max(0.0, neg_sim)
        
        # Geometry bonuses
        iou = float(m.get("predicted_iou", 0.0))
        stability = float(m.get("stability_score", 0.0))
        score += 0.05 * iou + 0.05 * stability 
        
        aspect = max(h, w) / float(min(h, w))
        elongation_score = np.clip((aspect - 1.5) / 6.0, 0.0, 1.0)
        sparse_fill_score = np.clip((0.65 - bbox_fill) / 0.65, 0.0, 1.0)
        score += 0.12 * float(elongation_score)
        score += 0.06 * float(sparse_fill_score)
        
        # Color bonuses
        mask_pixels = img_model[iy0:iy1, ix0:ix1][tile_seg[ly0:ly1, lx0:lx1]]
        if mask_pixels.size > 0:
            pixels_norm = mask_pixels.astype(np.float32) / 255.0
            hsv = rgb_to_hsv(pixels_norm.reshape(-1, 1, 3)).reshape(-1, 3)
            sat = float(hsv[:, 1].mean())
            val = float(hsv[:, 2].mean())
            
            max_sat = prompt_config.get("max_saturation", 0.28)
            min_val = prompt_config.get("min_value", 0.35)
            
            low_sat_score = np.clip((max_sat - sat) / max_sat, 0.0, 1.0)
            bright_enough_score = np.clip((val - min_val) / (1.0 - min_val), 0.0, 1.0)
            
            score += 0.05 * float(low_sat_score)
            score += 0.04 * float(bright_enough_score)
        
        scores[orig_idx] = score
    
    return scores


def serial_score_masks_clip(
    masks_input: list,
    pos_text_feature: torch.Tensor,
    neg_text_feature: torch.Tensor | None,
    prompt_config: dict,
) -> list[float]:
    """Fallback CLIP scorer that processes one mask at a time."""
    scores: list[float] = []
    for mask_dict in masks_input:
        try:
            scores.append(score_mask(mask_dict, pos_text_feature, neg_text_feature, prompt_config))
        except Exception as exc:
            print(f"[ERROR] Serial CLIP scoring failed for one mask: {exc}")
            import traceback
            traceback.print_exc()
            scores.append(-1.0)
    return scores


def select_masks_for_prompt(
    masks_input: list,
    score_key: str,
    pos_text_feature: torch.Tensor,
    neg_text_feature: torch.Tensor | None,
    prompt_name: str,
    prompt_config: dict,
) -> tuple[list, float]:
    stage_start = perf_counter()
    print(f"[INFO] Scoring {len(masks_input)} masks for '{prompt_name}'")

    # Use batch CLIP scoring for ~2-4x speedup, but fall back to serial scoring if needed.
    try:
        scores = batch_score_masks_clip(masks_input, pos_text_feature, neg_text_feature, prompt_config)
    except Exception as exc:
        print(f"[ERROR] Batch CLIP scoring failed for '{prompt_name}': {exc}")
        import traceback
        traceback.print_exc()
        print(f"[INFO] Retrying CLIP scoring serially for '{prompt_name}'")
        try:
            scores = serial_score_masks_clip(masks_input, pos_text_feature, neg_text_feature, prompt_config)
        except Exception as retry_exc:
            print(f"[ERROR] Serial CLIP scoring also failed for '{prompt_name}': {retry_exc}")
            import traceback
            traceback.print_exc()
            print(f"[ERROR] No CLIP scoring method worked for '{prompt_name}'; skipping prompt")
            return [], float(prompt_config.get("clip_score_threshold", -0.03))
    
    for idx, score in enumerate(scores):
        masks_input[idx][score_key] = score
        if (idx + 1) == 1 or (idx + 1) % 50 == 0 or (idx + 1) == len(masks_input):
            print(
                f"[PROGRESS] [{prompt_name}] Scored {idx + 1}/{len(masks_input)} masks"
            )

    log_stage(f"Mask scoring complete for '{prompt_name}'", stage_start)

    filtered = [m for m in masks_input if m[score_key] >= 0.0]
    filtered.sort(key=lambda m: m[score_key], reverse=True)

    if not filtered:
        print(f"[WARN] No non-negative CLIP scores for '{prompt_name}'. Selecting 0 masks.")
        return [], float(prompt_config.get("clip_score_threshold", -0.03))

    top_k = prompt_config.get("clip_top_k", 45)
    score_threshold = prompt_config.get("clip_score_threshold", -0.03)
    relative_margin = prompt_config.get("clip_relative_score_margin", 0.10)

    if filtered:
        best_score = float(filtered[0][score_key])
        dynamic_threshold = max(score_threshold, best_score - relative_margin)
    else:
        dynamic_threshold = score_threshold

    selected = [m for m in filtered if m[score_key] >= dynamic_threshold][:top_k]

    if not selected and filtered:
        selected = filtered[:1]

    print(f"Prompt: {prompt_name}")
    print(f"Selection threshold used: {dynamic_threshold:.3f}")
    print(f"Selected {len(selected)} / {len(masks_input)} masks by CLIP score")
    print("Top scores:", ", ".join(f"{m[score_key]:.3f}" for m in selected[:5]))

    return selected, dynamic_threshold


selected_masks_by_prompt = {}
auto_fallback_masks = [m for m in masks if m.get("dino_prompt_group") in (None, "auto")]
prompt_linked_masks = [m for m in masks if m.get("dino_prompt_group") not in (None, "auto")]
is_auto_mask_run = len(prompt_linked_masks) == 0

if is_auto_mask_run:
    print(
        "[INFO] SAM-only conceptual per-prompt mode: cloning auto-SAM masks into "
        "independent prompt candidate pools before CLIP scoring"
    )

for prompt_name, features in clip_features.items():
    if is_auto_mask_run:
        # In SAM-only mode, each prompt evaluates an independent copy of the auto-SAM pool.
        # This keeps prompt scoring isolated even though mask geometry came from one SAM run.
        prompt_masks = [{**m, "dino_prompt_group": prompt_name} for m in auto_fallback_masks]
        print(
            f"[INFO] Prompt '{prompt_name}' evaluating {len(prompt_masks)} independent SAM-only candidates"
        )
    else:
        # In DINO-guided mode, keep strict prompt isolation from DINO group labels.
        prompt_masks = [m for m in masks if m.get("dino_prompt_group") == prompt_name]
        if auto_fallback_masks:
            prompt_masks.extend({**m, "dino_prompt_group": prompt_name} for m in auto_fallback_masks)
        if not prompt_masks:
            print(
                f"[WARN] No SAM masks linked to prompt '{prompt_name}'; selecting 0 masks for this prompt."
            )
            selected_masks_by_prompt[prompt_name] = []
            continue
        print(
            f"[INFO] Prompt '{prompt_name}' evaluating {len(prompt_masks)} prompt-linked SAM masks"
        )

    selected, _ = select_masks_for_prompt(
        prompt_masks,
        f"clip_score_{prompt_name}",
        features["pos"],
        features["neg"],
        prompt_name,
        features["config"],
    )
    selected_masks_by_prompt[prompt_name] = selected

# Compute combined masks for each prompt
combined_masks = {}
for prompt_name, selected_masks in selected_masks_by_prompt.items():
    combined = np.zeros(img_model.shape[:2], dtype=bool)
    for m in selected_masks:
        combined |= expand_mask_to_full_image(m)
    combined_masks[prompt_name] = combined


def _apply_tier_thresholds(
    combined_masks: dict[str, np.ndarray],
    selected_masks_by_prompt: dict[str, list[dict]],
    image_shape: tuple[int, int],
    e_threshold: float = 0.25,
    c_threshold: float = 0.15,
) -> dict[str, np.ndarray]:
    """Apply confidence thresholds to tier assignments.
    
    Only assign E (uncomfortable) tier if score > e_threshold.
    Only assign C (pedestrian) tier if score > c_threshold.
    Otherwise default to A (comfortable) tier.
    
    Args:
        combined_masks: Dict of prompt_name -> full-image boolean mask
        selected_masks_by_prompt: Dict of prompt_name -> list of selected mask dicts (with scores)
        image_shape: (height, width) tuple
        e_threshold: Minimum CLIP score for E tier assignment
        c_threshold: Minimum CLIP score for C tier assignment
    
    Returns:
        Refined combined_masks dict with threshold-based tier reassignment
    """
    h, w = image_shape
    
    # Build full-image score maps for each tier
    tier_scores = {
        "nen_cat_a": np.full((h, w), -np.inf, dtype=np.float32),
        "nen_cat_c": np.full((h, w), -np.inf, dtype=np.float32),
        "nen_cat_e": np.full((h, w), -np.inf, dtype=np.float32),
    }
    
    # Fill score maps from selected masks
    for prompt_name, masks in selected_masks_by_prompt.items():
        if prompt_name not in tier_scores:
            continue
        
        for mask_dict in masks:
            mask_bool = expand_mask_to_full_image(mask_dict)
            if not np.any(mask_bool):
                continue
            
            score_key = f"clip_score_{prompt_name}"
            score = float(mask_dict.get(score_key, mask_dict.get("score", 0.0)))
            
            # Update pixels where this score is better than previously seen
            update_pixels = mask_bool & (score > tier_scores[prompt_name])
            if np.any(update_pixels):
                tier_scores[prompt_name][update_pixels] = score
    
    # For each pixel, apply threshold-based tier selection: E -> C -> A
    e_scores = tier_scores["nen_cat_e"]
    c_scores = tier_scores["nen_cat_c"]
    a_scores = tier_scores["nen_cat_a"]
    
    # Build refined masks based on thresholds
    e_confident = e_scores > e_threshold
    c_confident = c_scores > c_threshold
    
    # Pixel assignment logic:
    # 1. If E score is high enough, assign to E
    # 2. Else if C score is high enough, assign to C
    # 3. Else assign to A (or leave as default if A score is too low)
    
    refined_masks = {
        "nen_cat_a": np.zeros((h, w), dtype=bool),
        "nen_cat_c": np.zeros((h, w), dtype=bool),
        "nen_cat_e": np.zeros((h, w), dtype=bool),
    }
    
    # Assign E tier (only if confident)
    refined_masks["nen_cat_e"] = e_confident.copy()
    
    # Assign C tier (only if confident AND not already assigned to E)
    c_and_not_e = c_confident & ~e_confident
    refined_masks["nen_cat_c"] = c_and_not_e.copy()
    
    # Assign A tier (fallback: default to A if not C or E, or if A score is reasonable)
    not_e_or_c = ~e_confident & ~c_confident
    a_or_unassigned = (a_scores > -np.inf) | not_e_or_c
    refined_masks["nen_cat_a"] = a_or_unassigned.copy()
    
    # Ensure pixel-level winner-takes-all to avoid overlaps
    # Give priority: E > C > A
    final_masks = {
        "nen_cat_e": refined_masks["nen_cat_e"].copy(),
        "nen_cat_c": refined_masks["nen_cat_c"].copy() & ~refined_masks["nen_cat_e"],
        "nen_cat_a": refined_masks["nen_cat_a"].copy() & ~refined_masks["nen_cat_e"] & ~refined_masks["nen_cat_c"],
    }
    
    # Print statistics
    print("[INFO] Tier threshold refinement:")
    print(f"  E threshold: {e_threshold:.3f}, pixels assigned: {final_masks['nen_cat_e'].sum():,}")
    print(f"  C threshold: {c_threshold:.3f}, pixels assigned: {final_masks['nen_cat_c'].sum():,}")
    print(f"  A (fallback): pixels assigned: {final_masks['nen_cat_a'].sum():,}")
    
    return final_masks


# Apply tier thresholds to reduce false positive uncomfortable (E) tier assignments
tier_e_threshold = float(getattr(cfg, "tier_e_threshold", 0.25))
tier_c_threshold = float(getattr(cfg, "tier_c_threshold", 0.15))
if tier_e_threshold > 0 or tier_c_threshold > 0:
    print(f"[INFO] Applying tier thresholds: E={tier_e_threshold:.3f}, C={tier_c_threshold:.3f}")
    combined_masks = _apply_tier_thresholds(
        combined_masks,
        selected_masks_by_prompt,
        img_model.shape[:2],
        e_threshold=tier_e_threshold,
        c_threshold=tier_c_threshold,
    )


def _build_full_image_prompt_masks(
    selected_masks_by_prompt: dict[str, list[dict]],
    image_shape: tuple[int, int],
    prompt_names_ordered: list[str],
) -> dict[str, np.ndarray]:
    """Build a full-image, winner-takes-all mask for each prompt.

    Pixels covered by one or more prompt masks are assigned to the prompt with the highest
    score at that pixel. Uncovered pixels are filled by the nearest labeled region so the
    output covers the entire image.
    """
    h, w = image_shape
    if not prompt_names_ordered:
        return {}

    # Memory-safe winner-takes-all pass: avoid stacking per-prompt full-size score maps.
    best_score_map = np.full((h, w), -np.inf, dtype=np.float32)
    winner_indices = np.zeros((h, w), dtype=np.uint8)
    covered = np.zeros((h, w), dtype=bool)
    prompt_best_scores: dict[str, float] = {name: float("-inf") for name in prompt_names_ordered}

    for prompt_idx, prompt_name in enumerate(prompt_names_ordered):
        score_key = f"clip_score_{prompt_name}"
        for mask in selected_masks_by_prompt.get(prompt_name, []):
            mask_bool = expand_mask_to_full_image(mask)
            if not np.any(mask_bool):
                continue

            score = float(mask.get(score_key, mask.get("score", 0.0)))
            prompt_best_scores[prompt_name] = max(prompt_best_scores[prompt_name], score)

            update_pixels = mask_bool & (score > best_score_map)
            if not np.any(update_pixels):
                continue

            best_score_map[update_pixels] = score
            winner_indices[update_pixels] = np.uint8(prompt_idx)
            covered[update_pixels] = True

    # Ensure full-image assignment for pixels that were never covered by any selected mask.
    if not np.all(covered):
        best_prompt_idx = 0
        best_prompt_score = float("-inf")
        for idx, prompt_name in enumerate(prompt_names_ordered):
            score = prompt_best_scores.get(prompt_name, float("-inf"))
            if score > best_prompt_score:
                best_prompt_idx = idx
                best_prompt_score = score

        # If all prompts are empty, keep index 0 as deterministic fallback.
        winner_indices[~covered] = np.uint8(best_prompt_idx)

    full_image_masks: dict[str, np.ndarray] = {}
    for idx, prompt_name in enumerate(prompt_names_ordered):
        full_image_masks[prompt_name] = winner_indices == idx

    return full_image_masks

prompt_order = list(cfg.dino_prompt_configs) if hasattr(cfg, "dino_prompt_configs") else []
prompt_names_ordered = [p["name"] for p in prompt_order]

if bool(getattr(cfg, "full_image_mask_mode", False)):
    print("[INFO] Full-image mask mode enabled: assigning every pixel to the highest-scoring prompt")
    combined_masks = _build_full_image_prompt_masks(
        selected_masks_by_prompt,
        img_model.shape[:2],
        prompt_names_ordered,
    )
    for prompt_name in prompt_names_ordered:
        mask_pixels = int(combined_masks.get(prompt_name, np.zeros(img_model.shape[:2], dtype=bool)).sum())
        print(f"[DEBUG] {prompt_name}: full-image pixels = {mask_pixels:,}")
else:
    # Apply prompt priority: earlier prompts in ACTIVE_PROMPTS take precedence
    # (so sports_court masks remove overlapping building_roof, etc.)
    combined_masks_with_priority = {}
    accumulated_mask = np.zeros(img_model.shape[:2], dtype=bool)

    for prompt_name in prompt_names_ordered:
        if prompt_name in combined_masks:
            original_mask = combined_masks[prompt_name]
            priority_mask = original_mask.copy()
            priority_mask &= ~accumulated_mask
            combined_masks_with_priority[prompt_name] = priority_mask

            original_pixels = original_mask.sum()
            remaining_pixels = priority_mask.sum()
            removed_pixels = original_pixels - remaining_pixels
            print(
                f"[DEBUG] {prompt_name}: {original_pixels:,} pixels → {remaining_pixels:,} pixels "
                f"(removed {removed_pixels:,} due to priority masking)"
            )

            accumulated_mask |= priority_mask
        else:
            combined_masks_with_priority[prompt_name] = np.zeros(img_model.shape[:2], dtype=bool)
            print(f"[DEBUG] {prompt_name}: NOT FOUND in combined_masks - creating empty mask")

    combined_masks = combined_masks_with_priority

# Compute overlap mask (union of all masks)
union_of_all = np.zeros(img_model.shape[:2], dtype=bool)
for combined in combined_masks.values():
    union_of_all |= combined

# Visualize each prompt separately and create combined overlay
viz_stride = max(1, int(np.ceil(max(img_model.shape[:2]) / 2000)))
if viz_stride > 1:
    print(f"[INFO] Downsampling visualization by stride={viz_stride} to reduce memory use")

# DEBUG: Show what's in combined_masks before visualization
print(f"[DEBUG] combined_masks keys: {list(combined_masks.keys())}")
for pname, pmask in combined_masks.items():
    print(f"[DEBUG]   {pname}: {pmask.sum():,} pixels ({pmask.sum() / pmask.size * 100:.2f}% coverage)")

viz_img = img_display[::viz_stride, ::viz_stride]

# Colors for different prompts (explicit and high-contrast for reliable legend matching)
prompt_colors = {
    # Merged NEN 8100 Wind Comfort Categories
    "nen_cat_a": [0.00, 0.72, 0.38],   # green - comfortable, sheltered, vegetated
    "nen_cat_c": [1.00, 0.68, 0.10],   # amber - pedestrian-accessible but uncomfortable
    "nen_cat_e": [0.95, 0.18, 0.18],   # red - highways, roofs, no-access hardscape
    # Legacy amenity prompts
    "sports_court": [1.00, 0.84, 0.00],   # gold
    "transit_hub": [0.95, 0.35, 0.80],    # magenta-pink
    "pedestrian_features": [0.45, 0.95, 0.25],  # lime green
    "sidewalk_surface": [0.00, 0.90, 0.90],  # cyan
    "road_surface": [0.65, 0.65, 0.65],   # medium gray
    "park": [0.00, 0.65, 1.00],           # vivid blue
    "warehouse_roof": [1.00, 0.55, 0.00], # bold orange
    "building_roof": [1.00, 0.20, 0.20],  # bold red
    "outdoor_seating": [0.10, 0.90, 0.90],
    "standing_gathering": [1.00, 0.65, 0.10],
    "furniture": [0.90, 0.20, 0.20],
    "seated_dining": [0.85, 0.30, 0.85],
}


def get_prompt_color(prompt_name: str) -> np.ndarray:
    # Magenta fallback makes missing color mappings immediately obvious.
    return np.array(prompt_colors.get(prompt_name, [1.0, 0.0, 1.0]), dtype=np.float32)

# Save individual visualizations for each prompt
log_image_stage("Building visualizations", 5, PIPELINE_STAGE_TOTAL)
for prompt_name, combined_mask in combined_masks.items():
    viz_mask = combined_mask[::viz_stride, ::viz_stride]
    color = get_prompt_color(prompt_name)
    
    viz_overlay = np.zeros((*viz_img.shape[:2], 4), dtype=np.float32)
    viz_overlay[viz_mask, :3] = color
    viz_overlay[viz_mask, 3] = 0.50  # Increased alpha from 0.35 to 0.50 for better visibility

    plt.figure(figsize=(10, 10))
    plt.imshow(viz_img)
    plt.imshow(viz_overlay)
    plt.axis("off")
    plt.title(f"SAM + CLIP {prompt_name.replace('_', ' ').title()} Masks: '{clip_features[prompt_name]['text']}'")
    save_current_figure(f"{run_id}_{prompt_name}_masks.png", "individual_masks")

# Create combined visualization with all prompts overlaid
viz_combined_overlay = np.zeros((*viz_img.shape[:2], 4), dtype=np.float32)
for prompt_name, combined_mask in combined_masks.items():
    viz_mask = combined_mask[::viz_stride, ::viz_stride]
    color = get_prompt_color(prompt_name)
    viz_combined_overlay[viz_mask, :3] = color
    viz_combined_overlay[viz_mask, 3] = 0.45  # Increased alpha from 0.32 to 0.45 for better visibility

# Create figure with extra space for legend on the right
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(viz_img)
ax.imshow(viz_combined_overlay)
ax.axis("off")

# Create legend with custom patches for each prompt
from matplotlib.patches import Patch
legend_elements = []
for prompt_name in sorted(combined_masks.keys()):
    color_rgb = get_prompt_color(prompt_name).tolist()
    label = prompt_name.replace('_', ' ').title()
    legend_elements.append(Patch(facecolor=color_rgb, edgecolor='black', label=label))

# Add legend outside the plot area on the right side
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
         fontsize=11, frameon=True, fancybox=True, shadow=True)

fig.suptitle(f"Combined Masks: {', '.join(p.replace('_', ' ').title() for p in combined_masks.keys())}", 
            fontsize=14, weight='bold')
plt.tight_layout()
save_current_figure(f"{run_id}_combined_masks.png", "combined_masks")

heatmap_excluded_prompts = set(getattr(cfg, "amenity_heatmap_excluded_prompts", ["building_roof"]))
heatmap_mask_union = np.zeros(img_model.shape[:2], dtype=bool)
heatmap_prompt_names: list[str] = []
for prompt_name, combined_mask in combined_masks.items():
    if prompt_name in heatmap_excluded_prompts:
        continue
    heatmap_mask_union |= combined_mask
    heatmap_prompt_names.append(prompt_name)

if not heatmap_prompt_names:
    print("[WARN] Heatmap excluded all prompts; falling back to union of all prompt masks.")
    amenity_mask_union = union_of_all
else:
    print(
        "[INFO] Heatmap amenities from prompts: "
        f"{', '.join(heatmap_prompt_names)} (excluded: {', '.join(sorted(heatmap_excluded_prompts))})"
    )
    amenity_mask_union = heatmap_mask_union

if getattr(cfg, "build_amenity_heatmap", False) and loaded_extent_m is not None:
    stage_start = perf_counter()
    print(
        "[INFO] Building amenity heatmap grid from mask union "
        f"(cell area: {cfg.amenity_grid_cell_area_m2:.2f} m^2)"
    )
    amenity_heatmap, cell_px_w, cell_px_h, cell_side_m = build_amenity_heatmap(
        amenity_mask_union,
        img_model.shape,
        loaded_extent_m,
        cfg.amenity_grid_cell_area_m2,
        taper_sigma_cells=float(getattr(cfg, "amenity_heatmap_taper_sigma_cells", 0.90)),
        taper_blend=float(getattr(cfg, "amenity_heatmap_taper_blend", 0.75)),
    )
    log_stage("Amenity heatmap complete", stage_start)

    print(
        "[INFO] Heatmap grid cell size (approx): "
        f"{cell_side_m:.2f}m x {cell_side_m:.2f}m "
        f"(~{cell_px_w} x {cell_px_h} px)"
    )

    plt.figure(figsize=(10, 10))
    viz_amenity_heatmap = amenity_heatmap[::viz_stride, ::viz_stride]
    viz_alpha_map = np.where(
        viz_amenity_heatmap > 0,
        np.clip(viz_amenity_heatmap, 0.10, 0.90),
        0.0,
    )
    plt.imshow(viz_img)
    plt.imshow(viz_amenity_heatmap, cmap="Reds", alpha=viz_alpha_map, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Amenity Heatmap (Red = Higher Amenity Coverage)")
    save_current_figure(f"{run_id}_amenity_heatmap.png", "heatmaps")
elif not getattr(cfg, "build_amenity_heatmap", False):
    print("[INFO] Amenity heatmap generation disabled in config (build_amenity_heatmap=False).")
else:
    print("[WARN] Skipping amenity heatmap because real-world extent is unavailable.")

print(f"[INFO] Total figures saved this run: {len(saved_figure_paths)}")
for p in saved_figure_paths:
    print(f"[INFO] -> {p}")

log_image_stage("Completed", 6, PIPELINE_STAGE_TOTAL)
log_stage("Pipeline complete", script_start)
