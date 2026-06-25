from datetime import datetime
import faulthandler
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import warnings

# Reduce noisy cache warnings on Windows where symlinks are often unavailable.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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
from models import caching as cache
from models import mask_assignment as assign
from models.clip_scoring import ClipScorer

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
    warnings.filterwarnings(
        "ignore",
        message=r"Failed to load custom C\+\+ ops\. Running on CPU mode Only!",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\..*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch\.load` with `weights_only=False`.*",
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
    compose_visualization_with_side_panel,
)
from models.sam_processing import (
    build_sam_predictor,
    generate_sam_masks_from_detections,
    generate_sam_masks_automatic,
    generate_sam_masks_automatic_tiled,
    load_sam_model,
    ensure_checkpoint,
)


def _resolved_cpu_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return max(1, os.cpu_count() or 1)


script_start = perf_counter()
print(f"[INFO] Resolved CPU workers: {_resolved_cpu_workers()}")
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


def _visualizations_exist(results_dir: Path, run_id: str) -> bool:
    """Return True if the key visualizations and IoU report already exist."""
    # Combined mask image
    combined_dir = results_dir / "combined_masks"
    combined_file = combined_dir / f"{run_id}_combined_masks.png"

    # Annotation IoU report (any *_iou_report.json in annotation_iou)
    iou_dir = results_dir / "annotation_iou"
    iou_exists = False
    if iou_dir.exists() and any(iou_dir.glob("*_iou_report.json")):
        iou_exists = True

    return combined_file.exists() and iou_exists

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
    cache_key_dino = cache.get_cache_key_for_tile_dino(image_path, image_hash, tile_idx, len(tile_coords))
    cached_dino = cache.load_dino_cache(image_path, image_hash, cache_key_dino)
    
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
            cache.save_dino_cache(image_path, image_hash, tile_dino_records, cache_key_dino)
        except Exception as exc:
            log_pipeline_error(f"[Tile {tile_idx}/{len(tile_coords)}] DINO failed", exc)
            print(f"[INFO] [Tile {tile_idx}/{len(tile_coords)}] Retrying with automatic SAM instead")

    if tile_dino_records:
        # Try loading SAM masks from cache
        cache_key_masks = cache.get_cache_key_for_tile_masks(image_path, image_hash, tile_idx, len(tile_coords))
        cached_masks = cache.load_masks_cache(image_path, image_hash, cache_key_masks)
        
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
            cache.save_masks_cache(image_path, image_hash, tile_masks, cache_key_masks)
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
        cache.save_masks_cache(image_path, image_hash, tile_masks, cache.get_cache_key_for_tile_masks(image_path, image_hash, tile_idx, len(tile_coords)))
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
        cache.save_masks_cache(image_path, image_hash, tile_masks, cache.get_cache_key_for_tile_masks(image_path, image_hash, tile_idx, len(tile_coords)))
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

img_display = img_model if img_raw.dtype == np.uint8 else normalize_to_uint8(img_raw)
del img_raw
print(
    f"[INFO] Model image shape: {img_model.shape[0]}x{img_model.shape[1]} "
    f"({img_model.shape[2]} channels), dtype={img_model.dtype}"
)

# Save input image using PIL to avoid matplotlib memory issues with large arrays
input_image_filename = f"{Path(active_tif_file).stem}_input_image.png"
input_image_out = cfg.results_dir / "input_images" / input_image_filename
input_image_out.parent.mkdir(parents=True, exist_ok=True)
img_pil = Image.fromarray(img_display, mode="RGB")
if bool(getattr(cfg, "save_input_images", False)):
    img_pil.save(str(input_image_out))
    saved_figure_paths.append(input_image_out)
    print(f"[INFO] Saved input image: {input_image_out}")
else:
    print(f"[DEBUG] Skipping saving input image (cfg.save_input_images=False)")
log_stage("Image preprocessing complete", preprocessing_start)

# Compute image hash for cache keying
image_hash = cache.compute_image_hash(img_model)
print(f"[INFO] Image hash for caching: {image_hash}")

# If requested, skip heavy processing when visualizations + IoU already exist.
if bool(getattr(cfg, "skip_if_visualizations_exist", True)):
    try:
        if _visualizations_exist(cfg.results_dir, run_id):
            finish_pipeline_early("Visualizations and IoU report already exist; skipping heavy processing")
    except Exception as exc:
        print(f"[WARN] Visualizations exist check failed: {exc}")

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
    cache_key_dino_full = cache.get_cache_key_for_image_dino(active_tif_file, image_hash)
    cached_dino_records = cache.load_dino_cache(active_tif_file, image_hash, cache_key_dino_full)
    
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
            cache.save_dino_cache(active_tif_file, image_hash, dino_records, cache_key_dino_full)
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
        cache_key_masks_full = cache.get_cache_key_for_image_masks(active_tif_file, image_hash)
        cached_masks = cache.load_masks_cache(active_tif_file, image_hash, cache_key_masks_full)
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
                cache.save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
            except Exception as exc:
                log_pipeline_error("Automatic SAM tiled fallback failed", exc)
                print("[INFO] Retrying with non-tiled automatic SAM")
                try:
                    masks = generate_sam_masks_automatic(sam, cfg, img_model)
                    cache.save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
                except Exception as retry_exc:
                    log_pipeline_error("Automatic SAM retry failed", retry_exc)
                    print("[ERROR] No SAM fallback method worked for this image")
                    masks = []
        sam_device = device
    else:
        print(f"[INFO] Grounding DINO total kept detections for SAM: {len(dino_records)}")

        log_image_stage("Running SAM refinement", 3, PIPELINE_STAGE_TOTAL)
        
        # Try loading masks from cache
        cache_key_masks_full = cache.get_cache_key_for_image_masks(active_tif_file, image_hash)
        cached_masks = cache.load_masks_cache(active_tif_file, image_hash, cache_key_masks_full)
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
                cache.save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
            except Exception as exc:
                log_pipeline_error("SAM refinement from DINO failed", exc)
                print("[INFO] Retrying with automatic SAM fallback")
                sam_fallback = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
                sam_fallback.to(device=sam_device)
                try:
                    masks = generate_sam_masks_automatic_tiled(sam_fallback, cfg, img_model)
                    cache.save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
                except Exception as retry_exc:
                    log_pipeline_error("Automatic SAM fallback after DINO-SAM failure failed", retry_exc)
                    print("[INFO] Retrying with non-tiled automatic SAM")
                    try:
                        masks = generate_sam_masks_automatic(sam_fallback, cfg, img_model)
                        cache.save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
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
    cache_key_masks_full = cache.get_cache_key_for_image_masks(active_tif_file, image_hash)
    cached_masks = cache.load_masks_cache(active_tif_file, image_hash, cache_key_masks_full)
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
            cache.save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
        except Exception as exc:
            log_pipeline_error("Automatic SAM tiled generation failed", exc)
            print("[INFO] Retrying with non-tiled automatic SAM")
            try:
                masks = generate_sam_masks_automatic(sam, cfg, img_model)
                cache.save_masks_cache(active_tif_file, image_hash, masks, cache_key_masks_full)
            except Exception as retry_exc:
                log_pipeline_error("Automatic SAM retry failed", retry_exc)
                print("[ERROR] No SAM method worked for this image")
                masks = []
    sam_device = device

if not masks:
    print("[ERROR] No masks were produced for selected prompts after all fallbacks.")

log_image_stage("Running CLIP scoring", 4, PIPELINE_STAGE_TOTAL)
clip_device = sam_device if sam_device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")

scorer = ClipScorer(device=clip_device)

stage_start = perf_counter()
print("[INFO] Encoding CLIP prompts for active amenities")
# Build text features for all active prompts using their captions
clip_features = {}
for prompt_config in cfg.dino_prompt_configs:
    prompt_name = prompt_config["name"]
    caption = prompt_config["caption"]
    neg_captions = prompt_config.get("negative_captions", [])
    pos_feature, neg_feature = scorer.build_text_features(caption, neg_captions)
    clip_features[prompt_name] = {
        "pos": pos_feature,
        "neg": neg_feature,
        "text": caption,
        "config": prompt_config,
    }
log_stage("Text prompt encoding complete", stage_start)

selected_masks_by_prompt = {}
auto_fallback_masks = [m for m in masks if m.get("dino_prompt_group") in (None, "auto")]
prompt_linked_masks = [m for m in masks if m.get("dino_prompt_group") not in (None, "auto")]
is_auto_mask_run = len(prompt_linked_masks) == 0
pixel_assignment_mode = str(getattr(cfg, "pixel_assignment_mode", "legacy")).strip().lower()
prompt_order = list(cfg.dino_prompt_configs) if hasattr(cfg, "dino_prompt_configs") else []
prompt_names_ordered = [p["name"] for p in prompt_order]

if is_auto_mask_run:
    print(
        "[INFO] SAM-only conceptual per-prompt mode: cloning auto-SAM masks into "
        "independent prompt candidate pools before CLIP scoring"
    )

total_scored_masks = 0
region_context_masks = None
region_context_prompt_signature = cache.build_region_context_prompt_signature(prompt_order)
region_context_cache_key = cache.get_cache_key_for_region_context_scoring(
    active_tif_file,
    image_hash,
    region_context_prompt_signature,
)
cached_region_context_masks = None
if pixel_assignment_mode == "region_context":
    cached_region_context_masks = cache.load_region_context_scoring_cache(
        active_tif_file,
        image_hash,
        region_context_cache_key,
        region_context_prompt_signature,
    )

if cached_region_context_masks is not None:
    region_context_masks = cached_region_context_masks
    total_scored_masks = len(region_context_masks)
    print(
        f"[INFO] Using cached region-context CLIP scores ({total_scored_masks} shared SAM regions)"
    )

if cached_region_context_masks is None:
    for prompt_name, features in clip_features.items():
        if pixel_assignment_mode == "region_context":
            if region_context_masks is None:
                region_context_masks = [{**m} for m in masks]
            prompt_masks = region_context_masks
            total_scored_masks += len(prompt_masks)
            print(
                f"[INFO] Prompt '{prompt_name}' evaluating {len(prompt_masks)} shared SAM regions"
            )
        elif is_auto_mask_run:
            # In SAM-only mode, each prompt evaluates an independent copy of the auto-SAM pool.
            # This keeps prompt scoring isolated even though mask geometry came from one SAM run.
            prompt_masks = [{**m, "dino_prompt_group": prompt_name} for m in auto_fallback_masks]
            # prompt evaluates an independent copy of auto masks
            total_scored_masks += len(prompt_masks)
            print(
                f"[INFO] Prompt '{prompt_name}' evaluating {len(prompt_masks)} independent SAM-only candidates"
            )
        if is_auto_mask_run:
            pass
        elif pixel_assignment_mode != "region_context":
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
            total_scored_masks += len(prompt_masks)
            print(
                f"[INFO] Prompt '{prompt_name}' evaluating {len(prompt_masks)} prompt-linked SAM masks"
            )

        if pixel_assignment_mode in {"contrastive", "region_context"}:
            selected_masks_by_prompt[prompt_name] = scorer.select_masks_for_prompt_contrastive(
                prompt_masks,
                f"clip_score_{prompt_name}",
                features["pos"],
                features["neg"],
                prompt_name,
                features["config"],
                img_model,
            )
        else:
            selected, _ = scorer.select_masks_for_prompt(
                prompt_masks,
                f"clip_score_{prompt_name}",
                features["pos"],
                features["neg"],
                prompt_name,
                features["config"],
                img_model,
            )
            selected_masks_by_prompt[prompt_name] = selected

    if pixel_assignment_mode == "region_context" and region_context_masks is not None:
        cache.save_region_context_scoring_cache(
            active_tif_file,
            image_hash,
            region_context_cache_key,
            region_context_masks,
            total_scored_masks,
            region_context_prompt_signature,
        )

if pixel_assignment_mode == "region_context":
    contrastive_prompt_weights = dict(getattr(cfg, "contrastive_prompt_weights", {}))
    selected_masks_by_prompt = assign._assign_region_context_masks(
        region_context_masks or [],
        prompt_names_ordered,
        contrastive_prompt_weights,
    )
    combined_masks = assign._build_region_context_full_image_masks(
        selected_masks_by_prompt,
        img_model.shape[:2],
        prompt_names_ordered,
    )
    print("[INFO] Region-context scoring, assignment, and mask build finished")

# Compute combined masks for each prompt (legacy/thresholded paths only).
if pixel_assignment_mode not in {"contrastive", "region_context"}:
    combined_masks = {}
    for prompt_name, selected_masks in selected_masks_by_prompt.items():
        combined = np.zeros(img_model.shape[:2], dtype=bool)
        for m in selected_masks:
            combined |= assign.expand_mask_to_full_image(m)
        combined_masks[prompt_name] = combined

prompt_order = list(cfg.dino_prompt_configs) if hasattr(cfg, "dino_prompt_configs") else []
prompt_names_ordered = [p["name"] for p in prompt_order]

if pixel_assignment_mode == "contrastive":
    contrastive_prompt_weights = dict(getattr(cfg, "contrastive_prompt_weights", {}))
    combined_masks = assign._build_contrastive_full_image_masks(
        selected_masks_by_prompt,
        img_model.shape[:2],
        prompt_names_ordered,
        contrastive_prompt_weights,
    )
    # one summary printed inside the contrastive builder
elif bool(getattr(cfg, "full_image_mask_mode", False)) and pixel_assignment_mode != "region_context":
    combined_masks = assign._build_full_image_prompt_masks(
        selected_masks_by_prompt,
        img_model.shape[:2],
        prompt_names_ordered,
    )
    # suppressed per-prompt debug outputs
else:
    if pixel_assignment_mode != "region_context":
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

        # Always ensure the final output is a full-image label map, even if we capped or
        # filtered masks aggressively upstream.
        combined_masks = assign._fill_full_image_mask_gaps(
            combined_masks,
            img_model.shape[:2],
            prompt_names_ordered,
        )

        coarse_to_fine_cell_px = int(getattr(cfg, "coarse_to_fine_cell_px", 0))
        if coarse_to_fine_cell_px > 1:
            combined_masks = assign._apply_coarse_to_fine_pass(
                combined_masks,
                img_model.shape[:2],
                prompt_names_ordered,
                coarse_to_fine_cell_px,
            )

if pixel_assignment_mode == "legacy":
    # Apply tier thresholds to reduce false positive uncomfortable (E) tier assignments
    tier_e_threshold = float(getattr(cfg, "tier_e_threshold", 0.25))
    tier_c_threshold = float(getattr(cfg, "tier_c_threshold", 0.15))
    tier_d_threshold = float(getattr(cfg, "tier_d_threshold", 0.0))
    tier_b_threshold = float(getattr(cfg, "tier_b_threshold", 0.0))
    if tier_e_threshold > 0 or tier_c_threshold > 0 or tier_d_threshold > 0 or tier_b_threshold > 0:
        print(
            f"[INFO] Applying tier thresholds: E={tier_e_threshold:.3f}, D={tier_d_threshold:.3f}, "
            f"C={tier_c_threshold:.3f}, B={tier_b_threshold:.3f}"
        )
        combined_masks = assign._apply_tier_thresholds(
            combined_masks,
            selected_masks_by_prompt,
            img_model.shape[:2],
            e_threshold=tier_e_threshold,
            c_threshold=tier_c_threshold,
            d_threshold=tier_d_threshold,
            b_threshold=tier_b_threshold,
        )

if bool(getattr(cfg, "enable_annotation_iou_check", False)):
    print("[INFO] Starting annotation IoU check")
    try:
        from models.iou_comparator import compare_annotation_iou
        from visualizations.annotation_iou_visuals import save_annotation_iou_comparison_figure

        iou_report = compare_annotation_iou(
            Path(active_tif_file).name,
            combined_masks,
            xml_path=getattr(cfg, "annotation_iou_xml_path", None),
            output_dir=getattr(cfg, "annotation_iou_output_dir", None),
            search_root=PROJECT_ROOT / "Maps" / "Tiles",
            class_mode=getattr(cfg, "annotation_iou_class_mode", None),
        )
        print(
            "[INFO] Annotation IoU check: "
            f"mIoU={iou_report['mean_iou']:.3f}, "
            f"pixel_accuracy={iou_report['pixel_accuracy']:.3f}, "
            f"evaluated_pixels={iou_report['evaluated_pixels']:,}, "
            f"ignored_invalid_pixels={iou_report['ignored_invalid_pixels']:,}"
        )

        comparison_output_path = (
            Path(getattr(cfg, "annotation_iou_output_dir", cfg.results_dir / "annotation_iou"))
            / f"{Path(active_tif_file).stem}_comparison.png"
        )
        save_annotation_iou_comparison_figure(
            img_display,
            np.asarray(iou_report["ground_truth_class_map"]),
            np.asarray(iou_report["prediction_class_map"]),
            np.asarray(iou_report["valid_mask"]),
            comparison_output_path,
            title=f"{Path(active_tif_file).name}  |  IoU comparison",
        )
        print(f"[INFO] Saved annotation comparison figure to: {comparison_output_path}")
        for class_name, class_stats in iou_report["class_results"].items():
            print(
                f"[INFO]   {class_name}: IoU={class_stats['iou']:.3f}, "
                f"pred={int(class_stats['pred_pixels']):,}, gt={int(class_stats['gt_pixels']):,}"
            )
        if iou_report.get("output_path"):
            print(f"[INFO]   IoU report saved to: {iou_report['output_path']}")
        print("[INFO] Annotation IoU check complete")
    except FileNotFoundError:
        print(f"[INFO] No annotations found for tile '{Path(active_tif_file).name}'; skipping IoU check")
    except Exception as exc:
        log_pipeline_error("Annotation IoU comparison failed", exc)

# Compute overlap mask (union of all masks)
print("[INFO] Computing union of all prompt masks")
union_of_all = np.zeros(img_model.shape[:2], dtype=bool)
for combined in combined_masks.values():
    union_of_all |= combined
print("[INFO] Union of all prompt masks complete")

# Visualize each prompt separately and create combined overlay
combined_viz_max_dim = int(getattr(cfg, "combined_visualization_max_dim", 1200))
viz_stride = max(1, int(np.ceil(max(img_model.shape[:2]) / combined_viz_max_dim)))
if viz_stride > 1:
    print(f"[INFO] Downsampling visualization by stride={viz_stride} to reduce memory use")

# DEBUG: Show what's in combined_masks before visualization
print(f"[DEBUG] combined_masks keys: {list(combined_masks.keys())}")
for pname, pmask in combined_masks.items():
    print(f"[DEBUG]   {pname}: {pmask.sum():,} pixels ({pmask.sum() / pmask.size * 100:.2f}% coverage)")

print("[INFO] Starting visualization generation")
faulthandler.dump_traceback_later(120, repeat=False)
viz_img = img_display[::viz_stride, ::viz_stride]

# Colors for different prompts (explicit and high-contrast for reliable legend matching)
prompt_colors = {
    # Merged NEN 8100 Wind Comfort Categories
    "nen_cat_a": [0.00, 0.45, 0.20],   # dark green - comfortable, sheltered, vegetated
    "nen_cat_b": [0.45, 0.80, 0.25],   # light green - pedestrian-friendly / walkable
    "nen_cat_c": [1.00, 0.90, 0.20],   # yellow - pedestrian-accessible but uncomfortable
    "nen_cat_d": [1.00, 0.55, 0.10],   # orange - exposed hardscape / parking
    "nen_cat_e": [0.90, 0.12, 0.12],   # red - highways, roofs, no-access hardscape
    "nen_a": [0.00, 0.45, 0.20],
    "nen_b": [0.45, 0.80, 0.25],
    "nen_c": [1.00, 0.90, 0.20],
    "nen_d": [1.00, 0.55, 0.10],
    "nen_e": [0.90, 0.12, 0.12],
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
    color = (get_prompt_color(prompt_name) * 255).astype(np.uint8)

    # Build RGBA overlay for this prompt
    h_vis, w_vis = viz_img.shape[:2]
    overlay = np.zeros((h_vis, w_vis, 4), dtype=np.uint8)
    mask_idx = viz_mask.astype(bool)
    overlay[mask_idx, :3] = color
    overlay[mask_idx, 3] = int(0.50 * 255)

    base_pil = Image.fromarray(viz_img.astype(np.uint8)).convert("RGBA")
    overlay_pil = Image.fromarray(overlay, mode="RGBA")
    result_pil = Image.alpha_composite(base_pil, overlay_pil)

    # Save via PIL
    save_current_figure(f"{run_id}_{prompt_name}_masks.png", "individual_masks", pil_img=result_pil)

# Create legend data (not drawn) for consistent labeling elsewhere if needed
legend_elements = []
for prompt_name in sorted(combined_masks.keys()):
    color_rgb = get_prompt_color(prompt_name).tolist()
    label = prompt_name.replace('_', ' ').title()
    legend_elements.append((label, color_rgb))

assignment_mode_label = {
    "contrastive": "Contrastive",
    "region_context": "Region Context",
}.get(pixel_assignment_mode, "Legacy")

if bool(getattr(cfg, "build_prompt_strength_heatmaps", False)):
    print("[INFO] Building per-prompt CLIP strength heatmaps")

    heatmap_source_by_prompt = selected_masks_by_prompt
    heatmap_cmap = plt.get_cmap(str(getattr(cfg, "prompt_strength_heatmap_cmap", "magma")))
    heatmap_alpha = float(getattr(cfg, "prompt_strength_heatmap_alpha", 0.55))
    heatmap_low_pct = float(getattr(cfg, "prompt_strength_heatmap_percentile_low", 5.0))
    heatmap_high_pct = float(getattr(cfg, "prompt_strength_heatmap_percentile_high", 95.0))
    heatmap_viz_max_dim = int(getattr(cfg, "prompt_strength_heatmap_max_dim", 1000))
    heatmap_viz_stride = max(1, int(np.ceil(max(img_model.shape[:2]) / heatmap_viz_max_dim)))
    if heatmap_viz_stride > 1:
        print(f"[INFO] Downsampling prompt strength heatmaps by stride={heatmap_viz_stride} to reduce memory use")

    for prompt_name in prompt_names_ordered:
        heatmap_scores, heatmap_covered = assign._build_prompt_strength_heatmap(
            heatmap_source_by_prompt,
            img_model.shape[:2],
            prompt_name,
        )
        if heatmap_scores is None or heatmap_covered is None:
            print(f"[WARN] Skipping prompt strength heatmap for '{prompt_name}' because no pixels were covered")
            continue

        finite_scores = heatmap_scores[np.isfinite(heatmap_scores)]
        if finite_scores.size == 0:
            print(f"[WARN] Skipping prompt strength heatmap for '{prompt_name}' because no finite scores were found")
            continue

        low = float(np.percentile(finite_scores, heatmap_low_pct))
        high = float(np.percentile(finite_scores, heatmap_high_pct))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(finite_scores))
            high = float(np.nanmax(finite_scores))
        if high <= low:
            high = low + 1e-6

        heatmap_scores_viz = heatmap_scores[::heatmap_viz_stride, ::heatmap_viz_stride]
        heatmap_covered_viz = heatmap_covered[::heatmap_viz_stride, ::heatmap_viz_stride]
        normalized = np.clip((heatmap_scores_viz - low) / (high - low), 0.0, 1.0)
        normalized = np.where(np.isfinite(heatmap_scores_viz), normalized, 0.0)
        rgba = (heatmap_cmap(normalized) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(heatmap_covered_viz, int(round(255 * heatmap_alpha)), 0).astype(np.uint8)

        base_pil = Image.fromarray(img_display[::heatmap_viz_stride, ::heatmap_viz_stride].astype(np.uint8)).convert("RGBA")
        overlay_pil = Image.fromarray(rgba, mode="RGBA")
        result_pil = Image.alpha_composite(base_pil, overlay_pil)
        result_pil = compose_visualization_with_side_panel(
            result_pil,
            f"{prompt_name.replace('_', ' ').title()} CLIP Strength Heatmap",
            gradient_cmap=heatmap_cmap,
            gradient_low_label="Low score",
            gradient_high_label="High score",
        )
        save_current_figure(f"{run_id}_{prompt_name}_strength_heatmap.png", "prompt_strength_heatmaps", pil_img=result_pil)
        print(
            f"[INFO] Saved prompt strength heatmap for '{prompt_name}' "
            f"(p{heatmap_low_pct:.0f}={low:.3f}, p{heatmap_high_pct:.0f}={high:.3f})"
        )

# Save combined visualization using PIL compositing to avoid matplotlib figure hangs
combined_h, combined_w = viz_img.shape[:2]
combined_overlay = np.zeros((combined_h, combined_w, 4), dtype=np.uint8)
for prompt_name, combined_mask in combined_masks.items():
    viz_mask = combined_mask[::viz_stride, ::viz_stride]
    color = (get_prompt_color(prompt_name) * 255).astype(np.uint8)
    # apply color where mask is True; last prompt wins for overlapping pixels
    mask_idx = viz_mask.astype(bool)
    combined_overlay[mask_idx, :3] = color
    combined_overlay[mask_idx, 3] = int(0.45 * 255)

base_pil = Image.fromarray(viz_img.astype(np.uint8)).convert("RGBA")
overlay_pil = Image.fromarray(combined_overlay, mode="RGBA")
result_pil = Image.alpha_composite(base_pil, overlay_pil)
result_pil = compose_visualization_with_side_panel(
    result_pil,
    "Combined Mask",
    legend_rows=[
        ("A", get_prompt_color("nen_cat_a")),
        ("B", get_prompt_color("nen_cat_b")),
        ("C", get_prompt_color("nen_cat_c")),
        ("D", get_prompt_color("nen_cat_d")),
        ("E", get_prompt_color("nen_cat_e")),
    ],
    footer_lines=[
        f"mIoU: {iou_report['mean_iou']:.3f}" if 'iou_report' in locals() and iou_report else None,
        f"Pixel accuracy: {iou_report['pixel_accuracy']:.3f}" if 'iou_report' in locals() and iou_report else None,
    ] if 'iou_report' in locals() and iou_report else None,
)

save_current_figure(f"{run_id}_combined_masks.png", "combined_masks", pil_img=result_pil)
print("[INFO] Combined visualization saved")

heatmap_excluded_prompts = set(getattr(cfg, "amenity_heatmap_excluded_prompts", ["building_roof"]))
heatmap_mask_union = np.zeros(img_model.shape[:2], dtype=bool)
heatmap_prompt_names = []
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
    print("[INFO] Starting amenity heatmap build")
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
    print("[INFO] Amenity heatmap saved")
elif not getattr(cfg, "build_amenity_heatmap", False):
    print("[INFO] Amenity heatmap generation disabled in config (build_amenity_heatmap=False).")
else:
    print("[WARN] Skipping amenity heatmap because real-world extent is unavailable.")

faulthandler.cancel_dump_traceback_later()

print(f"[INFO] Total figures saved this run: {len(saved_figure_paths)}")
for p in saved_figure_paths:
    print(f"[INFO] -> {p}")

log_image_stage("Completed", 6, PIPELINE_STAGE_TOTAL)
log_stage("Pipeline complete", script_start)

print(f"[INFO] Total masks evaluated for CLIP scoring: {total_scored_masks}")
