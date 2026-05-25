from pathlib import Path

# Input image settings
tif_files = [
    "Maps/gtcampus.tif",
    "Maps/Tiles/Atlanta_split/tile_000_000.tif",
    "Maps/NYC/NYC(small).tif",
    "Maps/GTCampus/CampusFullSize.tif",
    "Maps/Grove(LA)/The Grove.tif"

]
# Use "all" to run every file in tif_files, or "single" to run just one selection.
tif_run_mode = "single"  # "all" or "single"
tif_single_index = 0  # 0-based index into tif_files when tif_run_mode="single"
tif_single_file = "Maps/Tiles/Atlanta_split_google/tile_002_003.tif"  # Optional explicit single file path override when tif_run_mode="single"
tif_file = tif_files[tif_single_index]  # Backward compatibility for modules still reading cfg.tif_file
use_bbox_crop = False
bbox_lonlat = (-84.4010, 33.7720, -84.3950, 33.7760)

# Output settings
results_dir = Path("results")
output_dpi = 100  # Effective native-pixel export when combined with figure size matching
dino_visualization_dpi = 100  # Lower DPI for DINO detections to reduce memory usage during render (still ~high quality)
dino_visualization_backend = "pil"  # "matplotlib" or "pil" - PIL is faster, smaller files, more efficient

# Heatmap settings
build_amenity_heatmap = False  # Set to True to generate amenity heatmap visualization (requires sufficient memory)
build_prompt_strength_heatmaps = True  # Set to True to save one full-image CLIP strength heatmap per prompt
combined_visualization_max_dim = 1200
prompt_strength_heatmap_max_dim = 1000
annotation_iou_visualization_max_dim = 900
amenity_grid_cell_area_m2 = 256.0
amenity_heatmap_excluded_prompts = ["building_roof", "warehouse_roof"]
amenity_heatmap_taper_sigma_cells = 0.90
amenity_heatmap_taper_blend = 0.75
prompt_strength_heatmap_alpha = 0.55
prompt_strength_heatmap_cmap = "magma"
prompt_strength_heatmap_percentile_low = 5.0
prompt_strength_heatmap_percentile_high = 95.0
dino_heatmap_mode = "average"  # "average" or "sum" - average shows per-pixel detection confidence, sum shows detection density
dino_enable_diagnostic_visualizations = True
dino_diagnostic_max_pixels = 500_000_000  # Skip heavy DINO diagnostics above this image size

# Large-image tiled pipeline settings
large_image_tile_max_pixels = 250_000_000  # Switch to tiled DINO+SAM processing above this size
large_image_tile_size_px = 4096
large_image_tile_overlap_px = 384

# Caching settings - saves DINO and SAM results to disk to avoid re-computation on error recovery
enable_pipeline_caching = True  # Set to True to cache DINO/SAM intermediate results between runs
overwrite_pipeline_cache = True  # Force recomputation after prompt changes so the cache is refreshed

# Annotation IoU comparison settings
enable_annotation_iou_check = True  # Set to True to compare final masks against CVAT annotations
annotation_iou_class_mode = "split"  # "grouped" (A/C/E) or "split" (A/B/C/D/E)
annotation_iou_xml_path = None  # Optional explicit annotation JSON/XML path or directory
annotation_iou_output_dir = Path("results") / "annotation_iou"

# DINO settings - Set use_dino=False to skip DINO and use SAM's automatic mask generation instead
use_dino = False
dino_only = False  # Temporary debug mode: run only DINO and skip SAM + CLIP stages
dino_suppress_low_risk_warnings = True
dino_full_resolution = False  # If True, skip global DINO resize and run at native pixel dimensions; False uses resize (faster, avoids OOM)
dino_resize_short_side = 1200
dino_resize_max_size = 2000
dino_device = "auto"  # "auto", "cpu", "cuda"
cache_empty_dino_results = False  # Keep False to avoid reusing empty DINO caches from failed runs
dino_enable_tiled_fallback = True
dino_tile_size_px = 4096
dino_tile_overlap_px = 384
#dino_tiled_max_detections_per_prompt = 24
dino_enable_area_split = False  # Keep original DINO boxes; do not subdivide into smaller boxes
dino_validate_split_boxes = False
dino_validate_split_max_candidates = 120
dino_nms_iou_threshold = 0.55
dino_negative_overlap_iou_threshold = 0.35
dino_max_boxes_per_prompt_for_sam = 200
dino_refine_bounds = False  # Disable extra per-box DINO passes to reduce runtime and memory spikes
dino_refine_bounds_max_depth = 1  # Keep refinement shallow to avoid excessive compute
dino_refine_bounds_min_area_ratio = 0.80  # Accept refinements only when they shrink meaningfully

# DINO prompt selection - see prompts.py for all available prompts
# Each prompt gets its own DINO run, then all detections are merged before SAM.
from models.prompts import AVAILABLE_PROMPTS

# NEN 8100 Wind Comfort Categories merged into three practical groups
# A/B: comfortable, vegetated, sheltered outdoor areas
# C/D: pedestrian-accessible but uncomfortable, exposed, and hardscape-heavy areas
# E: highways, roofs, parking areas, and other no-access hardscape
ACTIVE_PROMPTS = [
    "nen_cat_a",
    "nen_cat_b",
    "nen_cat_c",
    "nen_cat_d",
    "nen_cat_e",
]

# When enabled, every pixel is assigned to the active prompt with the highest score,
# and uncovered pixels are filled by the nearest labeled region.
full_image_mask_mode = True

# Pixel assignment mode for the final full-image mask. Use "legacy" to keep the
# current thresholded/full-image behavior, "contrastive" to assign pixels by
# contrastive CLIP scores without the later tier thresholds, or "region_context"
# to score each SAM region against all prompts and assign the whole region once.
pixel_assignment_mode = "region_context"  # "legacy", "contrastive", or "region_context"

# Tier threshold refinement - prevents excessive uncomfortable (E) tier assignments
# Only assign E tier if CLIP score > tier_e_threshold; otherwise fall back to C or A
# Only assign C tier if CLIP score > tier_c_threshold; otherwise fall back to A
# Set to 0 to disable threshold filtering
tier_e_threshold = 0.16  # Only assign E if it is clearly transport-infrastructure-like
tier_d_threshold = 0.05  # Make D easier to select for exposed hardscape and parking
tier_c_threshold = 0.06  # Only assign C if moderately confident
tier_b_threshold = 0.03  # Make B easier to select for pedestrian-friendly campus/residential areas

# Assignment weights - lower A so it is harder to win, boost B while keeping D more moderate so
# pedestrian-friendly classes win more often without letting exposed-hardscape dominate.
contrastive_prompt_weights = {
    "nen_cat_a": 0.68,
    "nen_cat_b": 1.25,
    "nen_cat_c": 1.10,
    "nen_cat_d": 1.10,
    "nen_cat_e": 0.95,
}

# Auto-build dino_prompt_configs from selected prompts
dino_prompt_configs = [
    {"name": name, **AVAILABLE_PROMPTS[name]}
    for name in ACTIVE_PROMPTS
    if name in AVAILABLE_PROMPTS
]

# SAM settings
sam_model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"
sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
sam_device = "auto"  # "auto", "cpu", "cuda"
sam_full_resolution = True  # Keep SAM input at full image resolution
runtime_profile = "fast"  # options: "fast", "balanced", "quality"
sam_prompt_box_expand_factors = {
    "park": 1.7,  # Expand DINO park seed boxes before SAM to capture broader park context
}

# When use_dino=False, SAM will auto-generate masks. These settings control the auto-generation:
sam_points_per_side = 16  # Grid density for automatic mask generation
sam_pred_iou_thresh = 0.70  # Prediction IoU threshold for filtering masks
sam_stability_score_thresh = 0.80  # Stability score threshold for filtering masks
sam_min_mask_area_px = 50000  # Drop very small SAM masks so the output stays coarse
# Coarse-to-fine smoothing for the final label map. Set to 0 or 1 to disable.
coarse_to_fine_cell_px = 0

# Low-memory automatic SAM fallback settings for large rasters
sam_auto_tile_size_px = 1200
sam_auto_tile_overlap_px = 240
sam_auto_max_points_per_side = 12
sam_auto_max_total_masks = 5000

# Model input conversion settings (applied when source image is not uint8)
model_input_use_robust_uint8 = True
model_input_percentile_low = 1.0
model_input_percentile_high = 99.0

# CLIP settings - all per-prompt settings are now in prompts.py
