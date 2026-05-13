import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
from matplotlib.colors import rgb_to_hsv
from time import perf_counter
from datetime import datetime
import gc

# --------------------------
# 1. Path to your large GeoTIFF
# --------------------------
tif_file = "gtcampus.tif"

# Optional crop control. Set to False to run segmentation on the full image.
use_bbox_crop = False
bbox_lonlat = (-84.4010, 33.7720, -84.3950, 33.7760)

# Heatmap grid cell area in square meters. Example: 20m x 20m = 400 m^2.
amenity_grid_cell_area_m2 = 400.0

# SAM memory controls (preserve resolution, reduce peak RAM).
# SAM works at 1024px internally; tiles larger than this cause huge upsampling allocations.
runtime_profile = "fast"  # options: "fast", "balanced", "quality"
sam_points_per_batch = 16
sam_tile_size_px = 1024
sam_tile_overlap_px = 0
sam_enable_tiled_fallback = True
max_masks_per_tile = 80
max_masks_for_clip_scoring = 900


script_start = perf_counter()


def log_stage(message: str, start_time: float | None = None) -> None:
    if start_time is None:
        print(f"[INFO] {message}")
        return
    elapsed = perf_counter() - start_time
    print(f"[INFO] {message} ({elapsed:.2f}s)")


log_stage("Starting satellite sidewalk segmentation pipeline")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Saving output figures to: {results_dir.resolve()}")
saved_figure_paths: list[Path] = []


def save_current_figure(filename: str, category: str) -> None:
    category_dir = results_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    output_path = category_dir / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_figure_paths.append(output_path)
    print(f"[INFO] Saved figure: {output_path}")


def get_runtime_profile_settings(profile_name: str, run_device: str) -> dict:
    profile = profile_name.lower().strip()

    if profile == "quality":
        return {
            "points_per_side": 64,
            "crop_n_layers": 1,
            "pred_iou_thresh": 0.75,
            "stability_score_thresh": 0.85,
            "box_nms_thresh": 0.60,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 80,
        }

    if profile == "balanced":
        return {
            "points_per_side": 40,
            "crop_n_layers": 0,
            "pred_iou_thresh": 0.78,
            "stability_score_thresh": 0.87,
            "box_nms_thresh": 0.65,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 120,
        }

    # Fast profile is tuned for CPU throughput.
    return {
        "points_per_side": 24 if run_device == "cpu" else 32,
        "crop_n_layers": 0,
        "pred_iou_thresh": 0.80,
        "stability_score_thresh": 0.90,
        "box_nms_thresh": 0.70,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 180,
    }


def is_out_of_memory_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "not enough memory" in msg or "out of memory" in msg


def iter_tile_coords(height: int, width: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    step = max(1, tile_size - overlap)
    coords: list[tuple[int, int, int, int]] = []
    for y0 in range(0, height, step):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, step):
            x1 = min(width, x0 + tile_size)
            coords.append((y0, y1, x0, x1))
    return coords


def score_sam_quality(mask_dict: dict) -> float:
    iou = float(mask_dict.get("predicted_iou", 0.0))
    stability = float(mask_dict.get("stability_score", 0.0))
    area = float(mask_dict.get("area", 0.0))
    return 0.6 * iou + 0.4 * stability - 1e-7 * area


def prune_masks_by_quality(mask_list: list[dict], max_count: int) -> list[dict]:
    if max_count <= 0 or len(mask_list) <= max_count:
        return mask_list
    ranked = sorted(mask_list, key=score_sam_quality, reverse=True)
    return ranked[:max_count]


def remap_mask_to_full_image(mask_dict: dict, full_shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> dict:
    tile_seg = mask_dict["segmentation"]
    remapped = dict(mask_dict)
    # Keep segmentation tile-local to avoid allocating one full-image array per mask.
    remapped["tile_bounds"] = (y0, y1, x0, x1)
    remapped["full_shape"] = full_shape
    remapped["area"] = int(tile_seg.sum())

    if "bbox" in remapped and len(remapped["bbox"]) == 4:
        bx, by, bw, bh = remapped["bbox"]
        remapped["bbox"] = [float(bx + x0), float(by + y0), float(bw), float(bh)]

    if "point_coords" in remapped and remapped["point_coords"] is not None:
        remapped["point_coords"] = [
            [float(px + x0), float(py + y0)] for px, py in remapped["point_coords"]
        ]

    if "crop_box" in remapped and len(remapped["crop_box"]) == 4:
        c0, c1, c2, c3 = remapped["crop_box"]
        remapped["crop_box"] = [float(c0 + x0), float(c1 + y0), float(c2 + x0), float(c3 + y0)]

    return remapped


def expand_mask_to_full_image(mask_dict: dict) -> np.ndarray:
    """Expand a tile-local mask segmentation to a full-image bool array on demand."""
    if "tile_bounds" not in mask_dict:
        return mask_dict["segmentation"]
    ty0, ty1, tx0, tx1 = mask_dict["tile_bounds"]
    full_h, full_w = mask_dict["full_shape"]
    full_seg = np.zeros((full_h, full_w), dtype=bool)
    full_seg[ty0:ty1, tx0:tx1] = mask_dict["segmentation"]
    return full_seg


def generate_masks_tiled(mask_generator: "SamAutomaticMaskGenerator", image: np.ndarray) -> list[dict]:
    h, w = image.shape[:2]
    tile_coords = iter_tile_coords(h, w, sam_tile_size_px, sam_tile_overlap_px)
    print(
        "[INFO] Running tiled SAM generation at full resolution "
        f"({len(tile_coords)} tiles, tile={sam_tile_size_px}px, overlap={sam_tile_overlap_px}px)"
    )

    all_masks: list[dict] = []
    stage_start = perf_counter()
    for idx, (y0, y1, x0, x1) in enumerate(tile_coords, start=1):
        tile_img = image[y0:y1, x0:x1]
        tile_masks = mask_generator.generate(tile_img)
        tile_masks = prune_masks_by_quality(tile_masks, max_masks_per_tile)

        for tm in tile_masks:
            all_masks.append(remap_mask_to_full_image(tm, (h, w), y0, y1, x0, x1))

        if idx == 1 or idx % 2 == 0 or idx == len(tile_coords):
            elapsed = perf_counter() - stage_start
            rate = idx / max(elapsed, 1e-6)
            eta = (len(tile_coords) - idx) / max(rate, 1e-6)
            print(
                f"[PROGRESS] [SAM tiled] Tile {idx}/{len(tile_coords)} "
                f"({rate:.2f} tiles/s, ETA {eta:.1f}s, masks={len(all_masks)})"
            )

        del tile_img
        del tile_masks
        gc.collect()

    log_stage(f"Tiled SAM generation complete: {len(all_masks)} masks", stage_start)
    return all_masks


def generate_masks_memory_safe(mask_generator: "SamAutomaticMaskGenerator", image: np.ndarray) -> list[dict]:
    stage_start = perf_counter()
    print("[INFO] Generating masks with SAM (memory-safe mode)")
    try:
        masks_local = mask_generator.generate(image)
        masks_local = prune_masks_by_quality(masks_local, max_masks_for_clip_scoring)
        log_stage(f"SAM mask generation complete: {len(masks_local)} masks", stage_start)
        return masks_local
    except RuntimeError as exc:
        if not sam_enable_tiled_fallback or not is_out_of_memory_error(exc):
            raise
        print(
            "[WARN] SAM ran out of memory on full image. "
            "Falling back to tiled generation to preserve full-resolution efficacy."
        )
        gc.collect()
        return generate_masks_tiled(mask_generator, image)


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

        # Report full-image extent in meters using a projected CRS.
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


report_geotiff_spatial_info(tif_file, use_crop=use_bbox_crop, lonlat_bbox=bbox_lonlat)


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

            # Input bbox is lon/lat (EPSG:4326); convert to dataset CRS when needed.
            if src.crs and src.crs.to_string() != "EPSG:4326":
                data_bounds = transform_bounds("EPSG:4326", src.crs, lon_min, lat_min, lon_max, lat_max)
            else:
                data_bounds = (lon_min, lat_min, lon_max, lat_max)

            # Ensure requested bounds overlap raster extent.
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


img = load_rgb_image(tif_file, use_crop=use_bbox_crop, lonlat_bbox=bbox_lonlat)
loaded_extent_m = get_loaded_extent_meters(tif_file, use_crop=use_bbox_crop, lonlat_bbox=bbox_lonlat)

if loaded_extent_m is None:
    print("[WARN] Could not estimate loaded image extent in meters; amenity heatmap will be skipped.")
else:
    print(
        "[INFO] Loaded image real size (approx meters): "
        f"{loaded_extent_m[0]:.2f} m x {loaded_extent_m[1]:.2f} m"
    )

if img.size == 0:
    raise RuntimeError("Cropped image is empty. Check bbox values and raster coverage.")

# --------------------------
# 5. Normalize to 0-255
# --------------------------
img = img.astype(float)
img_min = img.min()
img_max = img.max()
if img_max == img_min:
    raise RuntimeError("Cropped image has constant pixel values; cannot normalize display range.")
img = (img - img_min) / (img_max - img_min)
img = (img * 255).astype(np.uint8)
print(f"[INFO] Cropped image shape: {img.shape[0]}x{img.shape[1]} ({img.shape[2]} channels)")

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title("Input Aerial Image")
input_image_filename = f"{Path(tif_file).stem}_input_image.png"
save_current_figure(input_image_filename, "input_images")

# --------------------------
# 6. Display cropped image
# --------------------------
'''
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title("Georgia Tech Campus Aerial Image (Cropped)")
plt.show()
'''

# --------------------------
# Grounding DINO (text → bounding boxes)
# --------------------------
from groundingdino.util.inference import load_model, predict
import groundingdino
import groundingdino.datasets.transforms as T
from PIL import Image
import torch

print("[INFO] Running Grounding DINO detection")


def ensure_download(path: str, urls: list[str]) -> None:
    target = Path(path)
    if target.exists():
        return

    print(f"[INFO] Downloading model checkpoint: {target.name}")
    last_error: Exception | None = None
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                with target.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print(f"[INFO] Downloaded checkpoint to: {target.resolve()}")
            return
        except Exception as exc:
            last_error = exc
            if target.exists():
                target.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to download checkpoint: {target.name}") from last_error


dino_pkg_dir = Path(groundingdino.__file__).resolve().parent
dino_config_path = dino_pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
dino_checkpoint_path = Path("groundingdino_swint_ogc.pth")
dino_checkpoint_urls = [
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/alpha/groundingdino_swint_ogc.pth",
]
ensure_download(str(dino_checkpoint_path), dino_checkpoint_urls)

dino_model = load_model(
    str(dino_config_path),
    str(dino_checkpoint_path),
)

TEXT_PROMPT = (
    "sidewalk . pedestrian walkway . pavement . "
    "bench . park bench . outdoor table . plaza seating"
)
dino_device = "cuda" if torch.cuda.is_available() else "cpu"

dino_transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img_pil = Image.fromarray(img)
img_for_dino, _ = dino_transform(img_pil, None)

boxes, logits, phrases = predict(
    model=dino_model,
    image=img_for_dino,
    caption=TEXT_PROMPT,
    box_threshold=0.3,
    text_threshold=0.25,
    device=dino_device,
)

# Convert boxes to pixel coords
h, w, _ = img.shape
boxes_scaled = boxes * torch.tensor([w, h, w, h])
boxes_xyxy = torch.zeros_like(boxes_scaled)
boxes_xyxy[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2.0
boxes_xyxy[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2.0
boxes_xyxy[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2.0
boxes_xyxy[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2.0
boxes_xyxy = boxes_xyxy.cpu().numpy().astype(int)

print(f"[INFO] Grounding DINO found {len(boxes_xyxy)} objects")

dino_keep_keywords = (
    "sidewalk",
    "walkway",
    "pavement",
    "bench",
    "table",
    "seating",
    "plaza",
    "pedestrian",
)

filtered_dino_pairs: list[tuple[np.ndarray, str]] = []
for box, phrase in zip(boxes_xyxy, phrases):
    phrase_lc = phrase.lower()
    if any(keyword in phrase_lc for keyword in dino_keep_keywords):
        filtered_dino_pairs.append((box, phrase))

if filtered_dino_pairs:
    boxes_xyxy = np.stack([pair[0] for pair in filtered_dino_pairs]).astype(int)
    filtered_phrases = [pair[1] for pair in filtered_dino_pairs]
    print(
        "[INFO] DINO phrase prefilter kept "
        f"{len(filtered_dino_pairs)}/{len(phrases)} detections for SAM"
    )
else:
    filtered_phrases = list(phrases)
    print(
        "[WARN] DINO phrase prefilter removed all detections; "
        "falling back to all DINO boxes for SAM"
    )

# --------------------------
# SAM + CLIP (text prompt) workflow
# --------------------------
from segment_anything import sam_model_registry
from PIL import Image
import open_clip

import torch

# --------------------------
# 1. Load SAM2 weights (vit_h recommended)
# --------------------------
model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"
checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
profile_settings = get_runtime_profile_settings(runtime_profile, device)
print(f"[INFO] Runtime profile: {runtime_profile} ({device})")


def ensure_checkpoint(path: str, url: str) -> None:
    ckpt = Path(path)
    if ckpt.exists():
        return

    print(f"SAM checkpoint not found: {ckpt}. Downloading from {url} ...")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with ckpt.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded checkpoint to {ckpt.resolve()}")


ensure_checkpoint(sam_checkpoint, checkpoint_url)
stage_start = perf_counter()
print(f"[INFO] Loading SAM model '{model_type}' on device '{device}'")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
log_stage("SAM model ready", stage_start)

# --------------------------
# 2. Generate candidate masks with SAM
# --------------------------
from segment_anything import SamPredictor

predictor = SamPredictor(sam)
predictor.set_image(img)

masks = []

print("[INFO] Running SAM refinement on DINO boxes")

for i, (box, phrase) in enumerate(zip(boxes_xyxy, filtered_phrases)):
    x1, y1, x2, y2 = box

    try:
        sam_masks, scores, logits = predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )

        mask_dict = {
            "segmentation": sam_masks[0],
            "predicted_iou": float(scores[0]),
            "stability_score": float(scores[0]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "dino_phrase": phrase,
        }

        masks.append(mask_dict)

    except Exception as e:
        print(f"[WARN] SAM failed on box {i}: {e}")

print(f"[INFO] Generated {len(masks)} masks from DINO + SAM")
# --------------------------
# 3. Rank masks with CLIP text similarity
# --------------------------
sidewalk_prompt = "aerial overhead sidewalk"
sitting_prompt = "aerial overhead outdoor sitting area"
negative_prompt = "building roof, parking lot, trees, grass, water, shadow"
# Favor recall: keep more plausible sidewalk masks before fusion.
top_k = 45
score_threshold = -0.03
relative_score_margin = 0.10
min_area_ratio = 0.00015
max_area_ratio = 0.35
max_saturation_for_sidewalk = 0.28
min_value_for_sidewalk = 0.35

stage_start = perf_counter()
print("[INFO] Loading CLIP model and preprocessing")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
)
clip_model = clip_model.to(device)
clip_model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")
log_stage("CLIP model ready", stage_start)

def build_text_features(target_prompt: str):
    with torch.no_grad():
        pos_templates = [
            f"aerial image of {target_prompt}",
            f"overhead view of {target_prompt}",
            f"satellite photo of {target_prompt}",
            f"{target_prompt} in an urban area",
        ]
        neg_templates = [
            f"aerial image of {negative_prompt}",
            f"overhead view of {negative_prompt}",
        ]

        pos_tokens = tokenizer(pos_templates).to(device)
        neg_tokens = tokenizer(neg_templates).to(device)

        pos_features = clip_model.encode_text(pos_tokens)
        neg_features = clip_model.encode_text(neg_tokens)

        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

        # Template ensemble for more stable text conditioning.
        pos_text_feature = pos_features.mean(dim=0, keepdim=True)
        neg_text_feature = neg_features.mean(dim=0, keepdim=True)
        pos_text_feature = pos_text_feature / pos_text_feature.norm(dim=-1, keepdim=True)
        neg_text_feature = neg_text_feature / neg_text_feature.norm(dim=-1, keepdim=True)

    return pos_text_feature, neg_text_feature


stage_start = perf_counter()
print("[INFO] Encoding sidewalk and sitting-area prompts with CLIP")
sidewalk_pos_feature, sidewalk_neg_feature = build_text_features(sidewalk_prompt)
sitting_pos_feature, sitting_neg_feature = build_text_features(sitting_prompt)
log_stage("Text prompt encoding complete", stage_start)


def score_mask(mask_dict: dict, pos_text_feature: torch.Tensor, neg_text_feature: torch.Tensor) -> float:
    tile_seg = mask_dict["segmentation"]
    tile_bounds = mask_dict.get("tile_bounds")

    ys, xs = np.where(tile_seg)

    if ys.size == 0 or xs.size == 0:
        return -1.0

    # Bounding box in tile-local coordinates.
    ly0, ly1 = int(ys.min()), int(ys.max()) + 1
    lx0, lx1 = int(xs.min()), int(xs.max()) + 1

    # Map to full-image coordinates for img crops.
    if tile_bounds is not None:
        ty0, _, tx0, _ = tile_bounds
        full_h, full_w = mask_dict["full_shape"]
        iy0, iy1 = ly0 + ty0, ly1 + ty0
        ix0, ix1 = lx0 + tx0, lx1 + tx0
    else:
        full_h, full_w = tile_seg.shape[:2]
        iy0, iy1, ix0, ix1 = ly0, ly1, lx0, lx1

    area_ratio = float(tile_seg.sum()) / float(full_h * full_w)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return -1.0

    crop = img[iy0:iy1, ix0:ix1]
    crop_mask = tile_seg[ly0:ly1, lx0:lx1][..., None]

    # Keep only masked pixels; set background white to preserve object boundaries.
    object_only = np.where(crop_mask, crop, 255).astype(np.uint8)
    pil_img = Image.fromarray(object_only)

    with torch.no_grad():
        image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        pos_sim = float((image_features @ pos_text_feature.T).item())
        neg_sim = float((image_features @ neg_text_feature.T).item())
        # Contrastive score: higher means closer to prompt and farther from negatives.
        score = pos_sim - neg_sim

        # Favor SAM masks with stronger internal quality signals.
        iou = float(mask_dict.get("predicted_iou", 0.0))
        stability = float(mask_dict.get("stability_score", 0.0))
        score += 0.05 * iou + 0.05 * stability

    # Geometry prior: sidewalks are often elongated and sparse within their bbox.
    h = max(1, iy1 - iy0)
    w = max(1, ix1 - ix0)
    aspect = max(h, w) / float(min(h, w))
    bbox_fill = float(tile_seg.sum()) / float(h * w)

    elongation_score = np.clip((aspect - 1.5) / 6.0, 0.0, 1.0)
    sparse_fill_score = np.clip((0.65 - bbox_fill) / 0.65, 0.0, 1.0)

    # Keep geometry influence modest so wider/irregular sidewalks are not excluded.
    score += 0.12 * float(elongation_score)
    score += 0.06 * float(sparse_fill_score)

    # Color prior: sidewalks are usually low-saturation and medium/high brightness.
    mask_pixels = img[iy0:iy1, ix0:ix1][tile_seg[ly0:ly1, lx0:lx1]]
    if mask_pixels.size > 0:
        pixels_norm = mask_pixels.astype(np.float32) / 255.0
        hsv = rgb_to_hsv(pixels_norm.reshape(-1, 1, 3)).reshape(-1, 3)
        sat = float(hsv[:, 1].mean())
        val = float(hsv[:, 2].mean())

        low_sat_score = np.clip((max_saturation_for_sidewalk - sat) / max_saturation_for_sidewalk, 0.0, 1.0)
        bright_enough_score = np.clip((val - min_value_for_sidewalk) / (1.0 - min_value_for_sidewalk), 0.0, 1.0)

        # Use soft color priors; lighting and material vary across scenes.
        score += 0.05 * float(low_sat_score)
        score += 0.04 * float(bright_enough_score)

    return score


def select_masks_for_prompt(
    masks_input: list,
    score_key: str,
    pos_text_feature: torch.Tensor,
    neg_text_feature: torch.Tensor,
    prompt_name: str,
) -> tuple[list, float]:
    stage_start = perf_counter()
    print(f"[INFO] Scoring {len(masks_input)} masks for '{prompt_name}'")

    for idx, m in enumerate(masks_input, start=1):
        m[score_key] = score_mask(m, pos_text_feature, neg_text_feature)

        if idx == 1 or idx % 25 == 0 or idx == len(masks_input):
            elapsed = perf_counter() - stage_start
            rate = idx / max(elapsed, 1e-6)
            eta = (len(masks_input) - idx) / max(rate, 1e-6)
            print(
                f"[PROGRESS] [{prompt_name}] Scored {idx}/{len(masks_input)} masks "
                f"({rate:.2f} masks/s, ETA {eta:.1f}s)"
            )
    log_stage(f"Mask scoring complete for '{prompt_name}'", stage_start)

    filtered = [m for m in masks_input if m[score_key] >= 0.0]
    filtered.sort(key=lambda m: m[score_key], reverse=True)

    if filtered:
        best_score = float(filtered[0][score_key])
        dynamic_threshold = max(score_threshold, best_score - relative_score_margin)
    else:
        dynamic_threshold = score_threshold

    selected = [m for m in filtered if m[score_key] >= dynamic_threshold][:top_k]

    # Keep enough candidates when scene scores are tightly clustered.
    if len(selected) < 10:
        selected = filtered[: min(top_k, max(10, len(filtered)))]

    if not selected:
        selected = filtered[: min(top_k, 5)]

    if not selected:
        raise RuntimeError(f"No valid masks were generated/scored for prompt '{prompt_name}'.")

    print(f"Prompt: {prompt_name}")
    print(f"Negative prompt: {negative_prompt}")
    print(f"Selection threshold used: {dynamic_threshold:.3f}")
    print(f"Selected {len(selected)} / {len(masks_input)} masks by CLIP score")
    print(
        "Top scores:",
        ", ".join(f"{m[score_key]:.3f}" for m in selected[:5]),
    )

    return selected, dynamic_threshold


def build_amenity_heatmap(
    amenity_mask: np.ndarray,
    image_shape: tuple[int, int, int],
    extent_meters: tuple[float, float],
    cell_area_m2: float,
) -> tuple[np.ndarray, int, int, float]:
    """Convert amenity mask into a square-grid heatmap based on real-world cell area."""
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

    # Edge cells are clipped to image bounds when dimensions are not divisible.
    for y0 in range(0, h, cell_px_h):
        y1 = min(h, y0 + cell_px_h)
        for x0 in range(0, w, cell_px_w):
            x1 = min(w, x0 + cell_px_w)
            cell = amenity_mask[y0:y1, x0:x1]
            density = float(cell.mean()) if cell.size > 0 else 0.0
            heatmap[y0:y1, x0:x1] = density

    return heatmap, cell_px_w, cell_px_h, side_m


selected_sidewalk_masks, sidewalk_threshold = select_masks_for_prompt(
    masks,
    "clip_score_sidewalk",
    sidewalk_pos_feature,
    sidewalk_neg_feature,
    sidewalk_prompt,
)

selected_sitting_masks, sitting_threshold = select_masks_for_prompt(
    masks,
    "clip_score_sitting",
    sitting_pos_feature,
    sitting_neg_feature,
    sitting_prompt,
)


# def weighted_fusion_mask(masks_with_scores: list, image_shape: tuple) -> np.ndarray:
#     """Build a single automatic sidewalk mask from top-scoring candidates."""
#     h, w = image_shape[:2]
#     vote_map = np.zeros((h, w), dtype=np.float32)
#
#     valid_scores = [m["clip_score"] for m in masks_with_scores if m["clip_score"] > 0]
#     if not valid_scores:
#         return np.zeros((h, w), dtype=bool)
#
#     max_score = max(valid_scores)
#     min_score = min(valid_scores)
#     denom = max(1e-6, max_score - min_score)
#
#     for m in masks_with_scores:
#         s = float(m["clip_score"])
#         if s <= 0:
#             continue
#         norm_s = (s - min_score) / denom
#         weight = 0.25 + 0.75 * norm_s
#         vote_map[m["segmentation"]] += weight
#
#     # Adaptive threshold keeps masks robust across scenes with different score scales.
#     nonzero_votes = vote_map[vote_map > 0]
#     if nonzero_votes.size == 0:
#         return np.zeros((h, w), dtype=bool)
#
#     # Lower threshold to preserve more borderline sidewalk segments.
#     threshold = float(np.quantile(nonzero_votes, 0.40))
#     fused = vote_map >= threshold
#     return fused


# Fused sidewalk-only model is intentionally disabled for now.
# sidewalk_mask = weighted_fusion_mask(selected_sidewalk_masks, img.shape)
# print(f"[INFO] Fused sidewalk mask pixels: {int(sidewalk_mask.sum())}")

# --------------------------
# 4. Visualize CLIP-selected masks
# --------------------------
sidewalk_overlay = np.zeros((*img.shape[:2], 4), dtype=np.float32)
for m in selected_sidewalk_masks:
    seg = expand_mask_to_full_image(m)
    sidewalk_overlay[seg, :3] = np.array([0.1, 0.9, 0.9], dtype=np.float32)
    sidewalk_overlay[seg, 3] = 0.35

sitting_overlay = np.zeros((*img.shape[:2], 4), dtype=np.float32)
for m in selected_sitting_masks:
    seg = expand_mask_to_full_image(m)
    sitting_overlay[seg, :3] = np.array([1.0, 0.65, 0.1], dtype=np.float32)
    sitting_overlay[seg, 3] = 0.35

combined_overlay = np.zeros((*img.shape[:2], 4), dtype=np.float32)
combined_overlay[..., :3] = 0.0
combined_overlay[..., 3] = 0.0

sidewalk_mask_combined = sidewalk_overlay[..., 3] > 0
sitting_mask_combined = sitting_overlay[..., 3] > 0
overlap_mask = sidewalk_mask_combined & sitting_mask_combined

combined_overlay[sidewalk_mask_combined, :3] = np.array([0.1, 0.9, 0.9], dtype=np.float32)
combined_overlay[sidewalk_mask_combined, 3] = 0.32
combined_overlay[sitting_mask_combined, :3] = np.array([1.0, 0.65, 0.1], dtype=np.float32)
combined_overlay[sitting_mask_combined, 3] = 0.32
combined_overlay[overlap_mask, :3] = np.array([0.8, 0.3, 0.95], dtype=np.float32)
combined_overlay[overlap_mask, 3] = 0.45

viz_stride = max(1, int(np.ceil(max(img.shape[:2]) / 2000)))
if viz_stride > 1:
    print(f"[INFO] Downsampling visualization by stride={viz_stride} to reduce memory use")
viz_img = img[::viz_stride, ::viz_stride]
viz_sidewalk_overlay = sidewalk_overlay[::viz_stride, ::viz_stride]
viz_sitting_overlay = sitting_overlay[::viz_stride, ::viz_stride]
viz_combined_overlay = combined_overlay[::viz_stride, ::viz_stride]

plt.figure(figsize=(10, 10))
plt.imshow(viz_img)
plt.imshow(viz_sidewalk_overlay)
plt.axis("off")
plt.title(f"SAM + CLIP Sidewalk Masks: '{sidewalk_prompt}'")
save_current_figure(f"{run_id}_sidewalk_masks.png", "individual_masks")

plt.figure(figsize=(10, 10))
plt.imshow(viz_img)
plt.imshow(viz_sitting_overlay)
plt.axis("off")
plt.title(f"SAM + CLIP Sitting-Area Masks: '{sitting_prompt}'")
save_current_figure(f"{run_id}_sitting_area_masks.png", "individual_masks")

plt.figure(figsize=(10, 10))
plt.imshow(viz_img)
plt.imshow(viz_combined_overlay)
plt.axis("off")
plt.title("Combined Sidewalk + Sitting-Area Masks")
save_current_figure(f"{run_id}_combined_masks.png", "combined_masks")

# --------------------------
# 5. Amenity heatmap (grid-based)
# --------------------------
amenity_mask_union = sidewalk_mask_combined | sitting_mask_combined

if loaded_extent_m is not None:
    stage_start = perf_counter()
    print(
        "[INFO] Building amenity heatmap grid from mask union "
        f"(cell area: {amenity_grid_cell_area_m2:.2f} m^2)"
    )
    amenity_heatmap, cell_px_w, cell_px_h, cell_side_m = build_amenity_heatmap(
        amenity_mask_union,
        img.shape,
        loaded_extent_m,
        amenity_grid_cell_area_m2,
    )
    log_stage("Amenity heatmap complete", stage_start)

    alpha_map = np.where(amenity_heatmap > 0, np.clip(amenity_heatmap, 0.10, 0.90), 0.0)
    print(
        "[INFO] Heatmap grid cell size (approx): "
        f"{cell_side_m:.2f}m x {cell_side_m:.2f}m "
        f"(~{cell_px_w} x {cell_px_h} px)"
    )

    plt.figure(figsize=(10, 10))
    viz_amenity_heatmap = amenity_heatmap[::viz_stride, ::viz_stride]
    viz_alpha_map = alpha_map[::viz_stride, ::viz_stride]
    plt.imshow(viz_img)
    plt.imshow(viz_amenity_heatmap, cmap="Reds", alpha=viz_alpha_map, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title("Amenity Heatmap (Red = Higher Amenity Coverage)")
    save_current_figure(f"{run_id}_amenity_heatmap.png", "heatmaps")
else:
    print("[WARN] Skipping amenity heatmap because real-world extent is unavailable.")

print(f"[INFO] Total figures saved this run: {len(saved_figure_paths)}")
for p in saved_figure_paths:
    print(f"[INFO] -> {p}")

log_stage("Pipeline complete", script_start)