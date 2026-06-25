"""
Prediction model script using Grounding DINO to segment custom objects.
Reads prompt descriptions and thresholds from config.json.
"""

import os
import sys
import argparse
import json
import warnings
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Path Setup
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT, ".."))
GSAM_ROOT = os.path.join(REPO_ROOT, "GroundingSAM pipeline")

# Put GSAM_ROOT at the front of sys.path so 'import models' finds GroundingSAM's models package
if GSAM_ROOT not in sys.path:
    sys.path.insert(0, GSAM_ROOT)

try:
    from models.dino_processing import build_dino_model_and_transform, run_dino_prompts
except ImportError as err:
    print(f"[-] Could not import GroundingDINO helpers from GroundingSAM pipeline: {err}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Config Class
# ─────────────────────────────────────────────────────────────────────────────
class DINOConfigWrapper:
    def __init__(self, prompt_configs, dino_device="auto", dino_tile_size_px=4096, dino_tile_overlap_px=384):
        self.dino_prompt_configs = prompt_configs
        self.dino_device = dino_device
        self.dino_full_resolution = False
        self.dino_resize_short_side = 1200
        self.dino_resize_max_size = 2000
        self.dino_enable_tiled_fallback = True
        self.dino_tile_size_px = dino_tile_size_px
        self.dino_tile_overlap_px = dino_tile_overlap_px
        self.dino_validate_split_boxes = False
        self.dino_validate_split_max_candidates = 120
        self.dino_enable_area_split = False
        self.dino_nms_iou_threshold = 0.55
        self.dino_max_boxes_per_prompt_for_sam = 200
        self.dino_negative_overlap_iou_threshold = 0.35
        self.dino_refine_bounds = False


# Geometric constraints to filter out DINO hallucinations and false positives
DINO_CONSTRAINTS = {
    "seating": {
        "min_m2": 0.2,
        "max_m2": 250.0,           # Benches are small; this max area filters out parking lots
        "max_chip_fraction": 0.20 # Rejects boxes larger than 20% of the chip
    },
    "garden": {
        "min_m2": 3.0,
        "max_m2": 2500.0,          
        "max_chip_fraction": 0.80 # Rejects giant "global fallback" boxes
    }
}

def load_dino_configs_from_json(config_path):
    """Load config.json and extract DINO classes, thresholds, and prompts."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)

    dino_model_cfg = None
    for model in cfg_data.get("models", []):
        if model.get("name") == "dino_model":
            dino_model_cfg = model
            break

    if not dino_model_cfg:
        raise ValueError("Could not find 'dino_model' configuration section in config.json")

    classes_cfg = dino_model_cfg.get("classes", {})
    prompt_configs = []

    for class_key, class_info in classes_cfg.items():
        name = class_info["name"]
        threshold = float(class_info["threshold"])
        prompt = class_info["prompt"]

        # Parse keywords to validate local matches
        raw_keywords = [k.strip().lower() for k in prompt.split(".") if k.strip()]
        keywords = []
        for kw in raw_keywords:
            keywords.append(kw)
            for w in kw.split():
                keywords.append(w)
        keywords = tuple(set(keywords))

        prompt_configs.append({
            "name": name,
            "caption": prompt,
            "box_threshold": threshold,
            "text_threshold": max(0.05, threshold - 0.05),
            "keywords": keywords,
            "min_box_side_px": 8,
            "enable_tiled_fallback": True,
        })

    return prompt_configs, classes_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Execution Logic
# ─────────────────────────────────────────────────────────────────────────────
def run_dino_prediction(image_path, output_path, config_path, overlap, cell_size):
    print(f"[*] Grounding DINO detection started")
    print(f"    image     : {os.path.basename(image_path)}")
    print(f"    config    : {config_path}")

    # Load prompts
    prompt_configs, classes_cfg = load_dino_configs_from_json(config_path)
    class_names = sorted(list(classes_cfg.keys()))
    num_classes = len(class_names)
    class_to_id = {name: classes_cfg[name].get("id", i + 12) for i, name in enumerate(class_names)}
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"[*] Configured DINO prompts:")
    for pc in prompt_configs:
        print(f"    - {pc['name']}: '{pc['caption']}' (th={pc['box_threshold']})")

    # Resample image to cell_size
    with rasterio.open(image_path) as src:
        native_res = abs(src.transform.a)
        crs = src.crs

        if cell_size <= 0:
            cell_size = native_res
            w, h = src.width, src.height
            tx = src.transform
            print(f"Using native resolution: {native_res:.4f}m/px ({w}x{h})")
            img_data = src.read([1, 2, 3])
        else:
            scale = native_res / cell_size
            w, h = round(src.width * scale), round(src.height * scale)
            tx = Affine(cell_size, 0, src.transform.c, 0, -cell_size, src.transform.f)
            print(f"Resampling native {native_res:.4f}m/px -> target {cell_size:.4f}m/px ({src.width}x{src.height} -> {w}x{h})")
            img_data = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)

    # Convert to standard HWC RGB layout
    img_rgb = np.moveaxis(img_data, 0, -1)

    # Wrap config and build DINO model
    dino_cfg = DINOConfigWrapper(prompt_configs)
    print("Loading Grounding DINO model...")
    dino_model, dino_transform, device = build_dino_model_and_transform(dino_cfg)

    if dino_model is None:
        print("[-] Skipping DINO prediction because GroundingDINO is not available.")
        # Save empty geojson
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
        return

    # Run predictions
    detections = run_dino_prompts(
        img_rgb,
        dino_cfg,
        dino_model,
        dino_transform,
        device,
        pixels_per_meter_sq=None,
        return_unfiltered=False,
    )

    print(f"[*] DINO finished. Received {len(detections)} raw bounding boxes.")

    # Format output features and generate probability grids
    features = []
    prob_accum = np.zeros((num_classes, h, w), dtype=np.float32)

    for det in detections:
        box_coords = det["box"]  # [x1, y1, x2, y2]
        score = det["score"]
        prompt_group = det["prompt_group"]

        if prompt_group not in class_to_idx:
            continue

        c_idx = class_to_idx[prompt_group]
        x1, y1, x2, y2 = [int(v) for v in box_coords]

        # ─────────────────────────────────────────────────────────────
        # Geometric Screening Filters
        # ─────────────────────────────────────────────────────────────
        constraints = DINO_CONSTRAINTS.get(prompt_group, {})
        w_b = x2 - x1
        h_b = y2 - y1
        
        # Filter 1: Reject DINO global fallback boxes (too large for the tile)
        max_frac = constraints.get("max_chip_fraction", 1.0)
        tile_size = dino_cfg.dino_tile_size_px
        if (w_b / tile_size) > max_frac or (h_b / tile_size) > max_frac:
            continue 
        
        # Filter 2: Strict Area Screening (in square meters)
        area_m2 = (w_b * cell_size) * (h_b * cell_size)
        min_m2 = constraints.get("min_m2", 0.0)
        max_m2 = constraints.get("max_m2", 99999.0)
        
        if area_m2 < min_m2 or area_m2 > max_m2:
            continue
        # ─────────────────────────────────────────────────────────────

        # Rasterize confidence score onto probability map
        y0_c, y1_c = max(0, y1), min(h, y2)
        x0_c, x1_c = max(0, x1), min(w, x2)
        if y1_c > y0_c and x1_c > x0_c:
            prob_accum[c_idx, y0_c:y1_c, x0_c:x1_c] = np.maximum(
                prob_accum[c_idx, y0_c:y1_c, x0_c:x1_c],
                float(score)
            )

        # Convert to map coordinates
        map_left, map_top = tx * (x1, y1)
        map_right, map_bottom = tx * (x2, y2)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [map_left, map_bottom],
                    [map_right, map_bottom],
                    [map_right, map_top],
                    [map_left, map_top],
                    [map_left, map_bottom],
                ]],
            },
            "properties": {
                "Class": prompt_group,
                "Class_ID": class_to_id[prompt_group],
                "Confidence": float(score),
            },
        })

    # Save geojson
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
    print(f"[+] Saved {len(features)} DINO features to: {output_path}")

    # Save multi-band probability raster
    prob_output_path = output_path.replace(".geojson", ".prob.tif")
    try:
        with rasterio.open(
            prob_output_path, "w",
            driver="GTiff",
            height=h, width=w,
            count=num_classes,
            dtype=rasterio.float32,
            crs=crs,
            transform=tx,
            compress="lzw"
        ) as dst:
            for idx in range(num_classes):
                dst.write(prob_accum[idx], idx + 1)
        print(f"[+] Saved DINO probability map to: {prob_output_path}")
    except Exception as err:
        print(f"[-] WARNING: Failed to write DINO probability map: {err}")


def main():
    parser = argparse.ArgumentParser(description="DINO Prediction wrapper for Amenity Pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=os.path.join(PIPELINE_ROOT, "config.json"))
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--cell_size", type=float, default=0.3)
    args = parser.parse_args()

    run_dino_prediction(
        image_path=args.image,
        output_path=args.output,
        config_path=args.config,
        overlap=args.overlap,
        cell_size=args.cell_size,
    )


if __name__ == "__main__":
    main()
