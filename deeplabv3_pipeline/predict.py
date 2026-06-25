from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GroundingSAMpipeline.models.iou_comparator import compare_annotation_iou
from yoloseg_pipeline.common import CLASS_NAMES, PROJECT_ROOT, load_rgb_from_tif

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

# ImageNet normalisation (must match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREDICTION_CLASS_COLORS = {
    "NEN_A": np.array([0.00, 0.45, 0.20], dtype=np.float32),
    "NEN_B": np.array([0.45, 0.80, 0.25], dtype=np.float32),
    "NEN_C": np.array([1.00, 0.90, 0.20], dtype=np.float32),
    "NEN_D": np.array([1.00, 0.55, 0.10], dtype=np.float32),
    "Uncomfortable": np.array([0.90, 0.12, 0.12], dtype=np.float32),
}

PRED_TO_IOU_CLASS_NAMES = {
    "NEN_A": "nen_cat_a",
    "NEN_B": "nen_cat_b",
    "NEN_C": "nen_cat_c",
    "NEN_D": "nen_cat_d",
    "Uncomfortable": "nen_cat_e",
}

PREVIEW_MAX_DIM = 1200


# --------------------------------------------------
# CLI
# --------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepLabV3+ segmentation inference on a tif or folder of tifs."
    )
    parser.add_argument("--weights", required=True, help="Path to a trained DeepLabV3+ checkpoint (best.pt).")
    parser.add_argument("--source", required=True, help="A tif file or a folder containing tif files.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "deeplabv3_predictions"))
    parser.add_argument("--imgsz", type=int, default=512, help="Model input resolution (tiles are processed at this size).")
    parser.add_argument("--tile-size", type=int, default=1024, help="Sliding-window tile size in source pixels.")
    parser.add_argument("--tile-overlap", type=int, default=128, help="Pixel overlap between adjacent tiles.")
    parser.add_argument("--annotation-path", default=None, help="Optional annotation JSON/XML for IoU evaluation.")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--force-best-guess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fill any remaining background pixels with the nearest predicted class.",
    )
    return parser.parse_args()


# --------------------------------------------------
# Model loading
# --------------------------------------------------


def load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained DeepLabV3+ model from a checkpoint file."""
    from deeplabv3_pipeline.train import build_model

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    run_args = ckpt.get("run_args", {})
    backbone = run_args.get("backbone", "resnet101")

    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# --------------------------------------------------
# Tiled inference
# --------------------------------------------------


def _tile_ranges(length: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """Generate tile coordinates, safely stepping back to avoid edge-clipping."""
    if length <= tile_size:
        return [(0, length)]
        
    step = max(1, tile_size - overlap)
    ranges: list[tuple[int, int]] = []
    start = 0
    
    while start < length - tile_size:
        ranges.append((start, start + tile_size))
        start += step
        
    # Force the final tile to snap to the right/bottom edge perfectly,
    # guaranteeing no non-square crops that would get deformed.
    ranges.append((length - tile_size, length))
    
    # Remove duplicates if step perfectly aligns with the edge
    return list(dict.fromkeys(ranges))


def _preprocess_chip(chip: np.ndarray, imgsz: int, device: torch.device) -> torch.Tensor:
    """Convert an RGB numpy chip to a normalised model-input tensor."""
    pil_img = Image.fromarray(chip).resize((imgsz, imgsz), Image.BILINEAR)
    tensor = transforms.ToTensor()(pil_img)
    tensor = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    return tensor.unsqueeze(0).to(device)


def predict_tiled(
    model: torch.nn.Module,
    rgb: np.ndarray,
    tile_size: int,
    overlap: int,
    imgsz: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Run sliding-window inference and stitch softmax probabilities."""
    height, width = rgb.shape[:2]

    # Use float32 to prevent massive RAM overhead on gigapixel images
    prob_sum = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    y_ranges = _tile_ranges(height, tile_size, overlap)
    x_ranges = _tile_ranges(width, tile_size, overlap)

    with torch.no_grad():
        for y0, y1 in y_ranges:
            for x0, x1 in x_ranges:
                chip = rgb[y0:y1, x0:x1]
                if chip.size == 0:
                    continue

                chip_h, chip_w = chip.shape[:2]
                input_tensor = _preprocess_chip(chip, imgsz, device)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    # SMP models return the tensor directly, not an OrderedDict
                    output = model(input_tensor)

                # Resize logits back to chip spatial dimensions
                logits = F.interpolate(output, size=(chip_h, chip_w), mode="bilinear", align_corners=False)
                
                # Safely cast to float() before cpu().numpy() to avoid float16 accumulation errors
                probs = F.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()

                prob_sum[:, y0:y1, x0:x1] += probs
                count_map[y0:y1, x0:x1] += 1.0

    # Average probabilities in overlap regions
    count_map = np.maximum(count_map, 1.0)
    avg_probs = prob_sum / count_map[np.newaxis, :, :]

    # Per-pixel argmax
    class_map = avg_probs.argmax(axis=0).astype(np.uint8) + 1  # 1-indexed (0 = background)
    best_conf = avg_probs.max(axis=0).astype(np.float32)

    # Build per-class boolean masks
    class_masks = {}
    for idx, name in enumerate(CLASS_NAMES):
        class_masks[name] = (class_map == (idx + 1))

    return class_masks, class_map, best_conf


# --------------------------------------------------
# Post-processing
# --------------------------------------------------

from scipy.ndimage import label

def _remove_noise(
    class_masks: dict[str, np.ndarray], 
    min_size: int = 256
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Removes isolated pixel clusters smaller than min_size (fixes TV static)."""
    cleaned_masks = {}
    
    # Grab dimensions to reconstruct the map
    height, width = next(iter(class_masks.values())).shape
    cleaned_map = np.zeros((height, width), dtype=np.uint8)
    
    for idx, (name, mask) in enumerate(class_masks.items()):
        # Find contiguous blobs of pixels
        labeled_mask, num_features = label(mask)
        if num_features == 0:
            cleaned_masks[name] = mask
            continue
            
        # Count pixels in each blob
        sizes = np.bincount(labeled_mask.ravel())
        sizes[0] = 0  # Ignore the background size
        
        # Keep only blobs larger than min_size
        valid_labels = np.where(sizes >= min_size)[0]
        cleaned_mask = np.isin(labeled_mask, valid_labels)
        
        cleaned_masks[name] = cleaned_mask
        cleaned_map[cleaned_mask] = idx + 1
        
    return cleaned_masks, cleaned_map


def _fill_best_guess(
    class_masks: dict[str, np.ndarray],
    class_map: np.ndarray,
    best_conf: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Fill background pixels (class_map == 0) using nearest-neighbour EDT."""
    background = class_map == 0
    if not np.any(background) or not np.any(~background):
        return class_masks, class_map, best_conf

    _, indices = distance_transform_edt(background, return_indices=True)
    nearest_class_map = class_map[tuple(indices)]
    nearest_conf = best_conf[tuple(indices)]

    filled_map = class_map.copy()
    filled_map[background] = nearest_class_map[background]

    filled_conf = best_conf.copy()
    filled_conf[background] = nearest_conf[background]

    filled_masks = {
        name: (filled_map == (idx + 1)) for idx, name in enumerate(CLASS_NAMES)
    }
    return filled_masks, filled_map, filled_conf


# --------------------------------------------------
# Preview rendering
# --------------------------------------------------


def _preview_stride(shape: tuple[int, ...], max_dim: int = PREVIEW_MAX_DIM) -> int:
    return max(1, int(np.ceil(max(shape[:2]) / max_dim)))


def _save_preview(
    output_path: Path,
    rgb: np.ndarray,
    class_map: np.ndarray,
    footer_lines: list[str] | None = None,
) -> None:
    color_lookup = np.array(
        [
            [0, 0, 0],         # 0 = no prediction
            [26, 122, 26],     # NEN_A
            [125, 200, 125],   # NEN_B
            [245, 208, 32],    # NEN_C
            [245, 130, 13],    # NEN_D
            [214, 40, 40],     # Uncomfortable
        ],
        dtype=np.uint8,
    )
    safe_map = np.clip(class_map, 0, len(color_lookup) - 1)
    overlay = color_lookup[safe_map]
    blended = (0.65 * rgb.astype(np.float32) + 0.35 * overlay.astype(np.float32)).clip(0, 255).astype(np.uint8)

    base_img = Image.fromarray(blended, mode="RGB").convert("RGBA")
    font = ImageFont.load_default()
    pad = 14
    panel_width = 280
    panel_x = base_img.width + pad * 2
    canvas_width = base_img.width + panel_width + pad * 3
    canvas_height = max(base_img.height + pad * 2, 260)
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (16, 16, 18, 255))
    canvas.paste(base_img, (pad, pad))

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (panel_x - 10, pad - 6, canvas_width - pad, canvas_height - pad),
        radius=12,
        fill=(0, 0, 0, 160),
        outline=(255, 255, 255, 180),
        width=1,
    )

    title = f"DeepLabV3+ Prediction | {Path(output_path).stem}"
    draw.text((panel_x, pad), title, font=font, fill=(255, 255, 255, 255))
    title_bbox = draw.textbbox((0, 0), title, font=font)
    cursor_y = pad + (title_bbox[3] - title_bbox[1]) + 12

    legend_rows = [
        ("NEN A", PREDICTION_CLASS_COLORS["NEN_A"]),
        ("NEN B", PREDICTION_CLASS_COLORS["NEN_B"]),
        ("NEN C", PREDICTION_CLASS_COLORS["NEN_C"]),
        ("NEN D", PREDICTION_CLASS_COLORS["NEN_D"]),
        ("Uncomfortable", PREDICTION_CLASS_COLORS["Uncomfortable"]),
    ]

    box_size = 14
    row_gap = 8
    text_gap = 8
    for label, color in legend_rows:
        color_rgb = tuple(int(round(float(c) * 255)) for c in color)
        draw.rectangle(
            (panel_x, cursor_y + 1, panel_x + box_size, cursor_y + 1 + box_size),
            fill=color_rgb,
            outline=(255, 255, 255, 220),
        )
        draw.text((panel_x + box_size + text_gap, cursor_y), label, font=font, fill=(255, 255, 255, 255))
        cursor_y += box_size + row_gap

    if footer_lines:
        cursor_y += 16
        draw.line((panel_x, cursor_y, canvas_width - pad - 8, cursor_y), fill=(255, 255, 255, 120), width=1)
        cursor_y += 12
        for line in footer_lines:
            draw.text((panel_x, cursor_y), line, font=font, fill=(255, 255, 255, 255))
            line_bbox = draw.textbbox((0, 0), line, font=font)
            cursor_y += (line_bbox[3] - line_bbox[1]) + 6

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG", compress_level=6)


# --------------------------------------------------
# Source iteration
# --------------------------------------------------


def _iter_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(source.glob("*.tif"))
    return [source]


def _annotation_path_for_source(source: Path, explicit_path: str | None) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path)
        return candidate if candidate.exists() else None
    sibling_json = source.with_suffix(".json")
    sibling_xml = source.with_suffix(".xml")
    if sibling_json.exists():
        return sibling_json
    if sibling_xml.exists():
        return sibling_xml
    return None


def _to_iou_class_masks(class_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {iou_name: class_masks[pred_name] for pred_name, iou_name in PRED_TO_IOU_CLASS_NAMES.items()}


# --------------------------------------------------
# Process a single source file
# --------------------------------------------------


def _process_source(
    model: torch.nn.Module,
    source: Path,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    rgb = load_rgb_from_tif(source)

    class_masks, class_map, best_conf = predict_tiled(
        model, rgb, args.tile_size, args.tile_overlap, args.imgsz, device
    )
    # Clean up small isolated pixel clusters (TV static)
    # (Adjust the min_size up or down depending on how large your real features are)
    class_masks, class_map = _remove_noise(class_masks, min_size=256)

    if args.force_best_guess:
        class_masks, class_map, best_conf = _fill_best_guess(class_masks, class_map, best_conf)

    source_out = output_dir / source.stem
    source_out.mkdir(parents=True, exist_ok=True)

    class_map_path = source_out / f"{source.stem}_class_map.npy"
    np.save(class_map_path, class_map)
    np.save(source_out / f"{source.stem}_best_conf.npy", best_conf)

    report: dict[str, object] = {
        "source": str(source),
        "model": "DeepLabV3+",
        "imgsz": int(args.imgsz),
        "tile_size": int(args.tile_size),
        "class_map_path": str(class_map_path),
        "preview_path": str(source_out / f"{source.stem}_preview.png"),
        "class_pixel_counts": {name: int(mask.sum()) for name, mask in class_masks.items()},
        "force_best_guess": bool(args.force_best_guess),
    }

    # IoU evaluation against ground truth
    annotation_path = _annotation_path_for_source(source, args.annotation_path)
    if annotation_path is not None:
        comparison = compare_annotation_iou(
            image_name=source.name,
            predicted_masks=_to_iou_class_masks(class_masks),
            xml_path=annotation_path,
            output_dir=source_out / "annotation_iou",
            class_mode="split",
        )
        report.update(
            {
                key: value
                for key, value in comparison.items()
                if key not in {"ground_truth_class_map", "prediction_class_map", "valid_mask", "label_lookup"}
            }
        )

        # --- NEW: Print per-class metrics to console ---
        print(f"\n  [EVALUATION] mIoU: {float(comparison.get('mean_iou', 0.0)):.4f} | Pixel Acc: {float(comparison.get('pixel_accuracy', 0.0)):.4f}")
        
        class_ious = comparison.get("class_iou", {})
        if class_ious:
            print("  Per-Class IoU:")
            for cls_name, iou_val in class_ious.items():
                print(f"    - {cls_name}: {iou_val:.4f}")
    # Build preview
    footer_lines = None
    if report.get("mean_iou") is not None and report.get("pixel_accuracy") is not None:
        footer_lines = [
            f"mIoU: {float(report['mean_iou']):.3f}",
            f"Pixel accuracy: {float(report['pixel_accuracy']):.3f}",
        ]

    preview_stride = _preview_stride(rgb.shape)
    if preview_stride > 1:
        print(f"[INFO] Downsampling preview by stride={preview_stride}")
    preview_rgb = rgb[::preview_stride, ::preview_stride]
    preview_class_map = class_map[::preview_stride, ::preview_stride]

    _save_preview(
        source_out / f"{source.stem}_preview.png",
        preview_rgb,
        preview_class_map,
        footer_lines=footer_lines,
    )

    report_path = source_out / f"{source.stem}_prediction_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


# --------------------------------------------------
# Main
# --------------------------------------------------


def main() -> int:
    args = _parse_args()

    # Device
    if args.device in ("auto", ""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    print(f"[INFO] Device: {device}")

    # Allow large images
    Image.MAX_IMAGE_PIXELS = None

    model = load_model(Path(args.weights), device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    for source in _iter_sources(Path(args.source)):
        print(f"\n[INFO] Processing: {source.name}")
        reports.append(_process_source(model, source, args, output_dir, device))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\n[INFO] Wrote DeepLabV3+ predictions to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())