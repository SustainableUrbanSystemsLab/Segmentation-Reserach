from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.iou_comparator import compare_annotation_iou

from yoloseg_pipeline.common import CLASS_NAMES, PROJECT_ROOT, load_rgb_from_tif


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv11 segmentation inference on a tif or folder of tifs.")
    parser.add_argument("--weights", required=True, help="Path to a trained YOLO segmentation checkpoint.")
    parser.add_argument("--source", required=True, help="A tif file or a folder containing tif files.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "yoloseg_predictions"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--fallback-conf",
        type=float,
        default=0.001,
        help="Lower confidence used once if the initial pass returns no masks.",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--annotation-path", default=None, help="Optional annotation JSON/XML for IoU evaluation.")
    parser.add_argument(
        "--force-best-guess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fill any remaining background pixels with the nearest predicted class.",
    )
    return parser.parse_args()


def _iter_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(source.glob("*.tif"))
    return [source]


def _tile_ranges(length: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    if length <= tile_size:
        return [(0, length)]
    step = max(1, tile_size - overlap)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(length, start + tile_size)
        ranges.append((start, end))
        if end >= length:
            break
        start += step
    return ranges


def _effective_tile_size(tile_size: int, imgsz: int) -> int:
    return max(1, min(int(tile_size), int(imgsz)))

def _predict_tiled(
    model: YOLO,
    rgb: np.ndarray,
    tile_size: int,
    overlap: int,
    conf: float,
    imgsz: int,
    fallback_conf: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    num_classes = len(CLASS_NAMES)

    # 1. Initialize continuous accumulation canvases
    canvas_probs = np.zeros((num_classes, height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    # 2. Build a base 2D linear blending window matching your effective tile size
    ramp_y = np.minimum(np.arange(tile_size), np.arange(tile_size)[::-1])
    ramp_x = np.minimum(np.arange(tile_size), np.arange(tile_size)[::-1])
    if np.max(ramp_y) > 0: ramp_y = ramp_y / np.max(ramp_y)
    if np.max(ramp_x) > 0: ramp_x = ramp_x / np.max(ramp_x)
    base_window = np.outer(ramp_y, ramp_x)
    
    # Apply a tiny baseline floor value (0.01) to prevent absolute zero edge lockups
    base_window = np.maximum(base_window, 0.01)

    y_ranges = _tile_ranges(height, tile_size, overlap)
    x_ranges = _tile_ranges(width, tile_size, overlap)

    # 3. Slide windows across the map matrix
    for y0, y1 in y_ranges:
        for x0, x1 in x_ranges:
            chip = rgb[y0:y1, x0:x1]
            if chip.size == 0:
                continue

            # Run YOLO prediction pass
            result = model.predict(chip, imgsz=imgsz, conf=conf, verbose=False, retina_masks=True)[0]
            if (result.masks is None or result.boxes is None or len(result.boxes) == 0) and fallback_conf >= 0.0:
                result = model.predict(chip, imgsz=imgsz, conf=min(conf, fallback_conf), verbose=False, retina_masks=True)[0]

            if result.masks is None or result.boxes is None or len(result.boxes) == 0:
                continue

            # Handle edge chips that might be cut smaller than the target tile_size
            ch, cw = y1 - y0, x1 - x0
            window_mask = base_window[:ch, :cw]

            # Grab soft continuous mask probabilities directly from YOLO tensor outputs
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy().astype(float)

            # 4. Sum up soft mask assignments weighted by window masks
            for mask, class_index, score in zip(masks, classes, scores):
                if class_index < 0 or class_index >= num_classes:
                    continue
                
                # Multiply mask weights by structural coordinates bounding confidence
                weighted_mask = mask * score * window_mask
                canvas_probs[class_index, y0:y1, x0:x1] += weighted_mask
            
            # Accumulate processing window configuration footprints
            weight_map[y0:y1, x0:x1] += window_mask

    # 5. Normalize accumulated probabilities across overlapping passes
    safe_weight = np.where(weight_map == 0, 1e-6, weight_map)
    for c in range(num_classes):
        canvas_probs[c] /= safe_weight

    # 6. Reconstruct the precise downstream return variables expected by predict.py
    max_probs = np.max(canvas_probs, axis=0)
    class_map = np.argmax(canvas_probs, axis=0) + 1  # 1-indexed to match your convention

    # Isolate unpredicted or low-probability background regions
    background_mask = (weight_map == 0) | (max_probs == 0)
    class_map[background_mask] = 0

    best_conf = max_probs
    best_conf[background_mask] = 0.0

    class_masks = {
        name: (class_map == (idx + 1)) 
        for idx, name in enumerate(CLASS_NAMES)
    }

    return class_masks, class_map, best_conf


def _fill_best_guess(class_masks: dict[str, np.ndarray], class_map: np.ndarray, best_conf: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    background = class_map == 0
    if not np.any(background) or not np.any(~background):
        return class_masks, class_map, best_conf

    _, indices = distance_transform_edt(background, return_indices=True)
    nearest_class_map = class_map[tuple(indices)]
    nearest_best_conf = best_conf[tuple(indices)]

    filled_class_map = class_map.copy()
    filled_class_map[background] = nearest_class_map[background]

    filled_best_conf = best_conf.copy()
    filled_best_conf[background] = nearest_best_conf[background]

    filled_class_masks = {
        name: filled_class_map == (index + 1)
        for index, name in enumerate(CLASS_NAMES)
    }
    return filled_class_masks, filled_class_map, filled_best_conf


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


def _save_preview(output_path: Path, rgb: np.ndarray, class_map: np.ndarray, footer_lines: list[str] | None = None) -> None:
    from PIL import Image, ImageDraw, ImageFont

    color_lookup = np.array(
        [
            [0, 0, 0],
            [26, 122, 26],
            [125, 200, 125],
            [245, 208, 32],
            [245, 130, 13],
            [214, 40, 40],
        ],
        dtype=np.uint8,
    )
    overlay = color_lookup[class_map]
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

    title = f"YOLO Prediction | {Path(output_path).stem}"
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
        draw.rectangle((panel_x, cursor_y + 1, panel_x + box_size, cursor_y + 1 + box_size), fill=color_rgb, outline=(255, 255, 255, 220))
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


def _preview_stride(shape: tuple[int, int, int], max_dim: int = PREVIEW_MAX_DIM) -> int:
    return max(1, int(np.ceil(max(shape[:2]) / max_dim)))


def _to_iou_class_masks(class_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {iou_name: class_masks[pred_name] for pred_name, iou_name in PRED_TO_IOU_CLASS_NAMES.items()}


def _process_source(model: YOLO, source: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    rgb = load_rgb_from_tif(source)
    effective_tile_size = _effective_tile_size(args.tile_size, args.imgsz)
    if effective_tile_size != args.tile_size:
        print(
            f"[WARN] tile-size={args.tile_size} is larger than imgsz={args.imgsz}; "
            f"using effective tile-size={effective_tile_size} to avoid extra downsampling"
        )

    class_masks, class_map, best_conf = _predict_tiled(
        model,
        rgb,
        effective_tile_size,
        args.tile_overlap,
        args.conf,
        args.imgsz,
        args.fallback_conf,
    )

    if args.force_best_guess:
        class_masks, class_map, best_conf = _fill_best_guess(class_masks, class_map, best_conf)

    source_out = output_dir / source.stem
    source_out.mkdir(parents=True, exist_ok=True)

    class_map_path = source_out / f"{source.stem}_class_map.npy"
    np.save(class_map_path, class_map)
    np.save(source_out / f"{source.stem}_best_conf.npy", best_conf)

    report: dict[str, object] = {
        "source": str(source),
        "imgsz": int(args.imgsz),
        "tile_size": int(args.tile_size),
        "effective_tile_size": int(effective_tile_size),
        "class_map_path": str(class_map_path),
        "preview_path": str(source_out / f"{source.stem}_preview.png"),
        "class_pixel_counts": {name: int(mask.sum()) for name, mask in class_masks.items()},
        "force_best_guess": bool(args.force_best_guess),
    }

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

    footer_lines = None
    if report.get("mean_iou") is not None and report.get("pixel_accuracy") is not None:
        footer_lines = [
            f"mIoU: {float(report['mean_iou']):.3f}",
            f"Pixel accuracy: {float(report['pixel_accuracy']):.3f}",
        ]

    preview_stride = _preview_stride(rgb.shape)
    if preview_stride > 1:
        print(f"[INFO] Downsampling preview by stride={preview_stride} to match combined-mask display sizing")
    preview_rgb = rgb[::preview_stride, ::preview_stride]
    preview_class_map = class_map[::preview_stride, ::preview_stride]

    _save_preview(source_out / f"{source.stem}_preview.png", preview_rgb, preview_class_map, footer_lines=footer_lines)

    report_path = source_out / f"{source.stem}_prediction_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> int:
    args = _parse_args()
    model = YOLO(args.weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for source in _iter_sources(Path(args.source)):
        reports.append(_process_source(model, source, args, output_dir))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"[INFO] Wrote YOLO predictions to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())