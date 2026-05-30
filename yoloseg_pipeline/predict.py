from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.iou_comparator import compare_annotation_iou

from yoloseg_pipeline.common import CLASS_NAMES, PROJECT_ROOT, load_rgb_from_tif


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv11 segmentation inference on a tif or folder of tifs.")
    parser.add_argument("--weights", required=True, help="Path to a trained YOLO segmentation checkpoint.")
    parser.add_argument("--source", required=True, help="A tif file or a folder containing tif files.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "yoloseg_predictions"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--annotation-path", default=None, help="Optional annotation JSON/XML for IoU evaluation.")
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


def _predict_tiled(model: YOLO, rgb: np.ndarray, tile_size: int, overlap: int, conf: float, imgsz: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    class_masks = {name: np.zeros((height, width), dtype=bool) for name in CLASS_NAMES}
    best_conf = np.zeros((height, width), dtype=np.float32)
    class_map = np.zeros((height, width), dtype=np.uint8)

    y_ranges = _tile_ranges(height, tile_size, overlap)
    x_ranges = _tile_ranges(width, tile_size, overlap)

    for y0, y1 in y_ranges:
        for x0, x1 in x_ranges:
            chip = rgb[y0:y1, x0:x1]
            if chip.size == 0:
                continue

            result = model.predict(chip, imgsz=imgsz, conf=conf, verbose=False)[0]
            if result.masks is None or result.boxes is None or len(result.boxes) == 0:
                continue

            masks = result.masks.data.cpu().numpy() > 0.5
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy().astype(float)

            local_best = best_conf[y0:y1, x0:x1]
            local_map = class_map[y0:y1, x0:x1]

            for mask, class_index, score in zip(masks, classes, scores):
                if class_index < 0 or class_index >= len(CLASS_NAMES):
                    continue
                update = mask & (score > local_best)
                if not np.any(update):
                    continue
                local_best[update] = float(score)
                local_map[update] = int(class_index + 1)
                class_view = class_masks[CLASS_NAMES[class_index]][y0:y1, x0:x1]
                class_view[update] = True

    return class_masks, class_map, best_conf


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


def _save_preview(output_path: Path, rgb: np.ndarray, class_map: np.ndarray) -> None:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(blended, mode="RGB").save(output_path)


def _process_source(model: YOLO, source: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    rgb = load_rgb_from_tif(source)
    class_masks, class_map, best_conf = _predict_tiled(model, rgb, args.tile_size, args.tile_overlap, args.conf, args.imgsz)

    source_out = output_dir / source.stem
    source_out.mkdir(parents=True, exist_ok=True)

    class_map_path = source_out / f"{source.stem}_class_map.npy"
    np.save(class_map_path, class_map)
    np.save(source_out / f"{source.stem}_best_conf.npy", best_conf)
    _save_preview(source_out / f"{source.stem}_preview.png", rgb, class_map)

    report: dict[str, object] = {
        "source": str(source),
        "class_map_path": str(class_map_path),
        "preview_path": str(source_out / f"{source.stem}_preview.png"),
        "class_pixel_counts": {name: int(mask.sum()) for name, mask in class_masks.items()},
    }

    annotation_path = _annotation_path_for_source(source, args.annotation_path)
    if annotation_path is not None:
        comparison = compare_annotation_iou(
            image_name=source.name,
            predicted_masks=class_masks,
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