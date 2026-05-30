from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import rasterio
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.cvat_mask_cleanup import normalize_label


CLASS_NAMES = ["NEN_A", "NEN_B", "NEN_C", "NEN_D", "Uncomfortable"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}


@dataclass(frozen=True)
class TilePair:
    image_path: Path
    annotation_path: Path

    @property
    def stem(self) -> str:
        return self.image_path.stem


def iter_tile_pairs(source_dir: Path) -> Iterator[TilePair]:
    for annotation_path in sorted(source_dir.rglob("*.json")):
        image_path = annotation_path.with_suffix(".tif")
        if image_path.exists():
            yield TilePair(image_path=image_path, annotation_path=annotation_path)


def split_items(items: list[TilePair], val_ratio: float, seed: int) -> tuple[list[TilePair], list[TilePair]]:
    if not items:
        return [], []

    shuffled = items[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = int(round(len(shuffled) * max(0.0, min(1.0, val_ratio))))
    val_count = max(1 if len(shuffled) > 1 else 0, min(val_count, len(shuffled) - 1 if len(shuffled) > 1 else len(shuffled)))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items, val_items


def _percentile_uint8(channel: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    arr = channel.astype(np.float32, copy=False)
    lo = float(np.nanpercentile(arr, low))
    hi = float(np.nanpercentile(arr, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - lo) * (255.0 / (hi - lo))
    np.clip(scaled, 0.0, 255.0, out=scaled)
    return scaled.astype(np.uint8)


def load_rgb_from_tif(tif_path: Path) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        if src.count >= 3:
            data = src.read([1, 2, 3])
        elif src.count == 1:
            band = src.read(1)
            data = np.stack([band, band, band], axis=0)
        else:
            raise ValueError(f"Unsupported band count for {tif_path}: {src.count}")

    rgb = np.transpose(data, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = np.stack([_percentile_uint8(rgb[..., index]) for index in range(rgb.shape[2])], axis=2)
    return rgb


def save_rgb_png(rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(output_path)


def _read_json_shapes(annotation_path: Path) -> tuple[int, int, list[dict[str, object]]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    width = int(data.get("imageWidth", 0))
    height = int(data.get("imageHeight", 0))
    shapes = data.get("shapes", [])
    if width <= 0 or height <= 0:
        raise ValueError(f"Annotation file is missing image dimensions: {annotation_path}")
    if not isinstance(shapes, list):
        raise ValueError(f"Annotation file has invalid shapes payload: {annotation_path}")
    return width, height, shapes


def _clamp_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    x = min(max(float(x), 0.0), float(width))
    y = min(max(float(y), 0.0), float(height))
    return x, y


def annotation_shapes_to_yolo_segments(annotation_path: Path) -> list[tuple[int, list[float]]]:
    width, height, shapes = _read_json_shapes(annotation_path)
    segments: list[tuple[int, list[float]]] = []

    for shape in shapes:
        if not isinstance(shape, dict):
            continue

        raw_label = str(shape.get("label", ""))
        normalized_label = normalize_label(raw_label)
        if normalized_label not in CLASS_TO_INDEX:
            continue

        shape_type = str(shape.get("shape_type", "polygon")).lower()
        if shape_type not in {"polygon", "polyline"}:
            continue

        points = shape.get("points", [])
        if not isinstance(points, list):
            continue

        cleaned_points: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            cleaned_points.append(_clamp_point(point[0], point[1], width, height))

        if len(cleaned_points) < 3:
            continue

        if shape_type == "polyline" and cleaned_points[0] != cleaned_points[-1]:
            cleaned_points.append(cleaned_points[0])

        normalized_points: list[float] = []
        for x, y in cleaned_points:
            normalized_points.extend([x / float(width), y / float(height)])

        if len(normalized_points) >= 6:
            segments.append((CLASS_TO_INDEX[normalized_label], normalized_points))

    return segments


def write_yolo_label_file(annotation_path: Path, output_path: Path) -> int:
    segments = annotation_shapes_to_yolo_segments(annotation_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for class_index, points in segments:
        point_text = " ".join(f"{value:.6f}" for value in points)
        lines.append(f"{class_index} {point_text}")

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_dataset_yaml(output_dir: Path, include_test: bool = True) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    test_line = "test: images/test\n" if include_test else ""
    yaml_text = f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
{test_line}nc: {len(CLASS_NAMES)}
names:
{names_block}
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def dataset_summary_path(output_dir: Path) -> Path:
    return output_dir / "dataset_index.json"
