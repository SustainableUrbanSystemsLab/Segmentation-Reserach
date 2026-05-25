"""Utilities for turning CVAT annotation polygons into cleaned full-image masks."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


LABEL_PRIORITY = {
    "Uncomfortable": 0,
    "NEN_D": 1,
    "NEN_C": 2,
    "NEN_B": 3,
    "NEN_A": 4,
    "Invalid": 5,
}

LABEL_TO_COLOR = {
    "Uncomfortable": (214, 40, 40),
    "NEN_D": (245, 130, 13),
    "NEN_C": (245, 208, 32),
    "NEN_B": (125, 200, 125),
    "NEN_A": (26, 122, 26),
    "Invalid": (61, 61, 245),
}

_PRIORITY_ORDER = ["Uncomfortable", "NEN_D", "NEN_C", "NEN_B", "NEN_A", "Invalid"]


@dataclass(frozen=True)
class AnnotationShape:
    label: str
    points: list[tuple[float, float]]
    z_order: int
    index: int


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0

    x_coords = np.asarray([point[0] for point in points], dtype=np.float64)
    y_coords = np.asarray([point[1] for point in points], dtype=np.float64)
    x_next = np.roll(x_coords, -1)
    y_next = np.roll(y_coords, -1)
    return float(abs(np.dot(x_coords, y_next) - np.dot(y_coords, x_next)) * 0.5)


def _shape_render_key(shape: AnnotationShape) -> tuple[int, float, int, int]:
    # Smaller non-invalid shapes should overwrite larger overlapping shapes.
    # Invalid always renders last so it stays on top regardless of size.
    return (
        1 if shape.label == "Invalid" else 0,
        0.0 if shape.label == "Invalid" else -_polygon_area(shape.points),
        shape.z_order,
        shape.index,
    )


def normalize_label(raw_label: str) -> str | None:
    label = raw_label.strip()
    lowered = label.lower()

    if lowered == "invalid":
        return "Invalid"
    if label.startswith("NEN_A"):
        return "NEN_A"
    if label.startswith("NEN_B"):
        return "NEN_B"
    if label.startswith("NEN_C"):
        return "NEN_C"
    if label.startswith("NEN_D"):
        return "NEN_D"
    if label.startswith("NEN_U") or "uncomfortable" in lowered:
        return "Uncomfortable"
    return None


def _parse_points(points_text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for pair in points_text.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_text, y_text = pair.split(",")
        points.append((float(x_text), float(y_text)))
    return points


def _parse_json_points(points_value: object) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in points_value or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append((float(point[0]), float(point[1])))
    return points


def _resolve_annotation_file(annotation_path: str | Path, image_name: str) -> Path:
    candidate = Path(annotation_path)
    if candidate.is_file():
        return candidate

    if not candidate.exists():
        raise FileNotFoundError(f"Annotation file or directory not found: {candidate}")

    requested_stem = Path(image_name).stem
    preferred = [
        *sorted(candidate.glob(f"**/{requested_stem}.json")),
        *sorted(candidate.glob(f"**/{requested_stem}.xml")),
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        return preferred[0]

    matches = [*sorted(candidate.glob("**/*.json")), *sorted(candidate.glob("**/*.xml"))]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"No annotation file found for '{image_name}' under {candidate}"
    )


def _resolve_image_name(xml_root: ET.Element, image_name: str) -> ET.Element:
    requested_name = Path(image_name).name
    requested_stem = Path(image_name).stem

    exact_matches: list[ET.Element] = []
    stem_matches: list[ET.Element] = []

    for image_element in xml_root.findall("image"):
        current_name = image_element.get("name", "")
        if current_name == requested_name:
            exact_matches.append(image_element)
        elif Path(current_name).stem == requested_stem:
            stem_matches.append(image_element)

    if exact_matches:
        return exact_matches[0]
    if len(stem_matches) == 1:
        return stem_matches[0]
    if len(stem_matches) > 1:
        raise ValueError(f"Multiple annotation images match tile stem '{requested_stem}'")

    available = [image_element.get("name", "") for image_element in xml_root.findall("image")]
    raise FileNotFoundError(
        f"No annotation image found for '{image_name}'. Available images: {', '.join(available[:10])}"
    )


def _load_annotation_shapes(image_element: ET.Element) -> tuple[int, int, list[AnnotationShape]]:
    width = int(image_element.get("width", "0"))
    height = int(image_element.get("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError("Annotation image is missing a valid width/height")

    shapes: list[AnnotationShape] = []
    shape_index = 0
    for tag in ("polyline", "polygon"):
        for shape_element in image_element.findall(tag):
            raw_label = shape_element.get("label", "")
            label = normalize_label(raw_label)
            if label is None:
                continue

            points_text = shape_element.get("points", "")
            points = _parse_points(points_text)
            if len(points) < 3:
                shape_index += 1
                continue

            z_order = int(float(shape_element.get("z_order", "0")))
            shapes.append(
                AnnotationShape(
                    label=label,
                    points=points,
                    z_order=z_order,
                    index=shape_index,
                )
            )
            shape_index += 1

    return width, height, shapes


def _load_json_annotation_shapes(annotation_file: Path) -> tuple[int, int, list[AnnotationShape], str]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    width = int(data.get("imageWidth", 0))
    height = int(data.get("imageHeight", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Annotation JSON is missing a valid imageWidth/imageHeight")

    image_name = str(data.get("imagePath") or annotation_file.with_suffix(".tif").name)
    shapes: list[AnnotationShape] = []
    shape_index = 0

    for shape_element in data.get("shapes", []):
        if not isinstance(shape_element, dict):
            shape_index += 1
            continue

        raw_label = str(shape_element.get("label", ""))
        label = normalize_label(raw_label)
        if label is None:
            shape_index += 1
            continue

        shape_type = str(shape_element.get("shape_type", "polygon")).lower()
        if shape_type not in {"polygon", "polyline"}:
            shape_index += 1
            continue

        points = _parse_json_points(shape_element.get("points", []))
        if len(points) < 3:
            shape_index += 1
            continue

        z_order = int(shape_element.get("z_order", 0) or 0)
        shapes.append(
            AnnotationShape(
                label=label,
                points=points,
                z_order=z_order,
                index=shape_index,
            )
        )
        shape_index += 1

    return width, height, shapes, image_name


def _fill_background_by_nearest_label(label_mask: np.ndarray) -> np.ndarray:
    background = label_mask == 0
    if not np.any(background):
        return label_mask

    if not np.any(~background):
        raise ValueError("Annotation mask is empty; cannot fill by nearest label")

    _, indices = distance_transform_edt(background, return_indices=True)
    nearest_labels = label_mask[tuple(indices)]
    filled = label_mask.copy()
    filled[background] = nearest_labels[background]
    return filled


def build_cleaned_annotation_mask(
    xml_path: str | Path,
    image_name: str,
) -> dict[str, object]:
    """Build a cleaned, fully filled label mask and RGB color mask for one tile."""

    annotation_file = _resolve_annotation_file(xml_path, image_name)
    requested_stem = Path(image_name).stem

    if annotation_file.suffix.lower() == ".json":
        width, height, shapes, resolved_image_name = _load_json_annotation_shapes(annotation_file)
        resolved_stem = Path(resolved_image_name).stem
        if annotation_file.stem != requested_stem and resolved_stem != requested_stem:
            raise FileNotFoundError(
                f"Annotation JSON '{annotation_file}' does not match requested tile '{image_name}'"
            )
    else:
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation XML not found: {annotation_file}")
        root = ET.parse(annotation_file).getroot()
        image_element = _resolve_image_name(root, image_name)
        width, height, shapes = _load_annotation_shapes(image_element)
        resolved_image_name = image_element.get("name", image_name)

    if not shapes:
        raise ValueError(f"No supported annotations found for tile '{resolved_image_name}'")

    label_to_id = {label: idx + 1 for idx, label in enumerate(_PRIORITY_ORDER)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    label_raster = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(label_raster)

    for shape in sorted(shapes, key=_shape_render_key):
        draw.polygon(shape.points, fill=label_to_id[shape.label])

    label_mask = np.asarray(label_raster, dtype=np.uint8)
    label_mask = _fill_background_by_nearest_label(label_mask)

    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for label_name, label_id in label_to_id.items():
        color_mask[label_mask == label_id] = LABEL_TO_COLOR[label_name]

    return {
        "image_name": resolved_image_name,
        "width": width,
        "height": height,
        "label_mask": label_mask,
        "color_mask": color_mask,
        "label_lookup": id_to_label,
        "shape_count": len(shapes),
        "annotation_path": str(annotation_file),
        "xml_path": str(annotation_file),
    }