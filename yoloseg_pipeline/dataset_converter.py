from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloseg_pipeline.common import (
    CLASS_NAMES,
    PROJECT_ROOT,
    TilePair,
    iter_tile_pairs,
    load_rgb_from_tif,
    save_rgb_png,
    split_items,
    write_dataset_yaml,
    write_yolo_label_file,
    dataset_summary_path,
    annotation_shapes_to_pixel_polygons,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CVAT-style tif/json tiles into YOLOv11-seg format.")
    parser.add_argument(
        "--source-dir",
        default=str(PROJECT_ROOT / "Maps" / "Tiles" / "Atlanta_split_google"),
        help="Folder containing matching .tif and .json tile files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "yoloseg_windcomfort_rgb"),
        help="Destination YOLO dataset directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file with source_dir, output_dir, train_tiles, val_tiles, test_tiles, and skip_empty_label_tiles.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of labeled tiles to reserve for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the train/val split.")
    parser.add_argument("--train-tiles", default=None, help="Comma-separated tile stems or a JSON config file path.")
    parser.add_argument("--val-tiles", default=None, help="Comma-separated tile stems or a JSON config file path.")
    parser.add_argument("--test-tiles", default=None, help="Comma-separated tile stems or a JSON config file path.")
    parser.add_argument(
        "--skip-empty-label-tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tiles that have no valid wind-comfort labels after invalid annotations are removed.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Optional size to slice tiles into overlapping chips (e.g. 1024 or 1280). If omitted, saves full tiles.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=128,
        help="Pixel overlap between adjacent sliced chips. Only used if --tile-size is set.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the dataset even if files already exist.")
    return parser.parse_args()


def _prepare_split_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _parse_tile_list(value: object) -> set[str]:
    if not value:
        return set()
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str):
        text = value.strip()
        candidate = Path(text)
        if candidate.suffix.lower() == ".json" and candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("tiles", "train_tiles", "val_tiles", "test_tiles"):
                    items = data.get(key)
                    if isinstance(items, list):
                        return {str(item).strip() for item in items if str(item).strip()}
                    if isinstance(items, str):
                        return {item.strip() for item in items.split(",") if item.strip()}
            return set()
        return {item.strip() for item in text.split(",") if item.strip()}
    return {str(value).strip()}


def _load_run_config(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return config


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5


def _summarize_label_file(label_path: Path, image_shape: tuple[int, int]) -> tuple[dict[str, int], dict[str, float]]:
    height, width = image_shape
    class_counts = {name: 0 for name in CLASS_NAMES}
    class_area_px2 = {name: 0.0 for name in CLASS_NAMES}

    if not label_path.exists():
        return class_counts, class_area_px2

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            class_index = int(parts[0])
        except ValueError:
            continue

        if class_index < 0 or class_index >= len(CLASS_NAMES):
            continue

        coords = parts[1:]
        if len(coords) % 2 != 0:
            continue

        points: list[tuple[float, float]] = []
        for coord_index in range(0, len(coords), 2):
            try:
                x_norm = float(coords[coord_index])
                y_norm = float(coords[coord_index + 1])
            except ValueError:
                points = []
                break
            points.append((x_norm * width, y_norm * height))

        if len(points) < 3:
            continue

        class_name = CLASS_NAMES[class_index]
        class_counts[class_name] += 1
        class_area_px2[class_name] += _polygon_area(points)

    return class_counts, class_area_px2


def _copy_tile(pair: TilePair, output_dir: Path, split: str, skip_empty_label_tiles: bool) -> dict[str, object] | None:
    image_out = output_dir / "images" / split / f"{pair.stem}.png"
    label_out = output_dir / "labels" / split / f"{pair.stem}.txt"

    label_count = write_yolo_label_file(pair.annotation_path, label_out)
    if skip_empty_label_tiles and label_count == 0:
        if label_out.exists():
            label_out.unlink()
        return None

    rgb = load_rgb_from_tif(pair.image_path)
    save_rgb_png(rgb, image_out)
    class_counts, class_area_px2 = _summarize_label_file(label_out, (int(rgb.shape[0]), int(rgb.shape[1])))

    return {
        "stem": pair.stem,
        "split": split,
        "image_path": str(image_out),
        "label_path": str(label_out),
        "annotation_path": str(pair.annotation_path),
        "image_source_path": str(pair.image_path),
        "label_count": label_count,
        "class_counts": class_counts,
        "class_area_px2": class_area_px2,
        "image_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
    }


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


import os
import numpy as np
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import translate

def process_and_clip_polygon(
    global_polygon: Polygon, 
    chip_bounds: tuple, 
    chip_width: int = 640, 
    chip_height: int = 640, 
    normalize: bool = True
):
    """
    Helper function: Clips a single global-space polygon to a specific chip window,
    resolves topology anomalies, and maps coordinates to local chip space.
    """
    min_x, min_y, max_x, max_y = chip_bounds
    chip_box = box(min_x, min_y, max_x, max_y)
    
    if not global_polygon.intersects(chip_box):
        return []
        
    try:
        clipped_geometry = global_polygon.intersection(chip_box)
    except Exception as e:
        return []

    if clipped_geometry.is_empty:
        return []

    valid_polygons = []
    if isinstance(clipped_geometry, Polygon):
        valid_polygons.append(clipped_geometry)
    elif isinstance(clipped_geometry, MultiPolygon):
        valid_polygons.extend(list(clipped_geometry.geoms))
    elif isinstance(clipped_geometry, GeometryCollection):
        for geom in clipped_geometry.geoms:
            if isinstance(geom, Polygon):
                valid_polygons.append(geom)

    local_chip_labels = []

    for poly in valid_polygons:
        if poly.area < 1.0: # Filter out sub-pixel noise
            continue
            
        # Shift global coordinates to local chip space relative to top-left (min_x, min_y)
        local_poly = translate(poly, xoff=-min_x, yoff=-min_y)
        local_coords = np.array(local_poly.exterior.coords)
        
        # Drop duplicate closing vertex for line string conversion 
        if len(local_coords) > 1 and np.array_equal(local_coords[0], local_coords[-1]):
            local_coords = local_coords[:-1]

        if normalize:
            local_coords[:, 0] /= chip_width
            local_coords[:, 1] /= chip_height
            local_coords = np.clip(local_coords, 0.0, 1.0)
            
        local_chip_labels.append(local_coords.tolist())

    return local_chip_labels


def _slice_tile(
    tile_image: np.ndarray, 
    tile_name: str, 
    global_annotations: list, 
    output_dir: str, 
    chip_size: int = 640, 
    stride: int = 640
):
    """
    Orchestrates the sliding window processing across a large image tile.
    Extracts image chips and matches them with properly clipped local labels.
    
    Parameters:
    -----------
    tile_image : np.ndarray
        The large source image array (e.g., loaded from a large .tif).
    tile_name : str
        Base string name of the tile used to name output chips.
    global_annotations : list of dicts
        List of all ground-truth targets on this tile: 
        [{'class_id': int, 'polygon': shapely.geometry.Polygon}, ...]
    output_dir : str
        Directory where 'images' and 'labels' folders will be populated.
    """
    img_h, img_w = tile_image.shape[:2]
    
    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    
    chip_count = 0

    # Slide window horizontally and vertically across the large tile
    for y in range(0, img_h - chip_size + 1, stride):
        for x in range(0, img_w - chip_size + 1, stride):
            
            # Define global pixel space boundaries for the current chip window
            min_x, min_y = x, y
            max_x, max_y = x + chip_size, y + chip_size
            chip_bounds = (min_x, min_y, max_x, max_y)
            
            chip_labels_payload = []
            
            # Check all global annotations against this specific window
            for ann in global_annotations:
                class_id = ann['class_id']
                global_poly = ann['polygon']
                
                # Get clean, scaled, normalized local shapes
                local_polys = process_and_clip_polygon(
                    global_polygon=global_poly,
                    chip_bounds=chip_bounds,
                    chip_width=chip_size,
                    chip_height=chip_size,
                    normalize=True # Set to True for standard YOLO segmentation
                )
                
                # Append each valid segment component found within the boundaries
                for poly_coords in local_polys:
                    # Flatten out coordinate pairs to string tokens: class_id x1 y1 x2 y2 ...
                    coord_str = " ".join([f"{coord:.6f}" for pair in poly_coords for coord in pair])
                    chip_labels_payload.append(f"{class_id} {coord_str}")
            
            # Skip saving empty chips if you want to prevent background class dilution
            if not chip_labels_payload:
                continue
                
            # 1. Extract and save the corresponding image chip matrix
            chip_matrix = tile_image[min_y:max_y, min_x:max_x]
            chip_id = f"{tile_name}_chip_{chip_count}"
            
            # Using an arbitrary image write interface (adjust for cv2/PIL/tifffile as needed)
            # e.g., cv2.imwrite(os.path.join(images_out, f"{chip_id}.jpg"), chip_matrix)
            np.save(os.path.join(images_out, f"{chip_id}.npy"), chip_matrix) # Placeholder save method
            
            # 2. Write out the clean YOLO text label file
            label_file_path = os.path.join(labels_out, f"{chip_id}.txt")
            with open(label_file_path, "w") as f:
                f.write("\n".join(chip_labels_payload) + "\n")
                
            chip_count += 1

    print(f"Successfully sliced {tile_name} into {chip_count} valid annotated chips.")


def convert_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
    train_tiles: str | None = None,
    val_tiles: str | None = None,
    test_tiles: str | None = None,
    skip_empty_label_tiles: bool = True,
    tile_size: int | None = None,
    tile_overlap: int = 128,
) -> Path:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_split_dirs(output_dir)

    pairs = list(iter_tile_pairs(source_dir))
    train_tile_names = _parse_tile_list(train_tiles)
    val_tile_names = _parse_tile_list(val_tiles)
    test_tile_names = _parse_tile_list(test_tiles)

    if train_tile_names or val_tile_names or test_tile_names:
        train_items = [pair for pair in pairs if pair.stem in train_tile_names]
        val_items = [pair for pair in pairs if pair.stem in val_tile_names]
        test_items = [pair for pair in pairs if pair.stem in test_tile_names]

        remaining = [
            pair
            for pair in pairs
            if pair.stem not in train_tile_names and pair.stem not in val_tile_names and pair.stem not in test_tile_names
        ]
        if remaining:
            extra_train, extra_val = split_items(remaining, val_ratio=val_ratio, seed=seed)
            train_items.extend(extra_train)
            val_items.extend(extra_val)
    else:
        train_items, val_items = split_items(pairs, val_ratio=val_ratio, seed=seed)
        test_items = []

    records = []
    class_counts = {name: 0 for name in CLASS_NAMES}
    class_area_px2 = {name: 0.0 for name in CLASS_NAMES}
    for split_name, split_items_list in (("train", train_items), ("val", val_items), ("test", test_items)):
        for pair in split_items_list:
            if tile_size is not None:
                sliced_records = _slice_tile(pair, output_dir, split_name, tile_size, tile_overlap, skip_empty_label_tiles)
                for rec in sliced_records:
                    records.append(rec)
                    copied_counts = rec.get("class_counts", {})
                    copied_areas = rec.get("class_area_px2", {})
                    for class_name in CLASS_NAMES:
                        class_counts[class_name] += int(copied_counts.get(class_name, 0))
                        class_area_px2[class_name] += float(copied_areas.get(class_name, 0.0))
            else:
                copied = _copy_tile(pair, output_dir, split_name, skip_empty_label_tiles)
                if copied is not None:
                    records.append(copied)
                    copied_counts = copied.get("class_counts", {})
                    copied_areas = copied.get("class_area_px2", {})
                    for class_name in CLASS_NAMES:
                        class_counts[class_name] += int(copied_counts.get(class_name, 0))
                        class_area_px2[class_name] += float(copied_areas.get(class_name, 0.0))

    yaml_path = write_dataset_yaml(output_dir, include_test=bool(test_items))
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "total_count": len(pairs),
        "train_tiles": sorted(train_tile_names) if train_tile_names else None,
        "val_tiles": sorted(val_tile_names) if val_tile_names else None,
        "test_tiles": sorted(test_tile_names) if test_tile_names else None,
        "skip_empty_label_tiles": skip_empty_label_tiles,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "sliced_chips_count": len(records) if tile_size is not None else None,
        "class_counts": class_counts,
        "class_area_px2": class_area_px2,
        "records": records,
        "dataset_yaml": str(yaml_path),
    }
    dataset_summary_path(output_dir).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[INFO] Class mask counts:")
    for class_name in CLASS_NAMES:
        print(f"[INFO] - {class_name}: {class_counts[class_name]}")
    print("[INFO] Total area by class (pixel^2):")
    for class_name in CLASS_NAMES:
        print(f"[INFO] - {class_name}: {class_area_px2[class_name]:.2f}")
    return yaml_path


def main() -> int:
    args = _parse_args()
    if args.config:
        config_path = Path(args.config)
        config = _load_run_config(config_path)
        source_dir = Path(config.get("source_dir", args.source_dir))
        output_dir = Path(config.get("output_dir", args.output_dir))
        val_ratio = float(config.get("val_ratio", args.val_ratio))
        seed = int(config.get("seed", args.seed))
        overwrite = bool(config.get("overwrite", args.overwrite))
        train_tiles = config.get("train_tiles", args.train_tiles)
        val_tiles = config.get("val_tiles", args.val_tiles)
        test_tiles = config.get("test_tiles", args.test_tiles)
        skip_empty_label_tiles = bool(config.get("skip_empty_label_tiles", args.skip_empty_label_tiles))
        tile_size_val = config.get("tile_size", args.tile_size)
        tile_size = int(tile_size_val) if tile_size_val is not None else None
        tile_overlap = int(config.get("tile_overlap", args.tile_overlap))
    else:
        source_dir = Path(args.source_dir)
        output_dir = Path(args.output_dir)
        val_ratio = float(args.val_ratio)
        seed = int(args.seed)
        overwrite = bool(args.overwrite)
        train_tiles = args.train_tiles
        val_tiles = args.val_tiles
        test_tiles = args.test_tiles
        skip_empty_label_tiles = bool(args.skip_empty_label_tiles)
        tile_size = int(args.tile_size) if args.tile_size is not None else None
        tile_overlap = int(args.tile_overlap)

    yaml_path = convert_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        val_ratio=val_ratio,
        seed=seed,
        overwrite=overwrite,
        train_tiles=train_tiles,
        val_tiles=val_tiles,
        test_tiles=test_tiles,
        skip_empty_label_tiles=skip_empty_label_tiles,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    print(f"[INFO] Wrote YOLO dataset config: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())