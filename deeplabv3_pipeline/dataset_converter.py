from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloseg_pipeline.common import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    PROJECT_ROOT,
    TilePair,
    annotation_shapes_to_pixel_polygons,
    dataset_summary_path,
    iter_tile_pairs,
    load_rgb_from_tif,
    save_rgb_png,
    split_items,
)

IGNORE_INDEX = 255


# ------------------------------------------------------------------
# Mask rasterization
# ------------------------------------------------------------------


def rasterize_mask(annotation_path: Path, height: int, width: int) -> np.ndarray:
    """Rasterize polygon annotations into a single-channel class mask.

    Each pixel is set to its class index (0 .. len(CLASS_NAMES)-1).
    Pixels that belong to no polygon remain at ``IGNORE_INDEX`` (255).
    """
    mask = np.full((height, width), IGNORE_INDEX, dtype=np.uint8)
    polygons = annotation_shapes_to_pixel_polygons(annotation_path)

    for class_index, points in polygons:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=int(class_index))

    return mask


def save_mask_png(mask: np.ndarray, output_path: Path) -> None:
    """Write a single-channel uint8 mask as a grayscale PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)


# ------------------------------------------------------------------
# Tile slicing (mirrors yoloseg_pipeline/dataset_converter.py)
# ------------------------------------------------------------------


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


def _mask_class_summary(mask: np.ndarray) -> tuple[dict[str, int], dict[str, float]]:
    """Count pixels per class and approximate area (== pixel count for unit-pixel tiles)."""
    counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    areas: dict[str, float] = {name: 0.0 for name in CLASS_NAMES}
    for idx, name in enumerate(CLASS_NAMES):
        n = int((mask == idx).sum())
        counts[name] = n
        areas[name] = float(n)
    return counts, areas


# ------------------------------------------------------------------
# Per-tile processing
# ------------------------------------------------------------------


def _copy_tile(
    pair: TilePair,
    output_dir: Path,
    split: str,
    skip_empty: bool,
) -> dict[str, object] | None:
    """Convert a single tile pair into an image PNG + mask PNG."""
    rgb = load_rgb_from_tif(pair.image_path)
    height, width = rgb.shape[:2]
    mask = rasterize_mask(pair.annotation_path, height, width)

    label_count = int((mask != IGNORE_INDEX).sum())
    if skip_empty and label_count == 0:
        return None

    image_out = output_dir / "images" / split / f"{pair.stem}.png"
    mask_out = output_dir / "masks" / split / f"{pair.stem}.png"

    save_rgb_png(rgb, image_out)
    save_mask_png(mask, mask_out)

    class_counts, class_area = _mask_class_summary(mask)
    return {
        "stem": pair.stem,
        "split": split,
        "image_path": str(image_out),
        "mask_path": str(mask_out),
        "annotation_path": str(pair.annotation_path),
        "image_source_path": str(pair.image_path),
        "label_pixel_count": label_count,
        "class_counts": class_counts,
        "class_area_px2": class_area,
        "image_shape": [height, width],
    }


def _slice_tile(
    pair: TilePair,
    output_dir: Path,
    split: str,
    tile_size: int,
    tile_overlap: int,
    skip_empty: bool,
) -> list[dict[str, object]]:
    """Slice a large tile into overlapping chips with matching mask chips."""
    rgb = load_rgb_from_tif(pair.image_path)
    height, width = rgb.shape[:2]
    mask = rasterize_mask(pair.annotation_path, height, width)

    y_ranges = _tile_ranges(height, tile_size, tile_overlap)
    x_ranges = _tile_ranges(width, tile_size, tile_overlap)

    records: list[dict[str, object]] = []
    chip_idx = 0

    for y0, y1 in y_ranges:
        for x0, x1 in x_ranges:
            chip_rgb = rgb[y0:y1, x0:x1]
            chip_mask = mask[y0:y1, x0:x1]

            label_count = int((chip_mask != IGNORE_INDEX).sum())
            if skip_empty and label_count == 0:
                continue

            chip_stem = f"{pair.stem}_chip_{chip_idx:04d}"
            image_out = output_dir / "images" / split / f"{chip_stem}.png"
            mask_out = output_dir / "masks" / split / f"{chip_stem}.png"

            save_rgb_png(chip_rgb, image_out)
            save_mask_png(chip_mask, mask_out)

            class_counts, class_area = _mask_class_summary(chip_mask)
            records.append({
                "stem": chip_stem,
                "split": split,
                "image_path": str(image_out),
                "mask_path": str(mask_out),
                "annotation_path": str(pair.annotation_path),
                "image_source_path": str(pair.image_path),
                "label_pixel_count": label_count,
                "class_counts": class_counts,
                "class_area_px2": class_area,
                "image_shape": [int(chip_rgb.shape[0]), int(chip_rgb.shape[1])],
                "chip_origin": [y0, x0],
            })
            chip_idx += 1

    print(f"[INFO] Sliced {pair.stem} into {len(records)} annotated chips.")
    return records


# ------------------------------------------------------------------
# Config / CLI parsing (mirrors yoloseg_pipeline)
# ------------------------------------------------------------------


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


def _prepare_split_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "masks" / split).mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CVAT-style tif/json tiles into DeepLabV3+ pixel-mask format."
    )
    parser.add_argument(
        "--source-dir",
        default=str(PROJECT_ROOT / "Maps" / "Tiles" / "Atlanta_split_google"),
        help="Folder containing matching .tif and .json tile files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "deeplabv3_windcomfort_rgb"),
        help="Destination dataset directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config file with source_dir, output_dir, train_tiles, etc.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-tiles", default=None)
    parser.add_argument("--val-tiles", default=None)
    parser.add_argument("--test-tiles", default=None)
    parser.add_argument(
        "--skip-empty-label-tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------
# Main conversion entry point
# ------------------------------------------------------------------


def convert_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
    train_tiles: str | list | None = None,
    val_tiles: str | list | None = None,
    test_tiles: str | list | None = None,
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
        train_items = [p for p in pairs if p.stem in train_tile_names]
        val_items = [p for p in pairs if p.stem in val_tile_names]
        test_items = [p for p in pairs if p.stem in test_tile_names]

        remaining = [
            p for p in pairs
            if p.stem not in train_tile_names
            and p.stem not in val_tile_names
            and p.stem not in test_tile_names
        ]
        if remaining:
            extra_train, extra_val = split_items(remaining, val_ratio=val_ratio, seed=seed)
            train_items.extend(extra_train)
            val_items.extend(extra_val)
    else:
        train_items, val_items = split_items(pairs, val_ratio=val_ratio, seed=seed)
        test_items = []

    # Also prepare test split dirs if needed
    if test_items:
        (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (output_dir / "masks" / "test").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    class_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    class_area_px2: dict[str, float] = {name: 0.0 for name in CLASS_NAMES}

    for split_name, split_items_list in (("train", train_items), ("val", val_items), ("test", test_items)):
        for pair in split_items_list:
            if tile_size is not None:
                sliced = _slice_tile(pair, output_dir, split_name, tile_size, tile_overlap, skip_empty_label_tiles)
                for rec in sliced:
                    records.append(rec)
                    for cn in CLASS_NAMES:
                        class_counts[cn] += int(rec.get("class_counts", {}).get(cn, 0))
                        class_area_px2[cn] += float(rec.get("class_area_px2", {}).get(cn, 0.0))
            else:
                rec = _copy_tile(pair, output_dir, split_name, skip_empty_label_tiles)
                if rec is not None:
                    records.append(rec)
                    for cn in CLASS_NAMES:
                        class_counts[cn] += int(rec.get("class_counts", {}).get(cn, 0))
                        class_area_px2[cn] += float(rec.get("class_area_px2", {}).get(cn, 0.0))

    # Write dataset summary (replaces YOLO's dataset.yaml — DeepLab reads dirs directly)
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "format": "deeplabv3_pixel_mask",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "train_count": sum(1 for r in records if r["split"] == "train"),
        "val_count": sum(1 for r in records if r["split"] == "val"),
        "test_count": sum(1 for r in records if r["split"] == "test"),
        "total_tiles": len(pairs),
        "total_chips": len(records),
        "train_tiles": sorted(train_tile_names) if train_tile_names else None,
        "val_tiles": sorted(val_tile_names) if val_tile_names else None,
        "test_tiles": sorted(test_tile_names) if test_tile_names else None,
        "skip_empty_label_tiles": skip_empty_label_tiles,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "class_counts": class_counts,
        "class_area_px2": class_area_px2,
        "records": records,
    }
    summary_path = dataset_summary_path(output_dir)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[INFO] Class mask pixel counts:")
    for cn in CLASS_NAMES:
        print(f"  {cn}: {class_counts[cn]:,}")
    print(f"\n[INFO] Total chips: {len(records)}")
    print(f"[INFO] Summary written to: {summary_path}")
    return summary_path


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main() -> int:
    args = _parse_args()

    # 1. Establish your absolute base defaults
    config_dict = {
        "source_dir": args.source_dir,
        "output_dir": args.output_dir,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "overwrite": args.overwrite,
        "train_tiles": args.train_tiles,
        "val_tiles": args.val_tiles,
        "test_tiles": args.test_tiles,
        "skip_empty_label_tiles": args.skip_empty_label_tiles,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
    }

    # 2. If a JSON config is provided, let it overwrite the base defaults
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            print(f"[INFO] Loading baseline configurations from: {config_path}")
            file_config = _load_run_config(config_path)
            
            # Normalize keys to use underscores so they match our dictionary
            normalized_file_config = {
                k.replace("-", "_"): v for k, v in file_config.items()
            }
            config_dict.update(normalized_file_config)
        else:
            print(f"[WARN] Config file {config_path} not found. Using CLI arguments.")

    # 3. Explicitly check the sys.argv strings to see what the user actually typed in the terminal.
    # If the user explicitly typed an override, force that override to take precedence over the JSON.
    provided_args = sys.argv[1:]
    
    if any(flag in provided_args for flag in ["--source-dir"]):
        config_dict["source_dir"] = args.source_dir
    if any(flag in provided_args for flag in ["--output-dir"]):
        config_dict["output_dir"] = args.output_dir
    if any(flag in provided_args for flag in ["--val-ratio"]):
        config_dict["val_ratio"] = args.val_ratio
    if any(flag in provided_args for flag in ["--seed"]):
        config_dict["seed"] = args.seed
    if "--overwrite" in provided_args:
        config_dict["overwrite"] = True
    if any(flag in provided_args for flag in ["--train-tiles"]):
        config_dict["train_tiles"] = args.train_tiles
    if any(flag in provided_args for flag in ["--val-tiles"]):
        config_dict["val_tiles"] = args.val_tiles
    if any(flag in provided_args for flag in ["--test-tiles"]):
        config_dict["test_tiles"] = args.test_tiles
    if "--skip-empty-label-tiles" in provided_args:
        config_dict["skip_empty_label_tiles"] = True
    if "--no-skip-empty-label-tiles" in provided_args:
        config_dict["skip_empty_label_tiles"] = False
    if any(flag in provided_args for flag in ["--tile-size"]):
        config_dict["tile_size"] = args.tile_size
    if any(flag in provided_args for flag in ["--tile-overlap"]):
        config_dict["tile_overlap"] = args.tile_overlap

    # 4. Safely extract and type-cast parameters out to execution variables
    source_dir = Path(config_dict["source_dir"])
    output_dir = Path(config_dict["output_dir"])
    val_ratio = float(config_dict["val_ratio"])
    seed = int(config_dict["seed"])
    overwrite = bool(config_dict["overwrite"])
    train_tiles = config_dict["train_tiles"]
    val_tiles = config_dict["val_tiles"]
    test_tiles = config_dict["test_tiles"]
    skip_empty = bool(config_dict["skip_empty_label_tiles"])
    
    tile_size_val = config_dict["tile_size"]
    tile_size = int(tile_size_val) if tile_size_val is not None else None
    tile_overlap = int(config_dict["tile_overlap"])

    # 5. Run conversion
    summary_path = convert_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        val_ratio=val_ratio,
        seed=seed,
        overwrite=overwrite,
        train_tiles=train_tiles,
        val_tiles=val_tiles,
        test_tiles=test_tiles,
        skip_empty_label_tiles=skip_empty,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    print(f"[INFO] Wrote DeepLabV3+ dataset summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
