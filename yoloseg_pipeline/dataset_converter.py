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
    PROJECT_ROOT,
    TilePair,
    iter_tile_pairs,
    load_rgb_from_tif,
    save_rgb_png,
    split_items,
    write_dataset_yaml,
    write_yolo_label_file,
    dataset_summary_path,
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

    return {
        "stem": pair.stem,
        "split": split,
        "image_path": str(image_out),
        "label_path": str(label_out),
        "annotation_path": str(pair.annotation_path),
        "image_source_path": str(pair.image_path),
        "label_count": label_count,
        "image_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
    }


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
    for split_name, split_items_list in (("train", train_items), ("val", val_items), ("test", test_items)):
        for pair in split_items_list:
            copied = _copy_tile(pair, output_dir, split_name, skip_empty_label_tiles)
            if copied is not None:
                records.append(copied)

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
        "records": records,
        "dataset_yaml": str(yaml_path),
    }
    dataset_summary_path(output_dir).write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    )
    print(f"[INFO] Wrote YOLO dataset config: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())