from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloseg_pipeline.common import PROJECT_ROOT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv11 segmentation model on the wind comfort dataset.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "yoloseg_windcomfort_rgb" / "dataset.yaml"))
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Starting checkpoint or model name.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=str(PROJECT_ROOT / "results" / "yoloseg"))
    parser.add_argument("--name", default="wind_comfort_seg")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--allow-large-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Pillow decompression-bomb checks for trusted local training datasets.",
    )
    parser.add_argument(
        "--clear-label-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete labels/*.cache files before training to avoid stale corrupt-image cache state.",
    )
    return parser.parse_args()


def _delete_label_caches(dataset_yaml: Path) -> None:
    if not dataset_yaml.exists():
        return

    yaml_lines = dataset_yaml.read_text(encoding="utf-8").splitlines()
    base_path: Path | None = None
    for line in yaml_lines:
        stripped = line.strip()
        if stripped.startswith("path:"):
            base_path = Path(stripped.split(":", 1)[1].strip())
            break

    if base_path is None:
        base_path = dataset_yaml.parent

    labels_dir = base_path / "labels"
    if not labels_dir.exists():
        return

    for cache_file in labels_dir.rglob("*.cache"):
        cache_file.unlink(missing_ok=True)


def _preflight_dependencies() -> None:
    # Ultralytics can auto-install this package mid-run, but that is noisy and racy with workers.
    if importlib.util.find_spec("pi_heif") is None:
        print("[WARN] Missing optional dependency 'pi-heif'. Install once with: pip install pi-heif")


def main() -> int:
    args = _parse_args()
    _preflight_dependencies()

    if args.allow_large_images:
        Image.MAX_IMAGE_PIXELS = None

    dataset_yaml = Path(args.data)
    if args.clear_label_cache:
        _delete_label_caches(dataset_yaml)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache=args.cache,
        pretrained=True,
        patience=25,
        amp=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180.0,
        translate=0.08,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.05,
        copy_paste=0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())