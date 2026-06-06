from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yoloseg_pipeline.common import PROJECT_ROOT

# Default config file — lives next to this script.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "train_config.json"


def _default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return max(1, os.cpu_count() or 1)


def _load_config(config_path: Path) -> dict:
    """Load JSON config, stripping keys that start with '_' (comments)."""
    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}. Using built-in defaults.")
        return {}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        cleaned = {k: v for k, v in raw.items() if not k.startswith("_")}
        print(f"[INFO] Loaded training config: {config_path}")
        return cleaned
    except Exception as exc:
        print(f"[WARN] Could not parse config file {config_path}: {exc}. Using built-in defaults.")
        return {}


def _resolve_path(value: str) -> str:
    """Resolve a config path relative to PROJECT_ROOT if it is not absolute."""
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def _parse_args(config: dict) -> argparse.Namespace:
    # Pull config values (falling back to hard-coded defaults) so the config
    # file acts as the default layer and CLI flags always win.
    parser = argparse.ArgumentParser(
        description="Train a YOLOv11 segmentation model on the wind comfort dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to JSON training config file.",
    )
    parser.add_argument(
        "--data",
        default=_resolve_path(config.get("data", "data/yoloseg_windcomfort_rgb/dataset.yaml")),
    )
    parser.add_argument(
        "--model",
        default=config.get("model", "yolo11n-seg.pt"),
        help="Starting checkpoint or model name.",
    )
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 100))
    parser.add_argument("--batch", type=int, default=config.get("batch", 8))
    parser.add_argument("--imgsz", type=int, default=config.get("imgsz", 1280))
    parser.add_argument("--device", default=str(config.get("device", "0")))
    parser.add_argument("--workers", type=int, default=config.get("workers", _default_workers()))
    parser.add_argument(
        "--project",
        default=_resolve_path(config.get("project", "results/yoloseg")),
    )
    parser.add_argument("--name", default=config.get("name", "wind_comfort_seg"))
    parser.add_argument("--resume", action="store_true", default=config.get("resume", False))
    parser.add_argument("--cache", action="store_true", default=config.get("cache", False))

    # --- Class imbalance controls ---
    parser.add_argument(
        "--cls",
        type=float,
        default=config.get("cls", 0.5),
        help="Classification loss weight. Raise to ~1.0–2.0 to boost rare-class sensitivity.",
    )
    parser.add_argument(
        "--cls-pw",
        type=float,
        default=config.get("cls_pw", 0.0),
        help="Class weights power for handling class imbalance. "
             "0.0=disabled (uniform), 1.0=full inverse-frequency weighting. "
             "Use 1.0 to automatically upweight NEN_C and Uncomfortable.",
    )
    parser.add_argument(
        "--overlap-mask",
        action=argparse.BooleanOptionalAction,
        default=config.get("overlap_mask", True),
        help="Allow overlapping masks. Set --no-overlap-mask for small adjacent classes.",
    )
    parser.add_argument(
        "--allow-large-images",
        action=argparse.BooleanOptionalAction,
        default=config.get("allow_large_images", True),
        help="Disable Pillow decompression-bomb checks for trusted local training datasets.",
    )
    parser.add_argument(
        "--clear-label-cache",
        action=argparse.BooleanOptionalAction,
        default=config.get("clear_label_cache", True),
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
    if importlib.util.find_spec("pi_heif") is None:
        print("[WARN] Missing optional dependency 'pi-heif'. Install once with: pip install pi-heif")


def main() -> int:
    # Two-pass approach: load config file first, then parse CLI (CLI always wins).
    # We do a lightweight pre-parse just to find --config before the real parse.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()

    config = _load_config(Path(pre_args.config))
    args = _parse_args(config)

    _preflight_dependencies()
    print(f"[INFO] Resolved training workers: {args.workers}")

    if args.allow_large_images:
        Image.MAX_IMAGE_PIXELS = None

    dataset_yaml = Path(args.data)
    if args.clear_label_cache:
        _delete_label_caches(dataset_yaml)

    model = YOLO(args.model)

    train_kwargs: dict = dict(
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
        cls=args.cls,
        cls_pw=args.cls_pw,
        overlap_mask=args.overlap_mask,
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

    print(f"[INFO] cls={args.cls} | cls_pw={args.cls_pw} | overlap_mask={args.overlap_mask}")
    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
