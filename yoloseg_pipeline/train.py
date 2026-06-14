from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------
# Environment Setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return max(1, os.cpu_count() or 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(
            PROJECT_ROOT / "data" / "yoloseg_windcomfort_rgb_sliced" / "dataset.yaml"
        ),
    )

    parser.add_argument("--model", default="yolo11l-seg.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=_default_workers())

    parser.add_argument("--name", default="wind_comfort_seg")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true")

    parser.add_argument(
        "--allow-large-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--clear-label-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def _preflight_dependencies() -> None:
    if importlib.util.find_spec("pi_heif") is None:
        print(
            "[WARN] Missing optional dependency pi-heif. "
            "Install with: pip install pi-heif"
        )


def _delete_label_caches(dataset_yaml: Path) -> None:
    if not dataset_yaml.exists():
        return

    yaml_lines = dataset_yaml.read_text(encoding="utf-8").splitlines()

    base_path = None

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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(config_path="yoloseg_pipeline/train_config.json"):

    _preflight_dependencies()

    args = _parse_args()
    run_args = vars(args)

    config_file = Path(config_path)

    if config_file.exists():
        print(f"--> Loading config: {config_file}")

        with open(config_file, "r") as f:
            config_data = json.load(f)

        run_args.update(config_data)

    # ----------------------------------------------
    # Large image support
    # ----------------------------------------------

    if run_args.get("allow_large_images", True):
        print("--> Disabling Pillow decompression limits")
        Image.MAX_IMAGE_PIXELS = None

    # ----------------------------------------------
    # Optional cache cleanup
    # ----------------------------------------------

    if run_args.get("clear_label_cache", False):
        print("--> Clearing label cache files")
        _delete_label_caches(Path(run_args["data"]))

    # ----------------------------------------------
    # Canonical directories
    # ----------------------------------------------

    run_name = run_args.get("name", "wind_comfort_seg")
    resume_requested = run_args.get("resume", False)

    #
    # SINGLE SOURCE OF TRUTH
    #
    project_dir = (PROJECT_ROOT / "results" / "yoloseg").resolve()

    run_dir = project_dir / run_name

    weights_dir = run_dir / "weights"

    last_checkpoint = weights_dir / "last.pt"

    archive_root = project_dir / "old_yoloseg_results"

    project_dir.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    print("\n--> Canonical paths")
    print(f"Project dir : {project_dir}")
    print(f"Run dir     : {run_dir}")
    print(f"Checkpoint  : {last_checkpoint}")
    print(f"Exists      : {last_checkpoint.exists()}")

    # ----------------------------------------------
    # Resume handling
    # ----------------------------------------------

    model_weights = run_args.get("model", "yolo11l-seg.pt")

    if resume_requested:

        if last_checkpoint.exists():

            print("\n--> [RESUME]")
            print(f"Checkpoint found:")
            print(f"    {last_checkpoint}")

            model_weights = str(last_checkpoint)

        else:

            print("\n--> [WARN]")
            print("Resume requested but checkpoint not found.")
            print(f"Expected: {last_checkpoint}")

            resume_requested = False
            run_args["resume"] = False

    # ----------------------------------------------
    # Archive previous run
    # ----------------------------------------------

    if run_dir.exists() and not resume_requested:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        archive_dir = archive_root / f"{run_name}_{timestamp}"

        print("\n--> [ARCHIVE]")
        print(f"FROM: {run_dir}")
        print(f"TO  : {archive_dir}")

        shutil.move(str(run_dir), str(archive_dir))

    # ----------------------------------------------
    # Build train args
    # ----------------------------------------------

    ignored_keys = {
        "_comment",
        "_comment_dataset",
        "_comment_cache",
        "project",                 # ignored on purpose
        "allow_large_images",
        "clear_label_cache",
        "model",
    }

    clean_train_args = {
        k: v
        for k, v in run_args.items()
        if not k.startswith("_")
        and k not in ignored_keys
    }

    clean_train_args.update(
        {
            "project": str(project_dir),
            "name": run_name,
            "exist_ok": True,
        }
    )

    print("\n--> Final YOLO args")
    print(f"project = {clean_train_args['project']}")
    print(f"name    = {clean_train_args['name']}")
    print(f"resume  = {resume_requested}")

    # ----------------------------------------------
    # Train
    # ----------------------------------------------

    if resume_requested:

        print("\n--> Launching resumed training")

        model = YOLO(str(last_checkpoint))

        model.train(resume=True)

    else:

        print("\n--> Launching fresh training")

        model = YOLO(model_weights)

        model.train(**clean_train_args)


if __name__ == "__main__":
    main()