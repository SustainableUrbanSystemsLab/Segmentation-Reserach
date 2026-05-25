from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATE = PROJECT_ROOT / "runs" / "calibrate_configs.py"

# Run on the labeled tiles only. Tile 004 is intentionally omitted.
TILES = [
    "Maps/Tiles/Atlanta_split_google/tile_002_000.tif",
    "Maps/Tiles/Atlanta_split_google/tile_002_001.tif",
    "Maps/Tiles/Atlanta_split_google/tile_002_002.tif",
    "Maps/Tiles/Atlanta_split_google/tile_002_003.tif",
    "Maps/Tiles/Atlanta_split_google/tile_002_005.tif",
    "Maps/Tiles/Atlanta_split_google/tile_002_006.tif",
]

BASE_WEIGHTS = {
    "nen_cat_a": 0.68,
    "nen_cat_b": 1.25,
    "nen_cat_c": 1.10,
    "nen_cat_d": 1.10,
    "nen_cat_e": 0.95,
}

PROMPTS = list(BASE_WEIGHTS.keys())
DELTA_STEP = 0.02
STEPS_EACH_SIDE = 25


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def _tile_stem(tile_path: str) -> str:
    return Path(tile_path).stem


def _build_weight_values(prompt_name: str) -> list[float]:
    base_value = BASE_WEIGHTS[prompt_name]
    values = []
    for step in range(-STEPS_EACH_SIDE, STEPS_EACH_SIDE + 1):
        value = round(base_value + (step * DELTA_STEP), 3)
        values.append(max(0.05, value))
    return values


def _build_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for prompt_name in PROMPTS:
        for value in _build_weight_values(prompt_name):
            weights = copy.deepcopy(BASE_WEIGHTS)
            weights[prompt_name] = value
            variants.append(
                {
                    "prompt": prompt_name,
                    "weight": value,
                    "weights": weights,
                    "trial_name": f"{prompt_name}__w{value:.3f}".replace(".", "p"),
                }
            )
    return variants


def _run_worker(trial_name: str, tif_file: str, output_dir: Path, weights: dict[str, float]) -> dict[str, Any]:
    spec = {
        "trial_name": trial_name,
        "tif_file": tif_file,
        "xml_path": None,
        "output_dir": str(output_dir),
        "annotation_iou_class_mode": "split",
        "overrides": {
            "contrastive_prompt_weights": weights,
            "pixel_assignment_mode": "region_context",
            "build_prompt_strength_heatmaps": True,
            "coarse_to_fine_cell_px": 0,
            "sam_auto_max_total_masks": 5000,
            "output_dpi": 90,
            "dino_visualization_dpi": 90,
            "combined_visualization_max_dim": 900,
            "prompt_strength_heatmap_max_dim": 900,
            "annotation_iou_visualization_max_dim": 900,
        },
    }

    cmd = [sys.executable, str(CALIBRATE), "--worker", "--spec", json.dumps(spec, default=_json_default)]
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return {
        "returncode": int(completed.returncode),
        "spec": spec,
    }


def main() -> int:
    output_root = PROJECT_ROOT / "results" / "contrastive_calibration"
    output_root.mkdir(parents=True, exist_ok=True)

    variants = _build_variants()
    total_configs = len(variants)
    total_trials = total_configs * len(TILES)
    start_time = datetime.now()
    print(f"[INFO] Base weights: {BASE_WEIGHTS}")
    print(f"[INFO] Tiles: {TILES}")
    print(f"[INFO] One-at-a-time configs: {total_configs} ({len(PROMPTS)} prompts x {2 * STEPS_EACH_SIDE + 1} values)")
    print(f"[INFO] Total tile runs: {total_trials}")
    print(f"[INFO] Output root: {output_root}")

    results: list[dict[str, Any]] = []
    started = datetime.now()

    for index, variant in enumerate(variants, start=1):
        prompt_name = variant["prompt"]
        weight = variant["weight"]
        trial_name_base = f"trial_{index:03d}"
        trial_folder = output_root / trial_name_base
        trial_folder.mkdir(parents=True, exist_ok=True)
        per_tile_reports: list[dict[str, Any]] = []

        print(f"[INFO] [{index}/{total_configs}] Running config {trial_name_base} ({prompt_name}={weight:.3f})")
        for tile_index, tif_file in enumerate(TILES, start=1):
            tile_stem = _tile_stem(tif_file)
            tile_folder = trial_folder / tile_stem
            tile_folder.mkdir(parents=True, exist_ok=True)

            run_result = _run_worker(tile_stem, tif_file, trial_folder, variant["weights"])
            report_path = tile_folder / "annotation_iou" / f"{tile_stem}_iou_report.json"
            if run_result["returncode"] != 0 or not report_path.exists():
                print(f"[WARN] [{index}/{total_configs}] Tile failed or report missing for {tile_stem} in {trial_name_base}")
                per_tile_reports.append(
                    {
                        "tile": tile_stem,
                        "failed": True,
                        "returncode": run_result["returncode"],
                        "report_path": str(report_path),
                    }
                )
                continue

            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
            per_tile_reports.append(
                {
                    "tile": tile_stem,
                    "mean_iou": float(report.get("mean_iou", 0.0)),
                    "pixel_accuracy": float(report.get("pixel_accuracy", 0.0)),
                    "report_path": str(report_path),
                }
            )
            print(f"[INFO]   Tile {tile_index}/{len(TILES)} done: mIoU={per_tile_reports[-1]['mean_iou']:.4f}")

        valid_reports = [item for item in per_tile_reports if not item.get("failed")]
        summary = {
            "trial_name": trial_name_base,
            "prompt": prompt_name,
            "weight": weight,
            "weights": variant["weights"],
            "trial_folder": str(trial_folder),
            "mean_iou_avg": float(mean([item["mean_iou"] for item in valid_reports])) if valid_reports else float("nan"),
            "mean_pixel_accuracy_avg": float(mean([item["pixel_accuracy"] for item in valid_reports])) if valid_reports else float("nan"),
            "per_tile_reports": per_tile_reports,
            "failed_tiles": len(per_tile_reports) - len(valid_reports),
        }
        results.append(summary)

        trial_summary_path = trial_folder / "trial_summary.json"
        with trial_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        elapsed = (datetime.now() - started).total_seconds()
        remaining = max(0, total_configs - index)
        eta_seconds = (elapsed / index) * remaining if index else 0.0
        print(
            f"[INFO] [{index}/{total_configs}] Finished {trial_name_base}: "
            f"mIoU={summary['mean_iou_avg']:.4f} | ETA ~ {_format_duration(eta_seconds)}"
        )

    payload = {
        "generated_at": start_time.isoformat(),
        "tiles": TILES,
        "base_weights": BASE_WEIGHTS,
        "delta_step": DELTA_STEP,
        "steps_each_side": STEPS_EACH_SIDE,
        "sweep_mode": "one_at_a_time",
        "results": results,
    }
    summary_path = output_root / f"contrastive_calibration_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    best_result = max((item for item in results if item.get("mean_iou_avg") == item.get("mean_iou_avg")), key=lambda item: item["mean_iou_avg"], default=None)
    if best_result is not None:
        print(
            f"[INFO] Best config so far: {best_result['trial_name']} "
            f"(mIoU={best_result['mean_iou_avg']:.4f}, pixel_accuracy={best_result['mean_pixel_accuracy_avg']:.4f})"
        )
    print(f"[INFO] Saved calibration summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())