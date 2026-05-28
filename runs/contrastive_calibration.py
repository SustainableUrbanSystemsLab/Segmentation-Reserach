from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATE = PROJECT_ROOT / "runs" / "calibrate_configs.py"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)

# Run on the labeled tiles only. Tile 004 is intentionally omitted.
TILES = [
    "Maps/Tiles/Atlanta_split_google/tile_002_002.tif",
    "Maps/Tiles/Atlanta_split_google/tile_002_003.tif",
    "Maps/Tiles/Atlanta_split_google/tile_002_004.tif",
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
# Default steps each side for interactive runs (5 => 11 values per prompt => 55 configs)
STEPS_EACH_SIDE = _env_int("STEPS_EACH_SIDE", 5)

# Calibration resume/cache controls.
# Change CALIBRATION_CACHE_KEY to start a fresh run folder.
CALIBRATION_CACHE_KEY = os.environ.get("CALIBRATION_CACHE_KEY", "split_3tiles_steps5_v2")
CALIBRATION_RESUME_FROM_CACHE = _env_bool("CALIBRATION_RESUME_FROM_CACHE", True)
CALIBRATION_FORCE_RERUN = _env_bool("CALIBRATION_FORCE_RERUN", False)

# Runtime/GPU controls.
PREFER_CUDA = _env_bool("PREFER_CUDA", True)
REQUIRE_CUDA = _env_bool("REQUIRE_CUDA", True)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Keep pipeline cache so reruns can reuse prior expensive intermediates.
OVERWRITE_PIPELINE_CACHE = _env_bool("OVERWRITE_PIPELINE_CACHE", False)

# Runtime estimate controls for progress output.
ESTIMATED_MINUTES_PER_TILE = _env_float("ESTIMATED_MINUTES_PER_TILE", 5.0)


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


def _load_report_if_exists(report_path: Path) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    if PREFER_CUDA:
        env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    # More stable CUDA memory behavior for long sweeps.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    return env


def _run_worker(trial_name: str, tif_file: str, output_dir: Path, weights: dict[str, float]) -> dict[str, Any]:
    requested_device = "cuda" if PREFER_CUDA else "auto"
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
            "enable_pipeline_caching": True,
            "overwrite_pipeline_cache": OVERWRITE_PIPELINE_CACHE,
            "skip_mask_caching": _env_bool("SKIP_MASK_CACHING", True),
            "skip_if_visualizations_exist": _env_bool("SKIP_IF_VISUALIZATIONS_EXIST", True),
            "save_input_images": _env_bool("SAVE_INPUT_IMAGES", False),
            "sam_device": requested_device,
            "dino_device": requested_device,
            "output_dpi": 90,
            "dino_visualization_dpi": 90,
            "combined_visualization_max_dim": 900,
            "prompt_strength_heatmap_max_dim": 900,
            "annotation_iou_visualization_max_dim": 900,
        },
    }

    cmd = [sys.executable, str(CALIBRATE), "--worker", "--spec", json.dumps(spec, default=_json_default)]
    started = perf_counter()
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_build_worker_env())
    elapsed_seconds = perf_counter() - started
    return {
        "returncode": int(completed.returncode),
        "spec": spec,
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> int:
    # Hard-coded settings for interactive/local runs
    global STEPS_EACH_SIDE
    STEPS_EACH_SIDE = 5
    per_config_minutes = ESTIMATED_MINUTES_PER_TILE

    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if PREFER_CUDA and REQUIRE_CUDA and not cuda_available:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False. Set REQUIRE_CUDA=False to allow CPU fallback.")

    output_root = PROJECT_ROOT / "results" / "contrastive_calibration" / CALIBRATION_CACHE_KEY
    output_root.mkdir(parents=True, exist_ok=True)

    variants = _build_variants()
    total_configs = len(variants)
    total_trials = total_configs * len(TILES)
    start_time = datetime.now()
    print(f"[INFO] Base weights: {BASE_WEIGHTS}")
    print(f"[INFO] Tiles: {TILES}")
    print(
        f"[INFO] Cache key: {CALIBRATION_CACHE_KEY} | "
        f"resume={CALIBRATION_RESUME_FROM_CACHE} | force_rerun={CALIBRATION_FORCE_RERUN}"
    )
    print(
        f"[INFO] CUDA preference: prefer_cuda={PREFER_CUDA}, require_cuda={REQUIRE_CUDA}, "
        f"torch.cuda.is_available()={cuda_available}, CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}"
    )
    print(f"[INFO] One-at-a-time configs: {total_configs} ({len(PROMPTS)} prompts x {2 * STEPS_EACH_SIDE + 1} values)")
    est_total_minutes = total_configs * len(TILES) * per_config_minutes
    print(f"[INFO] Estimated total runtime (based on {per_config_minutes} min/config/tile): {_format_duration(est_total_minutes*60)}")
    print(f"[INFO] Total tile runs: {total_trials}")
    print(f"[INFO] Output root: {output_root}")

    cached_tile_runs = 0
    if CALIBRATION_RESUME_FROM_CACHE and not CALIBRATION_FORCE_RERUN:
        for idx, _variant in enumerate(variants, start=1):
            trial_folder = output_root / f"trial_{idx:03d}"
            for tif_file in TILES:
                tile_stem = _tile_stem(tif_file)
                report_path = trial_folder / tile_stem / "annotation_iou" / f"{tile_stem}_iou_report.json"
                if _load_report_if_exists(report_path) is not None:
                    cached_tile_runs += 1

    pending_tile_runs = max(0, total_trials - cached_tile_runs)
    pending_est_minutes = pending_tile_runs * per_config_minutes
    print(
        f"[INFO] Cached tile runs found: {cached_tile_runs}/{total_trials} | "
        f"estimated remaining runtime: {_format_duration(pending_est_minutes * 60)}"
    )

    results: list[dict[str, Any]] = []
    started = datetime.now()
    processed_tile_runs = 0

    for index, variant in enumerate(variants, start=1):
        prompt_name = variant["prompt"]
        weight = variant["weight"]
        trial_name_base = f"trial_{index:03d}"
        trial_folder = output_root / trial_name_base
        trial_folder.mkdir(parents=True, exist_ok=True)
        per_tile_reports: list[dict[str, Any]] = []
        trial_started = perf_counter()

        print(f"[INFO] [{index}/{total_configs}] Running config {trial_name_base} ({prompt_name}={weight:.3f})")
        for tile_index, tif_file in enumerate(TILES, start=1):
            tile_stem = _tile_stem(tif_file)
            tile_folder = trial_folder / tile_stem
            tile_folder.mkdir(parents=True, exist_ok=True)
            report_path = tile_folder / "annotation_iou" / f"{tile_stem}_iou_report.json"

            if CALIBRATION_RESUME_FROM_CACHE and not CALIBRATION_FORCE_RERUN:
                cached_report = _load_report_if_exists(report_path)
                if cached_report is not None:
                    per_tile_reports.append(
                        {
                            "tile": tile_stem,
                            "mean_iou": float(cached_report.get("mean_iou", 0.0)),
                            "pixel_accuracy": float(cached_report.get("pixel_accuracy", 0.0)),
                            "report_path": str(report_path),
                            "cached": True,
                            "elapsed_seconds": 0.0,
                        }
                    )
                    processed_tile_runs += 1
                    elapsed_tile = (datetime.now() - started).total_seconds()
                    remaining_tile_runs = max(0, total_trials - processed_tile_runs)
                    tile_eta_seconds = (elapsed_tile / processed_tile_runs) * remaining_tile_runs if processed_tile_runs else 0.0
                    print(
                        f"[INFO]   Tile {tile_index}/{len(TILES)} cache hit: {tile_stem} "
                        f"mIoU={per_tile_reports[-1]['mean_iou']:.4f} | "
                        f"overall remaining ~ {_format_duration(tile_eta_seconds)}"
                    )
                    continue

            print(f"[INFO]   Starting tile {tile_index}/{len(TILES)}: {tile_stem} (config {index}/{total_configs}) at {datetime.now().isoformat()}")
            run_result = _run_worker(tile_stem, tif_file, trial_folder, variant["weights"])
            if run_result["returncode"] != 0 or not report_path.exists():
                print(f"[WARN] [{index}/{total_configs}] Tile failed or report missing for {tile_stem} in {trial_name_base}")
                per_tile_reports.append(
                    {
                        "tile": tile_stem,
                        "failed": True,
                        "returncode": run_result["returncode"],
                        "report_path": str(report_path),
                        "elapsed_seconds": float(run_result.get("elapsed_seconds", 0.0)),
                    }
                )
                processed_tile_runs += 1
                elapsed_tile = (datetime.now() - started).total_seconds()
                remaining_tile_runs = max(0, total_trials - processed_tile_runs)
                tile_eta_seconds = (elapsed_tile / processed_tile_runs) * remaining_tile_runs if processed_tile_runs else 0.0
                print(f"[INFO]   Overall remaining after failed tile ~ {_format_duration(tile_eta_seconds)}")
                continue

            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
            per_tile_reports.append(
                {
                    "tile": tile_stem,
                    "mean_iou": float(report.get("mean_iou", 0.0)),
                    "pixel_accuracy": float(report.get("pixel_accuracy", 0.0)),
                    "report_path": str(report_path),
                    "cached": False,
                    "elapsed_seconds": float(run_result.get("elapsed_seconds", 0.0)),
                }
            )
            processed_tile_runs += 1
            elapsed_tile = (datetime.now() - started).total_seconds()
            remaining_tile_runs = max(0, total_trials - processed_tile_runs)
            tile_eta_seconds = (elapsed_tile / processed_tile_runs) * remaining_tile_runs if processed_tile_runs else 0.0
            print(
                f"[INFO]   Tile {tile_index}/{len(TILES)} done: mIoU={per_tile_reports[-1]['mean_iou']:.4f} | "
                f"overall remaining ~ {_format_duration(tile_eta_seconds)}"
            )

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
            "cache_key": CALIBRATION_CACHE_KEY,
            "used_cached_tiles": sum(1 for item in per_tile_reports if item.get("cached")),
            "elapsed_seconds": float(perf_counter() - trial_started),
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