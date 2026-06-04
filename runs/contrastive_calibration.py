from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATE = PROJECT_ROOT / "runs" / "calibrate_configs.py"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


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


def _default_parallel_jobs() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return max(1, os.cpu_count() or 1)

# Run on the labeled tiles only. Tile 004 is intentionally omitted.
_tiles_override = os.environ.get("CONTRASTIVE_TILES", "").strip()
if _tiles_override:
    TILES = [tile.strip() for tile in _tiles_override.split(",") if tile.strip()]
else:
    # Default tile stems relative to the tile base directory.
    _DEFAULT_TILE_STEMS = [
        "tile_002_002.tif",
        "tile_002_003.tif",
        "tile_002_004.tif",
    ]
    # If CONTRASTIVE_TILE_BASE_DIR is set (exported by the PACE batch script),
    # build absolute paths so the worker can find the files regardless of cwd.
    # Fall back to the known absolute PACE path, then to repo-relative for local runs.
    _tile_base_dir = os.environ.get("CONTRASTIVE_TILE_BASE_DIR", "").strip()
    _PACE_TILE_BASE = "/storage/project/r-pkastner3-0/ibaracskay3/Segmentation-Reserach-Manual/Maps/Tiles/Atlanta_split_google"
    if _tile_base_dir:
        TILES = [str(Path(_tile_base_dir) / stem) for stem in _DEFAULT_TILE_STEMS]
    elif Path(_PACE_TILE_BASE).is_dir():
        TILES = [str(Path(_PACE_TILE_BASE) / stem) for stem in _DEFAULT_TILE_STEMS]
    else:
        TILES = [
            str(PROJECT_ROOT / "Maps" / "Tiles" / "Atlanta_split_google" / stem)
            for stem in _DEFAULT_TILE_STEMS
        ]

BASE_WEIGHTS = {
    "nen_cat_a": 0.68,
    "nen_cat_b": 1.25,
    "nen_cat_c": 1.10,
    "nen_cat_d": 1.10,
    "nen_cat_e": 0.95,
}

_prompts_override = os.environ.get("CONTRASTIVE_PROMPTS", "").strip()
if _prompts_override:
    PROMPTS = [prompt.strip() for prompt in _prompts_override.split(",") if prompt.strip() in BASE_WEIGHTS]
    if not PROMPTS:
        raise ValueError("CONTRASTIVE_PROMPTS was set but did not contain any valid prompt names")
else:
    PROMPTS = list(BASE_WEIGHTS.keys())

_annotation_override = os.environ.get("CONTRASTIVE_ANNOTATION_PATH", "").strip()
DELTA_STEP = 0.03
# Default steps each side for interactive runs (5 => 11 values per prompt => 55 configs)
STEPS_EACH_SIDE = _env_int("STEPS_EACH_SIDE", 8)

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
    if STEPS_EACH_SIDE == 0:
        # Quick verification mode uses one small non-zero offset per prompt so the
        # prompt-specific configs are distinct instead of repeating the same baseline.
        value = round(base_value + 0.05, 3)
        values.append(max(0.05, value))
    else:
        for step in range(-STEPS_EACH_SIDE, STEPS_EACH_SIDE + 1):
            value = round(base_value + (step * DELTA_STEP), 3)
            values.append(max(0.05, value))
    return values


def _build_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen_weights = set()
    for prompt_name in PROMPTS:
        for value in _build_weight_values(prompt_name):
            weights = copy.deepcopy(BASE_WEIGHTS)
            weights[prompt_name] = value
            
            weights_key = tuple(sorted(weights.items()))
            if weights_key in seen_weights:
                continue
            seen_weights.add(weights_key)
            
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
        "xml_path": _annotation_override or None,
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
    STEPS_EACH_SIDE = _env_int("STEPS_EACH_SIDE", 5)
    per_config_minutes = ESTIMATED_MINUTES_PER_TILE

    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if PREFER_CUDA and REQUIRE_CUDA and not cuda_available:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False. Set REQUIRE_CUDA=False to allow CPU fallback.")

    output_root = Path(os.environ.get("CALIBRATION_OUTPUT_DIR", PROJECT_ROOT / "results" / "contrastive_calibration" / CALIBRATION_CACHE_KEY))
    output_root.mkdir(parents=True, exist_ok=True)
    parallel_jobs = max(1, _env_int("CALIBRATION_PARALLEL_JOBS", _default_parallel_jobs()))

    variants = _build_variants()
    total_configs = len(variants)
    total_trials = total_configs * len(TILES)
    start_time = datetime.now()
    print(f"[INFO] Base weights: {BASE_WEIGHTS}", flush=True)
    print(f"[INFO] Tiles: {TILES}", flush=True)
    print(
        f"[INFO] Cache key: {CALIBRATION_CACHE_KEY} | "
        f"resume={CALIBRATION_RESUME_FROM_CACHE} | force_rerun={CALIBRATION_FORCE_RERUN}"
    , flush=True)
    print(
        f"[INFO] CUDA preference: prefer_cuda={PREFER_CUDA}, require_cuda={REQUIRE_CUDA}, "
        f"torch.cuda.is_available()={cuda_available}, CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}"
    , flush=True)
    print(f"[INFO] Unique configs to run: {total_configs} (originally {len(PROMPTS)} prompts x {2 * STEPS_EACH_SIDE + 1} values, duplicates removed)", flush=True)
    print(f"[INFO] Total tile runs: {total_trials}", flush=True)
    print(f"[INFO] Parallel calibration jobs: {parallel_jobs}", flush=True)
    est_total_minutes = (total_configs * len(TILES) * per_config_minutes) / parallel_jobs
    print(
        f"[INFO] Estimated wall time with {parallel_jobs} worker(s) (based on {per_config_minutes} min/config/tile): "
        f"{_format_duration(est_total_minutes * 60)}",
        flush=True,
    )
    print(f"[INFO] Output root: {output_root}", flush=True)

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
    pending_est_minutes = (pending_tile_runs * per_config_minutes) / parallel_jobs
    print(
        f"[INFO] Cached tile runs found: {cached_tile_runs}/{total_trials} | "
        f"estimated remaining runtime: {_format_duration(pending_est_minutes * 60)}"
    , flush=True)

    results: list[dict[str, Any]] = []
    trial_states: list[dict[str, Any]] = []
    pending_jobs: list[dict[str, Any]] = []
    started = datetime.now()
    run_start_perf = perf_counter()
    processed_tile_runs = cached_tile_runs

    for index, variant in enumerate(variants, start=1):
        prompt_name = variant["prompt"]
        weight = variant["weight"]
        trial_name_base = f"trial_{index:03d}"
        trial_folder = output_root / trial_name_base
        trial_folder.mkdir(parents=True, exist_ok=True)
        per_tile_reports: list[dict[str, Any] | None] = [None] * len(TILES)
        trial_started = perf_counter()

        trial_state = {
            "trial_name": trial_name_base,
            "prompt_name": prompt_name,
            "weight": weight,
            "weights": variant["weights"],
            "trial_folder": trial_folder,
            "per_tile_reports": per_tile_reports,
            "trial_started": trial_started,
        }
        trial_states.append(trial_state)

        print(f"[INFO] [{index}/{total_configs}] Running config {trial_name_base} ({prompt_name}={weight:.3f})", flush=True)
        for tile_index, tif_file in enumerate(TILES, start=1):
            tile_stem = _tile_stem(tif_file)
            tile_folder = trial_folder / tile_stem
            tile_folder.mkdir(parents=True, exist_ok=True)
            report_path = tile_folder / "annotation_iou" / f"{tile_stem}_iou_report.json"
            tile_report_index = tile_index - 1

            if CALIBRATION_RESUME_FROM_CACHE and not CALIBRATION_FORCE_RERUN:
                cached_report = _load_report_if_exists(report_path)
                if cached_report is not None:
                    per_tile_reports[tile_report_index] = {
                        "tile": tile_stem,
                        "mean_iou": float(cached_report.get("mean_iou", 0.0)),
                        "pixel_accuracy": float(cached_report.get("pixel_accuracy", 0.0)),
                        "report_path": str(report_path),
                        "cached": True,
                        "elapsed_seconds": 0.0,
                    }
                    processed_tile_runs += 1
                    elapsed_tile = perf_counter() - run_start_perf
                    remaining_tile_runs = max(0, total_trials - processed_tile_runs)
                    tile_eta_seconds = (elapsed_tile / processed_tile_runs) * remaining_tile_runs if processed_tile_runs else 0.0
                    print(
                        f"[INFO]   Tile {tile_index}/{len(TILES)} cache hit: {tile_stem} "
                        f"mIoU={per_tile_reports[tile_report_index]['mean_iou']:.4f} | "
                        f"overall remaining ~ {_format_duration(tile_eta_seconds)}"
                    , flush=True)
                    continue

            pending_jobs.append(
                {
                    "index": index,
                    "total_configs": total_configs,
                    "trial_name_base": trial_name_base,
                    "trial_folder": trial_folder,
                    "tile_index": tile_index,
                    "tile_report_index": tile_report_index,
                    "tile_stem": tile_stem,
                    "tif_file": tif_file,
                    "report_path": report_path,
                    "weights": variant["weights"],
                }
            )

    if pending_jobs:
        worker_count = min(parallel_jobs, len(pending_jobs))
        if worker_count > 1 and cuda_available:
            print(
                f"[WARN] Parallel tile execution is enabled with a single visible GPU; "
                f"speedup may be limited by GPU contention.",
                flush=True,
            )
        print(f"[INFO] Launching {len(pending_jobs)} pending tile jobs with up to {worker_count} workers", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_worker, job["tile_stem"], job["tif_file"], job["trial_folder"], job["weights"]): job
                for job in pending_jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                trial_state = trial_states[job["index"] - 1]
                per_tile_reports = trial_state["per_tile_reports"]
                tile_report_index = job["tile_report_index"]
                tile_stem = job["tile_stem"]
                report_path = job["report_path"]

                try:
                    run_result = future.result()
                except Exception as exc:
                    run_result = {"returncode": 1, "elapsed_seconds": 0.0, "error": str(exc)}

                if run_result["returncode"] != 0 or not report_path.exists():
                    print(f"[WARN] [{job['index']}/{total_configs}] Tile failed or report missing for {tile_stem} in {job['trial_name_base']}")
                    per_tile_reports[tile_report_index] = {
                        "tile": tile_stem,
                        "failed": True,
                        "returncode": run_result["returncode"],
                        "report_path": str(report_path),
                        "elapsed_seconds": float(run_result.get("elapsed_seconds", 0.0)),
                    }
                else:
                    with report_path.open("r", encoding="utf-8") as f:
                        report = json.load(f)
                    per_tile_reports[tile_report_index] = {
                        "tile": tile_stem,
                        "mean_iou": float(report.get("mean_iou", 0.0)),
                        "pixel_accuracy": float(report.get("pixel_accuracy", 0.0)),
                        "report_path": str(report_path),
                        "cached": False,
                        "elapsed_seconds": float(run_result.get("elapsed_seconds", 0.0)),
                    }

                processed_tile_runs += 1
                elapsed_tile = perf_counter() - run_start_perf
                remaining_tile_runs = max(0, total_trials - processed_tile_runs)
                tile_eta_seconds = (elapsed_tile / processed_tile_runs) * remaining_tile_runs if processed_tile_runs else 0.0
                current_report = per_tile_reports[tile_report_index]
                if current_report and not current_report.get("failed"):
                    print(
                        f"[INFO]   Tile {job['tile_index']}/{len(TILES)} done: mIoU={current_report['mean_iou']:.4f} | "
                        f"overall remaining ~ {_format_duration(tile_eta_seconds)}",
                        flush=True,
                    )
                else:
                    print(f"[INFO]   Overall remaining after failed tile ~ {_format_duration(tile_eta_seconds)}", flush=True)

    for trial_state in trial_states:
        per_tile_reports = trial_state["per_tile_reports"]
        valid_reports = [item for item in per_tile_reports if item and not item.get("failed")]
        trial_name_base = trial_state["trial_name"]
        summary = {
            "trial_name": trial_name_base,
            "prompt": trial_state["prompt_name"],
            "weight": trial_state["weight"],
            "weights": trial_state["weights"],
            "trial_folder": str(trial_state["trial_folder"]),
            "mean_iou_avg": float(mean([item["mean_iou"] for item in valid_reports])) if valid_reports else float("nan"),
            "mean_pixel_accuracy_avg": float(mean([item["pixel_accuracy"] for item in valid_reports])) if valid_reports else float("nan"),
            "per_tile_reports": per_tile_reports,
            "cache_key": CALIBRATION_CACHE_KEY,
            "used_cached_tiles": sum(1 for item in per_tile_reports if item.get("cached")),
            "elapsed_seconds": float(perf_counter() - trial_state["trial_started"]),
            "failed_tiles": sum(1 for item in per_tile_reports if item and item.get("failed")),
        }
        results.append(summary)

        trial_summary_path = trial_state["trial_folder"] / "trial_summary.json"
        with trial_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        completed_trials = len(results)
        elapsed = perf_counter() - run_start_perf
        remaining = max(0, total_configs - completed_trials)
        eta_seconds = (elapsed / completed_trials) * remaining if completed_trials else 0.0
        print(
            f"[INFO] [{completed_trials}/{total_configs}] Finished {trial_name_base}: "
            f"mIoU={summary['mean_iou_avg']:.4f} | ETA ~ {_format_duration(eta_seconds)}"
        , flush=True)

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
        , flush=True)
    print(f"[INFO] Saved calibration summary to: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())