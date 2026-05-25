from __future__ import annotations

import argparse
import copy
import json
import re
import runpy
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Edit these arrays to widen or narrow the calibration search space.
# Use `--prompt-family grouped` for A/C/E grouped prompts or `--prompt-family split`
# for the full A/B/C/D/E taxonomy. `--prompt-family all` runs both families.
PROMPT_FAMILIES: dict[str, list[dict[str, Any]]] = {
    "grouped": [
        {
            "name": "baseline",
            "active_prompts": ["nen_cat_a", "nen_cat_c", "nen_cat_e"],
            "prompt_overrides": {},
        },
        {
            "name": "conservative",
            "active_prompts": ["nen_cat_a", "nen_cat_c", "nen_cat_e"],
            "prompt_overrides": {
                "nen_cat_a": {
                    "caption": "dense vegetation . tree canopy . sheltered park . wooded area . green space . lawn . garden",
                    "clip_score_threshold": 0.00,
                    "clip_relative_score_margin": 0.12,
                },
                "nen_cat_c": {
                    "caption": "sidewalk . pedestrian path . paved walkway . plaza . courtyard . promenade . transit plaza",
                    "clip_score_threshold": -0.10,
                    "clip_relative_score_margin": 0.10,
                },
                "nen_cat_e": {
                    "caption": "highway . freeway . parking lot . rooftop . roof . industrial yard . asphalt . hardscape",
                    "clip_score_threshold": -0.15,
                    "clip_relative_score_margin": 0.06,
                },
            },
        },
        {
            "name": "recall",
            "active_prompts": ["nen_cat_a", "nen_cat_c", "nen_cat_e"],
            "prompt_overrides": {
                "nen_cat_a": {
                    "caption": "park . plaza . courtyard . outdoor seating . tree-lined lawn . public square . garden . shaded pedestrian area",
                    "box_threshold": 0.15,
                    "text_threshold": 0.12,
                },
                "nen_cat_c": {
                    "caption": "sidewalk . pedestrian walkway . footpath . path . trail . promenade . open paved area",
                    "box_threshold": 0.13,
                    "text_threshold": 0.11,
                },
                "nen_cat_e": {
                    "caption": "road . parking lot . asphalt . concrete . roof . rooftop . highway . exposed hardscape",
                    "box_threshold": 0.18,
                    "text_threshold": 0.14,
                },
            },
        },
    ],
    "split": [
        {
            "name": "baseline",
            "active_prompts": ["nen_cat_a", "nen_cat_b", "nen_cat_c", "nen_cat_d", "nen_cat_e"],
            "prompt_overrides": {},
        },
        {
            "name": "conservative",
            "active_prompts": ["nen_cat_a", "nen_cat_b", "nen_cat_c", "nen_cat_d", "nen_cat_e"],
            "prompt_overrides": {
                "nen_cat_a": {
                    "caption": "dense vegetation . tree canopy . sheltered park . wooded area . green space . lawn . garden",
                    "clip_score_threshold": 0.00,
                    "clip_relative_score_margin": 0.12,
                },
                "nen_cat_b": {
                    "caption": "pedestrian-friendly space . shaded plaza . garden seating . walkable green area . park-like setting",
                    "clip_score_threshold": -0.02,
                    "clip_relative_score_margin": 0.10,
                },
                "nen_cat_c": {
                    "caption": "sidewalk . pedestrian path . paved walkway . plaza . courtyard . promenade . transit plaza",
                    "clip_score_threshold": -0.10,
                    "clip_relative_score_margin": 0.10,
                },
                "nen_cat_d": {
                    "caption": "open exposed plaza . windswept pedestrian area . minimally vegetated outdoor space . sparse hardscape",
                    "clip_score_threshold": -0.08,
                    "clip_relative_score_margin": 0.10,
                },
                "nen_cat_e": {
                    "caption": "highway . freeway . parking lot . rooftop . roof . industrial yard . asphalt . hardscape",
                    "clip_score_threshold": -0.15,
                    "clip_relative_score_margin": 0.06,
                },
            },
        },
        {
            "name": "recall",
            "active_prompts": ["nen_cat_a", "nen_cat_b", "nen_cat_c", "nen_cat_d", "nen_cat_e"],
            "prompt_overrides": {
                "nen_cat_a": {
                    "caption": "park . plaza . courtyard . outdoor seating . tree-lined lawn . public square . garden . shaded pedestrian area",
                    "box_threshold": 0.15,
                    "text_threshold": 0.12,
                },
                "nen_cat_b": {
                    "caption": "walkable plaza . pedestrian area with trees and benches . public square with shade . park-like setting",
                    "box_threshold": 0.20,
                    "text_threshold": 0.16,
                },
                "nen_cat_c": {
                    "caption": "sidewalk . pedestrian walkway . footpath . path . trail . promenade . open paved area",
                    "box_threshold": 0.13,
                    "text_threshold": 0.11,
                },
                "nen_cat_d": {
                    "caption": "exposed open area . windy plaza . minimally vegetated outdoor space . sparse pedestrian zone",
                    "box_threshold": 0.16,
                    "text_threshold": 0.12,
                },
                "nen_cat_e": {
                    "caption": "road . parking lot . asphalt . concrete . roof . rooftop . highway . exposed hardscape",
                    "box_threshold": 0.18,
                    "text_threshold": 0.14,
                },
            },
        },
    ],
}

GLOBAL_SEARCH_SPACE: dict[str, list[Any]] = {
    "full_image_mask_mode": [True, False],
    "tier_e_threshold": [0.24, 0.30, 0.36, 0.42],
    "tier_c_threshold": [0.12, 0.18],
    "dino_enable_tiled_fallback": [True],
    "dino_enable_area_split": [False],
    "dino_refine_bounds": [False, True],
    "dino_nms_iou_threshold": [0.45, 0.55],
}


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


def _deep_copy_prompt_map(prompt_module) -> dict[str, dict[str, Any]]:
    return {name: copy.deepcopy(settings) for name, settings in prompt_module.AVAILABLE_PROMPTS.items()}


def _trial_name(parts: list[str]) -> str:
    cleaned = [part.replace(" ", "_").replace("/", "_") for part in parts if part]
    return "__".join(cleaned)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate config/prompt combinations against annotation IoU.")
    parser.add_argument("--tif-file", required=False, help="Tile to evaluate. Defaults to cfg.tif_single_file.")
    parser.add_argument(
        "--annotation-path",
        "--xml-path",
        dest="xml_path",
        default=None,
        help="Optional annotation JSON/XML file or directory override.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "batch_runs"), help="Calibration output root.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python interpreter to use for subprocess trials.")
    parser.add_argument(
        "--prompt-family",
        choices=sorted(PROMPT_FAMILIES.keys()) + ["all"],
        default="split",
        help="Which prompt taxonomy to calibrate: grouped A/C/E, split A/B/C/D/E, or all.",
    )
    parser.add_argument("--max-trials", type=int, default=None, help="Optional limit on the number of combinations.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle trial order before running.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip trials that already have an IoU report.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned trials without running them.")
    parser.add_argument("--worker", action="store_true", help="Internal mode used to execute one trial.")
    parser.add_argument("--spec", default=None, help="Internal JSON spec for worker mode.")
    return parser.parse_args()


def _apply_overrides(cfg_module, prompts_module, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if key == "AVAILABLE_PROMPTS":
            for prompt_name, prompt_overrides in value.items():
                if prompt_name not in prompts_module.AVAILABLE_PROMPTS:
                    raise KeyError(f"Unknown prompt '{prompt_name}' in calibration spec")
                prompts_module.AVAILABLE_PROMPTS[prompt_name].update(copy.deepcopy(prompt_overrides))
            continue

        if not hasattr(cfg_module, key):
            raise AttributeError(f"Unknown config attribute '{key}' in calibration spec")
        setattr(cfg_module, key, copy.deepcopy(value))

    selected = list(getattr(cfg_module, "ACTIVE_PROMPTS", []))
    cfg_module.dino_prompt_configs = [
        {"name": name, **prompts_module.AVAILABLE_PROMPTS[name]}
        for name in selected
        if name in prompts_module.AVAILABLE_PROMPTS
    ]


def _estimate_eta(elapsed_seconds: float, completed_trials: int, total_trials: int) -> float:
    if completed_trials <= 0:
        return 0.0
    average_seconds = elapsed_seconds / completed_trials
    return average_seconds * max(0, total_trials - completed_trials)


def _prompt_families_for_mode(prompt_family: str) -> list[tuple[str, list[dict[str, Any]]]]:
    if prompt_family == "all":
        return list(PROMPT_FAMILIES.items())
    return [(prompt_family, PROMPT_FAMILIES[prompt_family])]


class _FilteredOutput:
    def __init__(self, sink) -> None:
        self._sink = sink
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        written = 0
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            written += len(line) + 1
            if self._should_forward(line):
                self._sink.write(line + "\n")
        return written

    def flush(self) -> None:
        if self._buffer:
            if self._should_forward(self._buffer):
                self._sink.write(self._buffer)
            self._buffer = ""
        self._sink.flush()

    def fileno(self) -> int:
        return self._sink.fileno()

    def isatty(self) -> bool:
        return bool(getattr(self._sink, "isatty", lambda: False)())

    def writable(self) -> bool:
        return True

    @staticmethod
    def _should_forward(line: str) -> bool:
        return bool(
            re.search(r"Saved figure: .*combined_masks|Saved annotation comparison figure|IoU report saved to:", line)
        )


def _worker_main(spec: dict[str, Any]) -> int:
    from models import config as cfg
    from models import prompts as prompts_module

    spec = copy.deepcopy(spec)
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_file = str(spec["tif_file"])
    trial_name = str(spec["trial_name"])
    trial_dir = output_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg.tif_run_mode = "single"
    cfg.tif_single_file = tif_file
    cfg.tif_file = tif_file
    cfg.tif_files = [tif_file]
    cfg.results_dir = trial_dir / "results"
    cfg.enable_annotation_iou_check = True
    cfg.annotation_iou_xml_path = spec.get("xml_path") or spec.get("annotation_path")
    cfg.annotation_iou_output_dir = trial_dir / "annotation_iou"
    cfg.overwrite_pipeline_cache = True
    cfg.annotation_iou_class_mode = spec.get("annotation_iou_class_mode", "grouped")

    prompts_module.AVAILABLE_PROMPTS = _deep_copy_prompt_map(prompts_module)
    _apply_overrides(cfg, prompts_module, spec["overrides"])

    cfg.enable_annotation_iou_check = True
    cfg.annotation_iou_xml_path = spec.get("xml_path") or spec.get("annotation_path")
    cfg.annotation_iou_output_dir = trial_dir / "annotation_iou"
    cfg.results_dir = trial_dir / "results"

    filtered_output = _FilteredOutput(sys.stdout)
    with redirect_stdout(filtered_output), redirect_stderr(filtered_output):
        runpy.run_path(str(RUNS_DIR / "get_satellite.py"), run_name="__main__")

    combined_mask_outputs = sorted((trial_dir / "results" / "combined_masks").glob("*.png"))
    if combined_mask_outputs:
        print(f"[INFO] Final combined mask saved to: {combined_mask_outputs[-1]}")

    report_paths = sorted((trial_dir / "annotation_iou").glob("*_iou_report.json"))
    if not report_paths:
        raise FileNotFoundError(f"No IoU report written for trial '{trial_name}'")

    report_path = report_paths[0]
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    summary = {
        "trial_name": trial_name,
        "report_path": str(report_path),
        "mean_iou": float(report.get("mean_iou", 0.0)),
        "pixel_accuracy": float(report.get("pixel_accuracy", 0.0)),
        "evaluated_pixels": int(report.get("evaluated_pixels", 0)),
        "ignored_invalid_pixels": int(report.get("ignored_invalid_pixels", 0)),
        "output_dir": str(trial_dir),
    }
    summary_path = trial_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(json.dumps(summary, indent=2))
    return 0


def _build_search_space() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_variants: list[dict[str, Any]] = []
    for family_name, presets in PROMPT_FAMILIES.items():
        for preset in presets:
            prompt_variants.append(
                {
                    "family": family_name,
                    "name": preset["name"],
                    "ACTIVE_PROMPTS": list(preset["active_prompts"]),
                    "AVAILABLE_PROMPTS": copy.deepcopy(preset.get("prompt_overrides", {})),
                }
            )

    global_keys = list(GLOBAL_SEARCH_SPACE.keys())
    global_variants: list[dict[str, Any]] = []
    for values in product(*(GLOBAL_SEARCH_SPACE[key] for key in global_keys)):
        global_variants.append({key: value for key, value in zip(global_keys, values)})

    return prompt_variants, global_variants


def _driver_main(args: argparse.Namespace) -> int:
    from models import config as cfg

    tif_file = args.tif_file or getattr(cfg, "tif_single_file", None) or getattr(cfg, "tif_file", None)
    if not tif_file:
        raise RuntimeError("No tif file provided. Use --tif-file or set cfg.tif_single_file.")

    xml_path = args.xml_path or getattr(cfg, "annotation_iou_xml_path", None)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    family_runs: list[tuple[str, list[dict[str, Any]]]] = _prompt_families_for_mode(str(args.prompt_family))
    global_trials = _build_search_space()[1]

    planned = []
    for family_name, prompt_trials in family_runs:
        combinations = list(product(prompt_trials, global_trials))
        if args.shuffle:
            import random

            random.Random(42).shuffle(combinations)

        if args.max_trials is not None:
            combinations = combinations[: max(0, int(args.max_trials))]

        for prompt_trial, global_trial in combinations:
            trial_name = _trial_name([family_name, prompt_trial["name"], Path(tif_file).stem, f"trial{len(planned) + 1:03d}"])
            planned.append(
                {
                    "trial_name": trial_name,
                    "prompt_family": family_name,
                    "prompt_preset": prompt_trial["name"],
                    "global_variant": global_trial,
                }
            )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "tif_file": tif_file,
                    "prompt_family": args.prompt_family,
                    "prompt_presets": [trial["name"] for _, trials in family_runs for trial in trials],
                    "global_search_space": {key: len(values) for key, values in GLOBAL_SEARCH_SPACE.items()},
                    "planned_trials": planned,
                },
                indent=2,
                default=_json_default,
            )
        )
        return 0

    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    total = len(planned)
    run_start = perf_counter()

    print(f"[INFO] Calibration prompt family: {args.prompt_family}")
    print(f"[INFO] Calibration search space: {len(planned)} trials")
    print(f"[INFO] Output root: {output_root.resolve()}")

    for index, planned_trial in enumerate(planned, start=1):
        prompt_family = planned_trial["prompt_family"]
        prompt_trial = next(
            trial for family_name, trials in family_runs if family_name == prompt_family for trial in trials if trial["name"] == planned_trial["prompt_preset"]
        )
        global_trial = planned_trial["global_variant"]
        trial_name = planned_trial["trial_name"]
        trial_dir = output_root / trial_name
        report_path = trial_dir / "annotation_iou" / f"{Path(tif_file).stem}_iou_report.json"
        elapsed_before_trial = perf_counter() - run_start
        eta_before_trial = _estimate_eta(elapsed_before_trial, index - 1, total)

        print(
            f"[INFO] [{index}/{total}] Starting {trial_name} | "
            f"elapsed {_format_duration(elapsed_before_trial)} | ETA ~ {_format_duration(eta_before_trial)}"
        )

        if args.skip_existing and report_path.exists():
            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
            summary = {
                "trial_name": trial_name,
                "prompt_family": prompt_family,
                "prompt_preset": prompt_trial["name"],
                "global_variant": global_trial,
                "mean_iou": float(report.get("mean_iou", 0.0)),
                "pixel_accuracy": float(report.get("pixel_accuracy", 0.0)),
                "evaluated_pixels": int(report.get("evaluated_pixels", 0)),
                "ignored_invalid_pixels": int(report.get("ignored_invalid_pixels", 0)),
                "report_path": str(report_path),
                "skipped_existing": True,
            }
        else:
            spec = {
                "trial_name": trial_name,
                "tif_file": tif_file,
                "xml_path": xml_path,
                "output_dir": str(output_root),
                "annotation_iou_class_mode": prompt_family,
                "overrides": {
                    "ACTIVE_PROMPTS": prompt_trial["ACTIVE_PROMPTS"],
                    "AVAILABLE_PROMPTS": prompt_trial.get("AVAILABLE_PROMPTS", {}),
                    **global_trial,
                },
            }
            command = [args.python_exe, str(Path(__file__).resolve()), "--worker", "--spec", json.dumps(spec, default=_json_default)]
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
            if completed.returncode != 0:
                summary = {
                    "trial_name": trial_name,
                    "prompt_family": prompt_family,
                    "prompt_preset": prompt_trial["name"],
                    "global_variant": global_trial,
                    "mean_iou": float("nan"),
                    "pixel_accuracy": float("nan"),
                    "evaluated_pixels": 0,
                    "ignored_invalid_pixels": 0,
                    "report_path": None,
                    "failed": True,
                    "returncode": int(completed.returncode),
                }
            else:
                if not report_path.exists():
                    raise FileNotFoundError(f"Expected IoU report missing after trial: {report_path}")
                with report_path.open("r", encoding="utf-8") as f:
                    report = json.load(f)
                summary = {
                    "trial_name": trial_name,
                    "prompt_family": prompt_family,
                    "prompt_preset": prompt_trial["name"],
                    "global_variant": global_trial,
                    "mean_iou": float(report.get("mean_iou", 0.0)),
                    "pixel_accuracy": float(report.get("pixel_accuracy", 0.0)),
                    "evaluated_pixels": int(report.get("evaluated_pixels", 0)),
                    "ignored_invalid_pixels": int(report.get("ignored_invalid_pixels", 0)),
                    "report_path": str(report_path),
                    "output_dir": str(trial_dir),
                }

        results.append(summary)
        if not summary.get("failed") and not summary.get("skipped_existing"):
            if best_result is None or summary["mean_iou"] > best_result["mean_iou"]:
                best_result = summary

        elapsed_after_trial = perf_counter() - run_start
        eta_after_trial = _estimate_eta(elapsed_after_trial, index, total)
        metric = summary.get("mean_iou")
        metric_text = "nan" if metric != metric else f"{metric:.4f}"
        print(
            f"[INFO] [{index}/{total}] Finished {trial_name}: mIoU={metric_text} | "
            f"elapsed {_format_duration(elapsed_after_trial)} | remaining ~ {_format_duration(eta_after_trial)}"
        )

    summary_path = output_root / f"calibration_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "tif_file": tif_file,
        "xml_path": xml_path,
        "prompt_family": args.prompt_family,
        "prompt_presets": [trial for _, trials in family_runs for trial in trials],
        "global_search_space": GLOBAL_SEARCH_SPACE,
        "results": results,
        "best_result": best_result,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    print(f"[INFO] Calibration summary saved to: {summary_path}")
    if best_result is not None:
        print(
            f"[INFO] Best trial: {best_result['trial_name']} "
            f"(mIoU={best_result['mean_iou']:.4f}, pixel_accuracy={best_result['pixel_accuracy']:.4f})"
        )
    return 0


def main() -> int:
    args = _parse_args()
    if args.worker:
        if not args.spec:
            raise RuntimeError("--worker requires --spec")
        spec = json.loads(args.spec)
        return _worker_main(spec)
    return _driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())