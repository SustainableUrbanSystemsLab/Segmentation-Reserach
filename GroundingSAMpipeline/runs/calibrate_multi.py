from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATE = PROJECT_ROOT / "runs" / "calibrate_configs.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate across multiple tiles and aggregate IoU.")
    parser.add_argument("--tif-files", nargs="+", required=True, help="List of tif files to evaluate (relative to project root).")
    parser.add_argument("--annotation-path", "--xml-path", dest="xml_path", default=None, help="Optional annotation JSON/XML file or directory override.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "batch_runs_multi"), help="Output root for aggregated calibration results.")
    parser.add_argument("--prompt-family", choices=["grouped", "split", "all"], default="split", help="Prompt family to evaluate.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional limit on the number of combinations.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle trial order before running.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip trials that already have an IoU report for all tiles.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python interpreter to use for subprocess trials.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tif_files = [Path(p) for p in args.tif_files]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Build a dry-run of planned trials by invoking calibrate_configs in dry-run mode
    dry_spec_cmd = [args.python_exe, str(CALIBRATE), "--tif-file", str(tif_files[0]), "--prompt-family", args.prompt_family, "--dry-run"]
    if args.xml_path:
        dry_spec_cmd += ["--annotation-path", args.xml_path]
    proc = subprocess.run(dry_spec_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("Failed to query planned trials from calibrate_configs")

    planned = json.loads(proc.stdout)
    planned_trials = planned.get("planned_trials", [])
    if args.max_trials is not None:
        planned_trials = planned_trials[: args.max_trials]

    results: list[dict[str, Any]] = []

    total_trials = len(planned_trials)
    start_time = perf_counter()
    def _format_duration(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    for idx, planned_trial in enumerate(planned_trials, start=1):
        trial_name_base = planned_trial["trial_name"]
        per_tile_reports = []
        failed = False
        print(f"[INFO] [{idx}/{total_trials}] Starting aggregated trial: {trial_name_base} (running {len(tif_files)} tiles)")
        for tindex, tif in enumerate(tif_files, start=1):
            tile_stem = tif.stem
            trial_name = f"{trial_name_base}__{tile_stem}"
            spec = {
                "trial_name": trial_name,
                "tif_file": str(tif),
                "xml_path": args.xml_path,
                "output_dir": str(output_root),
                "annotation_iou_class_mode": planned_trial.get("prompt_family", "split"),
                "overrides": {
                    "ACTIVE_PROMPTS": planned_trial.get("ACTIVE_PROMPTS", []),
                    "AVAILABLE_PROMPTS": planned_trial.get("AVAILABLE_PROMPTS", {}),
                    **planned_trial.get("global_variant", {}),
                },
            }
            cmd = [args.python_exe, str(CALIBRATE), "--worker", "--spec", json.dumps(spec, default=str)]

            print(f"[INFO]   [{idx}/{total_trials}] Tile {tindex}/{len(tif_files)}: running worker for {tif} -> trial {trial_name}")
            # Stream the worker output to console so the user can follow progress
            completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if completed.returncode != 0:
                print(f"[WARN]   Worker failed for {trial_name} (returncode={completed.returncode})")
                failed = True
                break

            report_path = output_root / trial_name / "annotation_iou" / f"{tif.stem}_iou_report.json"
            if not report_path.exists():
                print(f"[WARN]   Expected IoU report missing: {report_path}")
                failed = True
                break

            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
            per_tile_reports.append({"tile": tif.stem, "report": report})
            print(f"[INFO]   [{idx}/{total_trials}] Tile {tindex}/{len(tif_files)} done: mean_iou={report.get('mean_iou', 0.0):.4f}")

        if failed or not per_tile_reports:
            print(f"[WARN] [{idx}/{total_trials}] Aggregated trial failed or produced no reports: {trial_name_base}")
            results.append({"trial_name": trial_name_base, "failed": True})
            # update ETA even on failure
            elapsed = perf_counter() - start_time
            completed = idx
            remaining = max(0, total_trials - completed)
            eta = (elapsed / completed) * remaining if completed else 0
            print(f"[INFO] [{idx}/{total_trials}] ETA remaining ~ {_format_duration(eta)}")
            continue

        mean_ious = [r["report"].get("mean_iou", 0.0) for r in per_tile_reports]
        mean_pixel_acc = [r["report"].get("pixel_accuracy", 0.0) for r in per_tile_reports]
        summary = {
            "trial_name": trial_name_base,
            "mean_iou_avg": float(mean(mean_ious)),
            "mean_pixel_accuracy_avg": float(mean(mean_pixel_acc)),
            "per_tile_reports": per_tile_reports,
        }
        results.append(summary)
        elapsed = perf_counter() - start_time
        completed = idx
        remaining = max(0, total_trials - completed)
        eta = (elapsed / completed) * remaining if completed else 0
        print(f"[INFO] [{idx}/{total_trials}] Aggregated trial complete: mean_iou_avg={summary['mean_iou_avg']:.4f}")
        print(f"[INFO] [{idx}/{total_trials}] ETA remaining ~ {_format_duration(eta)}")

    # Save JSON summary and a simple IOU comparison graph
    payload = {
        "tif_files": [str(p) for p in tif_files],
        "prompt_family": args.prompt_family,
        "generated_at": datetime.now().isoformat(),
        "results": results,
    }
    summary_path = output_root / f"calibration_multi_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Try to plot the mean IoUs
    try:
        import matplotlib.pyplot as plt

        names = [r["trial_name"] for r in results if not r.get("failed")]
        values = [r["mean_iou_avg"] for r in results if not r.get("failed")]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(values)), values, color="tab:blue")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Mean IoU (avg across tiles)")
        ax.set_title("Calibration: Mean IoU per Trial (averaged across tiles)")
        plt.tight_layout()
        graph_path = output_root / f"calibration_multi_iou_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(graph_path)
        print(f"[INFO] Saved IOU comparison graph to: {graph_path}")
    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")

    print(f"[INFO] Saved calibration summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
