from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "results" / "contrastive_calibration" / "split_3tiles_steps5_v2"


@dataclass
class TrialRow:
    trial_name: str
    prompt: str
    weight: float
    mean_iou_avg: float
    mean_pixel_accuracy_avg: float
    elapsed_seconds: float
    runtime_is_estimated: bool
    used_cached_tiles: int
    failed_tiles: int
    cache_key: str
    trial_summary_path: Path
    tile_ious: dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visualization report from contrastive calibration results.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Calibration output root to analyze.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated report. Defaults to <input-root>/analysis_<timestamp>.",
    )
    parser.add_argument("--top-n", type=int, default=12, help="How many rows to include in the top tables.")
    return parser.parse_args()


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_duration(seconds: float) -> str:
    if seconds is None or math.isnan(seconds):
        return "n/a"
    total_seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _estimate_trial_runtime(summary: dict[str, Any], trial_summary_path: Path) -> tuple[float, bool]:
    elapsed = summary.get("elapsed_seconds")
    if elapsed is not None:
        return float(elapsed), False

    per_tile_reports = summary.get("per_tile_reports", [])
    report_paths: list[Path] = []
    for item in per_tile_reports:
        report_path = item.get("report_path")
        if report_path:
            candidate = Path(report_path)
            if candidate.exists():
                report_paths.append(candidate)

    if report_paths:
        earliest_tile_mtime = min(path.stat().st_mtime for path in report_paths)
        estimated = max(0.0, trial_summary_path.stat().st_mtime - earliest_tile_mtime)
        return estimated, True

    return 0.0, True


def _find_trial_summaries(input_root: Path) -> list[Path]:
    if not input_root.exists():
        return []
    return sorted(input_root.glob("trial_*/trial_summary.json"))


def _load_trials(input_root: Path) -> tuple[list[TrialRow], list[str]]:
    rows: list[TrialRow] = []
    tile_names: list[str] = []
    summary_paths = _find_trial_summaries(input_root)

    for summary_path in summary_paths:
        summary = _json_load(summary_path)
        per_tile_reports = summary.get("per_tile_reports", [])
        tile_ious: dict[str, float] = {}
        for item in per_tile_reports:
            tile_name = str(item.get("tile", ""))
            if tile_name and tile_name not in tile_names:
                tile_names.append(tile_name)
            if tile_name:
                tile_ious[tile_name] = _safe_float(item.get("mean_iou"))

        elapsed_seconds, runtime_is_estimated = _estimate_trial_runtime(summary, summary_path)
        rows.append(
            TrialRow(
                trial_name=str(summary.get("trial_name", summary_path.parent.name)),
                prompt=str(summary.get("prompt", "")),
                weight=_safe_float(summary.get("weight")),
                mean_iou_avg=_safe_float(summary.get("mean_iou_avg")),
                mean_pixel_accuracy_avg=_safe_float(summary.get("mean_pixel_accuracy_avg")),
                elapsed_seconds=elapsed_seconds,
                runtime_is_estimated=runtime_is_estimated,
                used_cached_tiles=int(summary.get("used_cached_tiles", 0)),
                failed_tiles=int(summary.get("failed_tiles", 0)),
                cache_key=str(summary.get("cache_key", "")),
                trial_summary_path=summary_path,
                tile_ious=tile_ious,
            )
        )

    tile_names = sorted(tile_names)
    return rows, tile_names


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_csv(rows: list[TrialRow], tile_names: list[str], output_dir: Path) -> Path:
    csv_path = output_dir / "calibration_summary_table.csv"
    fieldnames = [
        "trial_name",
        "prompt",
        "weight",
        "elapsed_seconds",
        "runtime_mode",
        "mean_iou_avg",
        "mean_pixel_accuracy_avg",
        "used_cached_tiles",
        "failed_tiles",
        "cache_key",
    ] + [f"{tile}_iou" for tile in tile_names]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "trial_name": row.trial_name,
                "prompt": row.prompt,
                "weight": row.weight,
                "elapsed_seconds": row.elapsed_seconds,
                "runtime_mode": "estimated" if row.runtime_is_estimated else "exact",
                "mean_iou_avg": row.mean_iou_avg,
                "mean_pixel_accuracy_avg": row.mean_pixel_accuracy_avg,
                "used_cached_tiles": row.used_cached_tiles,
                "failed_tiles": row.failed_tiles,
                "cache_key": row.cache_key,
            }
            for tile_name in tile_names:
                payload[f"{tile_name}_iou"] = row.tile_ious.get(tile_name, "")
            writer.writerow(payload)

    return csv_path


def _rows_to_table(rows: list[TrialRow], tile_names: list[str], top_n: int) -> list[TrialRow]:
    return rows[: max(0, top_n)]


def _table_html(rows: list[TrialRow], tile_names: list[str], title: str) -> str:
    headers = [
        "Trial",
        "Prompt",
        "Weight",
        "Elapsed",
        "Runtime",
        "Avg mIoU",
        "Avg pixel acc",
        "Cached tiles",
        "Failed tiles",
    ] + [f"{tile} IoU" for tile in tile_names]

    body_rows = []
    for row in rows:
        cells = [
            html.escape(row.trial_name),
            html.escape(row.prompt),
            f"{row.weight:.3f}",
            _format_duration(row.elapsed_seconds),
            "estimated" if row.runtime_is_estimated else "exact",
            f"{row.mean_iou_avg:.4f}",
            f"{row.mean_pixel_accuracy_avg:.4f}",
            str(row.used_cached_tiles),
            str(row.failed_tiles),
        ]
        for tile_name in tile_names:
            value = row.tile_ious.get(tile_name)
            cells.append("" if value is None or math.isnan(value) else f"{value:.4f}")
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    return f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <table>
        <thead>
          <tr>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}</tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </section>
    """


def _plot_prompt_effects(rows: list[TrialRow], output_dir: Path) -> list[Path]:
    grouped: dict[str, list[TrialRow]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt].append(row)

    plot_paths: list[Path] = []
    for prompt, prompt_rows in sorted(grouped.items()):
        prompt_rows = sorted(prompt_rows, key=lambda row: row.weight)
        if not prompt_rows:
            continue

        fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
        x = [row.weight for row in prompt_rows]
        y = [row.mean_iou_avg for row in prompt_rows]
        colors = [row.elapsed_seconds for row in prompt_rows]
        scatter = ax.scatter(x, y, c=colors, cmap="viridis", s=45, alpha=0.85, edgecolors="black", linewidths=0.3)
        ax.plot(x, y, alpha=0.55)
        ax.set_title(f"Average mIoU vs {prompt} weight")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Average mIoU")
        ax.grid(True, alpha=0.25)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Elapsed time (sec)")
        for row in prompt_rows:
            ax.annotate(row.trial_name, (row.weight, row.mean_iou_avg), fontsize=7, xytext=(4, 4), textcoords="offset points")

        plot_path = output_dir / f"{prompt}_weight_vs_miou.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def _plot_runtime_vs_iou(rows: list[TrialRow], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    x = [row.elapsed_seconds for row in rows]
    y = [row.mean_iou_avg for row in rows]
    colors = [row.weight for row in rows]
    scatter = ax.scatter(x, y, c=colors, cmap="plasma", s=45, alpha=0.85, edgecolors="black", linewidths=0.3)
    ax.set_title("Runtime vs average mIoU")
    ax.set_xlabel("Elapsed time (sec)")
    ax.set_ylabel("Average mIoU")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Weight")
    plot_path = output_dir / "runtime_vs_miou.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _prompt_sensitivity_table(rows: list[TrialRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TrialRow]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt].append(row)

    summary_rows: list[dict[str, Any]] = []
    for prompt, prompt_rows in sorted(grouped.items()):
        best = max(prompt_rows, key=lambda row: row.mean_iou_avg)
        worst = min(prompt_rows, key=lambda row: row.mean_iou_avg)
        summary_rows.append(
            {
                "prompt": prompt,
                "config_count": len(prompt_rows),
                "weight_min": min(row.weight for row in prompt_rows),
                "weight_max": max(row.weight for row in prompt_rows),
                "best_trial": best.trial_name,
                "best_weight": best.weight,
                "best_mean_iou_avg": best.mean_iou_avg,
                "worst_trial": worst.trial_name,
                "worst_weight": worst.weight,
                "worst_mean_iou_avg": worst.mean_iou_avg,
                "delta_mean_iou_avg": best.mean_iou_avg - worst.mean_iou_avg,
            }
        )
    return summary_rows


def _write_prompt_sensitivity_csv(summary_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    csv_path = output_dir / "prompt_sensitivity.csv"
    fieldnames = [
        "prompt",
        "config_count",
        "weight_min",
        "weight_max",
        "best_trial",
        "best_weight",
        "best_mean_iou_avg",
        "worst_trial",
        "worst_weight",
        "worst_mean_iou_avg",
        "delta_mean_iou_avg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return csv_path


def _build_html_report(
    rows: list[TrialRow],
    tile_names: list[str],
    fastest_rows: list[TrialRow],
    best_rows: list[TrialRow],
    prompt_summary_rows: list[dict[str, Any]],
    plot_paths: list[Path],
    runtime_plot_path: Path,
    output_dir: Path,
    input_root: Path,
) -> Path:
    best_overall = max(rows, key=lambda row: row.mean_iou_avg, default=None)
    fastest_overall = min(rows, key=lambda row: row.elapsed_seconds, default=None)
    exact_count = sum(1 for row in rows if not row.runtime_is_estimated)
    estimated_count = len(rows) - exact_count

    style = """
    <style>
      body { font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }
      h1, h2, h3 { margin-bottom: 0.4rem; }
      .meta { color: #555; margin-bottom: 1rem; }
      table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem 0; font-size: 13px; }
      th, td { border: 1px solid #d0d0d0; padding: 6px 8px; text-align: left; vertical-align: top; }
      th { background: #f4f6f8; position: sticky; top: 0; }
      tr:nth-child(even) { background: #fafafa; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
      .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; background: #fff; }
      img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
      .note { color: #666; font-size: 12px; }
      code { background: #f2f2f2; padding: 2px 4px; border-radius: 4px; }
    </style>
    """

    prompt_sensitivity_rows = []
    for item in prompt_summary_rows:
        prompt_sensitivity_rows.append(
            "<tr>" + "".join(
                [
                    f"<td>{html.escape(str(item['prompt']))}</td>",
                    f"<td>{item['config_count']}</td>",
                    f"<td>{item['weight_min']:.3f}</td>",
                    f"<td>{item['weight_max']:.3f}</td>",
                    f"<td>{html.escape(str(item['best_trial']))}</td>",
                    f"<td>{item['best_weight']:.3f}</td>",
                    f"<td>{item['best_mean_iou_avg']:.4f}</td>",
                    f"<td>{html.escape(str(item['worst_trial']))}</td>",
                    f"<td>{item['worst_weight']:.3f}</td>",
                    f"<td>{item['worst_mean_iou_avg']:.4f}</td>",
                    f"<td>{item['delta_mean_iou_avg']:.4f}</td>",
                ]
            ) + "</tr>"
        )

    plot_cards = []
    for plot_path in plot_paths:
        plot_cards.append(
            f"<div class='card'><h3>{html.escape(plot_path.stem)}</h3><img src='{html.escape(plot_path.name)}' alt='{html.escape(plot_path.stem)}'></div>"
        )
    plot_cards.append(
        f"<div class='card'><h3>{html.escape(runtime_plot_path.stem)}</h3><img src='{html.escape(runtime_plot_path.name)}' alt='{html.escape(runtime_plot_path.stem)}'></div>"
    )

    best_table = _table_html(best_rows, tile_names, f"Best runs by average mIoU (top {len(best_rows)})")
    fastest_table = _table_html(fastest_rows, tile_names, f"Fastest runs by elapsed time (top {len(fastest_rows)})")

    report_path = output_dir / "contrastive_calibration_report.html"
    best_overall_html = "n/a"
    if best_overall is not None:
        best_overall_html = f"{html.escape(best_overall.trial_name)} ({best_overall.prompt}, weight={best_overall.weight:.3f}, avg mIoU={best_overall.mean_iou_avg:.4f})"
    fastest_overall_html = "n/a"
    if fastest_overall is not None:
        fastest_overall_html = f"{html.escape(fastest_overall.trial_name)} ({fastest_overall.prompt}, weight={fastest_overall.weight:.3f}, elapsed={_format_duration(fastest_overall.elapsed_seconds)})"

    html_text = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Contrastive Calibration Report</title>
      {style}
    </head>
    <body>
      <h1>Contrastive Calibration Report</h1>
      <div class="meta">
        Source root: <code>{html.escape(str(input_root))}</code><br>
        Generated at: {html.escape(datetime.now().isoformat(timespec="seconds"))}<br>
        Completed trials: {len(rows)} | Exact runtimes: {exact_count} | Estimated runtimes: {estimated_count}<br>
        Best overall: {best_overall_html}<br>
        Fastest overall: {fastest_overall_html}
      </div>

      <section>
        <h2>Prompt sensitivity</h2>
        <table>
          <thead>
            <tr>
              <th>Prompt</th>
              <th>Configs</th>
              <th>Weight min</th>
              <th>Weight max</th>
              <th>Best trial</th>
              <th>Best weight</th>
              <th>Best avg mIoU</th>
              <th>Worst trial</th>
              <th>Worst weight</th>
              <th>Worst avg mIoU</th>
              <th>Delta avg mIoU</th>
            </tr>
          </thead>
          <tbody>
            {''.join(prompt_sensitivity_rows)}
          </tbody>
        </table>
      </section>

      {best_table}
      {fastest_table}

      <section>
        <h2>Plots</h2>
        <div class="grid">
          {''.join(plot_cards)}
        </div>
      </section>

      <section>
        <h2>Notes</h2>
        <div class="note">
          Runtime is exact for runs that recorded <code>elapsed_seconds</code>; otherwise it is estimated from result file timestamps.
          Per-tile IoU columns in the tables are included so you can spot tile-specific regressions quickly.
        </div>
      </section>
    </body>
    </html>
    """

    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def main() -> int:
    args = _parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = _ensure_output_dir(output_dir)

    rows, tile_names = _load_trials(input_root)
    if not rows:
        raise FileNotFoundError(f"No trial_summary.json files found under {input_root}")

    rows_by_runtime = sorted(rows, key=lambda row: row.elapsed_seconds)
    rows_by_miou = sorted(rows, key=lambda row: row.mean_iou_avg, reverse=True)
    fastest_rows = rows_by_runtime[: max(0, args.top_n)]
    best_rows = rows_by_miou[: max(0, args.top_n)]

    summary_rows = _prompt_sensitivity_table(rows)
    csv_path = _write_csv(rows, tile_names, output_dir)
    prompt_csv_path = _write_prompt_sensitivity_csv(summary_rows, output_dir)
    prompt_plot_paths = _plot_prompt_effects(rows, output_dir)
    runtime_plot_path = _plot_runtime_vs_iou(rows, output_dir)
    report_path = _build_html_report(
        rows=rows,
        tile_names=tile_names,
        fastest_rows=fastest_rows,
        best_rows=best_rows,
        prompt_summary_rows=summary_rows,
        plot_paths=prompt_plot_paths,
        runtime_plot_path=runtime_plot_path,
        output_dir=output_dir,
        input_root=input_root,
    )

    print(f"[INFO] Loaded {len(rows)} calibration trials from {input_root}")
    print(f"[INFO] Report written to: {report_path}")
    print(f"[INFO] Summary CSV: {csv_path}")
    print(f"[INFO] Prompt sensitivity CSV: {prompt_csv_path}")
    print(f"[INFO] Runtime plot: {runtime_plot_path}")
    for plot_path in prompt_plot_paths:
        print(f"[INFO] Prompt plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
