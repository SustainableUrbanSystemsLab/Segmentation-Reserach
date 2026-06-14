"""
Local runner for contrastive calibration — cross-platform equivalent of
pace_run_contrastive_calibration.sh, tuned for a local GPU (e.g. RTX 3080 Ti, 12 GB VRAM).

Usage:
    python run_contrastive_calibration_local.py
    python run_contrastive_calibration_local.py --mode quick
    python run_contrastive_calibration_local.py --mode quick --tile tile_002_003
    python run_contrastive_calibration_local.py --steps 3   # smaller sweep for quick test

Key differences from the PACE batch script:
- Single parallel worker (12 GB VRAM is tight; running 2+ SAM workers simultaneously
  risks OOM — use --workers 1 or carefully try --workers 2).
- Pipeline cache stored locally under Maps/Tiles/Atlanta_split_google/.segmentation_cache
  (or override with --cache-root).
- Output stored under results/contrastive_calibration_local/.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Defaults tuned for RTX 3080 Ti (12 GB VRAM)
# ---------------------------------------------------------------------------
# One worker at a time to stay comfortably within 12 GB.
# SAM-ViT-B peaks at ~6–8 GB with region-context scoring; running two in
# parallel would likely OOM. Flip to 2 if you find memory headroom.
DEFAULT_WORKERS = 1

# Local tile directory (relative to project root).
LOCAL_TILE_BASE = PROJECT_ROOT / "Maps" / "Tiles" / "Atlanta_split_google"

# Local output directory (separate from the PACE scratch path).
LOCAL_OUTPUT_BASE = PROJECT_ROOT / "results" / "contrastive_calibration_local"

# Local pipeline cache root — lives next to the tiles so it's easy to find.
LOCAL_CACHE_ROOT = LOCAL_TILE_BASE / ".segmentation_cache"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local contrastive calibration runner (cross-platform).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick"],
        default="full",
        help="'full' runs the complete sweep; 'quick' runs 3 configs on 1 tile.",
    )
    parser.add_argument(
        "--tile",
        default="tile_002_003",
        help="Tile stem or full .tif path for quick mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Steps each side of base weight (full mode). 5 → 11 values per prompt → 51 configs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel calibration workers. Keep at 1 for 12 GB VRAM; try 2 cautiously.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=LOCAL_CACHE_ROOT,
        help="Directory for the shared SAM/DINO pipeline cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_OUTPUT_BASE,
        help="Root directory for calibration outputs.",
    )
    parser.add_argument(
        "--cache-key",
        default="local_3tiles_steps5",
        help="Cache key / subfolder name for this calibration run.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing IoU reports and re-run all trials.",
    )
    parser.add_argument(
        "--overwrite-pipeline-cache",
        action="store_true",
        help="Force DINO+SAM to recompute even if a shared cache exists.",
    )
    return parser.parse_args()


def _resolve_tile(tile_input: str) -> Path:
    """Resolve a tile stem or full path to an absolute .tif path."""
    p = Path(tile_input)
    if p.suffix == ".tif" and p.is_absolute():
        return p
    if p.suffix == ".tif" and p.exists():
        return p.resolve()
    # Treat as stem
    stem = p.stem if p.suffix == ".tif" else tile_input
    candidate = LOCAL_TILE_BASE / f"{stem}.tif"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find tile '{tile_input}'.\n"
        f"Expected: {candidate}\n"
        f"Pass the full path or place the tile in {LOCAL_TILE_BASE}"
    )


def main() -> int:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Validate tile directory exists
    # ------------------------------------------------------------------
    if not LOCAL_TILE_BASE.exists():
        print(
            f"[ERROR] Local tile directory not found: {LOCAL_TILE_BASE}\n"
            f"[HINT]  Place your .tif tiles there or pass --tile with a full path."
        )
        return 1

    # ------------------------------------------------------------------
    # Build the environment that contrastive_calibration.py reads via os.environ
    # ------------------------------------------------------------------
    env_overrides: dict[str, str] = {
        # GPU
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",  # Smaller splits for 12GB VRAM
        "PYTHONUNBUFFERED": "1",
        # CUDA required (change to 0 if you want CPU fallback)
        "REQUIRE_CUDA": "1",
        "PREFER_CUDA": "1",
        # Tile discovery
        "CONTRASTIVE_TILE_BASE_DIR": str(LOCAL_TILE_BASE),
        # Cache
        "PIPELINE_CACHE_ROOT": str(args.cache_root),
        "OVERWRITE_PIPELINE_CACHE": "1" if args.overwrite_pipeline_cache else "0",
        "SKIP_MASK_CACHING": "0",          # Store full SAM arrays for cross-trial reuse
        # SAM VRAM cap — 3080 Ti has 12GB; 2500 masks is safe, 5000 risks OOM on large tiles
        "SAM_AUTO_MAX_TOTAL_MASKS": "2500",
        # Calibration
        "CALIBRATION_CACHE_KEY": args.cache_key,
        "CALIBRATION_OUTPUT_DIR": str(args.output_dir / args.cache_key),
        "CALIBRATION_PARALLEL_JOBS": str(args.workers),
        "CALIBRATION_RESUME_FROM_CACHE": "1",
        "CALIBRATION_FORCE_RERUN": "1" if args.force_rerun else "0",
        "STEPS_EACH_SIDE": str(args.steps),
        # Visualization / output size controls (keep small locally)
        "SKIP_IF_VISUALIZATIONS_EXIST": "1",
        "SAVE_INPUT_IMAGES": "0",
    }

    if args.mode == "quick":
        # Resolve tile and set quick-mode env vars
        try:
            tile_path = _resolve_tile(args.tile)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return 1

        # Find annotation sidecar
        annotation_path = tile_path.with_suffix(".json")
        if annotation_path.exists():
            env_overrides["CONTRASTIVE_ANNOTATION_PATH"] = str(annotation_path)

        env_overrides["CONTRASTIVE_TILES"] = str(tile_path)
        env_overrides["CONTRASTIVE_PROMPTS"] = "nen_cat_a,nen_cat_c,nen_cat_e"
        env_overrides["STEPS_EACH_SIDE"] = "0"
        env_overrides["CALIBRATION_CACHE_KEY"] = f"quick_1tile_3configs_{tile_path.stem}"
        env_overrides["CALIBRATION_OUTPUT_DIR"] = str(
            args.output_dir / env_overrides["CALIBRATION_CACHE_KEY"]
        )

        print(f"[INFO] Quick mode: 3 configs on {tile_path.stem}")

    # ------------------------------------------------------------------
    # Apply env overrides (only for keys not already set by the user)
    # ------------------------------------------------------------------
    for key, value in env_overrides.items():
        os.environ[key] = value

    # Ensure output dirs exist
    Path(env_overrides["CALIBRATION_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"[INFO] Project root:      {PROJECT_ROOT}")
    print(f"[INFO] Tile base:         {LOCAL_TILE_BASE}")
    print(f"[INFO] Pipeline cache:    {args.cache_root}")
    print(f"[INFO] Output dir:        {env_overrides['CALIBRATION_OUTPUT_DIR']}")
    print(f"[INFO] Mode:              {args.mode}")
    print(f"[INFO] Workers:           {args.workers}")
    print(f"[INFO] Steps each side:   {env_overrides['STEPS_EACH_SIDE']}")
    print(f"[INFO] Cache key:         {env_overrides['CALIBRATION_CACHE_KEY']}")
    print(f"[INFO] Skip mask caching: {env_overrides['SKIP_MASK_CACHING']} (0=store full arrays)")
    print(f"[INFO] Overwrite cache:   {env_overrides['OVERWRITE_PIPELINE_CACHE']}")
    print()

    # ------------------------------------------------------------------
    # Run contrastive_calibration.py in-process (same Python, same imports)
    # ------------------------------------------------------------------
    calibration_script = PROJECT_ROOT / "runs" / "contrastive_calibration.py"
    if not calibration_script.exists():
        print(f"[ERROR] Could not find {calibration_script}")
        return 1

    print(f"[INFO] Running {calibration_script.name}...")
    try:
        runpy.run_path(str(calibration_script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if exc.code is not None else 0
        return int(code)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
