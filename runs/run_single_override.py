from __future__ import annotations

import argparse
import copy
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("tif", help="Path to tif file to run (relative to project root)")
    p.add_argument("--prompt-family", choices=["grouped", "split"], default="split", help="Prompt taxonomy to use for the run")
    p.add_argument("--no-full-fill", action="store_true", help="Disable full_image_mask_mode (do not force per-pixel assignment)")
    p.add_argument("--sam-only", action="store_true", help="Force SAM-only mode (disable DINO) for this run")
    p.add_argument("--raise-a-threshold", type=float, default=None, help="Temporarily set nen_cat_a.clip_score_threshold to this value")
    return p.parse_args()


def main():
    args = _parse_args()
    from models import config as cfg
    from models import prompts

    # Apply overrides
    if args.prompt_family == "split":
        cfg.ACTIVE_PROMPTS = ["nen_cat_a", "nen_cat_b", "nen_cat_c", "nen_cat_d", "nen_cat_e"]
        print("[INFO] Overrode cfg.ACTIVE_PROMPTS = split A-E taxonomy")
    else:
        cfg.ACTIVE_PROMPTS = ["nen_cat_a", "nen_cat_c", "nen_cat_e"]
        print("[INFO] Overrode cfg.ACTIVE_PROMPTS = grouped A/C/E taxonomy")
    cfg.dino_prompt_configs = [
        {"name": name, **copy.deepcopy(prompts.AVAILABLE_PROMPTS[name])}
        for name in cfg.ACTIVE_PROMPTS
        if name in prompts.AVAILABLE_PROMPTS
    ]

    if args.no_full_fill:
        cfg.full_image_mask_mode = False
        print("[INFO] Overrode cfg.full_image_mask_mode = False")
    if args.sam_only:
        cfg.use_dino = False
        print("[INFO] Overrode cfg.use_dino = False (SAM-only run)")
    if args.raise_a_threshold is not None:
        if "nen_cat_a" in prompts.AVAILABLE_PROMPTS:
            prompts.AVAILABLE_PROMPTS["nen_cat_a"]["clip_score_threshold"] = float(args.raise_a_threshold)
            print(f"[INFO] Overrode nen_cat_a.clip_score_threshold = {args.raise_a_threshold}")
        else:
            print("[WARN] nen_cat_a not found in prompts; skipping threshold override")

    # Set the active tile via env expected by get_satellite
    import os
    os.environ["SEGMENTATION_TIF_FILE"] = args.tif

    # Run the pipeline
    runpy.run_path(str(PROJECT_ROOT / "runs" / "get_satellite.py"), run_name="__main__")


if __name__ == "__main__":
    raise SystemExit(main())
