"""
Pipeline Evaluation.

Compares the amenity pipeline's unified mask against the CVAT ground truth
annotations (NEN wind comfort labels).

Outputs:
  - Results/Evaluations/<tile>_evaluation.png  : 2×2 comparison grid
  - Printed statistics: per-class pixel counts, accuracy, overlap summary

The CVAT annotations use NEN_A / NEN_B / NEN_C / NEN_D / Uncomfortable labels.
The pipeline prediction uses FinalClass IDs.

Mapping from CVAT NEN class → FinalClass group for comparison:
    NEN_A         → "comfortable"   (canopy, lowveg, sidewalk, crosswalk)
    NEN_B         → "neutral"       (water, building, pools, impervious at low exposure)
    NEN_C         → "mild"          (parking, cars, some impervious)
    NEN_D + Uncom → "uncomfortable" (road)
    Invalid       → excluded from metrics

This lets us assess whether the pipeline is correctly classifying urban comfort
at the semantic level, not just matching individual pixel class IDs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from rasterio.enums import Resampling

# ── Path setup: ensure both amenity_pipeline/ and repo root are importable ──
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in [str(SCRIPT_DIR), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocessing import build_cleaned_annotation_mask  # noqa: E402
from pipeline_config import FinalClass                   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# CVAT label → comfort tier mapping
# ─────────────────────────────────────────────────────────────────────────────
# label_mask values from build_cleaned_annotation_mask use 1-indexed IDs
# matching _PRIORITY_ORDER = ["Uncomfortable","NEN_D","NEN_C","NEN_B","NEN_A","Invalid"]
_CVAT_ID_TO_TIER = {
    1: 3,  # Uncomfortable → tier 3 (worst)
    2: 3,  # NEN_D         → tier 3
    3: 2,  # NEN_C         → tier 2
    4: 1,  # NEN_B         → tier 1
    5: 0,  # NEN_A         → tier 0 (best)
    6: -1, # Invalid       → excluded
}

# FinalClass → comfort tier (same 0-3 scale)
# Updated to match strictly the 11 active classes in the pipeline
_PRED_CLASS_TO_TIER = {
    FinalClass.WATER:      1,
    FinalClass.CANOPY:     0,   # comfortable
    FinalClass.LOWVEG:     0,
    FinalClass.IMPERVIOUS: 2,   # mild discomfort
    FinalClass.ROAD:       3,   # uncomfortable
    FinalClass.PARKING:    2,
    FinalClass.BUILDING:   1,   # neutral
    FinalClass.SIDEWALK:   0,   # comfortable
    FinalClass.CROSSWALK:  0,
    FinalClass.CAR:        2,
    FinalClass.POOL:       1,   # neutral
}

_TIER_LABELS = {0: "Comfortable (A)", 1: "Neutral (B)", 2: "Mild (C/D)", 3: "Uncomfortable (D/E)"}
_TIER_COLORS = {0: "#2ca02c", 1: "#98df8a", 2: "#ff7f0e", 3: "#d62728"}


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_annotation(image_name: str) -> Path:
    """Find the JSON annotation sidecar for a tile anywhere under Maps/Tiles."""
    stem = Path(image_name).stem
    for pattern in (f"**/{stem}.json", f"**/{stem}.xml"):
        hits = sorted((REPO_ROOT / "Maps" / "Tiles").glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No annotation file found for '{image_name}' under {REPO_ROOT / 'Maps' / 'Tiles'}"
    )


def find_image(annotation_path: Path, image_name: str) -> Path:
    """Find the .tif tile, preferring a sibling of the annotation."""
    sibling = annotation_path.with_suffix(".tif")
    if sibling.exists():
        return sibling
    hits = sorted((REPO_ROOT / "Maps" / "Tiles").glob(f"**/{Path(image_name).name}"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Could not find image '{image_name}' under Maps/Tiles")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_rgb(image_path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    with rasterio.open(image_path) as src:
        if target_shape:
            h, w = target_shape
            rgb = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)
        else:
            rgb = src.read([1, 2, 3])
    rgb = rgb.transpose(1, 2, 0).astype(np.float32)
    lo, hi = rgb.min(), rgb.max()
    if hi > lo:
        rgb = (rgb - lo) * (255.0 / (hi - lo))
    return rgb.clip(0, 255).astype(np.uint8)


def load_pred(pred_path: Path) -> np.ndarray:
    with rasterio.open(pred_path) as src:
        return src.read(1)


def colorize_pred(pred: np.ndarray) -> np.ndarray:
    """RGB visualization of the FinalClass prediction map."""
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # Updated to match the current 11 classes
    CLASS_RGB = {
        FinalClass.WATER:      [0,   196, 255],
        FinalClass.CANOPY:     [38,  115,   0],
        FinalClass.LOWVEG:     [163, 255, 115],
        FinalClass.IMPERVIOUS: [155, 155, 155],
        FinalClass.ROAD:       [76,   76,  76],
        FinalClass.PARKING:    [25,  188, 155],
        FinalClass.BUILDING:   [232,  76,  61],
        FinalClass.SIDEWALK:   [242, 155,  17],
        FinalClass.CROSSWALK:  [229, 125,  33],
        FinalClass.CAR:        [51,  150, 219],
        FinalClass.POOL:       [224,  64, 250],
    }
    for cls_id, rgb in CLASS_RGB.items():
        out[pred == cls_id] = rgb
    return out


def colorize_gt_tier(label_mask: np.ndarray) -> np.ndarray:
    """Colour the GT annotation by comfort tier for visual comparison."""
    h, w = label_mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    TIER_RGB = {
        0: [26,  122,  26],   # comfortable  green
        1: [125, 200, 125],   # neutral      light green
        2: [245, 130,  13],   # mild         orange
        3: [214,  40,  40],   # uncomfortable red
    }
    for cvat_id, tier in _CVAT_ID_TO_TIER.items():
        if tier >= 0:
            out[label_mask == cvat_id] = TIER_RGB[tier]
    return out


def colorize_diff(rgb: np.ndarray, gt_tier: np.ndarray, pred_tier: np.ndarray) -> np.ndarray:
    """Diff overlay: green=match, red=false_pos, blue=false_neg, yellow=mismatch."""
    valid = (gt_tier >= 0)
    match     = valid & (pred_tier == gt_tier)
    false_pos = valid & (pred_tier < gt_tier)   # predicted more comfortable than GT
    false_neg = valid & (pred_tier > gt_tier)   # predicted less comfortable than GT

    base = rgb.astype(np.float32) / 255.0 * 0.55
    out = base.copy()
    alpha = 0.75
    for mask, color in [
        (match,     [0.13, 0.55, 0.13]),
        (false_pos, [0.12, 0.56, 1.00]),
        (false_neg, [0.86, 0.08, 0.24]),
    ]:
        if np.any(mask):
            out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color)
    return (out.clip(0, 1) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(gt_tier: np.ndarray, pred_tier: np.ndarray) -> dict:
    valid = gt_tier >= 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"error": "No valid GT pixels"}

    gt_v = gt_tier[valid]
    pr_v = pred_tier[valid]

    overall_acc = float((gt_v == pr_v).sum()) / n_valid

    # Within-one-tier accuracy (common for comfort scoring)
    within_one = float((np.abs(gt_v.astype(int) - pr_v.astype(int)) <= 1).sum()) / n_valid

    per_tier: dict[int, dict] = {}
    for tier in range(4):
        gt_pos = gt_v == tier
        pr_pos = pr_v == tier
        tp = int((gt_pos & pr_pos).sum())
        fp = int((~gt_pos & pr_pos).sum())
        fn = int((gt_pos & ~pr_pos).sum())
        union = tp + fp + fn
        iou = tp / union if union > 0 else float("nan")
        recall = tp / gt_pos.sum() if gt_pos.sum() > 0 else float("nan")
        prec = tp / pr_pos.sum() if pr_pos.sum() > 0 else float("nan")
        per_tier[tier] = {
            "label": _TIER_LABELS[tier],
            "gt_px": int(gt_pos.sum()),
            "pred_px": int(pr_pos.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "iou": iou, "precision": prec, "recall": recall,
        }

    return {
        "n_valid_px": n_valid,
        "overall_accuracy": overall_acc,
        "within_one_tier_accuracy": within_one,
        "per_tier": per_tier,
    }


def print_stats(stats: dict, tile_name: str) -> None:
    if "error" in stats:
        print(f"[EVAL] Error: {stats['error']}")
        return
    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS — {tile_name}")
    print("=" * 65)
    print(f"  Valid GT pixels        : {stats['n_valid_px']:,}")
    print(f"  Overall accuracy       : {stats['overall_accuracy']:.1%}")
    print(f"  Within-one-tier acc.   : {stats['within_one_tier_accuracy']:.1%}")
    print("-" * 65)
    print(f"  {'Tier':<28} {'GT%':>6} {'Pred%':>6} {'IoU':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 65)
    n = stats["n_valid_px"]
    for tier, t in stats["per_tier"].items():
        gt_pct = t["gt_px"] / n * 100
        pr_pct = t["pred_px"] / n * 100
        iou_s  = f"{t['iou']:.3f}"     if not np.isnan(t["iou"])       else "  n/a"
        pre_s  = f"{t['precision']:.3f}" if not np.isnan(t["precision"]) else "  n/a"
        rec_s  = f"{t['recall']:.3f}"  if not np.isnan(t["recall"])    else "  n/a"
        print(f"  {t['label']:<28} {gt_pct:>5.1f}% {pr_pct:>5.1f}% {iou_s:>7} {pre_s:>7} {rec_s:>7}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def save_grid(
    rgb: np.ndarray,
    gt_color: np.ndarray,
    pred_color: np.ndarray,
    diff_color: np.ndarray,
    stats: dict,
    output_path: Path,
    tile_name: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), dpi=150)
    axes = axes.flatten()

    axes[0].imshow(rgb)
    axes[0].set_title("Original tile", fontsize=14, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_color)
    gt_legend = [
        mpatches.Patch(facecolor=_TIER_COLORS[t], label=_TIER_LABELS[t])
        for t in range(4)
    ]
    axes[1].legend(handles=gt_legend, loc="upper right", frameon=True,
                   facecolor="white", fontsize=8)
    axes[1].set_title("Ground truth (comfort tiers)", fontsize=14, weight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_color)
    axes[2].set_title("Pipeline prediction", fontsize=14, weight="bold")
    axes[2].axis("off")

    axes[3].imshow(diff_color)
    diff_legend = [
        mpatches.Patch(facecolor=[0.13, 0.55, 0.13], label="Correct tier"),
        mpatches.Patch(facecolor=[0.12, 0.56, 1.00], label="Over-predicted comfort"),
        mpatches.Patch(facecolor=[0.86, 0.08, 0.24], label="Under-predicted comfort"),
    ]
    axes[3].legend(handles=diff_legend, loc="upper right", frameon=True,
                   facecolor="white", fontsize=8)
    axes[3].set_title("Difference (tier comparison)", fontsize=14, weight="bold")
    axes[3].axis("off")

    # Add stats text below diff panel
    if "overall_accuracy" in stats:
        acc_text = (
            f"Overall acc: {stats['overall_accuracy']:.1%}  |  "
            f"Within-1-tier: {stats['within_one_tier_accuracy']:.1%}"
        )
        axes[3].set_xlabel(acc_text, fontsize=11)

    fig.suptitle(f"Pipeline Evaluation — {tile_name}", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVAL] Saved evaluation grid to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by predict.py
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    image_name: str,
    pred_path: Path | None = None,
    annotation_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Run evaluation for one tile.  Returns the stats dict.
    Safe to call even if annotation or prediction is missing — prints a
    warning and returns an empty dict rather than crashing the pipeline.
    """
    if pred_path is None:
        pred_path = REPO_ROOT / "Results" / "unified_clean_mask.tif"
    if output_dir is None:
        output_dir = REPO_ROOT / "Results" / "Evaluations"

    # Locate annotation
    try:
        ann_path = annotation_path or find_annotation(image_name)
    except FileNotFoundError as e:
        print(f"[EVAL] Skipping evaluation — {e}")
        return {}

    # Locate image
    try:
        img_path = find_image(ann_path, image_name)
    except FileNotFoundError as e:
        print(f"[EVAL] Skipping evaluation — {e}")
        return {}

    # Check prediction exists
    if not pred_path.exists():
        print(f"[EVAL] Skipping evaluation — prediction raster not found at {pred_path}")
        return {}

    print(f"[EVAL] Evaluating {image_name} against {ann_path.name}")

    # Load GT
    gt = build_cleaned_annotation_mask(ann_path, image_name)
    label_mask = np.asarray(gt["label_mask"], dtype=np.int32)
    gt_h, gt_w = label_mask.shape

    # Load prediction — resize to match GT dimensions
    pred_raw = load_pred(pred_path)
    if pred_raw.shape != (gt_h, gt_w):
        from PIL import Image as PILImage
        pred_pil = PILImage.fromarray(pred_raw.astype(np.uint8))
        pred_raw = np.asarray(
            pred_pil.resize((gt_w, gt_h), PILImage.NEAREST), dtype=np.uint8
        )

    # Load RGB (resized to GT dims for display)
    rgb = load_rgb(img_path, target_shape=(gt_h, gt_w))

    # Map both to comfort tiers
    gt_tier = np.vectorize(_CVAT_ID_TO_TIER.get)(label_mask, -1).astype(np.int8)
    pred_tier = np.vectorize(_PRED_CLASS_TO_TIER.get)(pred_raw, 1).astype(np.int8)

    # Compute stats
    stats = compute_stats(gt_tier, pred_tier)
    print_stats(stats, image_name)

    # Build visualizations
    gt_color   = colorize_gt_tier(label_mask)
    pred_color = colorize_pred(pred_raw)
    diff_color = colorize_diff(rgb, gt_tier, pred_tier)

    stem = Path(image_name).stem
    out_path = output_dir / f"{stem}_evaluation.png"
    save_grid(rgb, gt_color, pred_color, diff_color, stats, out_path, stem)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate amenity pipeline predictions against CVAT ground truth."
    )
    parser.add_argument("image_name", nargs="?", default=None,
                        help="Tile name, e.g. tile_002_003.tif  (auto-detected from pipeline_image if omitted)")
    parser.add_argument("--annotation-path", type=Path, default=None)
    parser.add_argument("--pred-path", type=Path,
                        default=REPO_ROOT / "Results" / "unified_clean_mask.tif")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "Results" / "Evaluations")
    args = parser.parse_args()

    image_name = args.image_name
    if not image_name:
        # Auto-detect from config.json
        config_path = SCRIPT_DIR / "config.json"
        if config_path.exists():
            import json
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            image_name = raw.get("pipeline_image", "")
        if not image_name:
            print("[EVAL] No image_name provided and none found in config.json")
            return 1

    stats = evaluate(
        image_name=image_name,
        pred_path=args.pred_path,
        annotation_path=args.annotation_path,
        output_dir=args.output_dir,
    )
    return 0 if stats else 1


if __name__ == "__main__":
    raise SystemExit(main())