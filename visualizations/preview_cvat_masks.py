"""Local preview for cleaned CVAT annotation masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import build_cleaned_annotation_mask


def find_default_annotation_file(image_name: str) -> Path:
    requested_stem = Path(image_name).stem
    candidates = sorted(PROJECT_ROOT.glob(f"Maps/Tiles/**/{requested_stem}.json"))
    if not candidates:
        candidates = sorted(PROJECT_ROOT.glob(f"Maps/Tiles/**/{requested_stem}.xml"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No annotation file found for '{image_name}' under Maps/Tiles")
    raise RuntimeError(
        f"Multiple annotation files found for '{image_name}'. Pass --annotation-path explicitly or provide a direct file path.\n"
        + "\n".join(str(path) for path in candidates)
    )


def resolve_image_path(annotation_path: Path, image_name: str) -> Path:
    sibling_candidate = annotation_path.with_suffix(".tif")
    if sibling_candidate.exists():
        return sibling_candidate

    direct_candidate = annotation_path.parent / Path(image_name).name
    if direct_candidate.exists():
        return direct_candidate

    candidates = sorted(PROJECT_ROOT.glob(f"Maps/Tiles/**/{Path(image_name).name}"))
    if not candidates:
        raise FileNotFoundError(f"Could not find image '{image_name}' under Maps/Tiles")
    return candidates[0]


def load_rgb_image(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype("float32")
        rgb_min = float(rgb.min())
        rgb_max = float(rgb.max())
        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) * (255.0 / (rgb_max - rgb_min))
        rgb = rgb.clip(0, 255).astype("uint8")
    return rgb


def save_review_figure(image_rgb: np.ndarray, color_mask: np.ndarray, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original tile", fontsize=14, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Cleaned annotation mask", fontsize=14, weight="bold")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview cleaned CVAT masks for one tile")
    parser.add_argument("image_name", help="Tile image name, for example tile_002_000.tif")
    parser.add_argument(
        "--annotation-path",
        "--xml-path",
        type=Path,
        default=None,
        help="Path to a tile annotation JSON or annotations.xml. Defaults to the matching file under Maps/Tiles.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional review figure output path.",
    )
    args = parser.parse_args()

    annotation_path = args.annotation_path if args.annotation_path is not None else find_default_annotation_file(args.image_name)
    image_path = resolve_image_path(annotation_path, args.image_name)

    result = build_cleaned_annotation_mask(annotation_path, args.image_name)
    image_rgb = load_rgb_image(image_path)
    color_mask = result["color_mask"]

    output_path = args.output_path
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "annotation_reviews" / f"{Path(result['image_name']).stem}_cleaned_mask.png"

    save_review_figure(
        image_rgb,
        color_mask,
        output_path,
        title=f"{result['image_name']}  |  {result['shape_count']} source annotations",
    )
    print(f"[INFO] Saved review figure to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())