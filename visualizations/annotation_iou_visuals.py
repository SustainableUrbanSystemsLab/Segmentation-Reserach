"""Visualization helpers for annotation-vs-prediction IoU review."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models import config as cfg


CLASS_COLORS = {
    1: np.array([0.00, 0.72, 0.38], dtype=np.float32),
    2: np.array([0.20, 0.55, 0.95], dtype=np.float32),
    3: np.array([1.00, 0.68, 0.10], dtype=np.float32),
    4: np.array([0.66, 0.36, 0.92], dtype=np.float32),
    5: np.array([0.95, 0.18, 0.18], dtype=np.float32),
}

CLASS_LABELS = {
    1: "nen_cat_a",
    2: "nen_cat_b",
    3: "nen_cat_c",
    4: "nen_cat_d",
    5: "nen_cat_e",
}


def _colorize_class_map(class_map: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    height, width = class_map.shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    for class_id, color in CLASS_COLORS.items():
        rgb[class_map == class_id] = color

    if valid_mask is not None:
        rgb[~valid_mask] = np.array([0.25, 0.25, 0.25], dtype=np.float32)

    return rgb


def _build_difference_overlay(
    gt_class_map: np.ndarray,
    pred_class_map: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    height, width = gt_class_map.shape[:2]
    overlay = np.zeros((height, width, 4), dtype=np.float32)

    correct = valid_mask & (gt_class_map == pred_class_map) & (gt_class_map > 0)
    false_positive = valid_mask & (pred_class_map > 0) & (gt_class_map == 0)
    false_negative = valid_mask & (gt_class_map > 0) & (pred_class_map == 0)
    class_mismatch = valid_mask & (gt_class_map > 0) & (pred_class_map > 0) & (gt_class_map != pred_class_map)

    overlay[correct, :3] = np.array([0.05, 0.75, 0.20], dtype=np.float32)
    overlay[correct, 3] = 0.45

    overlay[false_positive, :3] = np.array([0.95, 0.18, 0.18], dtype=np.float32)
    overlay[false_positive, 3] = 0.55

    overlay[false_negative, :3] = np.array([0.20, 0.45, 1.00], dtype=np.float32)
    overlay[false_negative, 3] = 0.55

    overlay[class_mismatch, :3] = np.array([1.00, 0.90, 0.20], dtype=np.float32)
    overlay[class_mismatch, 3] = 0.65

    return overlay


def save_annotation_iou_comparison_figure(
    image_rgb: np.ndarray,
    gt_class_map: np.ndarray,
    pred_class_map: np.ndarray,
    valid_mask: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_dim = max(1, int(getattr(cfg, "annotation_iou_visualization_max_dim", 900)))
    stride = max(1, int(np.ceil(max(image_rgb.shape[:2]) / max_dim)))
    if stride > 1:
        image_rgb = image_rgb[::stride, ::stride]
        gt_class_map = gt_class_map[::stride, ::stride]
        pred_class_map = pred_class_map[::stride, ::stride]
        valid_mask = valid_mask[::stride, ::stride]

    gt_overlay = _colorize_class_map(gt_class_map, valid_mask)
    pred_overlay = _colorize_class_map(pred_class_map, valid_mask)
    diff_overlay = _build_difference_overlay(gt_class_map, pred_class_map, valid_mask)

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=150)
    ax_image, ax_gt, ax_pred, ax_diff = axes.flatten()

    ax_image.imshow(image_rgb)
    ax_image.set_title("Original tile", fontsize=14, weight="bold")
    ax_image.axis("off")

    ax_gt.imshow(image_rgb)
    ax_gt.imshow(gt_overlay, alpha=0.82)
    ax_gt.set_title("Cleaned annotation", fontsize=14, weight="bold")
    ax_gt.axis("off")

    ax_pred.imshow(image_rgb)
    ax_pred.imshow(pred_overlay, alpha=0.82)
    ax_pred.set_title("Model prediction", fontsize=14, weight="bold")
    ax_pred.axis("off")

    ax_diff.imshow(image_rgb)
    ax_diff.imshow(diff_overlay)
    ax_diff.set_title("Difference view", fontsize=14, weight="bold")
    ax_diff.axis("off")

    legend_items = [
        ("green", "match"),
        ("red", "false positive"),
        ("blue", "false negative"),
        ("yellow", "class mismatch"),
    ]
    legend_text = ", ".join(f"{name}={label}" for name, label in legend_items)
    fig.suptitle(f"{title}\n{legend_text}", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
