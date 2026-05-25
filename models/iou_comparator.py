"""IoU comparison helpers for cleaned CVAT annotations versus model masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from preprocessing import build_cleaned_annotation_mask


GROUPED_CLASS_ORDER = ("nen_cat_a", "nen_cat_c", "nen_cat_e")
GROUPED_GT_TO_PRED_CLASS = {
    "NEN_A": "nen_cat_a",
    "NEN_B": "nen_cat_a",
    "NEN_C": "nen_cat_c",
    "NEN_D": "nen_cat_c",
    "Uncomfortable": "nen_cat_e",
}

SPLIT_CLASS_ORDER = ("nen_cat_a", "nen_cat_b", "nen_cat_c", "nen_cat_d", "nen_cat_e")
SPLIT_GT_TO_PRED_CLASS = {
    "NEN_A": "nen_cat_a",
    "NEN_B": "nen_cat_b",
    "NEN_C": "nen_cat_c",
    "NEN_D": "nen_cat_d",
    "Uncomfortable": "nen_cat_e",
}


def _resolve_annotations_xml(
    image_name: str,
    xml_path: str | Path | None = None,
    search_root: str | Path | None = None,
) -> Path:
    if xml_path is not None:
        candidate = Path(xml_path)
        if candidate.is_dir():
            matches = sorted(candidate.glob("**/*.json")) + sorted(candidate.glob("**/*.xml"))
            if not matches:
                raise FileNotFoundError(f"No annotation files found under {candidate}")
            if len(matches) == 1:
                return matches[0]
            for match in matches:
                try:
                    build_cleaned_annotation_mask(match, image_name)
                    return match
                except Exception:
                    continue
            raise FileNotFoundError(
                f"No annotation file under {candidate} contains image '{image_name}'"
            )

        if not candidate.exists():
            raise FileNotFoundError(f"Annotation file not found: {candidate}")
        return candidate

    root = Path(search_root) if search_root is not None else Path("Maps") / "Tiles"
    candidates = sorted(root.glob("**/*.json")) + sorted(root.glob("**/*.xml"))
    if not candidates:
        raise FileNotFoundError(f"No annotation files found under {root}")
    if len(candidates) == 1:
        return candidates[0]

    for candidate in candidates:
        try:
            build_cleaned_annotation_mask(candidate, image_name)
            return candidate
        except Exception:
            continue

    raise FileNotFoundError(f"No annotation file under {root} contains image '{image_name}'")


def _resolve_class_spec(class_mode: str | None) -> tuple[tuple[str, ...], dict[str, str]]:
    mode = (class_mode or "grouped").strip().lower()
    if mode in {"split", "full", "all", "nen5", "five"}:
        return SPLIT_CLASS_ORDER, SPLIT_GT_TO_PRED_CLASS
    return GROUPED_CLASS_ORDER, GROUPED_GT_TO_PRED_CLASS


def _build_ground_truth_maps(
    cleaned_result: dict[str, object],
    class_order: tuple[str, ...],
    gt_to_pred_class: dict[str, str],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    label_mask = np.asarray(cleaned_result["label_mask"], dtype=np.uint8)
    label_lookup = cleaned_result.get("label_lookup", {})

    gt_masks: dict[str, np.ndarray] = {
        class_name: np.zeros_like(label_mask, dtype=bool) for class_name in class_order
    }
    valid_mask = np.zeros_like(label_mask, dtype=bool)
    gt_class_map = np.zeros_like(label_mask, dtype=np.uint8)

    class_to_id = {name: idx + 1 for idx, name in enumerate(class_order)}

    for label_id, raw_label in label_lookup.items():
        label_name = str(raw_label)
        current_mask = label_mask == int(label_id)
        if not np.any(current_mask):
            continue

        if label_name == "Invalid":
            continue

        valid_mask |= current_mask
        class_name = gt_to_pred_class.get(label_name)
        if class_name is None:
            continue

        gt_masks[class_name] |= current_mask
        gt_class_map[current_mask] = class_to_id[class_name]

    return gt_masks, valid_mask, gt_class_map


def _build_predicted_class_map(
    predicted_masks: dict[str, np.ndarray],
    image_shape: tuple[int, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    return _build_predicted_class_map_for_order(predicted_masks, image_shape, GROUPED_CLASS_ORDER)


def _build_predicted_class_map_for_order(
    predicted_masks: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    class_order: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    pred_masks: dict[str, np.ndarray] = {}
    for class_name in class_order:
        mask = np.asarray(predicted_masks.get(class_name, np.zeros(image_shape, dtype=bool)), dtype=bool)
        pred_masks[class_name] = mask

    pred_class_map = np.zeros(image_shape, dtype=np.uint8)
    class_to_id = {name: idx + 1 for idx, name in enumerate(class_order)}
    for class_name in class_order:
        pred_class_map[pred_masks[class_name]] = class_to_id[class_name]

    return pred_masks, pred_class_map


def compare_annotation_iou(
    image_name: str,
    predicted_masks: dict[str, np.ndarray],
    xml_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    search_root: str | Path | None = None,
    class_mode: str | None = None,
) -> dict[str, object]:
    """Compare final model masks against cleaned CVAT labels for one tile.

    Invalid pixels are excluded from the metric. The comparison supports either
    the merged A / C / E taxonomy or the split A / B / C / D / E taxonomy.
    """

    resolved_annotation = _resolve_annotations_xml(image_name, xml_path=xml_path, search_root=search_root)
    cleaned = build_cleaned_annotation_mask(resolved_annotation, image_name)
    class_order, gt_to_pred_class = _resolve_class_spec(class_mode)
    gt_masks, valid_mask, gt_class_map = _build_ground_truth_maps(cleaned, class_order, gt_to_pred_class)

    image_shape = (int(cleaned["height"]), int(cleaned["width"]))
    pred_masks, pred_class_map = _build_predicted_class_map_for_order(predicted_masks, image_shape, class_order)

    if not np.any(valid_mask):
        raise ValueError(f"Annotation tile '{image_name}' contains no non-Invalid pixels")

    class_results: dict[str, dict[str, float]] = {}
    class_ious: list[float] = []
    correct_pixels = int(((pred_class_map == gt_class_map) & valid_mask).sum())
    evaluated_pixels = int(valid_mask.sum())

    for class_name in class_order:
        pred_eval = pred_masks[class_name] & valid_mask
        gt_eval = gt_masks[class_name] & valid_mask
        intersection = int(np.logical_and(pred_eval, gt_eval).sum())
        union = int(np.logical_or(pred_eval, gt_eval).sum())
        iou = 1.0 if union == 0 else float(intersection / union)

        pred_count = int(pred_eval.sum())
        gt_count = int(gt_eval.sum())
        class_results[class_name] = {
            "iou": iou,
            "intersection": float(intersection),
            "union": float(union),
            "pred_pixels": float(pred_count),
            "gt_pixels": float(gt_count),
        }
        class_ious.append(iou)

    mean_iou = float(np.mean(class_ious)) if class_ious else 0.0
    pixel_accuracy = float(correct_pixels / evaluated_pixels) if evaluated_pixels else 0.0

    report = {
        "image_name": cleaned["image_name"],
        "annotation_path": str(resolved_annotation),
        "xml_path": str(resolved_annotation),
        "shape": [int(cleaned["height"]), int(cleaned["width"])],
        "evaluated_pixels": evaluated_pixels,
        "ignored_invalid_pixels": int((~valid_mask).sum()),
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "class_results": class_results,
        "class_mode": (class_mode or "grouped").strip().lower(),
        "ground_truth_class_map": gt_class_map,
        "prediction_class_map": pred_class_map,
        "valid_mask": valid_mask,
        "label_lookup": cleaned.get("label_lookup", {}),
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{Path(cleaned['image_name']).stem}_iou_report.json"
        np.save(out_dir / f"{Path(cleaned['image_name']).stem}_ground_truth_class_map.npy", gt_class_map)
        np.save(out_dir / f"{Path(cleaned['image_name']).stem}_prediction_class_map.npy", pred_class_map)
        np.save(out_dir / f"{Path(cleaned['image_name']).stem}_valid_mask.npy", valid_mask)

        serializable_report = {
            key: value
            for key, value in report.items()
            if key not in {"ground_truth_class_map", "prediction_class_map", "valid_mask", "label_lookup"}
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(serializable_report, f, indent=2)
        report["output_path"] = str(output_path)

    return report