from pathlib import Path
from time import perf_counter

import groundingdino
import groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from groundingdino.util.inference import load_model, predict
from matplotlib.patches import Rectangle
from PIL import Image

from models import config as cfg


def ensure_download(path: str, urls: list[str]) -> None:
    target = Path(path)
    if target.exists():
        return

    print(f"[INFO] Downloading model checkpoint: {target.name}")
    last_error: Exception | None = None
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                with target.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print(f"[INFO] Downloaded checkpoint to: {target.resolve()}")
            return
        except Exception as exc:
            last_error = exc
            if target.exists():
                target.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to download checkpoint: {target.name}") from last_error


def iter_tile_coords(height: int, width: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    step = max(1, tile_size - overlap)
    coords: list[tuple[int, int, int, int]] = []
    for y0 in range(0, height, step):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, step):
            x1 = min(width, x0 + tile_size)
            coords.append((y0, y1, x0, x1))
    return coords


def cxcywh_to_xyxy(boxes_cxcywh: torch.Tensor, width: int, height: int) -> np.ndarray:
    boxes_scaled = boxes_cxcywh * torch.tensor([width, height, width, height])
    boxes_xyxy_local = torch.zeros_like(boxes_scaled)
    boxes_xyxy_local[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2.0
    boxes_xyxy_local[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2.0
    boxes_xyxy_local[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2.0
    boxes_xyxy_local[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2.0
    return boxes_xyxy_local.cpu().numpy().astype(int)


def _is_box_geometry_valid(
    box_xyxy: np.ndarray,
    image_shape: tuple[int, int, int],
    cfg: dict,
    pixels_per_meter_sq: float | None = None,
) -> bool:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Reject ultra-thin boxes (often edge artifacts that render as lines).
    min_box_side_px = int(cfg.get("min_box_side_px", 8))
    if min(bw, bh) < min_box_side_px:
        return False

    area_ratio = float(bw * bh) / float(max(1, w * h))
    min_area_ratio = float(cfg.get("min_area_ratio", 0.0))
    max_area_ratio = float(cfg.get("max_area_ratio", 1.0))
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False

    # Optional real-world area gate (m^2) derived from GeoTIFF spatial metadata.
    if pixels_per_meter_sq is not None and pixels_per_meter_sq > 0:
        area_meters_sq = float(bw * bh) / float(pixels_per_meter_sq)
        max_area_meters_sq = float(cfg.get("max_area_meters_sq", float("inf")))
        if area_meters_sq > max_area_meters_sq:
            return False

    aspect_ratio = max(bw, bh) / float(min(bw, bh))
    min_aspect_ratio = float(cfg.get("min_aspect_ratio", 1.0))
    if aspect_ratio < min_aspect_ratio:
        return False
    max_aspect_ratio = float(cfg.get("max_aspect_ratio", float("inf")))
    if aspect_ratio > max_aspect_ratio:
        return False

    return True


def build_dino_model_and_transform(config) -> tuple[object, object, str]:
    dino_pkg_dir = Path(groundingdino.__file__).resolve().parent
    dino_config_path = dino_pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
    dino_checkpoint_path = Path("groundingdino_swint_ogc.pth")
    dino_checkpoint_urls = [
        "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/alpha/groundingdino_swint_ogc.pth",
    ]
    ensure_download(str(dino_checkpoint_path), dino_checkpoint_urls)

    dino_model = load_model(str(dino_config_path), str(dino_checkpoint_path))

    requested_device = str(getattr(config, "dino_device", "auto")).strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "auto":
        dino_device = "cuda" if cuda_available else "cpu"
    elif requested_device == "cuda":
        if cuda_available:
            dino_device = "cuda"
        else:
            print(
                "[WARN] cfg.dino_device='cuda' but CUDA is unavailable in this PyTorch build; "
                "falling back to CPU"
            )
            dino_device = "cpu"
    else:
        dino_device = "cpu"

    print(f"[INFO] DINO device resolved to '{dino_device}'")

    if bool(getattr(config, "dino_full_resolution", False)):
        dino_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        dino_transform = T.Compose(
            [
                T.RandomResize([config.dino_resize_short_side], max_size=config.dino_resize_max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return dino_model, dino_transform, dino_device


def _run_dino_inference(
    image_rgb: np.ndarray,
    caption: str,
    box_th: float,
    text_th: float,
    dino_model,
    dino_transform,
    dino_device: str,
):
    image_pil = Image.fromarray(image_rgb)
    image_tensor, _ = dino_transform(image_pil, None)
    boxes_local, logits_local, phrases_local = predict(
        model=dino_model,
        image=image_tensor,
        caption=caption,
        box_threshold=box_th,
        text_threshold=text_th,
        device=dino_device,
    )
    logits_np_local = (
        logits_local.detach().cpu().numpy()
        if hasattr(logits_local, "detach")
        else np.array(logits_local)
    )
    return boxes_local, logits_np_local, list(phrases_local)


def _safe_run_dino_inference(
    image_rgb: np.ndarray,
    caption: str,
    box_th: float,
    text_th: float,
    dino_model,
    dino_transform,
    dino_device: str,
    context: str,
):
    """Run DINO inference with a one-shot stricter-threshold retry for known tile failures."""
    try:
        return _run_dino_inference(
            image_rgb,
            caption,
            box_th,
            text_th,
            dino_model,
            dino_transform,
            dino_device,
        )
    except Exception as exc:
        msg = str(exc)
        if "selected index k out of range" in msg.lower():
            strict_box_th = min(0.95, float(box_th) + 0.08)
            strict_text_th = min(0.95, float(text_th) + 0.08)
            print(
                f"[WARN] DINO {context} failed with k-index error; "
                f"retrying once with stricter thresholds "
                f"(box={strict_box_th:.2f}, text={strict_text_th:.2f})"
            )
            try:
                return _run_dino_inference(
                    image_rgb,
                    caption,
                    strict_box_th,
                    strict_text_th,
                    dino_model,
                    dino_transform,
                    dino_device,
                )
            except Exception as retry_exc:
                print(f"[WARN] DINO {context} retry failed: {retry_exc}")
                return None

        print(f"[WARN] DINO {context} failed: {exc}")
        return None


def _nms_records(
    records: list[dict],
    iou_threshold: float,
    prefer_smaller_boxes: bool = True,
) -> list[dict]:
    if not records:
        return []

    boxes = np.array([rec["box"] for rec in records], dtype=np.float32)
    scores = np.array([float(rec["score"]) for rec in records], dtype=np.float32)
    areas = np.maximum(1.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    # Prefer higher confidence first, then configurable area tie-break.
    # Smaller boxes are better for compact objects (courts/roofs), while larger boxes
    # preserve context for broad regions (e.g., parks).
    area_tiebreak = areas if prefer_smaller_boxes else -areas
    order = np.lexsort((area_tiebreak, -scores))
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        area_i = np.maximum(1.0, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        area_rest = np.maximum(
            1.0,
            (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]),
        )
        union = area_i + area_rest - inter
        iou = inter / np.maximum(union, 1e-6)

        order = rest[iou <= float(iou_threshold)]

    return [records[i] for i in keep]


def _box_area_xyxy(box: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(1.0, (x2 - x1) * (y2 - y1))


def _compute_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return float(inter_area / max(union, 1e-6))


def _compute_iomin_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over min(area(a), area(b)) for containment checks."""
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter_area / max(min(area_a, area_b), 1e-6))


def _remove_contained_records(
    records: list[dict],
    iomin_threshold: float,
) -> tuple[list[dict], list[dict]]:
    """Remove boxes that are largely contained within already-kept boxes.

    Assumes `records` are pre-sorted by preference (score/area tie-break).
    """
    if not records:
        return [], []

    kept: list[dict] = []
    removed: list[dict] = []
    for rec in records:
        rb = np.array(rec["box"], dtype=np.float32)
        is_contained = any(
            _compute_iomin_xyxy(rb, np.array(k["box"], dtype=np.float32)) >= float(iomin_threshold)
            for k in kept
        )
        if is_contained:
            removed.append(rec)
        else:
            kept.append(rec)

    return kept, removed


def _remove_records_overlapping_negative_boxes(
    records: list[dict],
    negative_boxes: list[np.ndarray],
    iou_threshold: float,
) -> tuple[list[dict], int]:
    if not records or not negative_boxes:
        return records, 0

    kept: list[dict] = []
    removed = 0
    for rec in records:
        rb = np.array(rec["box"], dtype=np.float32)
        overlaps_negative = any(
            _compute_iou_xyxy(rb, np.array(nb, dtype=np.float32)) >= float(iou_threshold)
            for nb in negative_boxes
        )
        if overlaps_negative:
            removed += 1
            continue
        kept.append(rec)

    return kept, removed


def run_dino_prompts(
    img: np.ndarray,
    config,
    dino_model,
    dino_transform,
    dino_device: str,
    pixels_per_meter_sq: float | None = None,
    return_unfiltered: bool = False,
    show_timing_summary: bool = True,
) -> list[dict] | tuple[list[dict], list[dict], list[dict]]:
    print("[INFO] Running prompt-by-prompt DINO detection")
    dino_total_start = perf_counter()
    all_records: list[dict] = []
    all_unfiltered_records: list[dict] = []
    all_filtered_records: list[dict] = []
    prompt_timing_rows: list[dict] = []
    validate_split_boxes = bool(getattr(config, "dino_validate_split_boxes", True))
    validate_split_max_candidates = int(getattr(config, "dino_validate_split_max_candidates", 120))
    enable_area_split = bool(getattr(config, "dino_enable_area_split", True))
    nms_iou_threshold = float(getattr(config, "dino_nms_iou_threshold", 0.70))
    max_boxes_for_sam = int(getattr(config, "dino_max_boxes_per_prompt_for_sam", 60))
    negative_overlap_iou_threshold = float(getattr(config, "dino_negative_overlap_iou_threshold", 0.35))
    top_k_filtered_debug = int(getattr(config, "dino_debug_top_k_filtered_scores", 15))
    refine_bounds = bool(getattr(config, "dino_refine_bounds", True))
    refine_max_depth = int(getattr(config, "dino_refine_bounds_max_depth", 1))
    refine_min_area_ratio = float(getattr(config, "dino_refine_bounds_min_area_ratio", 0.80))

    def _print_top_filtered(stage_label: str, dropped_records: list[dict]) -> None:
        if not dropped_records:
            print(f"[INFO] DINO {stage_label}: filtered out 0 boxes")
            return
        sorted_dropped = sorted(dropped_records, key=lambda r: float(r.get("score", 0.0)), reverse=True)
        top_records = sorted_dropped[:top_k_filtered_debug]
        top_scores = ", ".join(f"{float(r.get('score', 0.0)):.3f}" for r in top_records)
        print(
            f"[INFO] DINO {stage_label}: filtered out {len(dropped_records)} boxes; "
            f"top-{len(top_records)} filtered scores: [{top_scores}]"
        )

    def _split_record_by_area_limit(record: dict, prompt_cfg: dict) -> list[dict]:
        """Split oversized DINO boxes into the largest possible sub-boxes under area limit."""
        if pixels_per_meter_sq is None or pixels_per_meter_sq <= 0:
            return [record]

        max_area_m2 = float(prompt_cfg.get("max_area_meters_sq", float("inf")))
        if not np.isfinite(max_area_m2) or max_area_m2 <= 0:
            return [record]

        max_area_px = max_area_m2 * float(pixels_per_meter_sq)
        if not np.isfinite(max_area_px) or max_area_px <= 0:
            return [record]

        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in record["box"]]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_px = float(bw * bh)
        if area_px <= max_area_px:
            return [record]

        min_side = int(prompt_cfg.get("min_box_side_px", 8))
        max_tiles = int(prompt_cfg.get("max_split_boxes_per_detection", 256))

        # Search for a balanced grid that minimizes tile count while keeping tile area <= max_area_px.
        best: tuple[int, int, int, int, int] | None = None  # (tiles, nx, ny, tile_w, tile_h)
        max_ny = max(1, min(bh, int(np.ceil(bh / max(1, min_side)))))

        for ny in range(1, max_ny + 1):
            tile_h = int(np.ceil(bh / ny))
            if tile_h < min_side:
                break

            nx = max(1, int(np.ceil((bw * tile_h) / max_area_px)))
            tile_w = int(np.ceil(bw / nx))
            if tile_w < min_side:
                continue

            if (tile_w * tile_h) > max_area_px:
                continue

            tiles = nx * ny
            if tiles > max_tiles:
                continue

            if best is None or tiles < best[0] or (tiles == best[0] and (tile_w * tile_h) > (best[3] * best[4])):
                best = (tiles, nx, ny, tile_w, tile_h)

        if best is None:
            return [record]

        _, nx, ny, _, _ = best
        split_records: list[dict] = []
        for iy in range(ny):
            sy1 = y1 + (iy * bh) // ny
            sy2 = y1 + ((iy + 1) * bh) // ny
            if sy2 <= sy1:
                continue
            for ix in range(nx):
                sx1 = x1 + (ix * bw) // nx
                sx2 = x1 + ((ix + 1) * bw) // nx
                if sx2 <= sx1:
                    continue
                split_records.append({**record, "box": np.array([sx1, sy1, sx2, sy2], dtype=int)})

        return split_records if split_records else [record]

    def _refine_record_bounds(record: dict, prompt_cfg: dict, depth: int = 0) -> dict:
        """Recursively tighten a DINO box by re-running DINO inside the current box."""
        prompt_refine_bounds = bool(prompt_cfg.get("refine_bounds", refine_bounds))
        if not prompt_refine_bounds or depth >= refine_max_depth:
            return record

        x1, y1, x2, y2 = [int(v) for v in record["box"]]
        min_side = int(prompt_cfg.get("min_box_side_px", 8))
        if (x2 - x1) < min_side or (y2 - y1) < min_side:
            return record

        tile_img = img[y1:y2, x1:x2]
        if tile_img.size == 0:
            return record

        infer_out = _safe_run_dino_inference(
            tile_img,
            prompt_cfg["caption"],
            prompt_cfg["box_threshold"],
            prompt_cfg["text_threshold"],
            dino_model,
            dino_transform,
            dino_device,
            context=f"[{prompt_cfg['name']}] refine depth {depth + 1}",
        )
        if infer_out is None:
            return record

        t_boxes, t_logits, t_phrases = infer_out
        t_xyxy = cxcywh_to_xyxy(t_boxes, tile_img.shape[1], tile_img.shape[0])

        candidates: list[dict] = []
        for lbox, phrase, score in zip(t_xyxy, t_phrases, t_logits):
            phrase_l = phrase.lower()
            if not any(k in phrase_l for k in prompt_cfg.get("keywords", ())):
                continue

            lx1, ly1, lx2, ly2 = [int(v) for v in lbox]
            gx1, gy1 = x1 + lx1, y1 + ly1
            gx2, gy2 = x1 + lx2, y1 + ly2
            candidate = {
                "box": np.array([gx1, gy1, gx2, gy2], dtype=int),
                "phrase": phrase,
                "score": float(score),
                "prompt_group": record["prompt_group"],
            }
            if _is_box_geometry_valid(candidate["box"], img.shape, prompt_cfg, pixels_per_meter_sq):
                candidates.append(candidate)

        if not candidates:
            return record

        candidates.sort(key=lambda r: (-float(r["score"]), _box_area_xyxy(r["box"])))
        best = candidates[0]

        parent_area = _box_area_xyxy(record["box"])
        best_area = _box_area_xyxy(best["box"])
        if best_area >= parent_area * float(refine_min_area_ratio):
            return record

        # Recurse once more only if the box meaningfully shrank.
        return _refine_record_bounds(best, prompt_cfg, depth + 1)

    def _apply_prompt_filters(records: list[dict], prompt_cfg: dict) -> list[dict]:
        positive_keywords = tuple(k.lower() for k in prompt_cfg.get("keywords", ()))
        negative_keywords = tuple(k.lower() for k in prompt_cfg.get("negative_keywords", ()))

        keyword_filtered: list[dict] = []
        dropped_negative = 0
        for rec in records:
            phrase_l = rec["phrase"].lower()
            if positive_keywords and not any(k in phrase_l for k in positive_keywords):
                continue
            if negative_keywords and any(k in phrase_l for k in negative_keywords):
                dropped_negative += 1
                continue
            keyword_filtered.append(rec)

        if dropped_negative > 0:
            print(
                f"[INFO] DINO [{prompt_cfg['name']}] early negative filter removed {dropped_negative} boxes"
            )

        do_split = enable_area_split and bool(prompt_cfg.get("enable_area_split", True))
        area_split_records: list[dict] = []
        split_children = 0
        if do_split:
            for rec in keyword_filtered:
                children = _split_record_by_area_limit(rec, prompt_cfg)
                split_children += max(0, len(children) - 1)
                area_split_records.extend(children)
        else:
            area_split_records = keyword_filtered

        if do_split and split_children > 0:
            print(
                f"[INFO] DINO [{prompt_cfg['name']}] split {len(keyword_filtered)} boxes into "
                f"{len(area_split_records)} boxes (added {split_children} children) for area limiting"
            )

        # Re-run DINO on split boxes to confirm local prompt match instead of inheriting parent box label.
        validated_records = area_split_records
        if validate_split_boxes and do_split and split_children > 0:
            validated_records = []
            candidates = area_split_records[:validate_split_max_candidates]
            dropped = len(area_split_records) - len(candidates)
            if dropped > 0:
                print(
                    f"[WARN] DINO [{prompt_cfg['name']}] validating only first {len(candidates)} split boxes "
                    f"(dropped {dropped} due to limit={validate_split_max_candidates})"
                )

            for rec in candidates:
                x1, y1, x2, y2 = [int(v) for v in rec["box"]]
                if x2 <= x1 or y2 <= y1:
                    continue

                tile_img = img[y1:y2, x1:x2]
                if tile_img.size == 0:
                    continue

                infer_out = _safe_run_dino_inference(
                    tile_img,
                    prompt_cfg["caption"],
                    prompt_cfg["box_threshold"],
                    prompt_cfg["text_threshold"],
                    dino_model,
                    dino_transform,
                    dino_device,
                    context=f"[{prompt_cfg['name']}] split-validation",
                )
                if infer_out is None:
                    continue
                t_boxes, t_logits, t_phrases = infer_out

                t_xyxy = cxcywh_to_xyxy(t_boxes, tile_img.shape[1], tile_img.shape[0])
                for lbox, phrase, score in zip(t_xyxy, t_phrases, t_logits):
                    if not any(k in phrase.lower() for k in prompt_cfg["keywords"]):
                        continue
                    lx1, ly1, lx2, ly2 = [int(v) for v in lbox]
                    gx1, gy1 = x1 + lx1, y1 + ly1
                    gx2, gy2 = x1 + lx2, y1 + ly2
                    validated_records.append(
                        {
                            "box": np.array([gx1, gy1, gx2, gy2], dtype=int),
                            "phrase": phrase,
                            "score": float(score),
                            "prompt_group": rec["prompt_group"],
                        }
                    )

            print(
                f"[INFO] DINO [{prompt_cfg['name']}] split-box local validation generated "
                f"{len(validated_records)} candidates from {len(area_split_records)} split boxes"
            )

        geom_filtered = [
            rec
            for rec in validated_records
            if _is_box_geometry_valid(rec["box"], img.shape, prompt_cfg, pixels_per_meter_sq)
        ]
        return geom_filtered

    for cfg in config.dino_prompt_configs:
        prompt_start = perf_counter()
        # Negative-first pass: detect forbidden categories and remove overlapping positive boxes.
        negative_dino_boxes: list[np.ndarray] = []
        negative_dino_caption = str(cfg.get("negative_dino_caption", "")).strip()
        if negative_dino_caption:
            neg_box_th = float(cfg.get("negative_box_threshold", max(0.30, cfg["box_threshold"])))
            neg_text_th = float(cfg.get("negative_text_threshold", max(0.28, cfg["text_threshold"])))
            neg_infer_out = _safe_run_dino_inference(
                img,
                negative_dino_caption,
                neg_box_th,
                neg_text_th,
                dino_model,
                dino_transform,
                dino_device,
                context=f"[{cfg['name']}] negative-first full-image",
            )
            if neg_infer_out is not None:
                neg_boxes_local, _, _ = neg_infer_out
                neg_xyxy = cxcywh_to_xyxy(neg_boxes_local, img.shape[1], img.shape[0])
                negative_dino_boxes = [np.array(b, dtype=int) for b in neg_xyxy]
            print(
                f"[INFO] DINO [{cfg['name']}] negative-first detections: {len(negative_dino_boxes)} "
                f"(caption='{negative_dino_caption}')"
            )

        infer_out = _safe_run_dino_inference(
            img,
            cfg["caption"],
            cfg["box_threshold"],
            cfg["text_threshold"],
            dino_model,
            dino_transform,
            dino_device,
            context=f"[{cfg['name']}] full-image",
        )
        if infer_out is None:
            print(f"[WARN] DINO [{cfg['name']}] full-image inference failed; continuing with empty detections")
            boxes_local = torch.empty((0, 4), dtype=torch.float32)
            logits_local = np.array([], dtype=np.float32)
            phrases_local = []
        else:
            boxes_local, logits_local, phrases_local = infer_out
        boxes_xyxy_local = cxcywh_to_xyxy(boxes_local, img.shape[1], img.shape[0])
        print(f"[INFO] DINO [{cfg['name']}] full-image detections: {len(boxes_xyxy_local)}")

        use_tiled_fallback = bool(cfg.get("enable_tiled_fallback", True))
        prompt_records: list[dict] = [
            {
                "box": box_xyxy,
                "phrase": phrase,
                "score": float(score),
                "prompt_group": cfg["name"],
            }
            for box_xyxy, phrase, score in zip(boxes_xyxy_local, phrases_local, logits_local)
        ]
        all_unfiltered_records.extend(prompt_records)

        if (
            config.dino_enable_tiled_fallback
            and use_tiled_fallback
            and len(prompt_records) == 0
        ):
            tile_records: list[dict] = []
            tile_coords = iter_tile_coords(
                img.shape[0],
                img.shape[1],
                config.dino_tile_size_px,
                config.dino_tile_overlap_px,
            )
            print(
                f"[INFO] DINO [{cfg['name']}] using tiled fallback "
                f"({len(tile_coords)} tiles, tile={config.dino_tile_size_px}, "
                f"overlap={config.dino_tile_overlap_px})"
            )

            for tile_idx, (y0, y1, x0, x1) in enumerate(tile_coords, start=1):
                tile_img = img[y0:y1, x0:x1]
                infer_out = _safe_run_dino_inference(
                    tile_img,
                    cfg["caption"],
                    cfg["box_threshold"],
                    cfg["text_threshold"],
                    dino_model,
                    dino_transform,
                    dino_device,
                    context=f"[{cfg['name']}] tile {tile_idx}/{len(tile_coords)}",
                )
                if infer_out is None:
                    continue
                t_boxes, t_logits, t_phrases = infer_out
                t_xyxy = cxcywh_to_xyxy(t_boxes, tile_img.shape[1], tile_img.shape[0])

                for box_xyxy, phrase, score in zip(t_xyxy, t_phrases, t_logits):
                    gx1, gy1, gx2, gy2 = box_xyxy
                    tile_records.append(
                        {
                            "box": np.array([gx1 + x0, gy1 + y0, gx2 + x0, gy2 + y0], dtype=int),
                            "phrase": phrase,
                            "score": float(score),
                            "prompt_group": cfg["name"],
                        }
                    )

            if tile_records:
                print(f"[INFO] DINO [{cfg['name']}] tiled detections: {len(tile_records)}")
                prompt_records = tile_records
                all_unfiltered_records.extend(tile_records)

        if negative_dino_boxes:
            before_neg = list(prompt_records)
            prompt_records, removed_neg = _remove_records_overlapping_negative_boxes(
                prompt_records,
                negative_dino_boxes,
                iou_threshold=negative_overlap_iou_threshold,
            )
            if removed_neg > 0:
                print(
                    f"[INFO] DINO [{cfg['name']}] negative-first overlap filter removed {removed_neg} boxes"
                )
                kept_ids = {id(rec) for rec in prompt_records}
                dropped_neg = [rec for rec in before_neg if id(rec) not in kept_ids]
                _print_top_filtered(f"[{cfg['name']}] negative-overlap", dropped_neg)

        before_filters = list(prompt_records)
        prompt_records = _apply_prompt_filters(prompt_records, cfg)
        if len(prompt_records) < len(before_filters):
            kept_ids = {id(rec) for rec in prompt_records}
            dropped_prompt = [rec for rec in before_filters if id(rec) not in kept_ids]
            _print_top_filtered(f"[{cfg['name']}] prompt/geometry", dropped_prompt)

        if prompt_records and refine_bounds:
            before_refine = list(prompt_records)
            refined_records = []
            refined_count = 0
            for rec in prompt_records:
                refined = _refine_record_bounds(rec, cfg)
                if _box_area_xyxy(refined["box"]) < _box_area_xyxy(rec["box"]):
                    refined_count += 1
                refined_records.append(refined)
            prompt_records = refined_records
            if refined_count > 0:
                print(
                    f"[INFO] DINO [{cfg['name']}] tightened {refined_count}/{len(before_refine)} boxes "
                    f"to smaller valid bounds"
                )

        # If full-image detections were all filtered out, try tiled fallback for smaller boxes.
        if (
            config.dino_enable_tiled_fallback
            and use_tiled_fallback
            and len(prompt_records) == 0
            and len(boxes_xyxy_local) > 0
        ):
            tile_records: list[dict] = []
            tile_coords = iter_tile_coords(
                img.shape[0],
                img.shape[1],
                config.dino_tile_size_px,
                config.dino_tile_overlap_px,
            )
            print(
                f"[INFO] DINO [{cfg['name']}] retrying tiled fallback after filtering "
                f"({len(tile_coords)} tiles, tile={config.dino_tile_size_px}, "
                f"overlap={config.dino_tile_overlap_px})"
            )

            for tile_idx, (y0, y1, x0, x1) in enumerate(tile_coords, start=1):
                tile_img = img[y0:y1, x0:x1]
                infer_out = _safe_run_dino_inference(
                    tile_img,
                    cfg["caption"],
                    cfg["box_threshold"],
                    cfg["text_threshold"],
                    dino_model,
                    dino_transform,
                    dino_device,
                    context=f"[{cfg['name']}] retry tile {tile_idx}/{len(tile_coords)}",
                )
                if infer_out is None:
                    continue
                t_boxes, t_logits, t_phrases = infer_out
                t_xyxy = cxcywh_to_xyxy(t_boxes, tile_img.shape[1], tile_img.shape[0])

                for box_xyxy, phrase, score in zip(t_xyxy, t_phrases, t_logits):
                    gx1, gy1, gx2, gy2 = box_xyxy
                    tile_records.append(
                        {
                            "box": np.array([gx1 + x0, gy1 + y0, gx2 + x0, gy2 + y0], dtype=int),
                            "phrase": phrase,
                            "score": float(score),
                            "prompt_group": cfg["name"],
                        }
                    )

            if tile_records:
                print(f"[INFO] DINO [{cfg['name']}] tiled retry detections: {len(tile_records)}")
                all_unfiltered_records.extend(tile_records)
                if negative_dino_boxes:
                    before_neg_tiled = list(tile_records)
                    tile_records, removed_neg = _remove_records_overlapping_negative_boxes(
                        tile_records,
                        negative_dino_boxes,
                        iou_threshold=negative_overlap_iou_threshold,
                    )
                    if removed_neg > 0:
                        print(
                            f"[INFO] DINO [{cfg['name']}] negative-first overlap filter removed "
                            f"{removed_neg} tiled-retry boxes"
                        )
                        kept_ids = {id(rec) for rec in tile_records}
                        dropped_neg_tiled = [rec for rec in before_neg_tiled if id(rec) not in kept_ids]
                        _print_top_filtered(f"[{cfg['name']}] tiled-retry negative-overlap", dropped_neg_tiled)
                before_filters_tiled = list(tile_records)
                prompt_records = _apply_prompt_filters(tile_records, cfg)
                if len(prompt_records) < len(before_filters_tiled):
                    kept_ids = {id(rec) for rec in prompt_records}
                    dropped_prompt_tiled = [rec for rec in before_filters_tiled if id(rec) not in kept_ids]
                    _print_top_filtered(f"[{cfg['name']}] tiled-retry prompt/geometry", dropped_prompt_tiled)

        # Keep only strongest boxes per prompt and suppress heavy overlap before SAM.
        if prompt_records:
            before_nms = list(prompt_records)
            prefer_smaller_boxes = bool(cfg.get("prefer_smaller_boxes", True))
            remove_contained_boxes = bool(cfg.get("remove_contained_boxes", False))
            contained_iomin_threshold = float(cfg.get("contained_iomin_threshold", 0.92))
            if prefer_smaller_boxes:
                prompt_records.sort(key=lambda r: (-float(r["score"]), _box_area_xyxy(r["box"])))
            else:
                prompt_records.sort(key=lambda r: (-float(r["score"]), -_box_area_xyxy(r["box"])))
            prompt_records = _nms_records(
                prompt_records,
                iou_threshold=nms_iou_threshold,
                prefer_smaller_boxes=prefer_smaller_boxes,
            )
            if len(prompt_records) < len(before_nms):
                kept_ids = {id(rec) for rec in prompt_records}
                dropped_nms = [rec for rec in before_nms if id(rec) not in kept_ids]
                _print_top_filtered(f"[{cfg['name']}] nms", dropped_nms)
                all_filtered_records.extend(dropped_nms)

            if remove_contained_boxes and prompt_records:
                prompt_records, dropped_contained = _remove_contained_records(
                    prompt_records,
                    iomin_threshold=contained_iomin_threshold,
                )
                if dropped_contained:
                    _print_top_filtered(f"[{cfg['name']}] contained", dropped_contained)
                    all_filtered_records.extend(dropped_contained)

            if len(prompt_records) > max_boxes_for_sam:
                print(
                    f"[INFO] DINO [{cfg['name']}] capping boxes for SAM: "
                    f"{len(prompt_records)} -> {max_boxes_for_sam}"
                )
                dropped_cap = prompt_records[max_boxes_for_sam:]
                _print_top_filtered(f"[{cfg['name']}] cap-for-sam", dropped_cap)
                all_filtered_records.extend(dropped_cap)
                prompt_records = prompt_records[:max_boxes_for_sam]

        print(f"[INFO] DINO [{cfg['name']}] kept for SAM: {len(prompt_records)}")
        all_records.extend(prompt_records)
        prompt_elapsed_s = perf_counter() - prompt_start
        prompt_timing_rows.append(
            {
                "name": str(cfg["name"]),
                "seconds": float(prompt_elapsed_s),
                "kept_for_sam": int(len(prompt_records)),
            }
        )
        print(f"[TIMING] DINO [{cfg['name']}] total time: {prompt_elapsed_s:.2f}s")

    if show_timing_summary and prompt_timing_rows:
        total_elapsed_s = perf_counter() - dino_total_start
        print("[TIMING] DINO prompt timing summary (sorted by runtime):")
        for idx, row in enumerate(
            sorted(prompt_timing_rows, key=lambda r: r["seconds"], reverse=True), start=1
        ):
            print(
                f"[TIMING] {idx:02d}. {row['name']:<24} {row['seconds']:>7.2f}s "
                f"(kept_for_sam={row['kept_for_sam']})"
            )
        print(
            f"[TIMING] DINO total runtime: {total_elapsed_s:.2f}s "
            f"across {len(prompt_timing_rows)} prompts"
        )

    if return_unfiltered:
        return all_records, all_unfiltered_records, all_filtered_records
    return all_records


def save_dino_debug_image(img: np.ndarray, run_id: str, save_figure_fn) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.title("DINO Input Debug Image", fontsize=28, weight="bold")
    save_figure_fn(f"{run_id}_dino_input_debug.png", "dino_detections")


def save_dino_detection_viz(
    img: np.ndarray,
    records: list[dict],
    run_id: str,
    save_figure_fn,
    title_suffix: str = "",
    filtered_records: list[dict] | None = None,
    dpi: int = 75,
) -> None:
    if records:
        dino_boxes_xyxy = np.stack([rec["box"] for rec in records]).astype(int)
        dino_phrases = [rec["phrase"] for rec in records]
        dino_logits = np.array([rec["score"] for rec in records], dtype=float)
        dino_prompt_groups = [rec["prompt_group"] for rec in records]
    else:
        dino_boxes_xyxy = np.empty((0, 4), dtype=int)
        dino_phrases = []
        dino_logits = np.array([], dtype=float)
        dino_prompt_groups = []

    # Get top-10 filtered detections (sorted by score, descending)
    filtered_boxes_xyxy = np.empty((0, 4), dtype=int)
    filtered_phrases = []
    filtered_logits = np.array([], dtype=float)
    if filtered_records:
        sorted_filtered = sorted(filtered_records, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:10]
        if sorted_filtered:
            filtered_boxes_xyxy = np.stack([rec["box"] for rec in sorted_filtered]).astype(int)
            filtered_phrases = [rec["phrase"] for rec in sorted_filtered]
            filtered_logits = np.array([rec["score"] for rec in sorted_filtered], dtype=float)

    dino_viz_stride = 1
    dino_viz_img = img
    dino_group_colors = {
        "sidewalk": "lime",
        "sitting": "orange",
        "transit_hub": "magenta",
        "pedestrian_features": "chartreuse",
        "sidewalk_surface": "cyan",
        "road_surface": "gray",
        "building_roof": "lime",
        "warehouse_roof": "orange",
        "outdoor_seating": "lime",
        "seated_dining": "orange",
        "standing_gathering": "cyan",
        "furniture": "yellow",
    }
    max_labels = 40

    h, w = dino_viz_img.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(dino_viz_img)

    # Draw passed detections in green
    order = np.argsort(-dino_logits) if len(dino_logits) > 0 else np.array([], dtype=int)
    for draw_idx, idx in enumerate(order.tolist()):
        box = dino_boxes_xyxy[idx]
        phrase = dino_phrases[idx]
        score = dino_logits[idx]
        group = dino_prompt_groups[idx]
        x1, y1, x2, y2 = box.astype(float) / float(dino_viz_stride)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        # Use green for passed detections - LARGE linewidth for high-res images
        ax.add_patch(
            Rectangle((x1, y1), box_w, box_h, fill=False, edgecolor="green", linewidth=12)
        )
        if draw_idx < max_labels:
            ax.text(
                x1,
                max(0.0, y1 - 15.0),
                f"[{group}] {phrase} {float(score):.2f}",
                color="white",
                fontsize=32,
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 3.0},
            )

    # Draw top-10 filtered detections in red
    for idx in range(len(filtered_boxes_xyxy)):
        box = filtered_boxes_xyxy[idx]
        phrase = filtered_phrases[idx]
        score = filtered_logits[idx]
        x1, y1, x2, y2 = box.astype(float) / float(dino_viz_stride)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        ax.add_patch(
            Rectangle((x1, y1), box_w, box_h, fill=False, edgecolor="red", linewidth=10, linestyle="--")
        )
        ax.text(
            x1,
            max(0.0, y1 - 15.0),
            f"FILTERED {phrase} {float(score):.2f}",
            color="white",
            fontsize=24,
            bbox={"facecolor": "red", "alpha": 0.7, "pad": 2.5},
        )

    if len(records) == 0 and len(filtered_boxes_xyxy) == 0:
        ax.text(
            10,
            20,
            "No DINO detections to draw",
            color="white",
            fontsize=36,
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 5},
        )

    ax.axis("off")
    fig.suptitle(f"Grounding DINO Detections (Before SAM){title_suffix}", fontsize=48, y=0.99, weight="bold")
    save_figure_fn(f"{run_id}_dino_detections.png", "dino_detections", dpi=dpi)


def save_dino_detection_viz_pil(
    img: np.ndarray,
    records: list[dict],
    run_id: str,
    save_figure_fn,
    title_suffix: str = "",
    filtered_records: list[dict] | None = None,
) -> None:
    """PIL/OpenCV-based DINO visualization - faster, smaller files, no memory issues."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Make a copy to draw on
    viz_img = img.copy()
    
    # Get top-10 filtered detections
    filtered_boxes_xyxy = np.empty((0, 4), dtype=int)
    filtered_phrases = []
    filtered_logits = np.array([], dtype=float)
    if filtered_records:
        sorted_filtered = sorted(filtered_records, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:10]
        if sorted_filtered:
            filtered_boxes_xyxy = np.stack([rec["box"] for rec in sorted_filtered]).astype(int)
            filtered_phrases = [rec["phrase"] for rec in sorted_filtered]
            filtered_logits = np.array([rec["score"] for rec in sorted_filtered], dtype=float)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(viz_img, mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    
    # Scale font size based on image dimensions
    img_h, img_w = viz_img.shape[:2]
    font_size = max(24, int(img_h / 400))
    line_width = max(8, int(img_h / 1000))
    
    # Try to load a larger font, fall back to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw passed detections in green
    for rec in records:
        box = rec["box"]
        phrase = rec["phrase"]
        score = float(rec["score"])
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # Draw thick green rectangle
        for i in range(line_width):
            draw.rectangle(
                [(x1 + i, y1 + i), (x2 - i, y2 - i)],
                outline=(0, 255, 0),
                width=1
            )
        
        # Draw label with black background
        label = f"{phrase} {score:.2f}"
        bbox = draw.textbbox((x1, max(0, y1 - 30)), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x1, max(0, y1 - 30)), label, fill=(255, 255, 255), font=font)
    
    # Draw top-10 filtered detections in red dashed
    for idx in range(len(filtered_boxes_xyxy)):
        box = filtered_boxes_xyxy[idx]
        phrase = filtered_phrases[idx]
        score = filtered_logits[idx]
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # Draw dashed red rectangle
        dash_len = 20
        for x in range(x1, x2, dash_len * 2):
            draw.line([(x, y1), (min(x + dash_len, x2), y1)], fill=(255, 0, 0), width=line_width)
            draw.line([(x, y2), (min(x + dash_len, x2), y2)], fill=(255, 0, 0), width=line_width)
        for y in range(y1, y2, dash_len * 2):
            draw.line([(x1, y), (x1, min(y + dash_len, y2))], fill=(255, 0, 0), width=line_width)
            draw.line([(x2, y), (x2, min(y + dash_len, y2))], fill=(255, 0, 0), width=line_width)
        
        # Draw label with red background
        label = f"FILTERED {phrase} {score:.2f}"
        bbox = draw.textbbox((x1, max(0, y1 - 30)), label, font=font)
        draw.rectangle(bbox, fill=(200, 0, 0))
        draw.text((x1, max(0, y1 - 30)), label, fill=(255, 255, 255), font=font)
    
    # Save via callback function
    save_figure_fn(f"{run_id}_dino_detections.png", "dino_detections", pil_img=pil_img)
