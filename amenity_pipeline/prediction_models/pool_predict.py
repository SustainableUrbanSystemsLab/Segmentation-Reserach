"""
Pool detection using Esri PoolSegmentation_USA model with explicit chip-based tiling.

Why explicit tiling:
    ArcGIS ModelExtension.predict() on a 7800x7700 px image processes the
    image in coarse internal tiles, typically missing detections in areas that
    don't align well with those tiles. Explicit tiling at CHIP_SIZE (400px)
    guarantees every region is seen at the model's training resolution.
    Result: ~10-20x more raw detections, better recall.

Coordinate flow:
    1. Resample full image → 0.15m/px, save temp GeoTIFF with correct transform tx.
    2. For each chip (row, col) offset: extract, save temp chip GeoTIFF, predict.
    3. Raw bboxes from ArcGIS are in [xmin, ymin, w, h] pixel space of the CHIP.
    4. Add chip offset → pixel space of the full resampled image.
    5. Convert to map coords using tx.
    6. Filter by area (m²), aspect, blue/turquoise color.
    7. NMS across all chips.
"""

import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

warnings.filterwarnings("ignore")

try:
    from arcgis.learn import ModelExtension
except ImportError:
    raise ImportError("arcgis required. conda install -c esri arcgis")

SCRIPT_DIR    = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL = str(PIPELINE_ROOT / "models" / "PoolSegmentation_USA.dlpk")

CHIP_SIZE    = 400     # px — model trained on 400×400 chips

# Pool constraints
POOL_MIN_M2    =    6.0   #  3m × 2m minimum
POOL_MAX_M2    = 2000.0   # 60m × 30m generous maximum
POOL_MAX_ASPECT =   5.0


# ─────────────────────────────────────────────────────────────────────────────
# Color filter — HSV-based, tuned for aerial pool appearance
# ─────────────────────────────────────────────────────────────────────────────

def _pool_color_fraction(img_rgb: np.ndarray, r0, c0, r1, c1) -> float:
    h, w = img_rgb.shape[:2]
    r0, r1 = max(0, r0), min(h, r1)
    c0, c1 = max(0, c0), min(w, c1)
    if r1 <= r0 or c1 <= c0:
        return 0.0

    patch = img_rgb[r0:r1, c0:c1].astype(np.float32) / 255.0
    if patch.size == 0:
        return 0.0

    r_ch, g_ch, b_ch = patch[..., 0], patch[..., 1], patch[..., 2]

    v   = np.maximum.reduce([r_ch, g_ch, b_ch])
    mn  = np.minimum.reduce([r_ch, g_ch, b_ch])
    delta = v - mn

    s = np.where(v > 1e-6, delta / v, 0.0)

    hue = np.zeros_like(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_r = (v == r_ch) & (delta > 0)
        m_g = (v == g_ch) & (delta > 0)
        m_b = (v == b_ch) & (delta > 0)
        hue[m_r] = ((g_ch[m_r] - b_ch[m_r]) / delta[m_r]) % 6.0
        hue[m_g] =  (b_ch[m_g] - r_ch[m_g]) / delta[m_g] + 2.0
        hue[m_b] =  (r_ch[m_b] - g_ch[m_b]) / delta[m_b] + 4.0
    hue /= 6.0

    is_pool_color = (hue >= 0.40) & (hue <= 0.71) & (s > 0.10) & (v > 0.15)
    return float(is_pool_color.sum()) / max(float(is_pool_color.size), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union  = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0

def _nms(detections: list, iou_threshold: float = 0.35) -> list:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: -d["score"])
    keep, dropped = [], set()
    for i, det in enumerate(detections):
        if i in dropped:
            continue
        keep.append(det)
        for j in range(i + 1, len(detections)):
            if j not in dropped and _iou(det["box"], detections[j]["box"]) > iou_threshold:
                dropped.add(j)
    return keep


def detect_pools(model_path, image_path, output_path, conf_threshold,
                 cell_size=0.15, min_blue_fraction=0.08, overlap = 256):

    print(f"[*] Pool detection")
    print(f"    model     : {Path(model_path).name}")
    print(f"    image     : {Path(image_path).name}")
    print(f"    threshold : {conf_threshold}  cell_size: {cell_size}m/px")
    print(f"    chip_size : {CHIP_SIZE}px  overlap: {overlap}px")

    with rasterio.open(image_path) as src:
        native_res = abs(src.transform.a)
        crs        = src.crs

        if cell_size <= 0:
            cell_size = native_res
            out_w     = src.width
            out_h     = src.height
            tx        = src.transform
            print(f"[*] Using native resolution: {native_res:.4f}m/px ({out_w}x{out_h})")
            img_data  = src.read([1, 2, 3])
        else:
            scale = native_res / cell_size
            out_w = round(src.width  * scale)
            out_h = round(src.height * scale)
            tx    = Affine(cell_size, 0, src.transform.c,
                           0, -cell_size, src.transform.f)
            print(
                f"[*] Resampling {native_res:.4f}m/px → {cell_size:.4f}m/px "
                f"({src.width}x{src.height} → {out_w}x{out_h})"
            )
            img_data = src.read(
                [1, 2, 3],
                out_shape=(3, out_h, out_w),
                resampling=Resampling.bilinear,
            )

    img_rgb = np.moveaxis(img_data, 0, -1)   # (H, W, 3) for color filter

    tmp_full_fd, tmp_full_path = tempfile.mkstemp(suffix="_full.tif")
    os.close(tmp_full_fd)
    with rasterio.open(
        tmp_full_path, "w",
        driver="GTiff", height=out_h, width=out_w,
        count=3, dtype=img_data.dtype, crs=crs, transform=tx,
    ) as dst:
        dst.write(img_data)

    print(f"[*] Loading Esri model…")
    model = ModelExtension.from_model(model_path)

    stride    = CHIP_SIZE - overlap
    row_starts = list(range(0, out_h, stride))
    col_starts = list(range(0, out_w, stride))
    total_chips = len(row_starts) * len(col_starts)
    print(f"[*] Tiling: {len(col_starts)} cols × {len(row_starts)} rows = {total_chips} chips")

    raw_detections = []
    chips_processed = 0
    chips_with_hits = 0

    for row0 in row_starts:
        for col0 in col_starts:
            row1 = min(row0 + CHIP_SIZE, out_h)
            col1 = min(col0 + CHIP_SIZE, out_w)

            chip = img_data[:, row0:row1, col0:col1]
            chip_h, chip_w = chip.shape[1], chip.shape[2]

            if chip_h < CHIP_SIZE or chip_w < CHIP_SIZE:
                padded_chip = np.zeros((3, CHIP_SIZE, CHIP_SIZE), dtype=chip.dtype)
                padded_chip[:, :chip_h, :chip_w] = chip
                chip = padded_chip
                chip_h, chip_w = CHIP_SIZE, CHIP_SIZE

            tmp_fd, tmp_chip = tempfile.mkstemp(suffix="_chip.tif")
            os.close(tmp_fd)
            try:
                with rasterio.open(
                    tmp_chip, "w",
                    driver="GTiff", height=chip_h, width=chip_w,
                    count=3, dtype=chip.dtype,
                ) as dst:
                    dst.write(chip)

                preds = model.predict(tmp_chip, threshold=conf_threshold)

                # Safely handle the SAM model tuple structure (Boxes and Labels, no Scores)
                if not preds or len(preds) == 0 or len(preds[0]) == 0:
                    bboxes = []
                    scores = []
                else:
                    bboxes = preds[0]
                    # Assign a placeholder score since the model drops it
                    scores = preds[2] if len(preds) > 2 else [conf_threshold] * len(bboxes)

            finally:
                if os.path.exists(tmp_chip):
                    os.unlink(tmp_chip)

            chips_processed += 1
            if not bboxes:
                continue

            chips_with_hits += 1
            for bbox, score in zip(bboxes, scores):
                xmin_c, ymin_c, w_b, h_b = [float(v) for v in bbox]

                # Slight inflation to capture full pool extent
                inflation = 1.1 
                new_w = w_b * inflation
                new_h = h_b * inflation

                xmin_c = xmin_c - ((new_w - w_b) / 2)
                ymin_c = ymin_c - ((new_h - new_h) / 2)

                xmin_f = col0 + xmin_c
                xmax_f = col0 + xmin_c + new_w
                ymin_f = row0 + ymin_c
                ymax_f = row0 + ymin_c + new_h

                cx = xmin_c + (w_b / 2)
                cy = ymin_c + (h_b / 2)
                if (col0 > 0 and cx < overlap / 2) or \
                   (row0 > 0 and cy < overlap / 2):
                    continue

                raw_detections.append({
                    "score": float(score),
                    "box":   (xmin_f, ymin_f, xmax_f, ymax_f),
                    "dims":  (w_b, h_b),
                })

        if (row_starts.index(row0) + 1) % 10 == 0:
            print(
                f"    Row {row_starts.index(row0)+1}/{len(row_starts)}  "
                f"raw_so_far={len(raw_detections)}"
            )

    if os.path.exists(tmp_full_path):
        os.unlink(tmp_full_path)

    print(f"[*] Tiling complete: {chips_processed} chips, "
          f"{chips_with_hits} had detections, {len(raw_detections)} raw detections")

    after_nms = _nms(raw_detections, iou_threshold=0.35)
    print(f"[*] After NMS: {len(after_nms)} detections")

    features   = []
    rej_area   = 0
    rej_aspect = 0
    rej_color  = 0

    for det in after_nms:
        xmin_f, ymin_f, xmax_f, ymax_f = det["box"]
        w_b, h_b = det["dims"]
        score    = det["score"]

        area_m2 = (w_b * cell_size) * (h_b * cell_size)
        if area_m2 < POOL_MIN_M2 or area_m2 > POOL_MAX_M2:
            rej_area += 1
            continue

        aspect = max(w_b, h_b) / max(min(w_b, h_b), 1e-6)
        if aspect > POOL_MAX_ASPECT:
            rej_aspect += 1
            continue

        r0, c0 = int(ymin_f), int(xmin_f)
        r1, c1 = int(ymax_f), int(xmax_f)
        blue_frac = _pool_color_fraction(img_rgb, r0, c0, r1, c1)
        if blue_frac < min_blue_fraction:
            rej_color += 1
            continue

        map_left,  map_top    = tx * (xmin_f, ymin_f)
        map_right, map_bottom = tx * (xmax_f, ymax_f)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [map_left,  map_bottom],
                    [map_right, map_bottom],
                    [map_right, map_top],
                    [map_left,  map_top],
                    [map_left,  map_bottom],
                ]],
            },
            "properties": {
                "Class":        "pool",
                "area_m2":      round(area_m2, 1),
                "confidence":   round(score, 4),
                "blue_fraction": round(blue_frac, 3),
                "debug_px":     f"col={int(xmin_f)}-{int(xmax_f)} row={int(ymin_f)}-{int(ymax_f)}",
            },
        })

    print(
        f"[*] Filter summary — "
        f"after_nms={len(after_nms)}  area_rej={rej_area}  "
        f"aspect_rej={rej_aspect}  color_rej={rej_color}  "
        f"accepted={len(features)}"
    )

    if features:
        print("[*] Accepted detections (for location verification):")
        for i, f in enumerate(features):
            p = f["properties"]
            print(
                f"    [{i}] {p['debug_px']}  "
                f"area={p['area_m2']}m²  "
                f"blue={p['blue_fraction']}  "
                f"conf={p['confidence']}"
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh, indent=2)

    print(f"[*] Saved {len(features)} pool detections → {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pool Detection — Esri FasterRCNN + explicit tiling")
    parser.add_argument("--image",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--model_path",  default=DEFAULT_MODEL)
    parser.add_argument("--threshold",   type=float, default=0.65,
                        help="Confidence threshold (0.65 gives more recall than 0.70)")
    parser.add_argument("--overlap",     type=int,   default=256)
    parser.add_argument("--cell_size",   type=float, default=0.15)
    parser.add_argument("--min_blue",    type=float, default=0.08,
                        help="Min fraction of bbox pixels that must be blue/cyan/turquoise")
    args = parser.parse_args()

    detect_pools(
        model_path         = args.model_path,
        image_path         = args.image,
        output_path        = args.output,
        conf_threshold     = args.threshold,
        cell_size          = args.cell_size,
        min_blue_fraction  = args.min_blue,
    )