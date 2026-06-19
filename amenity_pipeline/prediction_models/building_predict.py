"""
Predict building footprints using the ESRI Mask R-CNN model.
Model: MaskRCNN + ResNet50 FPN, single class (buildings), 512x512 tiles.

Fuses overlapping windows via spatial union and optionally regularizes 
borders to strict right angles.

Usage:
    uv run python building_predict.py --image input/your_image.tif --regularize
"""
import argparse
import json
import os
import warnings
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.features import shapes
from rasterio.enums import Resampling
from shapely.geometry import shape, mapping, Polygon
from shapely.affinity import rotate
from shapely.ops import unary_union
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT, ".."))

TILE_SIZE = 512
MODEL_PATH = os.path.join(PIPELINE_ROOT, "extracted_models", "usa_building_footprints.pth")
NUM_CLASSES = 2


def build_model():
    # min_size/max_size=512: prevents torchvision's internal transform from upscaling
    # tiles before the backbone. Model was trained on fixed 512x512 tiles.
    model = maskrcnn_resnet50_fpn(weights=None, min_size=512, max_size=512)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, NUM_CLASSES)
    return model


def remap_keys(state_dict):
    # rpn.head.conv changed from single Conv2d to Sequential in torchvision 0.14
    mapping = {
        "rpn.head.conv.weight": "rpn.head.conv.0.0.weight",
        "rpn.head.conv.bias":   "rpn.head.conv.0.0.bias",
    }
    return {mapping.get(k, k): v for k, v in state_dict.items()}


def load_model(device):
    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    state = remap_keys(state)
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}")
    model.to(device).eval()
    return model


def pixel_transform(src_transform, col_off, row_off):
    tx = src_transform
    return Affine(tx.a, tx.b, tx.c + col_off * tx.a,
                  tx.d, tx.e, tx.f + row_off * tx.e)


def mask_to_shapely(mask_np, tile_tx):
    m = (mask_np > 0.5).astype(np.uint8)
    polys = []
    for geom_dict, val in shapes(m, mask=m, transform=tile_tx):
        if val != 1:
            continue
        p = shape(geom_dict)
        if p.is_valid and not p.is_empty:
            polys.append(p)
    return polys


def dominant_angle(coords):
    """Angle of the longest edge, used as the building's principal direction."""
    edges = np.diff(np.vstack([coords, coords[0]]), axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    i = np.argmax(lengths)
    return np.degrees(np.arctan2(edges[i, 1], edges[i, 0]))


def snap_to_right_angles(coords):
    """
    Snap each vertex so that edges become axis-aligned.
    For each vertex: if the incoming edge is more horizontal, lock y to the
    previous vertex's y; if more vertical, lock x to the previous vertex's x.
    One pass is enough for convex shapes; two passes handles concave ones.
    """
    n = len(coords)
    c = coords.copy()
    for _ in range(2):
        for i in range(n):
            prev = c[(i - 1) % n]
            curr = c[i].copy()
            e = curr - prev
            if abs(e[0]) >= abs(e[1]):   # incoming edge is more horizontal
                curr[1] = prev[1]
            else:                        # incoming edge is more vertical
                curr[0] = prev[0]
            c[i] = curr
    return c


def regularize_polygon(poly, simplify_tol):
    poly = poly.simplify(simplify_tol, preserve_topology=True)
    if not poly.is_valid or poly.is_empty or poly.geom_type != "Polygon":
        return poly

    coords = np.array(poly.exterior.coords[:-1])
    if len(coords) < 3:
        return poly

    angle = dominant_angle(coords)
    centroid = poly.centroid

    # Rotate so the dominant direction aligns with x-axis
    rotated = rotate(poly, -angle, origin=centroid)
    rc = np.array(rotated.exterior.coords[:-1], dtype=float)

    snapped = snap_to_right_angles(rc)
    snapped = np.vstack([snapped, snapped[0]])

    reg = Polygon(snapped)
    if not reg.is_valid:
        reg = reg.buffer(0)
    if reg.is_empty:
        return poly

    # Rotate back to original orientation
    return rotate(reg, angle, origin=centroid)


def run(image_path, threshold, overlap, cell_size, batch_size, tolerance, regularize, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(device)
    print("Model loaded.")

    stride = TILE_SIZE - overlap
    all_polys = []

    with rasterio.open(image_path) as src:
        native_res = abs(src.transform.a)
        if cell_size <= 0:
            cell_size = native_res
            w = src.width
            h = src.height
            tx = src.transform
            print(f"Using native resolution: {native_res:.4f}m/px  ({w}x{h})")
            data = src.read([1, 2, 3])
        else:
            scale = native_res / cell_size
            w = round(src.width  * scale)
            h = round(src.height * scale)
            tx = Affine(cell_size, 0, src.transform.c,
                        0, -cell_size, src.transform.f)
            print(f"Native res: {native_res:.4f} -> resampling to {cell_size}m/px  ({src.width}x{src.height} -> {w}x{h})")
            data = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)

    print(f"Image: {w}x{h}, CRS: {src.crs}")

    # Build list of all tile positions
    tile_positions = [
        (row, col)
        for row in range(0, h, stride)
        for col in range(0, w, stride)
    ]
    total = len(tile_positions)

    def make_tile(row, col):
        win_h = min(TILE_SIZE, h - row)
        win_w = min(TILE_SIZE, w - col)
        tile  = data[:, row:row+win_h, col:col+win_w].astype(np.float32) / 255.0
        if win_h < TILE_SIZE or win_w < TILE_SIZE:
            pad = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
            pad[:, :win_h, :win_w] = tile
            tile = pad
        return tile, win_h, win_w

    print("Running batch sliding-window inference...")
    for batch_start in range(0, total, batch_size):
        batch_pos   = tile_positions[batch_start:batch_start + batch_size]
        tiles       = [make_tile(r, c) for r, c in batch_pos]
        tensor_batch = torch.stack([torch.from_numpy(t[0]) for t in tiles]).to(device)

        with torch.no_grad():
            outputs = model(list(tensor_batch))

        for (row, col), (tile, win_h, win_w), out in zip(batch_pos, tiles, outputs):
            scores  = out["scores"].cpu().numpy()
            masks   = out["masks"].cpu().numpy()
            tile_tx = pixel_transform(tx, col, row)
            half    = overlap // 2
            cx0, cx1 = col + half, col + win_w - half
            cy0, cy1 = row + half, row + win_h - half

            for score, mask in zip(scores, masks):
                if score < threshold:
                    continue
                for p in mask_to_shapely(mask[0], tile_tx):
                    px_col = (p.centroid.x - tx.c) / tx.a
                    px_row = (p.centroid.y - tx.f) / tx.e
                    if cx0 <= px_col < cx1 and cy0 <= px_row < cy1:
                        all_polys.append(p)

        done = min(batch_start + batch_size, total)
        if done % max(batch_size, 40) == 0 or done == total:
            print(f"  Tiles {done}/{total}, raw footprints found: {len(all_polys)}")

    if not all_polys:
        print("No buildings detected.")
        return

    # --- GEOMETRIC DISSOLVE/UNION STEP ---
    print(f"Dissolving {len(all_polys)} overlapping building windows into distinct blocks...")
    merged_geom = unary_union(all_polys)
    
    dissolved_polys = []
    if merged_geom.geom_type == "Polygon":
        dissolved_polys.append(merged_geom)
    elif merged_geom.geom_type == "MultiPolygon":
        dissolved_polys.extend(list(merged_geom.geoms))
        
    print(f"Fusing completed. Compiling {len(dissolved_polys)} standalone building layers...")

    # --- GENERATE RAW (UNREGULARIZED) OUTPUT ---
    raw_features = [
        {
            "type": "Feature",
            "properties": {"FID": i + 1, "Class": "building", "Confidence": 1.0},
            "geometry": mapping(poly)
        }
        for i, poly in enumerate(dissolved_polys)
    ]
    with open(output_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": raw_features}, f)
    print(f"[SUCCESS] Saved merged raw assets to: {output_path}")

    # --- CONDITIONAL RIGHT-ANGLE REGULARIZATION PASS ---
    if regularize:
        reg_output_path = output_path.replace(".geojson", "_reg.geojson")
        print("Running structural right-angle regularization...")
        reg_features = []
        skipped = 0
        
        for i, poly in enumerate(dissolved_polys):
            polys_to_process = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
            reg_sub_polys = []
            
            for p in polys_to_process:
                r = regularize_polygon(p, tolerance)
                if r and not r.is_empty:
                    reg_sub_polys.append(r)
                    
            if not reg_sub_polys:
                skipped += 1
                continue
                
            combined_res = reg_sub_polys[0] if len(reg_sub_polys) == 1 else reg_sub_polys[0].union(*reg_sub_polys[1:])
            reg_features.append({
                "type": "Feature",
                "properties": {"FID": i + 1, "Class": "building", "Confidence": 1.0},
                "geometry": mapping(combined_res)
            })

        with open(reg_output_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": reg_features}, f)
        print(f"[SUCCESS] Saved {len(reg_features)} right-angle regularized assets ({skipped} skipped) to: {reg_output_path}")
    else:
        print("[INFO] Regularization flag not set. Skipping geometric regularization pass.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",         required=True)
    parser.add_argument("--threshold",     type=float, default=0.9)
    parser.add_argument("--overlap",       type=int,   default=256)
    parser.add_argument("--batch_size",    type=int,   default=4,
                        help="Tiles per GPU batch — increase for larger images (try 8 or 16)")
    parser.add_argument("--cell_size",     type=float, default=0.3,
                        help="Target resolution in map units (metres for EPSG:3857)")
    parser.add_argument("--tolerance",    type=float, default=1.0,
                        help="Simplification tolerance in map units before snapping")
    parser.add_argument("--regularize",    action="store_true",
                        help="Enable structural right-angle post-processing and output a alternate '_reg.geojson' layout file")
    parser.add_argument("--output",        default=os.path.join(REPO_ROOT, "Results", "amenity_pipeline", "buildings.geojson"))
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    run(
        image_path=args.image,
        threshold=args.threshold,
        overlap=args.overlap,
        cell_size=args.cell_size,
        batch_size=args.batch_size,
        tolerance=args.tolerance,
        regularize=args.regularize,
        output_path=args.output
    )