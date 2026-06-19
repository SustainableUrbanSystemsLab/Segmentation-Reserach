"""
Predict car locations using the ESRI Car Detection Mask R-CNN model.
Model: MaskRCNN + ResNet50 FPN, single class (cars), 400x400 tiles.
"""
import argparse
import json
import os
import warnings
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape, mapping

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT, ".."))

TILE_SIZE = 400
MODEL_PATH = os.path.join(PIPELINE_ROOT, "extracted_models", "CarDetection_USA.pth")
NUM_CLASSES = 2

def build_model():
    # Model was trained on fixed 400x400 tiles.
    model = maskrcnn_resnet50_fpn(weights=None, min_size=TILE_SIZE, max_size=TILE_SIZE)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, NUM_CLASSES)
    return model

def remap_keys(state_dict):
    # Support torchvision version differences for RPN heads
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

def run(image_path, threshold, overlap, cell_size, batch_size, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(device)
    print("Model loaded.")

    stride = TILE_SIZE - overlap
    all_polys, all_scores = [], []

    with rasterio.open(image_path) as src:
        native_res = abs(src.transform.a)
        if cell_size <= 0:
            # Use native resolution directly
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
                        all_scores.append(float(score))

        done = min(batch_start + batch_size, total)
        if done % max(batch_size, 10) == 0 or done == total:
            print(f"  Tiles {done}/{total}, detections so far: {len(all_polys)}")

    features = [
        {
            "type": "Feature",
            "properties": {"FID": i + 1, "Class": "car", "Confidence": score},
            "geometry": mapping(poly)
        }
        for i, (poly, score) in enumerate(zip(all_polys, all_scores))
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    print(f"Done. {len(features)} cars saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlap",    type=int,   default=200)
    parser.add_argument("--batch_size", type=int,   default=4,
                        help="Tiles per GPU batch")
    parser.add_argument("--cell_size", type=float, default=0.3,
                        help="Target resolution in map units (use 0 for native resolution)")
    parser.add_argument("--output",    default=os.path.join(REPO_ROOT, "Results", "amenity_pipeline", "cars.geojson"))
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    run(args.image, args.threshold, args.overlap, args.cell_size, args.batch_size, args.output)
