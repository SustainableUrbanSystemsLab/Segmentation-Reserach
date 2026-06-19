"""
Predict tree canopy using the ESRI DeepForest + SAM model.
Model: RetinaNet + ResNet50 FPN + SAM, single class (trees), 400x400 tiles.
Output: Feature class containing separate irregular polygon masks for each tree.
"""
import argparse
import json
import os
import warnings
import numpy as np
import torch
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely.geometry import shape, mapping
from torchvision.models.detection.retinanet import RetinaNet, retinanet_resnet50_fpn
from torchvision.ops import nms

# Import SAM Components
from segment_anything import sam_model_registry, SamPredictor

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT, ".."))

TILE_SIZE = 400
MODEL_PATH = os.path.join(PIPELINE_ROOT, "extracted_models", "NEON.pt")
NUM_CLASSES = 1

def build_model():
    try:
        backbone = retinanet_resnet50_fpn(weights=None).backbone
    except TypeError:
        backbone = retinanet_resnet50_fpn(pretrained=False).backbone
    model = RetinaNet(backbone=backbone, num_classes=NUM_CLASSES)
    return model

def load_model(device):
    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}")
    model.to(device).eval()
    return model

def pixel_transform(src_transform, col_off, row_off):
    tx = src_transform
    return Affine(tx.a, tx.b, tx.c + col_off * tx.a,
                  tx.d, tx.e, tx.f + row_off * tx.e)

def run(image_path, threshold, overlap, cell_size, batch_size, nms_overlap, sam_checkpoint, sam_type, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Object Detector
    model = load_model(device)
    print("RetinaNet Detector loaded.")
    
    # 2. Initialize SAM Predictor
    print(f"Loading SAM ({sam_type}) from {sam_checkpoint}...")
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("SAM Predictor initialized.")

    stride = TILE_SIZE - overlap
    all_polys, all_scores = [], []

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
            boxes   = out["boxes"].cpu().numpy()
            
            tile_tx = pixel_transform(tx, col, row)
            half    = overlap // 2
            cx0, cx1 = col + half, col + win_w - half
            cy0, cy1 = row + half, row + win_h - half

            # Prepare image for SAM (expects HWC uint8 [0, 255])
            tile_np = (tile.transpose(1, 2, 0) * 255).astype(np.uint8)
            sam_predictor.set_image(tile_np)

            for idx, (score, box) in enumerate(zip(scores, boxes)):
                if score < threshold:
                    continue
                
                # Check tile-edge boundary using the anchor box centroid
                xmin, ymin, xmax, ymax = box
                bx_cx = (xmin + xmax) / 2.0 + col
                bx_cy = (ymin + ymax) / 2.0 + row
                
                if not (cx0 <= bx_cx < cx1 and cy0 <= bx_cy < cy1):
                    continue

                # Pass the detector's bounding box coordinates directly into SAM
                masks, _, _ = sam_predictor.predict(
                    box=box,
                    multimask_output=False
                )
                
                if masks is not None and len(masks) > 0:
                    mask_img = masks[0].astype(np.uint8)
                    shape_generator = shapes(mask_img, mask=mask_img, transform=tile_tx)
                    
                    for geom, val in shape_generator:
                        if val == 1: 
                            poly_shape = shape(geom)
                            if poly_shape.is_valid and not poly_shape.is_empty:
                                all_polys.append(poly_shape)
                                all_scores.append(float(score))

        done = min(batch_start + batch_size, total)
        if done % max(batch_size, 10) == 0 or done == total:
            print(f"  Tiles {done}/{total}, segments extracted so far: {len(all_polys)}")

    # Apply global NMS based on bounding extents
    if all_polys and len(all_polys) > 0:
        print(f"Applying NMS with threshold {nms_overlap} on {len(all_polys)} tree segments...")
        boxes_map = []
        for p in all_polys:
            bounds = p.bounds 
            boxes_map.append([bounds[0], bounds[1], bounds[2], bounds[3]])
        
        boxes_tensor = torch.tensor(boxes_map, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        
        keep_idxs = nms(boxes_tensor, scores_tensor, nms_overlap).numpy()
        all_polys = [all_polys[i] for i in keep_idxs]
        all_scores = [all_scores[i] for i in keep_idxs]
        print(f"NMS complete. Kept {len(all_polys)} refined tree objects.")

    features = [
        {
            "type": "Feature",
            "properties": {"FID": i + 1, "Class": "trees", "Confidence": score},
            "geometry": mapping(poly)
        }
        for i, (poly, score) in enumerate(zip(all_polys, all_scores))
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    print(f"Done. {len(features)} detailed tree masks saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--threshold", type=float, default=0.4) 
    parser.add_argument("--overlap",   type=int,   default=200)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--cell_size", type=float, default=0.10) 
    parser.add_argument("--nms_overlap", type=float, default=0.15)
    
    # Added arguments to direct the script to your downloaded SAM weights file
    # Updated to include the /models/ folder and the correct 01ec64 suffix
    parser.add_argument("--sam_checkpoint", default=os.path.join(PIPELINE_ROOT, "extracted_models", "segment-anything", "models", "sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam_type", default="vit_b", help="vit_h, vit_l, or vit_b")
    
    parser.add_argument("--output",    default=os.path.join(REPO_ROOT, "Results", "amenity_pipeline", "trees.geojson"))
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    run(args.image, args.threshold, args.overlap, args.cell_size, args.batch_size, args.nms_overlap, 
        args.sam_checkpoint, args.sam_type, args.output)