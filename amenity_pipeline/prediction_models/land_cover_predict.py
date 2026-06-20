"""
Predict land cover classification using the ESRI Unet model.
Model: Unet + ResNet34, 9 classes, 512x512 tiles.
"""
import argparse
import json
import os
import warnings
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape, mapping

warnings.filterwarnings("ignore")

try:
    from arcgis.learn import UnetClassifier
except ImportError:
    raise ImportError("ArcGIS API for Python ('arcgis') is required to run the Unet model. Please install it using pip: pip install arcgis")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT, ".."))

TILE_SIZE = 512
MODEL_PATH = os.path.join(PIPELINE_ROOT, "extracted_models", "HighResolutionLandCoverClassification_USA.emd")

# Native dataset class mappings
CLASS_MAPPING = {
    1: "Water",
    2: "Wetlands",
    3: "Tree Canopy",
    4: "Shrubland",
    5: "Low Vegetation",
    6: "Barren",
    7: "Structures",
    8: "Impervious Surfaces",
    9: "Impervious Roads"
}

def load_model(device):
    print(f"Loading UnetClassifier from EMD: {MODEL_PATH}")
    unet = UnetClassifier.from_emd(data=None, emd_path=MODEL_PATH)
    model = unet.learn.model.to(device)
    model.eval()
    return model

def normalize_batch(batch_np):
    tensors = torch.from_numpy(batch_np).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensors - mean) / std

def run(image_path, overlap, cell_size, batch_size, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(device)
    print("Model loaded.")

    # Automatically load config.json directly from disk to read thresholds/weights
    classes_config = {}
    config_path = os.path.join(PIPELINE_ROOT, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                for m in config_data.get("models", []):
                    if m.get("name") == "land_cover":
                        classes_config = m.get("classes", {})
                        print("[SUCCESS] Loaded multiclass configurations directly from config.json")
                        break
        except Exception as e:
            print(f"[WARNING] Failed to parse config.json for thresholds: {e}")

    stride = TILE_SIZE - overlap

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
    src_crs = src.crs
    src_tx = tx
    global_map = np.zeros((h, w), dtype=np.uint8)
    # Accumulate per-class softmax probabilities for confidence-based fusion
    # LC model has 9 classes (indices 0..8, where class label = index+1 for 1-based classes,
    # but the model outputs indices 0-8 matching CLASS_MAPPING keys 1-9 directly via argmax)
    NUM_LC_CLASSES = 9
    prob_accumulator = np.zeros((NUM_LC_CLASSES, h, w), dtype=np.float32)
    weight_accumulator = np.zeros((h, w), dtype=np.float32)

    tile_positions = [
        (row, col)
        for row in range(0, h, stride)
        for col in range(0, w, stride)
    ]
    total = len(tile_positions)

    def make_tile(row, col):
        win_h = min(TILE_SIZE, h - row)
        win_w = min(TILE_SIZE, w - col)
        tile  = data[:, row:row+win_h, col:col+win_w].astype(np.float32)
        if win_h < TILE_SIZE or win_w < TILE_SIZE:
            pad = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
            pad[:, :win_h, :win_w] = tile
            tile = pad
        return tile, win_h, win_w

    for batch_start in range(0, total, batch_size):
        batch_pos   = tile_positions[batch_start:batch_start + batch_size]
        tiles       = [make_tile(r, c) for r, c in batch_pos]
        
        batch_np = np.stack([t[0] for t in tiles])
        tensor_batch = normalize_batch(batch_np).to(device)

        with torch.no_grad():
            outputs = model(tensor_batch) # Shape: (B, num_classes, H, W)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1) # Shape: (B, H, W)
            
            # Apply confidence thresholds loaded from config.json
            for c_idx, c_name in CLASS_MAPPING.items():
                c_info = classes_config.get(str(c_idx), classes_config.get(c_name, {}))
                thresh = c_info.get("threshold", 0.0)
                
                if thresh > 0.0 and c_idx < probs.shape[1]:
                    # Revert pixels below their confidence floor to background (0)
                    failed_mask = (preds == c_idx) & (probs[:, c_idx] < thresh)
                    preds[failed_mask] = 0

            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()  # Shape: (B, 9, H, W) — for prob raster accumulation


        for (row, col), (tile, win_h, win_w), pred_tile, prob_tile in zip(batch_pos, tiles, preds_np, probs_np):
            half = overlap // 2
            
            r0 = row + half if row > 0 else row
            c0 = col + half if col > 0 else col
            r1 = row + win_h - half if row + TILE_SIZE < h else row + win_h
            c1 = col + win_w - half if col + TILE_SIZE < w else col + win_w
            
            tr0 = r0 - row
            tc0 = c0 - col
            tr1 = r1 - row
            tc1 = c1 - col
            
            if r1 > r0 and c1 > c0:
                global_map[r0:r1, c0:c1] = pred_tile[tr0:tr1, tc0:tc1]
                # Accumulate probabilities (all 9 channels) for the non-overlapping core
                prob_accumulator[:, r0:r1, c0:c1] += prob_tile[1:, tr0:tr1, tc0:tc1]
                weight_accumulator[r0:r1, c0:c1] += 1.0

        done = min(batch_start + batch_size, total)
        if done % max(batch_size, 10) == 0 or done == total:
            print(f"  Processed tiles {done}/{total}")

    print("Generating polygons from seamless global map...")
    features = []
    fid = 1
    
    for geom_dict, val in shapes(global_map, mask=(global_map > 0), transform=tx):
        val = int(val)
        val_str = str(val)
        default_name = CLASS_MAPPING.get(val, f"Class_{val}")
        
        # Check if custom values exist inside the config.json dictionary structure
        if val_str in classes_config:
            class_name = classes_config[val_str].get("name", default_name)
            feature_weight = classes_config[val_str].get("weight", 0.0)
            feature_thresh = classes_config[val_str].get("threshold", 0.0)
        else:
            class_name = default_name
            feature_weight = 0.0
            feature_thresh = 0.0
        
        p = shape(geom_dict)
        if p.is_valid and not p.is_empty:
            features.append({
                "type": "Feature",
                "properties": {
                    "FID": fid, 
                    "Class": class_name, 
                    "Class_Index": val,
                    "Weight": feature_weight,
                    "Threshold_Applied": feature_thresh,
                    "Confidence_Passing": 1.0
                },
                "geometry": mapping(p)
            })
            fid += 1

    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    print(f"Done. {len(features)} land cover classification zones saved to: {output_path}")

    # Normalize accumulated probability raster and save as multi-band float GeoTIFF
    # Band N corresponds to LC class N (1=Water, 2=Wetlands, ..., 7=Structures, 8=Impervious, 9=Roads)
    try:
        weight_safe = np.maximum(weight_accumulator, 1.0)
        for c in range(NUM_LC_CLASSES):
            prob_accumulator[c] /= weight_safe
        prob_output_path = output_path.replace(".geojson", ".prob.tif")
        with rasterio.open(
            prob_output_path, "w",
            driver="GTiff",
            height=h, width=w,
            count=NUM_LC_CLASSES,
            dtype=rasterio.float32,
            crs=src_crs,
            transform=src_tx,
            compress="lzw"
        ) as dst:
            for c in range(NUM_LC_CLASSES):
                dst.write(prob_accumulator[c], c + 1)  # Bands are 1-indexed
        print(f"[LandCover] Probability raster ({NUM_LC_CLASSES} bands) saved to: {prob_output_path}")
    except Exception as e:
        print(f"[LandCover] WARNING: Could not save probability raster: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",         required=True)
    parser.add_argument("--overlap",        type=int,   default=256)
    parser.add_argument("--batch_size",     type=int,   default=4, help="Tiles per GPU batch")
    parser.add_argument("--cell_size",      type=float, default=0.3, help="Target resolution in map units")
    parser.add_argument("--output",         default=os.path.join(REPO_ROOT, "Results", "amenity_pipeline", "land_cover.geojson"))
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    run(args.image, args.overlap, args.cell_size, args.batch_size, args.output)