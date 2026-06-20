"""
Pedestrian infrastructure prediction script.
Uses HRNetV2 + OCR (Tile2Net) to segment high-resolution satellite imagery
and exports detected assets (sidewalks, crosswalks, roads) to GeoJSON.
Supports tiling with overlap and cell-size resampling.
"""
import os
import sys
import argparse
import warnings
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import geopandas as gpd
import json

warnings.filterwarnings("ignore")

# Ensure local imports work and we can import from the extracted_models directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "extracted_models"))
if EXTRACTED_MODELS_DIR not in sys.path:
    sys.path.insert(0, EXTRACTED_MODELS_DIR)

from pedestrian_infra_utils import NumpyToTensor, forgiving_state_restore
from pedestrian_infra_model import ImageBasedCrossEntropyLoss2d, HRNet_Mscale
import torchvision.transforms as standard_transforms

# --- CLASS LABEL CONFIGURATION ---
CLASS_MAPPING = {
    0: "sidewalk",
    1: "road",
    2: "crosswalk"
}

TILE_SIZE = 1024

def load_model(weights_path, device):
    """Initializes HRNet_Mscale and loads the weights."""
    print(f"[i] Initializing HRNet_Mscale model on {device}...")
    
    # Initialize loss/criterion required by model structure
    criterion = ImageBasedCrossEntropyLoss2d(
        classes=0,
        ignore_index=-1,
        upper_bound=1.0, 
        fp16=False
    ).to(device)
    
    # Instantiate the model
    model = HRNet_Mscale(num_classes=4, criterion=criterion)
    model.to(device)
    
    print(f"[i] Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint else checkpoint['state_dict']
    
    model = forgiving_state_restore(model, state_dict)
    model.eval()
    return model

def run_tiled_prediction(model, image_path, threshold, overlap, cell_size, device):
    """
    Reads input image, resamples if cell_size is specified, performs tiled prediction,
    and returns prediction mask, transform, and CRS.
    """
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    val_input_transform = standard_transforms.Compose([
        NumpyToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    with rasterio.open(image_path) as src:
        native_res = abs(src.transform.a)
        crs = src.crs
        if cell_size <= 0:
            cell_size = native_res
            w = src.width
            h = src.height
            tx = src.transform
            print(f"Using native resolution: {native_res:.4f}m/px ({w}x{h})")
            img_data = src.read([1, 2, 3])
        else:
            scale = native_res / cell_size
            w = round(src.width * scale)
            h = round(src.height * scale)
            tx = Affine(cell_size, 0, src.transform.c,
                        0, -cell_size, src.transform.f)
            print(f"Resampling from native {native_res:.4f}m/px to target {cell_size:.4f}m/px ({src.width}x{src.height} -> {w}x{h})")
            img_data = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)

    print(f"[i] Running tiled inference (tile size: {TILE_SIZE}, overlap: {overlap})...")
    
    # Grid accumulators
    # 4 classes: 0=sidewalk, 1=road, 2=crosswalk, 3=others
    prob_accum = np.zeros((4, h, w), dtype=np.float32)
    count_accum = np.zeros((h, w), dtype=np.float32)
    
    stride = TILE_SIZE - overlap
    if stride <= 0:
        stride = TILE_SIZE // 2
        
    tile_positions = [
        (row, col)
        for row in range(0, h, stride)
        for col in range(0, w, stride)
    ]
    total_tiles = len(tile_positions)
    
    for idx, (row, col) in enumerate(tile_positions):
        win_h = min(TILE_SIZE, h - row)
        win_w = min(TILE_SIZE, w - col)
        
        tile = img_data[:, row:row+win_h, col:col+win_w]
        
        # Pad tile to TILE_SIZE x TILE_SIZE if needed
        if win_h < TILE_SIZE or win_w < TILE_SIZE:
            padded_tile = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=img_data.dtype)
            padded_tile[:, :win_h, :win_w] = tile
            tile_input = padded_tile
        else:
            tile_input = tile
            
        # Transform and predict
        with torch.no_grad():
            tensor_input = val_input_transform(tile_input)
            tensor_input = tensor_input.unsqueeze(0).to(device)
            
            # Dummy ground truth required by forward pass of MscaleOCR
            inputs = {
                'images': tensor_input,
                'gts': torch.randn(1, 3, TILE_SIZE, TILE_SIZE).to(device)
            }
            
            output_dict = model(inputs)
            pred = output_dict['pred'] # Shape: (1, 4, 1024, 1024)
            probs = torch.softmax(pred, dim=1).squeeze(0).cpu().numpy() # Shape: (4, 1024, 1024)
            
            # Crop back to original tile size
            probs = probs[:, :win_h, :win_w]
            
        prob_accum[:, row:row+win_h, col:col+win_w] += probs
        count_accum[row:row+win_h, col:col+win_w] += 1.0
        
        if (idx + 1) % max(1, total_tiles // 10) == 0 or (idx + 1) == total_tiles:
            print(f"  Processed {idx + 1}/{total_tiles} tiles...")
            
    # Normalize probabilities
    count_accum = np.maximum(count_accum, 1.0)
    avg_probs = prob_accum / count_accum
    
    # Binary threshold mapping: background class is index 3
    pred_mask = np.argmax(avg_probs, axis=0).astype(np.uint8)
    
    # Filter out weak detections below threshold (assign to background/others = 3)
    max_probs = np.max(avg_probs, axis=0)
    pred_mask[max_probs < threshold] = 3
    
    return pred_mask, avg_probs, tx, crs

def mask_to_geojson(pred_mask, transform, crs, output_path):
    """Vectorizes the classified numpy raster mask into spatial geometries."""
    print("[i] Vectorizing classification mask array to spatial geometries...")
    # mask = (pred_mask < 3) means we only keep values 0, 1, 2 (sidewalk, road, crosswalk)
    geoms = shapes(pred_mask, mask=(pred_mask < 3), transform=transform)
    
    features = []
    for geo, val in geoms:
        class_id = int(val)
        class_name = CLASS_MAPPING.get(class_id, "unknown")
        
        if class_name != "unknown":
            features.append({
                "type": "Feature",
                "geometry": mapping(shape(geo)),
                "properties": {
                    "Class": class_name,
                    "Class_ID": class_id
                }
            })
            
    if not features:
        print("[!] No pedestrian infrastructure assets detected above threshold.")
        gdf = gpd.GeoDataFrame(columns=["geometry", "Class", "Class_ID"], crs=crs)
    else:
        gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
        
    print(f"[+] Exporting {len(gdf)} vectorized features to: {output_path}")
    gdf.to_file(output_path, driver="GeoJSON")

def main():
    parser = argparse.ArgumentParser(description="Inference Script for Pedestrian Infrastructure")
    parser.add_argument("--image", required=True, help="Path to input satellite TIF imagery")
    parser.add_argument("--output", required=True, help="Path to save output GeoJSON results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold filter")
    parser.add_argument("--overlap", type=int, default=512, help="Overlap size for tiling")
    parser.add_argument("--cell_size", type=float, default=0.2, help="Resampling cell size in map units")
    parser.add_argument("--weights", default=os.path.join(EXTRACTED_MODELS_DIR, "satellite_2021.pth"),
                        help="Path to trained model weights checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.weights):
        print(f"[-] Model weights file not found at {args.weights}")
        sys.exit(1)

    try:
        model = load_model(args.weights, device)
        pred_mask, avg_probs, transform, crs = run_tiled_prediction(
            model, args.image, args.threshold, args.overlap, args.cell_size, device
        )
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        # Save per-class pedestrian probability raster for confidence-based fusion
        # Band 1=sidewalk, Band 2=road (ped model), Band 3=crosswalk
        try:
            prob_output_path = args.output.replace(".geojson", ".prob.tif")
            with rasterio.open(
                prob_output_path, "w",
                driver="GTiff",
                height=avg_probs.shape[1], width=avg_probs.shape[2],
                count=3,
                dtype=rasterio.float32,
                crs=crs,
                transform=transform,
                compress="lzw"
            ) as dst:
                dst.write(avg_probs[0].astype(np.float32), 1)  # sidewalk
                dst.write(avg_probs[1].astype(np.float32), 2)  # road
                dst.write(avg_probs[2].astype(np.float32), 3)  # crosswalk
            print(f"[Pedestrian] Probability raster saved to: {prob_output_path}")
        except Exception as prob_err:
            print(f"[Pedestrian] WARNING: Could not save probability raster: {prob_err}")

        mask_to_geojson(pred_mask, transform, crs, args.output)
        
        print("[+] Pedestrian prediction processing complete.")
    except Exception as e:
        import traceback
        print(f"[-] Execution stopped due to failure: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()