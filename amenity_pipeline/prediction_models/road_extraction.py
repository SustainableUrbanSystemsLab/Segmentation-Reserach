"""
Road extraction prediction script.
Uses Esri's MultiTaskRoadExtractor (North America model) to segment roads
from high-resolution satellite imagery (30-50cm) and exports to GeoJSON.
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

warnings.filterwarnings("ignore")

try:
    from arcgis.learn import MultiTaskRoadExtractor
except ImportError:
    print("[-] Missing 'arcgis' package. Ensure you are running this with the ArcGIS Pro Python interpreter (propy.bat).")
    sys.exit(1)

import torchvision.transforms as standard_transforms

# --- CLASS LABEL CONFIGURATION ---
# The North America model is a binary classifier: 1 = Road, 0 = Non-Road
CLASS_MAPPING = {
    1: "road"
}

TILE_SIZE = 1024

def load_model(emd_path, device):
    """Initializes the MultiTaskRoadExtractor from the Esri Model Definition (.emd)."""
    print(f"[i] Initializing Esri Road Extractor architecture on {device}...")
    
    # Load via the arcgis framework without passing a training databunch
    extractor = MultiTaskRoadExtractor.from_model(data=None, emd_path=emd_path)
    
    # Extract the raw underlying PyTorch model to plug into our custom tiling loop
    model = extractor.learn.model
    model.to(device)
    model.eval()
    
    print(f"[+] Loaded pretrained Road weights from {os.path.basename(emd_path)}")
    return model

def run_tiled_prediction(model, image_path, threshold, overlap, cell_size, device):
    """
    Reads input image, resamples to target cell_size (crucial for 30-50cm models),
    performs tiled prediction using raw [0, 255] float tensors, and returns the binary road mask.
    """
    with rasterio.open(image_path) as src:
        native_res = abs(src.transform.a)
        crs = src.crs
        
        if cell_size <= 0:
            cell_size = native_res
            w, h = src.width, src.height
            tx = src.transform
            print(f"Using native resolution: {native_res:.4f}m/px ({w}x{h})")
            img_data = src.read([1, 2, 3])
        else:
            scale = native_res / cell_size
            w, h = round(src.width * scale), round(src.height * scale)
            tx = Affine(cell_size, 0, src.transform.c,
                        0, -cell_size, src.transform.f)
            print(f"Resampling from native {native_res:.4f}m/px to target {cell_size:.4f}m/px ({src.width}x{src.height} -> {w}x{h})")
            img_data = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)

    print(f"[i] Running tiled inference (tile size: {TILE_SIZE}, overlap: {overlap})...")
    
    # Grid accumulators
    prob_accum = np.zeros((h, w), dtype=np.float32)
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
        win_h, win_w = min(TILE_SIZE, h - row), min(TILE_SIZE, w - col)
        tile = img_data[:, row:row+win_h, col:col+win_w]
        
        # Pad tile out to TILE_SIZE if it hits a boundary
        if win_h < TILE_SIZE or win_w < TILE_SIZE:
            padded_tile = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=img_data.dtype)
            padded_tile[:, :win_h, :win_w] = tile
            tile_input = padded_tile
        else:
            tile_input = tile
            
        with torch.no_grad():
            tile_input_hwc = tile_input.transpose(1, 2, 0) # Convert to HWC
            
            # FIX: Convert manually to avoid automatic [0, 1] division. 
            # Reorders to (C, H, W), casts to float32, and keeps values in [0, 255].
            tensor_input = torch.from_numpy(tile_input_hwc.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(tensor_input)
            
            # Unpack the verified segmentation logits head (Element 0)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            # Apply softmax across the two classes (0: background, 1: road)
            probs = torch.softmax(logits, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
            
            # Crop padding away
            probs = probs[:win_h, :win_w]
            
        prob_accum[row:row+win_h, col:col+win_w] += probs
        count_accum[row:row+win_h, col:col+win_w] += 1.0
        
        if (idx + 1) % max(1, total_tiles // 10) == 0 or (idx + 1) == total_tiles:
            print(f"   Processed {idx + 1}/{total_tiles} tiles...")
            
    # Normalize probabilities
    count_accum = np.maximum(count_accum, 1.0)
    avg_probs = prob_accum / count_accum
    
    # Binary threshold mapping
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    pred_mask[avg_probs >= threshold] = 1
    
    return pred_mask, tx, crs

def mask_to_geojson(pred_mask, transform, crs, output_path):
    """Vectorizes the binary road array into spatial geometries."""
    print("[i] Vectorizing road classification mask array to spatial geometries...")
    
    # Mask out background (0), keep only roads (1)
    geoms = shapes(pred_mask, mask=(pred_mask == 1), transform=transform)
    
    features = []
    for geo, val in geoms:
        class_id = int(val)
        class_name = CLASS_MAPPING.get(class_id, "unknown")
        
        # Filter out negligible noise polygons (less than ~10 pixels area)
        poly_shape = shape(geo)
        if poly_shape.area < 2.0:  
            continue

        if class_name != "unknown":
            features.append({
                "type": "Feature",
                "geometry": mapping(poly_shape),
                "properties": {
                    "Class": class_name,
                    "Class_ID": class_id
                }
            })
            
    if not features:
        print("[!] No road infrastructure elements detected above threshold.")
        gdf = gpd.GeoDataFrame(columns=["geometry", "Class", "Class_ID"], crs=crs)
    else:
        gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
        
    print(f"[+] Exporting {len(gdf)} vectorized road features to: {output_path}")
    gdf.to_file(output_path, driver="GeoJSON")

def main():
    parser = argparse.ArgumentParser(description="Esri Road Infrastructure Extraction")
    parser.add_argument("--image", required=True, help="Path to input satellite TIF imagery")
    parser.add_argument("--output", required=True, help="Path to save output road GeoJSON")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold filter")
    parser.add_argument("--overlap", type=int, default=256, help="Overlap size for tiling")
    parser.add_argument("--cell_size", type=float, default=0.3, help="Resampling cell size (Model tuned for 0.3-0.5m)")
    
    # Path configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_EMD = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "extracted_models", "RoadsExtraction_NorthAmerica.emd"))
    
    parser.add_argument("--emd", default=DEFAULT_EMD, help="Path to the extracted .emd metadata file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.emd):
        print(f"[-] Esri model definition file missing at: {args.emd}")
        print("[*] Please change the .dlpk extension to .zip, extract it, and point to the .emd file inside.")
        sys.exit(1)

    try:
        model = load_model(args.emd, device)
        pred_mask, transform, crs = run_tiled_prediction(
            model, args.image, args.threshold, args.overlap, args.cell_size, device
        )
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        mask_to_geojson(pred_mask, transform, crs, args.output)
        
        print("[+] Road prediction pipeline complete.")
    except Exception as e:
        import traceback
        print(f"[-] Execution stopped due to failure: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()