import os
import argparse
import json
import warnings
import rasterio

warnings.filterwarnings("ignore")

try:
    from arcgis.learn import ModelExtension
except ImportError:
    raise ImportError(
        "The 'arcgis' package is required to run Esri models. "
        "Install it in your environment using: conda install -c esri arcgis"
    )

# Dynamic path discovery matching your land cover model architecture
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT, ".."))

# DEFAULT POOL MODEL PATH
DEFAULT_MODEL_PATH = os.path.join(PIPELINE_ROOT, "extracted_models", "PoolDetection_USA.emd")


def detect_pools(model_path, image_path, output_path, conf_threshold):
    """
    Executes Esri's pool detection FasterRCNN model on a target high-res TIFF tile
    and exports standard GeoJSON polygons mapped to real-world coordinates.
    """
    print(f"[*] Loading Esri Pool Detection model from: {model_path}")
    model = ModelExtension.from_model(model_path)

    print(f"[*] Running inference on: {image_path}")
    predictions = model.predict(image_path, threshold=conf_threshold)

    # 1. EXTRACT THE GEOTRANSFORM FROM THE SOURCE IMAGE
    with rasterio.open(image_path) as src:
        transform = src.transform

    features = []
    # Handle the prediction tuple: typically (bboxes, labels, scores)
    if predictions and len(predictions) > 0:
        bboxes = predictions[0]
        scores = predictions[2] if len(predictions) > 2 else [conf_threshold] * len(bboxes)

        for bbox, score in zip(bboxes, scores):
            # Esri typically returns bboxes as [ymin, xmin, ymax, xmax] in PIXELS
            ymin_px, xmin_px, ymax_px, xmax_px = [float(coord) for coord in bbox]
            
            # 2. APPLY AFFINE TRANSFORM (Convert pixels -> real-world coordinates)
            # x is column (width), y is row (height)
            x_min, y_min = transform * (xmin_px, ymin_px)
            x_max, y_max = transform * (xmax_px, ymax_px)
            
            # Convert bounding box to a valid GeoJSON Polygon using mapped coordinates
            geom = {
                "type": "Polygon",
                "coordinates": [[
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                    [x_min, y_min]
                ]]
            }
            
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "Class": "pool",
                    "confidence": float(score)
                }
            })
            
    print(f"[*] Detected {len(features)} pools.")

    # Wrap in FeatureCollection
    geojson_out = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(geojson_out, f, indent=4)
        
    print(f"[*] Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pool Detection via ArcGIS FasterRCNN")
    
    # Standard Pipeline Arguments
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    # Model-Specific Arguments
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, 
                        help="Path to the Esri .dlpk or .emd model file")
    parser.add_argument("--threshold", type=float, default=0.5)
    
    # Catch arguments sent by the orchestrator so it doesn't crash
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--cell_size", type=float, default=0.15)

    args = parser.parse_args()

    detect_pools(
        model_path=args.model_path,
        image_path=args.image,
        output_path=args.output,
        conf_threshold=args.threshold
    )