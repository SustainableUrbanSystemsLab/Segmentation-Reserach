"""
Main orchestrator script for the amenity pipeline.
Loads config.json via pipeline_config and executes each enabled model's
prediction script in sequence, then triggers mask fusion and visualization.

To disable a model without deleting it, set "enabled": false in config.json.
"""
import os
import sys
import json
import argparse
import subprocess
from scipy.ndimage import binary_erosion

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

def resolve_path(path_str, relative_to):
    if not path_str:
        return None
    if os.path.isabs(path_str):
        return path_str
    # Try resolving relative to relative_to
    candidate = os.path.abspath(os.path.join(relative_to, path_str))
    if os.path.exists(candidate):
        return candidate
    # Try resolving relative to REPO_ROOT
    candidate_repo = os.path.abspath(os.path.join(REPO_ROOT, path_str))
    if os.path.exists(candidate_repo):
        return candidate_repo
    # Try resolving relative to SCRIPT_DIR
    candidate_script = os.path.abspath(os.path.join(SCRIPT_DIR, path_str))
    if os.path.exists(candidate_script):
        return candidate_script
    # Default to candidate relative to relative_to
    return candidate

def run_pipeline(config_path, image_override=None):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    print(f"Loading configuration from {config_path}")
    # Load structured config — single source of truth for weights/radii
    from pipeline_config import load_pipeline_config
    pipeline_cfg = load_pipeline_config(config_path)

    # Keep raw JSON for the subprocess arg paths
    with open(config_path, "r") as f:
        config = json.load(f)

    # Determine input image
    image_path = image_override or pipeline_cfg.pipeline_image
    if not image_path:
        raise ValueError("No input image specified in command line arguments or config.json")

    resolved_image = resolve_path(image_path, SCRIPT_DIR)
    if not os.path.exists(resolved_image):
        raise FileNotFoundError(f"Input image not found at {resolved_image}")
    print(f"Using input image: {resolved_image}")

    executed_outputs = {}

    # Run each enabled model
    for model_cfg in pipeline_cfg.enabled_models:
        name = model_cfg.name
        script_path = model_cfg.script_path
        output_path = model_cfg.output_path
        extra_args = model_cfg.extra_args

        if not script_path:
            print(f"Warning: Model '{name}' has no script_path specified. Skipping.")
            continue

        resolved_script = resolve_path(script_path, SCRIPT_DIR)
        if not os.path.exists(resolved_script):
            raise FileNotFoundError(f"Model script not found for '{name}' at {resolved_script}")

        resolved_output = resolve_path(output_path, REPO_ROOT)
        os.makedirs(os.path.dirname(resolved_output), exist_ok=True)

        print("\n" + "=" * 60)
        print(f"Evaluating model: {name.upper()}")
        print(f"Script:        {resolved_script}")
        print(f"Output:        {resolved_output}")
        print("=" * 60)

        interpreter = model_cfg.python_interpreter or sys.executable
        cmd = [interpreter, resolved_script, "--image", resolved_image, "--output", resolved_output]
        cmd.extend(extra_args)

        # ---------------------------------------------------------------------
        # SMART CACHING MECHANISM
        # Checks if output exists and arguments match the last successful run.
        # ---------------------------------------------------------------------
        meta_path = resolved_output + ".meta.json"
        cache_payload = {
            "image": resolved_image,
            "extra_args": extra_args
        }

        should_run = True
        if os.path.exists(resolved_output) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as mf:
                    previous_run = json.load(mf)
                if previous_run == cache_payload:
                    print(f"[+] CACHE HIT: Parameters unchanged for {name}. Skipping inference.")
                    should_run = False
            except Exception:
                pass # Meta file corrupted or unreadable, force rerun

        if should_run:
            print(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            # Save state to cache
            with open(meta_path, "w") as mf:
                json.dump(cache_payload, mf)

        if os.path.exists(resolved_output):
            executed_outputs[name] = resolved_output

    # =========================================================================
    # MASK FUSION STEP
    # Rasterizes all vector GeoJSON outputs into pixel masks, then fuses them
    # with the land cover classification into a single clean categorical raster.
    # =========================================================================
    print("\n" + "=" * 60)
    print("Running urban mask fusion...")
    print("=" * 60)
    try:
        import rasterio
        import geopandas as gpd
        import numpy as np
        from rasterio.features import rasterize
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
        from shapely.geometry import box
        sys.path.insert(0, SCRIPT_DIR)
        from mask_fusion import fuse_urban_masks
        from mask_fusion import generate_comfort_score_map

        # ----------------------------------------------------------------
        # Step 1: Build the reference raster grid from the input image.
        # ----------------------------------------------------------------
        TARGET_CELL = 0.3   # metres per pixel – must match land_cover output

        with rasterio.open(resolved_image) as src:
            native_res = abs(src.transform.a)
            crs = src.crs
            scale = native_res / TARGET_CELL
            ref_w = round(src.width  * scale)
            ref_h = round(src.height * scale)
            ref_tx = Affine(TARGET_CELL, 0, src.transform.c,
                            0, -TARGET_CELL, src.transform.f)

        print(f"[Fusion] Reference grid: {ref_w}x{ref_h} @ {TARGET_CELL}m/px  (CRS: {crs})")
        
        # ----------------------------------------------------------------
        # Helper: Load GeoDataFrame and handle CRS/coordinates
        # ----------------------------------------------------------------
        def load_gdf(path, crs_ref):
            if not path or not os.path.exists(path):
                return None
            try:
                gdf = gpd.read_file(path)
                if gdf.empty:
                    return gdf
                bounds = gdf.total_bounds
                is_meter_based = (
                    max(abs(bounds[0]), abs(bounds[2])) > 1000 or
                    max(abs(bounds[1]), abs(bounds[3])) > 1000
                )
                if is_meter_based:
                    gdf.crs = crs_ref
                else:
                    if gdf.crs is None:
                        gdf.crs = "EPSG:4326"
                    if crs_ref and gdf.crs != crs_ref:
                        gdf = gdf.to_crs(crs_ref)
                return gdf
            except Exception as e:
                print(f"[Fusion] Warning: could not load {path}: {e}")
            return None

        # ----------------------------------------------------------------
        # Step 2: Build the land cover integer array from GeoJSON.
        # ----------------------------------------------------------------
        lc_path = executed_outputs.get("land_cover")
        lc_array = np.zeros((ref_h, ref_w), dtype=np.uint8)
        if lc_path and os.path.exists(lc_path):
            print(f"[Fusion] Rasterizing land cover from: {lc_path}")
            lc_gdf = load_gdf(lc_path, crs)

            if lc_gdf is not None and not lc_gdf.empty:
                index_col = None
                for col in lc_gdf.columns:
                    if col.lower() in ["class_index", "class_id", "value", "val", "classindex"]:
                        index_col = col
                        break

                if index_col:
                    shapes_iter = (
                        (geom, int(val))
                        for geom, val in zip(lc_gdf.geometry, lc_gdf[index_col])
                        if geom is not None and not geom.is_empty
                    )
                    lc_array = rasterize(
                        shapes_iter,
                        out_shape=(ref_h, ref_w),
                        transform=ref_tx,
                        fill=0,
                        dtype=np.uint8
                    )
                else:
                    print("[Fusion] Warning: No class index column found in land cover features.")
            print(f"[Fusion]   LC unique values: {np.unique(lc_array).tolist()}")
        else:
            print("[Fusion] land_cover not run or missing. Using empty LC array.")

        # ----------------------------------------------------------------
        # Step 3: Rasterize each vector model output into a binary mask.
        # ----------------------------------------------------------------
        def rasterize_class(gdf, class_name_filter, h, w, tx, crs_ref):
            if gdf is None or gdf.empty:
                return np.zeros((h, w), dtype=np.uint8)
            
            filter_val = class_name_filter.lower().strip()
            
            def match_class(c_val):
                if not isinstance(c_val, str):
                    return False
                c_clean = c_val.lower().strip()
                return (
                    c_clean == filter_val or
                    c_clean + "s" == filter_val or
                    filter_val + "s" == c_clean or
                    c_clean == filter_val.rstrip('s')
                )
            
            if "Class" in gdf.columns:
                sub = gdf[gdf["Class"].apply(match_class)]
            elif "class" in gdf.columns:
                sub = gdf[gdf["class"].apply(match_class)]
            else:
                sub = gdf
                
            if sub.empty:
                return np.zeros((h, w), dtype=np.uint8)
                
            return rasterize(
                [(g, 1) for g in sub.geometry if g is not None and not g.is_empty],
                out_shape=(h, w),
                transform=tx,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )

        ped_gdf   = load_gdf(executed_outputs.get("pedestrian_infrastructure"), crs)
        bldg_gdf  = load_gdf(executed_outputs.get("buildings"), crs)
        bldg_reg  = executed_outputs.get("buildings", "").replace(".geojson", "_reg.geojson")
        if bldg_reg and os.path.exists(bldg_reg):
            bldg_gdf = load_gdf(bldg_reg, crs)
        cars_gdf  = load_gdf(executed_outputs.get("cars"), crs)
        pools_gdf = load_gdf(executed_outputs.get("pools"), crs)
        parking_gdf = load_gdf(executed_outputs.get("parking_lots"), crs)
        def rasterize_all(gdf, h, w, tx, crs_ref):
            if gdf is None or gdf.empty:
                return np.zeros((h, w), dtype=np.uint8)
            return rasterize(
                [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty],
                out_shape=(h, w),
                transform=tx,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )

        print("[Fusion] Rasterizing vector layers...")
        '''
        print("\nEXECUTED OUTPUTS:")
        for k, v in executed_outputs.items():
            print(f"{k} -> {v}")
        '''
        roads_gdf = load_gdf(executed_outputs.get("high_res_roads"), crs)  

        if ped_gdf is not None:
            print("\nPEDESTRIAN CLASSES:")
            if "Class" in ped_gdf.columns:
                print(ped_gdf["Class"].value_counts())
            elif "class" in ped_gdf.columns:
                print(ped_gdf["class"].value_counts())

        print("\nROAD GEOMETRY TYPES:")
        print(roads_gdf.geom_type.value_counts())

        #Adjust building logic for higher accuracy
        # 1. Rasterize the buildings normally first
        raw_building_mask = rasterize_all(bldg_gdf, ref_h, ref_w, ref_tx, crs)

        # 2. Apply the 1-pixel binary erosion recommendation (clears a 0.3m edge bleed)
        print("[Fusion] Applying binary erosion to building footprints...")
        cleaned_building_mask = binary_erosion(raw_building_mask, iterations=1).astype(np.uint8)


        # FIX: Align payload keys exactly with your JSON config and fusion lookups
        vector_masks_payload = {
            'sidewalk':        rasterize_class(ped_gdf,  "sidewalk",    ref_h, ref_w, ref_tx, crs),
            'ped_model_roads': rasterize_class(ped_gdf,  "road",        ref_h, ref_w, ref_tx, crs),
            'crosswalk':       rasterize_class(ped_gdf,  "crosswalk",   ref_h, ref_w, ref_tx, crs),
            'buildings':       cleaned_building_mask,
            'cars':            rasterize_all(cars_gdf,                  ref_h, ref_w, ref_tx, crs),
            'pools':           rasterize_all(pools_gdf,                 ref_h, ref_w, ref_tx, crs),
            'high_res_roads':  rasterize_all(roads_gdf,                 ref_h, ref_w, ref_tx, crs),
            'parking':         rasterize_all(parking_gdf,               ref_h, ref_w, ref_tx, crs)
        }
        for k, v in vector_masks_payload.items():
            print(f"[Fusion]   {k}: {v.sum()} pixels")

        # ----------------------------------------------------------------
        # Step 4: Run the matrix fusion AND comfort ecosystem generation
        # ----------------------------------------------------------------
        print("[Fusion] Running fuse_urban_masks...")
        clean_unified_map = fuse_urban_masks(lc_array, vector_masks_payload, cfg=pipeline_cfg)
        print(f"[Fusion] Fused map unique values: {np.unique(clean_unified_map).tolist()}")

        print("[Fusion] Generating comfort score heatmap...")
        from mask_fusion import generate_comfort_score_map
        comfort_heatmap = generate_comfort_score_map(clean_unified_map, pipeline_cfg)

        # ----------------------------------------------------------------
        # Step 5: Write both the fused raster and the comfort heatmap
        # ----------------------------------------------------------------
        output_dir = os.path.join(REPO_ROOT, "Results")
        os.makedirs(output_dir, exist_ok=True)
        
        fused_output_tiff = os.path.join(output_dir, "unified_clean_mask.tif")
        heatmap_output_tiff = os.path.join(output_dir, "comfort_heatmap.tif")

        # Write Categorical Mask
        with rasterio.open(
            fused_output_tiff, "w",
            driver="GTiff",
            height=ref_h, width=ref_w,
            count=1,
            dtype=rasterio.uint8,
            crs=crs,
            transform=ref_tx,
            compress="lzw"
        ) as dst:
            dst.write(clean_unified_map, 1)

        # Write Continuous Comfort Heatmap
        with rasterio.open(
            heatmap_output_tiff, "w",
            driver="GTiff",
            height=ref_h, width=ref_w,
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=ref_tx,
            compress="lzw"
        ) as dst:
            dst.write(comfort_heatmap.astype(rasterio.float32), 1)

        print(f"[Fusion] Fused mask saved to: {fused_output_tiff}")
        print(f"[Fusion] Comfort heatmap saved to: {heatmap_output_tiff}")

    except Exception as fusion_err:
        import traceback
        print(f"Warning: Mask fusion failed with error: {fusion_err}")
        traceback.print_exc()
        print("Moving to visualization.")

    # Trigger visualization
    visualize_script = os.path.join(SCRIPT_DIR, "visualize.py")
    if os.path.exists(visualize_script):
        print("\n" + "=" * 60)
        print("Running visualization and comfort score heatmap generation...")
        print("=" * 60)
        cmd_vis = [
            sys.executable, visualize_script,
            "--image", resolved_image,
            "--config", config_path
        ]
        print(f"Executing: {' '.join(cmd_vis)}")
        subprocess.run(cmd_vis, check=True)
    else:
        print("\nWarning: visualize.py not found in the pipeline directory.")

    # =========================================================================
    # ADDED: PIPELINE EVALUATION TRIGGER (WITH PATH FALLBACKS)
    # =========================================================================
    eval_candidates = [
        os.path.join(REPO_ROOT, "evaluate_pipeline.py"),
        os.path.join(SCRIPT_DIR, "evaluate_pipeline.py")
    ]
    
    evaluate_script = next((p for p in eval_candidates if os.path.exists(p)), None)

    if evaluate_script:
        print("\n" + "=" * 60)
        print("Running CVAT Ground Truth Evaluation...")
        print("=" * 60)
        
        tile_name = os.path.basename(resolved_image)
        
        cmd_eval = [
            sys.executable, evaluate_script,
            tile_name
        ]
        
        print(f"Executing: {' '.join(cmd_eval)}")
        subprocess.run(cmd_eval)
    else:
        print("\nWarning: evaluate_pipeline.py not found in REPO_ROOT or SCRIPT_DIR. Skipping evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for Amenity Detection Pipeline")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.json"),
                        help="Path to pipeline config.json")
    parser.add_argument("--image", default=None,
                        help="Override input TIF image path")
    args = parser.parse_args()

    try:
        run_pipeline(args.config, args.image)
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"\nPipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)