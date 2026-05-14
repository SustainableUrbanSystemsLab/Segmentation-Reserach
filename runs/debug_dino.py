#!/usr/bin/env python3
"""Debug script to isolate DINO detection issues"""

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("[DEBUG] Imports starting...", flush=True)

try:
    from models import config as cfg
    print("[DEBUG] Config imported", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import config: {e}", flush=True)
    sys.exit(1)

try:
    from models.dino_processing import run_dino_prompts
    print("[DEBUG] dino_processing imported", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import dino_processing: {e}", flush=True)
    sys.exit(1)

try:
    from models.image_processing import load_rgb_image, report_geotiff_spatial_info
    print("[DEBUG] image_processing imported", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import image_processing: {e}", flush=True)
    sys.exit(1)

# Load test image
print("[DEBUG] Loading test image...", flush=True)
try:
    tif_path = cfg.tif_file
    report_geotiff_spatial_info(tif_path)
    img_np = load_rgb_image(tif_path)
    print(f"[DEBUG] Image loaded: shape={img_np.shape}", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to load image: {e}", flush=True)
    sys.exit(1)

# Try to initialize DINO
print("[DEBUG] Initializing DINO...", flush=True)
try:
    from models.dino_processing import build_dino_model_and_transform
    dino_model, dino_transform, device = build_dino_model_and_transform(cfg)
    print(f"[DEBUG] DINO model loaded on {device}", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to initialize DINO: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try to run DINO detection
print("[DEBUG] Running DINO detection...", flush=True)
try:
    print(f"[DEBUG] Config prompts: {len(cfg.dino_prompt_configs)}", flush=True)
    if not cfg.dino_prompt_configs:
        print("[ERROR] No prompts configured!", flush=True)
        sys.exit(1)
    
    print("[DEBUG] About to call run_dino_prompts", flush=True)
    result = run_dino_prompts(
        img_np,
        cfg,
        dino_model,
        dino_transform,
        device,
        pixels_per_meter_sq=None,
        return_unfiltered=True,
    )
    print(f"[DEBUG] DINO returned: {type(result)}", flush=True)
    if isinstance(result, tuple):
        dino_records, unfiltered, filtered = result
        print(f"[DEBUG] Records: kept={len(dino_records)}, unfiltered={len(unfiltered)}, filtered={len(filtered)}", flush=True)
    else:
        print(f"[DEBUG] Got simple list with {len(result)} records", flush=True)
        
except Exception as e:
    print(f"[ERROR] DINO detection failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[DEBUG] SUCCESS - Debug script completed", flush=True)
