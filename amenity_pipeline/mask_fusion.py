"""
Urban mask fusion & Comfort Ecosystem Generator.
Strictly adheres to the 11-class schema defined in your config.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict
from pipeline_config import FinalClass, PipelineConfig

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: GEOMETRIC MASK FUSION (Crisp Categorical Raster)
# ─────────────────────────────────────────────────────────────────────────────
from scipy.ndimage import binary_erosion

def fuse_urban_masks(
    lc_array: np.ndarray,
    vector_masks_dict: Dict[str, np.ndarray],
    cfg: PipelineConfig | None = None,
) -> np.ndarray:
    h, w = lc_array.shape
    final_map = np.zeros((h, w), dtype=np.uint8)

    # 1. Base Mapping
    lc_map = {
        1: FinalClass.WATER, 2: FinalClass.WATER, 3: FinalClass.CANOPY,
        4: FinalClass.LOWVEG, 5: FinalClass.LOWVEG, 6: FinalClass.IMPERVIOUS,
        7: FinalClass.BUILDING, 8: FinalClass.IMPERVIOUS, 9: FinalClass.ROAD 
    }
    for lc_val, fc in lc_map.items():
        final_map[lc_array == lc_val] = fc

    # 2. Extract Vectors
    v = {
        "high_res_roads": vector_masks_dict.get("high_res_roads", np.zeros((h, w), dtype=np.uint8)) == 1,
        "sidewalk": vector_masks_dict.get("sidewalk", np.zeros((h, w), dtype=np.uint8)) == 1,
        "crosswalk": vector_masks_dict.get("crosswalk", np.zeros((h, w), dtype=np.uint8)) == 1,
        "ped_model_roads": vector_masks_dict.get("ped_model_roads", np.zeros((h, w), dtype=np.uint8)) == 1, 
        "parking": vector_masks_dict.get("parking", np.zeros((h, w), dtype=np.uint8)) == 1,
        "buildings": vector_masks_dict.get("buildings", np.zeros((h, w), dtype=np.uint8)) == 1,
        "cars": vector_masks_dict.get("cars", np.zeros((h, w), dtype=np.uint8)) == 1,
        "pools": vector_masks_dict.get("pools", np.zeros((h, w), dtype=np.uint8)) == 1
    }

    # 3. Priority Overwrites (High to Low)
    # Buildings are priority 1 (already eroded by predict.py)
    final_map[v["buildings"]] = FinalClass.BUILDING
    final_map[v["pools"]] = FinalClass.POOL
    final_map[v["cars"]] = FinalClass.CAR
    final_map[v["crosswalk"]] = FinalClass.CROSSWALK
    final_map[v["sidewalk"] & (final_map != FinalClass.BUILDING)] = FinalClass.SIDEWALK
    
    # 4. ROAD VALIDATION (Must be impervious/road land cover AND not occluded by high canopy)
    all_road_vectors = v["high_res_roads"] | v["ped_model_roads"]
    is_paved = np.isin(lc_array, [8, 9])
    is_canopy = (lc_array == 3)
    
    # Roads only get set if the land is paved AND the model didn't mistake tree shadow for road
    # This automatically suppresses roads in high-canopy areas
    road_mask = (
        all_road_vectors &
        is_paved &
        ~is_canopy &
        (final_map != FinalClass.BUILDING) &
        (final_map != FinalClass.CAR) &
        (final_map != FinalClass.POOL)
    )

    final_map[road_mask] = FinalClass.ROAD
    
    # 5. Fill remaining context
    parking_candidates = (
        v["parking"]
        & ~v["buildings"]
        & ~v["sidewalk"]
        & ~v["crosswalk"]
    )

    final_map[parking_candidates] = FinalClass.PARKING
    final_map[is_canopy & (final_map == 0)] = FinalClass.CANOPY
    
    return final_map
# ─────────────────────────────────────────────────────────────────────────────
# PART 2: COMFORT ECOSYSTEM GENERATION (Weighted Gradient)
# ─────────────────────────────────────────────────────────────────────────────
def generate_comfort_score_map(
    clean_map: np.ndarray,
    cfg: PipelineConfig
) -> np.ndarray:
    """
    Generates the comfort heatmap by applying specific weights and radii 
    from your config to each class in the clean categorical map.
    """
    h, w = clean_map.shape
    accumulated_heatmap = np.zeros((h, w), dtype=np.float32)

    # We iterate through every class in the final_classes registry
    for class_id, info in cfg.class_registry.items():
        
        # 1. Use your built-in classmethod to get the lowercase string name
        #class_name = FinalClass.name(class_id)
        
        # 2. Pull weight and radius using the string name
        weight = cfg.comfort_scores.get(class_id, 0.0)
        radius = info.radius
        
        # 3. Since class_id is already an integer, direct NumPy comparison works natively
        mask = (clean_map == class_id).astype(np.float32)

        print(
            f"{class_id:12}",
            "pixels=", int(mask.sum()),
            "weight=", weight,
            "radius=", radius
        )
            
        if np.any(mask) and weight != 0:
            weighted_mask = mask * weight
            
            # If radius exists, radiate the score; otherwise apply crisp value
            if radius > 0:
                accumulated_heatmap += gaussian_filter(weighted_mask, sigma=radius)
            else:
                accumulated_heatmap += weighted_mask
    print("\nHEATMAP STATS")
    print("min:", accumulated_heatmap.min())
    print("max:", accumulated_heatmap.max())
    print("mean:", accumulated_heatmap.mean())
    print("std:", accumulated_heatmap.std())
    valid = accumulated_heatmap[accumulated_heatmap != 0]
    if len(valid):
        print(f"\n[Score Distribution]")
        for p in [5, 20, 40, 50, 60, 80, 95]:
            print(f"  p{p:2d}: {np.percentile(valid, p):+.3f}")
    return accumulated_heatmap