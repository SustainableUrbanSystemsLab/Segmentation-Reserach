"""
Urban mask fusion & Comfort Ecosystem Generator.

Clamping strategy per class:
    FULL CLAMP  (buildings, parking):
        These are large-area features where tiled-inference prob_accumulator
        bleeds 10–30px beyond object edges. The binary mask at threshold is a
        reliable spatial anchor. Zeroing outside it prevents wrong argmax wins.

    NO CLAMP (roads, sidewalks, crosswalks):
        These are linear/thin features where the model IS the primary detection
        source, often superior to land cover. Clamping them causes under-detection
        and lets LC classes fill the vacuum. Use raw prob raster directly.

    LC-ONLY (water, canopy, lowveg, impervious):
        No dedicated detection model. Use LC confidence, but keep it below
        typical model detection confidence so model-detected classes beat them
        in pixels where both have signal.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from typing import Dict, Optional
from pipeline_config import FinalClass, PipelineConfig


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: GEOMETRIC MASK FUSION
# ─────────────────────────────────────────────────────────────────────────────

def fuse_urban_masks(
    lc_array: np.ndarray,
    vector_masks_dict: Dict[str, np.ndarray],
    cfg: PipelineConfig | None = None,
    prob_maps: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    h, w = lc_array.shape
    final_map = np.zeros((h, w), dtype=np.uint8)
    pm = prob_maps or {}

    BINARY_CONF = 0.5
    WARN = "[Fusion] WARNING: No prob raster for"

    print("[Fusion] Available prob_maps keys:", sorted(pm.keys()))
    if "lc_canopy" in pm:
        print(f"  lc_canopy: max={pm['lc_canopy'].max():.3f} nonzero={int((pm['lc_canopy']>0).sum()):,}")
    if "lc_lowveg" in pm:
        print(f"  lc_lowveg: max={pm['lc_lowveg'].max():.3f} nonzero={int((pm['lc_lowveg']>0).sum()):,}")

    def _bin(key: str) -> np.ndarray:
        return (vector_masks_dict.get(key, np.zeros((h, w), dtype=np.uint8)) == 1)

    # ── 1. Base land cover canvas ─────────────────────────────────────────────
    lc_map = {
        1: FinalClass.WATER,      2: FinalClass.WATER,
        3: FinalClass.CANOPY,     4: FinalClass.LOWVEG,
        5: FinalClass.LOWVEG,     6: FinalClass.IMPERVIOUS,
        7: FinalClass.BUILDING,   8: FinalClass.IMPERVIOUS,
        9: FinalClass.ROAD,
    }
    for lc_val, fc in lc_map.items():
        final_map[lc_array == lc_val] = fc

    # ── 2. Binary detection masks ─────────────────────────────────────────────
    # Road binary: trust model directly, NO is_paved constraint.
    # The Esri road model detects roads correctly even where LC says "Structures"
    # or "Low Veg" (tunnels, tree-lined streets, etc). Restricting by LC class
    # causes the model to lose confidence in areas it detected correctly.
    road_binary     = _bin("high_res_roads") | _bin("ped_model_roads")
    parking_binary  = _bin("parking")
    building_binary = _bin("buildings")
    sidewalk_binary = _bin("sidewalk")
    crosswalk_binary = _bin("crosswalk")

    # ── 3. Build confidence maps ──────────────────────────────────────────────

    def _raw(key: str, binary_fallback: np.ndarray) -> np.ndarray:
        """Raw prob raster — NO clamping. For road, sidewalk, crosswalk."""
        if key in pm:
            return pm[key].astype(np.float32)
        print(f"{WARN} '{key}' — using binary at {BINARY_CONF}.")
        return binary_fallback.astype(np.float32) * BINARY_CONF

    def _clamped(key: str, binary_mask: np.ndarray) -> np.ndarray:
        """Prob raster zeroed outside binary footprint. For building, parking."""
        raw = _raw(key, binary_mask)
        return raw * binary_mask.astype(np.float32)

    # ── ROAD: no clamp, raw model prob everywhere ─────────────────────────────
    # Also fuse lc_road channel at reduced weight (LC road class is coarser
    # than the dedicated model, only use where model has no signal).
    road_conf = _raw("road", road_binary)
    if "lc_road" in pm:
        lc_road = pm["lc_road"].astype(np.float32) * 0.60   # downweight LC
        road_conf = np.where(road_conf > 0, road_conf, lc_road)
    if "ped_road" in pm:
        ped_road = pm["ped_road"].astype(np.float32) * 0.70  # slightly downweight ped model
        road_conf = np.maximum(road_conf, ped_road)

    # ── PARKING: full clamp to binary footprint ───────────────────────────────
    parking_conf = _clamped("parking", parking_binary)

    # Parking proximity constraint: real parking lots are almost always within
    # 15m (50px @ 0.3m) of a road. Pixels where parking won't touch a road
    # are reclassified to impervious instead of parking.
    if road_binary.any() or road_conf.max() > 0:
        road_signal = (road_conf > 0.2) | road_binary
        near_road   = binary_dilation(road_signal, iterations=50)
        parking_conf = parking_conf * near_road.astype(np.float32)

    # ── BUILDING: LC is primary, dedicated model is precision booster ──────────
    #
    # LC Structures (4902 px) covers far more ground than the building model
    # (41 detections at threshold 0.95). Clamping lc_building to the binary
    # footprint throws away signal for ~99% of buildings. Instead:
    #   - lc_building runs UNCLAMPED at moderate confidence (0.68)
    #   - building model runs CLAMPED at full confidence (where it fires, it wins)
    #   - vegetation suppression handles false positives on sports fields etc.

    # Primary: LC Structures — unclamped, moderate confidence
    if "lc_building" in pm:
        building_conf_lc = pm["lc_building"].astype(np.float32) * 0.62
    else:
        # Fallback: derive from LC array if no prob raster
        building_conf_lc = (lc_array == 7).astype(np.float32) * 0.65
        print(f"{WARN} 'lc_building' — deriving from LC array at 0.65.")

    # Secondary: dedicated model — clamped to binary footprint
    # This wins over LC signal where the model fires confidently (e.g. 0.92 > 0.68)
    building_conf_model = _clamped("building", building_binary)

    # Combined: model boosts LC where it fires, LC covers everywhere else
    building_conf = np.maximum(building_conf_lc, building_conf_model)

    # Vegetation suppression: where LC says canopy/lowveg, the building model
    # almost certainly fired on a sports field, grass, or tree shadow.
    # Reduce building_conf so LC vegetation classes win argmax in those pixels.
    lc_is_vegetation = np.isin(lc_array, [3, 4, 5])
    building_conf = building_conf * np.where(lc_is_vegetation, 0.25, 1.0)

    # ── SIDEWALK / CROSSWALK: no clamp ───────────────────────────────────────
    # Thin linear features (~1–3m wide). Model is the primary detection source.
    # Their prob rasters don't bleed significantly because features are narrow.
    sidewalk_conf  = _raw("sidewalk",  sidewalk_binary)
    crosswalk_conf = _raw("crosswalk", crosswalk_binary)

    # ── LC-ONLY classes: set below typical model detection confidence ─────────
    # Goal: model-detected classes beat LC in pixels where both have signal.
    # lowveg was 0.80 which caused it to dominate when road/sidewalk were zeroed.
    # Reduced to 0.50 so it only wins where no model detection is present.
    water_conf      = np.isin(lc_array, [1, 2]).astype(np.float32) * 0.80

    canopy_conf = (lc_array == 3).astype(np.float32) * 0.75
    if "lc_canopy" in pm:
        canopy_conf = np.maximum(canopy_conf, pm["lc_canopy"].astype(np.float32) * 0.72)

    lowveg_conf = np.isin(lc_array, [4, 5]).astype(np.float32) * 0.50
    if "lc_lowveg" in pm:
        lowveg_conf = np.maximum(lowveg_conf, pm["lc_lowveg"].astype(np.float32) * 0.58)
    if "lc_impervious" in pm:
        impervious_conf = pm["lc_impervious"].astype(np.float32) * 0.55
    else:
        impervious_conf = np.isin(lc_array, [6, 8]).astype(np.float32) * 0.55

    # ── 4. Per-pixel argmax ───────────────────────────────────────────────────
    class_ids = np.array([
        FinalClass.ROAD,       FinalClass.PARKING,    FinalClass.BUILDING,
        FinalClass.SIDEWALK,   FinalClass.CROSSWALK,  FinalClass.IMPERVIOUS,
        FinalClass.WATER,      FinalClass.CANOPY,     FinalClass.LOWVEG,
    ], dtype=np.uint8)

    conf_arrays = [
        road_conf, parking_conf, building_conf,
        sidewalk_conf, crosswalk_conf, impervious_conf,
        water_conf, canopy_conf, lowveg_conf,
    ]
    conf_names = [
        "road", "parking", "building",
        "sidewalk", "crosswalk", "impervious",
        "water", "canopy", "lowveg",
    ]

    # Pre-argmax diagnostic
    print("[Fusion] Confidence summary (pre-argmax):")
    print(f"  {'class':<12} {'active_px':>10}  {'max_conf':>9}  {'clamped?':>8}")
    clamped_set = {"building", "parking"}
    for name, arr in zip(conf_names, conf_arrays):
        c = "yes" if name in clamped_set else "no"
        print(f"  {name:<12} {int((arr > 0).sum()):>10,}  {arr.max():>9.3f}  {c:>8}")

    conf_stack = np.stack(conf_arrays, axis=0)                          # (9, H, W)
    row_idx    = np.arange(h)[:, None]
    col_idx    = np.arange(w)[None, :]
    winner_idx = np.argmax(conf_stack, axis=0)                          # (H, W)
    max_conf   = conf_stack[winner_idx, row_idx, col_idx]               # (H, W)

    confident_px = max_conf > 0
    final_map[confident_px] = class_ids[winner_idx[confident_px]]

    # ── 5. Hard overrides: no semantic competition ────────────────────────────
    # Pools and cars are detected unambiguously — they sit on top of all surfaces.
    final_map[_bin("pools")] = FinalClass.POOL
    final_map[_bin("cars")]  = FinalClass.CAR

    # ── 6. Post-fusion diagnostic ─────────────────────────────────────────────
    print("[Fusion] Final pixel counts:")
    for cid in FinalClass.all_ids():
        count = int((final_map == cid).sum())
        pct   = 100 * count / (h * w)
        print(f"  {FinalClass.name(cid):<12} (id={cid:2d})  {count:>9,} px  ({pct:.1f}%)")

    return final_map


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: COMFORT ECOSYSTEM GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_comfort_score_map(
    clean_map: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    h, w = clean_map.shape
    accumulated_heatmap = np.zeros((h, w), dtype=np.float32)

    print("[Heatmap] Class contributions:")
    for class_id, info in cfg.class_registry.items():
        weight = cfg.comfort_scores.get(class_id, 0.0)
        radius = info.radius
        mask   = (clean_map == class_id).astype(np.float32)
        px     = int(mask.sum())
        print(f"  {FinalClass.name(class_id):<12} (id={class_id:2d})"
              f"  pixels={px:>9,}  weight={weight:+.2f}  radius={radius:.0f}")
        if px > 0 and weight != 0:
            weighted = mask * weight
            accumulated_heatmap += gaussian_filter(weighted, sigma=radius) if radius > 0 else weighted

    valid = accumulated_heatmap[accumulated_heatmap != 0]
    print(f"\n[Heatmap] min={accumulated_heatmap.min():.4f}  "
          f"max={accumulated_heatmap.max():.4f}  "
          f"mean={accumulated_heatmap.mean():.4f}  "
          f"std={accumulated_heatmap.std():.4f}")
    if len(valid):
        print("[Score Distribution]")
        for p in [5, 20, 40, 50, 60, 80, 95]:
            print(f"  p{p:2d}: {np.percentile(valid, p):+.3f}")

    return accumulated_heatmap