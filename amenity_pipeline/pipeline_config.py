"""
Single source of truth for the amenity pipeline configuration.

config.json has two sections:
  "models"        — script paths, output paths, thresholds, inference settings.
                    No weights or radii here. Set "enabled": false to skip.
  "final_classes" — the output classes with weight (comfort score) and radius.
                    This is the only place those values live.

Usage:
    from pipeline_config import load_pipeline_config, FinalClass, NEN_THRESHOLDS
    cfg = load_pipeline_config()
    cfg.class_registry[FinalClass.SIDEWALK].weight   # → 1.5
    cfg.class_registry[FinalClass.SIDEWALK].radius   # → 25
    cfg.comfort_scores                               # {0: 0.0, 1: 0.30, ...}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


# ─────────────────────────────────────────────────────────────────────────────
# Unified output class IDs
# Import FinalClass everywhere — never redefine these integers in other files.
# ─────────────────────────────────────────────────────────────────────────────
class FinalClass:
    # IDs defined strictly by your 11-class requirement
    WATER      = 1
    CANOPY     = 2
    LOWVEG     = 3
    IMPERVIOUS = 4
    ROAD       = 5
    PARKING    = 6
    BUILDING   = 7
    SIDEWALK   = 8
    CROSSWALK  = 9
    CAR        = 10
    POOL       = 11
    SEATING    = 12
    GARDEN     = 13

    @classmethod
    def name(cls, cid: int) -> str:
        _MAP = {
            1: "water",      2: "canopy",    3: "lowveg",
            4: "impervious", 5: "road",      6: "parking",
            7: "building",   8: "sidewalk",  9: "crosswalk",
            10: "car",       11: "pool",     12: "seating",
            13: "garden"
        }
        return _MAP.get(cid, f"class_{cid}")

    @classmethod
    def all_ids(cls) -> List[int]:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# ─────────────────────────────────────────────────────────────────────────────
# NEN-8100 thresholds
# Single definition — import from here in visualize.py.
# (lower_bound_inclusive, label, hex_color)  — last entry: lower_bound = None
# ─────────────────────────────────────────────────────────────────────────────
NEN_THRESHOLDS = [
    (0.20,  "A", "#2ca02c"),   # Excellent — canopy/sidewalk range
    (0,  "B", "#98df8a"),   # Good      — light green areas
    (-0.15, "C", "#ffdd57"),   # Moderate  — near-neutral surfaces
    (-0.27, "D", "#ff7f0e"),   # Poor      — impervious/parking
    (None,  "E", "#d62728"),   # Unacceptable — roads
]

# ─────────────────────────────────────────────────────────────────────────────
# Built-in defaults — used when a class is not listed in config.json
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_WEIGHT: Dict[int, float] = {
    FinalClass.WATER:      0.15,
    FinalClass.CANOPY:     0.55,
    FinalClass.LOWVEG:     0.40,
    FinalClass.IMPERVIOUS:-0.15,
    FinalClass.ROAD:      -0.35,
    FinalClass.PARKING:   -0.25,
    FinalClass.BUILDING:   0.00,
    FinalClass.SIDEWALK:   0.45,
    FinalClass.CROSSWALK:  0.25,
    FinalClass.CAR:       -0.20,
    FinalClass.POOL:      -0.10, # Added default for pool
    FinalClass.SEATING:    0.35,
    FinalClass.GARDEN:     0.40,
}

_DEFAULT_RADIUS: Dict[int, float] = {cid: 0.0 for cid in FinalClass.all_ids()}

# Map final_classes config key names → FinalClass IDs
_KEY_TO_FINAL: Dict[str, int] = {
    "water":      FinalClass.WATER,
    "canopy":     FinalClass.CANOPY,
    "lowveg":     FinalClass.LOWVEG,
    "impervious": FinalClass.IMPERVIOUS,
    "road":       FinalClass.ROAD,
    "parking":    FinalClass.PARKING,
    "building":   FinalClass.BUILDING,
    "sidewalk":   FinalClass.SIDEWALK,
    "crosswalk":  FinalClass.CROSSWALK,
    "car":        FinalClass.CAR,
    "pool":       FinalClass.POOL,
    "seating":    FinalClass.SEATING,
    "garden":     FinalClass.GARDEN,
}
# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FinalClassConfig:
    """Visualization/scoring settings for one output class."""
    id: int
    name: str
    weight: float = 0.0   # Comfort score: positive = pleasant, negative = stressful
    radius: float = 0.0   # Dilation radius in pixels applied during mask fusion
    # ── Confidence-based fusion scalars ───────────────────────────────────────
    # fusion_conf:        Base multiplier for LC-derived-only classes (water, canopy,
    #                     lowveg, impervious) competing in the per-pixel argmax.
    # lc_source_weight:   How much to weight the LC probability channel when blending
    #                     it into the composite confidence (road, building, canopy, lowveg).
    # ped_source_weight:  Weight of pedestrian model road channel fused into road_conf.
    # veg_suppress_factor: Building confidence multiplier in vegetation pixels (suppresses
    #                      false detections on sports fields and tree shadows).
    fusion_conf:          float = 1.0
    lc_source_weight:     float = 0.0
    ped_source_weight:    float = 0.0
    veg_suppress_factor:  float = 1.0


@dataclass
class FusionParams:
    """Global parameters for confidence-based mask fusion."""
    binary_fallback_conf:    float = 0.50  # Confidence for classes with no prob raster
    parking_road_dilation_px: int  = 50   # Parking proximity filter radius (0 = disabled)


@dataclass
class ModelClassConfig:
    """Per-class threshold for a multiclass prediction model."""
    key: str             # class key from config (e.g. "0", "1")
    name: str            # human-readable name (e.g. "sidewalk")
    threshold: float = 0.5


@dataclass
class ModelConfig:
    """One prediction model — script path, output, thresholds, inference settings."""
    name: str
    script_path: str
    output_path: str
    enabled: bool = True
    is_multiclass: bool = False
    classes: Dict[str, ModelClassConfig] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)
    python_interpreter: Optional[str] = None


@dataclass
class PipelineConfig:
    """Complete resolved pipeline configuration."""
    pipeline_image: str
    models: List[ModelConfig]
    class_registry: Dict[int, FinalClassConfig] = field(default_factory=dict)
    fusion_params: FusionParams = field(default_factory=FusionParams)

    @property
    def enabled_models(self) -> List[ModelConfig]:
        return [m for m in self.models if m.enabled]

    @property
    def comfort_scores(self) -> Dict[int, float]:
        """Per-class comfort weights — used in heatmap and NEN-8100 map."""
        return {cid: cc.weight for cid, cc in self.class_registry.items()}

    @property
    def radii(self) -> Dict[int, float]:
        """Per-class dilation radii — used in mask_fusion."""
        return {cid: cc.radius for cid, cc in self.class_registry.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline_config(config_path: Path | str = DEFAULT_CONFIG) -> PipelineConfig:
    """
    Load config.json and return a fully resolved PipelineConfig.

    The "final_classes" section in config.json sets weights and radii.
    Classes not listed there use the built-in defaults above.
    """
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    # Seed class registry with built-in defaults
    class_registry: Dict[int, FinalClassConfig] = {
        cid: FinalClassConfig(
            id=cid,
            name=FinalClass.name(cid),
            weight=_DEFAULT_WEIGHT.get(cid, 0.0),
            radius=_DEFAULT_RADIUS.get(cid, 0.0),
        )
        for cid in FinalClass.all_ids()
    }

    # Override with values from "final_classes" section
    for key, fc_raw in raw.get("final_classes", {}).items():
        final_id = _KEY_TO_FINAL.get(key.lower().strip())
        if final_id is None:
            print(f"[pipeline_config] Warning: unknown final_class key '{key}' — skipping.")
            continue
        class_registry[final_id] = FinalClassConfig(
            id=final_id,
            name=FinalClass.name(final_id),
            weight=float(fc_raw.get("weight", _DEFAULT_WEIGHT.get(final_id, 0.0))),
            radius=float(fc_raw.get("radius", 0.0)),
            fusion_conf=float(fc_raw.get("fusion_conf", 1.0)),
            lc_source_weight=float(fc_raw.get("lc_source_weight", 0.0)),
            ped_source_weight=float(fc_raw.get("ped_source_weight", 0.0)),
            veg_suppress_factor=float(fc_raw.get("veg_suppress_factor", 1.0)),
        )

    # Parse models section
    models: List[ModelConfig] = []
    for m in raw.get("models", []):
        classes_dict: Dict[str, ModelClassConfig] = {}
        for class_key, cr in m.get("classes", {}).items():
            classes_dict[str(class_key)] = ModelClassConfig(
                key=str(class_key),
                name=cr.get("name", class_key),
                threshold=float(cr.get("threshold", 0.5)),
            )

        models.append(ModelConfig(
            name=m.get("name", ""),
            script_path=m.get("script_path", ""),
            output_path=m.get("output_path", ""),
            enabled=bool(m.get("enabled", True)),
            is_multiclass=bool(m.get("is_multiclass", False)),
            classes=classes_dict,
            extra_args=m.get("extra_args", []),
            python_interpreter=m.get("python_interpreter"),
        ))

    # Parse fusion global params
    fusion_raw = raw.get("fusion", {})
    fusion_params = FusionParams(
        binary_fallback_conf=float(fusion_raw.get("binary_fallback_conf", 0.50)),
        parking_road_dilation_px=int(fusion_raw.get("parking_road_dilation_px", 50)),
    )

    return PipelineConfig(
        pipeline_image=raw.get("pipeline_image", ""),
        models=models,
        class_registry=class_registry,
        fusion_params=fusion_params,
    )
