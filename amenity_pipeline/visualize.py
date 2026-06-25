"""
Amenity pipeline visualizer.

Generates:
  1. overlay_combined.png         – raw vector detection outlines
  2. overlay_land_cover.png       – raw land cover polygons
  3. overlay_cleaned_mask.png     – unified fused mask (if fusion raster exists)
  4. overlay_comfort_heatmap.png  – continuous comfort score gradient
  5. overlay_nen8100.png          – NEN-8100 A-E pedestrian wind comfort classes

All weights, comfort scores, and NEN-8100 thresholds come from config.json
via pipeline_config — nothing is hardcoded here.
"""
import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
from rasterio.transform import Affine, rowcol
from shapely.geometry import shape
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from rasterio.features import rasterize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import (
    FinalClass,
    NEN_THRESHOLDS,
    load_pipeline_config,
    DEFAULT_CONFIG,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_path(path_str, relative_to):
    if not path_str:
        return None
    if os.path.isabs(path_str):
        return path_str
    candidate = os.path.abspath(os.path.join(relative_to, path_str))
    if os.path.exists(candidate):
        return candidate
    candidate_repo = os.path.abspath(os.path.join(REPO_ROOT, path_str))
    if os.path.exists(candidate_repo):
        return candidate_repo
    return candidate


def safe_load_json(file_path):
    """Load a JSON file trying UTF-8, CP1252, then byte-replacement."""
    for enc in ("utf-8", "cp1252"):
        try:
            with open(file_path, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, LookupError):
            pass
    print(f"Warning: Mixed encoding in {os.path.basename(file_path)} – using replacement.")
    with open(file_path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_features_in_meters(path, crs_ref="EPSG:3857"):
    """
    Loads a GeoJSON using GeoPandas, handles the degree/meter coordinate heuristic,
    and returns a standard list of feature dicts with geometries reprojected to crs_ref.
    """
    import geopandas as gpd
    if not os.path.exists(path):
        return []
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return []
        
        # Heuristic: check if coordinates are in meters or degrees
        bounds = gdf.total_bounds
        is_meter_based = (
            max(abs(bounds[0]), abs(bounds[2])) > 1000 or
            max(abs(bounds[1]), abs(bounds[3])) > 1000
        )
        
        if is_meter_based:
            # Override CRS to reference CRS directly (avoid corrupt reprojection)
            gdf.crs = crs_ref
        else:
            if gdf.crs is None:
                gdf.crs = "EPSG:4326"
            gdf = gdf.to_crs(crs_ref)
            
        # Convert back to standard features list with shapely geometries in meters
        features = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            props = {k: v for k, v in row.items() if k != "geometry"}
            features.append({
                "geometry": geom,
                "properties": props
            })
        return features
    except Exception as e:
        print(f"Warning: Failed to load {path} with geopandas: {e}. Falling back to safe_load_json.")
        try:
            geojson = safe_load_json(path)
            features = []
            for feat in geojson.get("features", []):
                features.append({
                    "geometry": shape(feat["geometry"]),
                    "properties": feat.get("properties", {})
                })
            return features
        except Exception as e2:
            print(f"Error: Fallback failed: {e2}")
            return []


def match_layer(props, layer):
    """Return True if properties match the filter criteria for a given layer."""
    feat_class = str(props.get("class", props.get("Class", props.get("label", "")))).strip()
    feat_class_id = str(props.get("Class_ID", props.get("class_id", ""))).strip()
    
    if layer["class_filter"] or layer["class_name_filter"]:
        match_id = (feat_class_id == layer["class_filter"])
        match_name = False
        if layer["class_name_filter"]:
            f_name = feat_class.lower().strip()
            filter_name = layer["class_name_filter"].lower().strip()
            match_name = (
                f_name == filter_name or 
                f_name + "s" == filter_name or 
                filter_name + "s" == f_name or
                f_name == filter_name.rstrip('s')
            )
        return (match_id or match_name)
    return True


def geo_to_pixel(geom, transform):
    coords = np.array(geom.exterior.coords)
    rows, cols = rowcol(transform, coords[:, 0], coords[:, 1])
    return np.column_stack([cols, rows])


def load_rgb(image_path, target_shape=None):
    from rasterio.enums import Resampling
    with rasterio.open(image_path) as src:
        native_transform = src.transform
        if target_shape is not None:
            target_h, target_w = target_shape
            rgb = src.read([1, 2, 3], out_shape=(3, target_h, target_w), resampling=Resampling.bilinear)
            scale_x = src.width / target_w
            scale_y = src.height / target_h
            transform = native_transform * Affine.scale(scale_x, scale_y)
        else:
            rgb = src.read([1, 2, 3])
            transform = native_transform
    rgb = np.moveaxis(rgb, 0, -1).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb, transform

def flatten_layers(models_config, skip_dino=False, only_dino=False):
    """Flatten model configs into individual rendering layers (skip land_cover)."""
    layers = []
    for mcfg in models_config:
        if mcfg.get("name") == "land_cover":
            continue
        is_dino = (mcfg.get("name") == "dino_model")
        if skip_dino and is_dino:
            continue
        if only_dino and not is_dino:
            continue
        output_path = mcfg.get("output_path")
        if mcfg.get("is_multiclass", False):
            for class_id, class_info in mcfg.get("classes", {}).items():
                layers.append({
                    "name": class_info.get("name", "unnamed_subclass"),
                    "output_path": output_path,
                    "weight": class_info.get("weight", 1.0),
                    "radius": class_info.get("radius", 30),
                    "class_filter": str(class_id),
                    "class_name_filter": class_info.get("name"),
                })
        else:
            layers.append({
                "name": mcfg.get("name", "unnamed"),
                "output_path": output_path,
                "weight": mcfg.get("weight", 1.0),
                "radius": mcfg.get("radius", 30),
                "class_filter": str(mcfg.get("class_filter", "")),
                "class_name_filter": None,
            })
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_comfort_heatmap_legacy(rgb_shape, transform, models_config) -> np.ndarray:
    """Fallback: centroid-based Gaussian heatmap from raw vector GeoJSONs."""
    h, w = rgb_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    layers = flatten_layers(models_config)

    for layer in layers:
        weight = layer["weight"]
        radius = layer["radius"]
        resolved_output = resolve_path(layer["output_path"], REPO_ROOT)
        if not os.path.exists(resolved_output):
            continue
        features = load_features_in_meters(resolved_output)
        for feat in features:
            props = feat.get("properties", {})
            if not match_layer(props, layer):
                continue
            geom = feat["geometry"]
            cx, cy = ~transform * (geom.centroid.x, geom.centroid.y)
            cx, cy = int(round(cx)), int(round(cy))
            window_size = int(3 * radius)
            x0, x1 = max(0, cx - window_size), min(w, cx + window_size)
            y0, y1 = max(0, cy - window_size), min(h, cy + window_size)
            if x1 <= x0 or y1 <= y0:
                continue
            ys, xs = np.ogrid[y0:y1, x0:x1]
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            heatmap[y0:y1, x0:x1] += weight * np.exp(-d2 / (2.0 * radius ** 2))

    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Rendering functions
# ─────────────────────────────────────────────────────────────────────────────

def render_combined_detections(rgb, transform, models_config, output_png):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(rgb)

    color_map = {
        "buildings": "#e74c3c",
        "trees":     "#2ecc71",
        "cars":      "#3498db",
        "sidewalks": "#f39c12",
        "crosswalks":"#e67e22",
        "parking":   "#1abc9c",
        "road":      "#ff00ff", # Bright neon magenta
    }
    palette = ["#3498db", "#9b59b6", "#f1c40f", "#1abc9c", "#e67e22"]
    legend_elements = []
    total_features  = 0
    layers = flatten_layers(models_config, skip_dino=True)

    for idx, layer in enumerate(layers):
        name = layer["name"]
        resolved_output = resolve_path(layer["output_path"], REPO_ROOT)
        if name == "buildings":
            reg_path = resolved_output.replace(".geojson", "_reg.geojson")
            if os.path.exists(reg_path):
                resolved_output = reg_path
                name = "buildings (regularized)"
        if not os.path.exists(resolved_output):
            continue
        features = load_features_in_meters(resolved_output)
        color   = color_map.get(name.split(" ")[0], palette[idx % len(palette)])
        patches = []
        for feat in features:
            props = feat.get("properties", {})
            if not match_layer(props, layer):
                continue
            geom = feat["geometry"]
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for poly in polys:
                raw_pixel_coords = geo_to_pixel(poly, transform)
                
                # FIX: Maintain standard (x, y) / (col, row) order so Matplotlib maps it correctly
                corrected_coords = [(x, y) for x, y in raw_pixel_coords]
                
                patches.append(MplPolygon(corrected_coords, closed=True))
                
        if patches:
            ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=color,
                                              linewidth=1.5, alpha=0.85))
            total_features += len(patches)
            legend_elements.append(mpatches.Patch(edgecolor=color, facecolor="none",
                                                  label=f"{name.capitalize()} ({len(patches)})",
                                                  linewidth=2))
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", frameon=True,
                  facecolor="white", edgecolor="gray")
    ax.set_title(f"Combined Raw Detections ({total_features} total features)",
                  fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
def render_dino_detections(rgb, transform, models_config, output_png):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(rgb)

    color_map = {
        "seating":   "#9b59b6", # Amethyst
        "garden":    "#27ae60", # Emerald Green
    }
    palette = ["#f1c40f", "#1abc9c", "#e67e22"]
    legend_elements = []
    total_features  = 0
    layers = flatten_layers(models_config, only_dino=True)

    for idx, layer in enumerate(layers):
        name = layer["name"]
        resolved_output = resolve_path(layer["output_path"], REPO_ROOT)
        if not os.path.exists(resolved_output):
            continue
        features = load_features_in_meters(resolved_output)
        color   = color_map.get(name, palette[idx % len(palette)])
        patches = []
        for feat in features:
            props = feat.get("properties", {})
            if not match_layer(props, layer):
                continue
            geom = feat["geometry"]
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for poly in polys:
                raw_pixel_coords = geo_to_pixel(poly, transform)
                corrected_coords = [(x, y) for x, y in raw_pixel_coords]
                patches.append(MplPolygon(corrected_coords, closed=True))
                
        if patches:
            ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=color,
                                              linewidth=2.0, alpha=0.9))
            total_features += len(patches)
            legend_elements.append(mpatches.Patch(edgecolor=color, facecolor="none",
                                                  label=f"{name.capitalize()} ({len(patches)})",
                                                  linewidth=2))
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", frameon=True,
                  facecolor="white", edgecolor="gray")
    ax.set_title(f"DINO Raw Detections ({total_features} total features)",
                  fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def render_cleaned_mask(rgb, clean_map, output_png):
    """Semi-transparent thematic visualization of the fused class map."""
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(rgb)

    h, w = clean_map.shape
    rgba_mask = np.zeros((h, w, 4), dtype=np.float32)

    class_config = {
        FinalClass.WATER:      ([0.00, 0.45, 1.00], 0.75, "Water"),        # Bright blue
        FinalClass.CANOPY:     ([0.00, 0.60, 0.00], 0.75, "Canopy"),       # Dark green
        FinalClass.LOWVEG:     ([0.50, 1.00, 0.00], 0.75, "Low Veg"),      # Lime green

        FinalClass.IMPERVIOUS: ([0.55, 0.55, 0.55], 0.75, "Impervious"),   # Medium gray
        FinalClass.ROAD:       ([1.00, 0.00, 1.00], 0.80, "Road"),         # Pure magenta
        FinalClass.PARKING:    ([1.00, 0.55, 0.00], 0.80, "Parking"),      # Orange

        FinalClass.BUILDING:   ([0.85, 0.00, 0.00], 0.80, "Building"),     # Strong red
        FinalClass.SIDEWALK:   ([1.00, 1.00, 0.00], 0.80, "Sidewalk"),     # Yellow
        FinalClass.CROSSWALK:  ([0.00, 1.00, 1.00], 0.90, "Crosswalk"),    # Cyan

        FinalClass.CAR:        ([0.00, 0.00, 0.50], 0.90, "Car"),          # Dark navy
        FinalClass.POOL:       ([0.70, 0.20, 1.00], 0.85, "Pool"),         # Purple
        FinalClass.SEATING:    ([0.60, 0.35, 0.15], 0.85, "Seating"),      # Brown
        FinalClass.GARDEN:     ([0.15, 0.70, 0.15], 0.80, "Garden"),       # Fresh Green
    }

    legend_elements = []
    for val, (rgb_c, alpha, label_name) in class_config.items():
        mask = clean_map == val
        if np.any(mask):
            rgba_mask[mask] = rgb_c + [alpha]
            legend_elements.append(mpatches.Patch(facecolor=rgb_c, alpha=alpha, label=label_name))

    ax.imshow(rgba_mask)
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", frameon=True,
                  facecolor="white", edgecolor="gray", fontsize=9)
    ax.set_title("Unified Cleaned Mask Visualization", fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def render_comfort_heatmap(rgb, heatmap, output_png):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(rgb)

    # 1. FIX: Auto-align transposed heatmap matrix to match RGB array layout
    if heatmap.shape != rgb.shape[:2] and heatmap.shape[::-1] == rgb.shape[:2]:
        heatmap = heatmap.T

    # 2. FIX: Isolate true valid data ranges (ignore typical NoData like -9999, inf, nan)
    valid_mask = np.isfinite(heatmap) & (heatmap > -9900) & (heatmap < 9900)
    
    if np.any(valid_mask):
        valid_data = heatmap[valid_mask]
        max_val = np.max(np.abs(valid_data))
        
        norm_heatmap = np.zeros_like(heatmap)
        if max_val > 1e-6:
            norm_heatmap[valid_mask] = heatmap[valid_mask] / max_val
        
        # Map values from [-1, 1] out to a [0, 1] range for the colormap
        mapped = (norm_heatmap + 1.0) / 2.0
    else:
        mapped = np.ones_like(heatmap) * 0.5
        valid_mask = np.ones_like(heatmap, dtype=bool)

    # 3. FIX: Build a direct RGBA map to keep NoData regions perfectly clear
    rgba_heatmap = plt.cm.get_cmap("RdYlGn")(mapped)
    rgba_heatmap[~valid_mask, 3] = 0.0   # Completely transparent background
    rgba_heatmap[valid_mask, 3] = 0.50   # Controlled overlay alpha for real data

    im = ax.imshow(rgba_heatmap, vmin=0, vmax=1)
    
    # Use ScalarMappable to keep colorbar generation independent of alpha array
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.04, shrink=0.65)
    cbar.set_label("Comfort Score  (red = exposed / built-up  →  green = vegetation / pedestrian)",
                   fontsize=11)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low comfort", "Neutral", "High comfort"])

    ax.set_title("Consolidated Comfort Score Heatmap", fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")

def render_nen8100(rgb, heatmap, output_png):
    """
    Render the NEN-8100 wind comfort A-E class map.
    Thresholds and colors come from NEN_THRESHOLDS in pipeline_config.
    Buildings now display their calculated comfort score.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(rgb)

    # 1. FIX: Auto-align transposed heatmap matrix to match RGB array layout
    if heatmap.shape != rgb.shape[:2] and heatmap.shape[::-1] == rgb.shape[:2]:
        heatmap = heatmap.T

    h, w = heatmap.shape
    nen_rgba = np.zeros((h, w, 4), dtype=np.float32)

    # Define a safe validity mask to screen out background fills
    valid_mask = np.isfinite(heatmap) & (heatmap > -9900) & (heatmap < 9900)

    alphas = {"A": 0.72, "B": 0.68, "C": 0.65, "D": 0.68, "E": 0.72}
    nen_entries = []
    
    for i, (lo, label, hex_color) in enumerate(NEN_THRESHOLDS):
        hi = NEN_THRESHOLDS[i - 1][0] if i > 0 else float("inf")
        actual_lo = lo if lo is not None else -float("inf")
        
        color = np.array([int(hex_color[j:j+2], 16) / 255.0 for j in (1, 3, 5)])
        nen_entries.append((label, color, actual_lo, hi, alphas.get(label, 0.68)))

    legend_elements = []
    for label, color, lo, hi, alpha in nen_entries:
        # 2. FIX: Prevent NoData values from being swallowed by the bottom threshold
        cat_mask = (heatmap >= lo) & (heatmap < hi) & valid_mask
        
        if np.any(cat_mask):
            nen_rgba[cat_mask] = np.append(color, alpha)
        
        range_str = f"≥ {lo:.2f}" if lo > -1e9 else f"< {hi:.2f}"
        legend_elements.append(mpatches.Patch(
            facecolor=color, alpha=0.75,
            label=f"NEN-8100 {label} (score {range_str})"
        ))

    ax.imshow(nen_rgba)
    ax.legend(handles=legend_elements[::-1], loc="upper right", frameon=True,
              facecolor="#1a1a1a", edgecolor="gray", fontsize=10, labelcolor="white")

    cmap_nen = ListedColormap([e[1] for e in nen_entries])
    boundaries = list(range(len(nen_entries) + 1))
    norm_nen = BoundaryNorm(boundaries, cmap_nen.N)
    
    sm = plt.cm.ScalarMappable(cmap=cmap_nen, norm=norm_nen)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.04, shrink=0.65)
    cbar.set_ticks([i + 0.5 for i in range(len(nen_entries))])
    cbar.set_ticklabels([f"{e[0]} {NEN_THRESHOLDS[i][1]}" for i, e in enumerate(nen_entries)])
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("NEN-8100 Pedestrian Wind Comfort Classes (A = most comfortable, E = least)", 
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")

def render_land_cover(rgb, transform, model_config, output_png):
    output_path = model_config.get("output_path")
    resolved_output = resolve_path(output_path, REPO_ROOT)
    if not os.path.exists(resolved_output):
        print(f"Warning: land_cover output not found at {resolved_output}. Skipping.")
        return
    print(f"Rendering land cover from {resolved_output}...")
    features = load_features_in_meters(resolved_output)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(rgb)

    # High-contrast color palette: Distinct hues to avoid visual blending
    color_map = {
        "Water":               "#0070FF", # Vivid Blue
        "Wetlands":            "#008C76", # Teal
        "Tree Canopy":         "#228B22", # Forest Green
        "Shrubland":           "#8BC34A", # Lime Green
        "Low Vegetation":      "#D4E157", # Yellow-Green
        "Barren":              "#FF9800", # Orange
        "Structures":          "#D32F2F", # Deep Red
        "Impervious Surfaces": "#9E9E9E", # Grey
        "Impervious Roads":    "#455A64", # Dark Slate
        "Pool":                "#E040FB", # Magenta/Purple
    }

    class_patches = {cls: [] for cls in color_map}
    for feat in features:
        cls_name = feat["properties"].get("Class", "Unknown").strip()
        # Add to existing map if found, else default to 'Unknown' bucket
        if cls_name not in class_patches:
            class_patches[cls_name] = []
            
        geom = feat["geometry"]
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            class_patches[cls_name].append(MplPolygon(geo_to_pixel(poly, transform), closed=True))

    legend_elements = []
    for cls_name, patches in class_patches.items():
        if not patches:
            continue
        # Use a safe fallback for unknown classes
        color = color_map.get(cls_name, "#FFFFFF")
        ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=color,
                                          linewidth=0.2, alpha=0.6))
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor="none",
                                              label=f"{cls_name} ({len(patches)})"))

    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", frameon=True,
                  facecolor="white", edgecolor="gray", fontsize=10)
    
    ax.set_title("Raw Land Cover Classification Map", fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")

def render_debug_overlay(rgb, geojson_path, transform, raster_crs, output_png, color, alpha, title):
    """
    Renders an isolated vector layer directly over the original RGB image.
    Ensures the vector's CRS is explicitly matched to the raster's native CRS 
    before rasterization to prevent spatial displacement.
    """
    if not os.path.exists(geojson_path):
        print(f"[-] Cannot render debug overlay, missing: {geojson_path}")
        return

    try:
        # 1. Load the vector data
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            print(f"[!] No features to render in {geojson_path}")
            return
            
        # 2. CRITICAL FIX: Ensure coordinate alignment
        if gdf.crs is None:
            gdf.crs = "EPSG:4326" # Safe geographic fallback fallback
            
        # Reproject vectors to explicitly match the native raster coordinate system
        gdf = gdf.to_crs(raster_crs)
        
        h, w = rgb.shape[:2]
        
        # 3. Rasterize reprojected geometries using the image's affine transform
        geom_shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
        mask = rasterize(geom_shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)

        # 4. FIX: Normalize target color to match float32 canvas [0.0 - 1.0] space
        color_arr = np.array(color, dtype=np.float32)
        if color_arr.max() > 1.0:
            color_arr /= 255.0

        overlay = rgb.copy()
        colored_mask = np.zeros_like(rgb, dtype=np.float32)
        colored_mask[:] = color_arr

        # Safely blend color mask with normalized RGB background array
        active_pixels = mask == 1
        overlay[active_pixels] = (1 - alpha) * overlay[active_pixels] + alpha * colored_mask[active_pixels]
        overlay = np.clip(overlay, 0.0, 1.0)

        # 5. Plot and save visualization output
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(overlay)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.axis("off")
        
        legend_patch = mpatches.Patch(color=color_arr, label='Detected Road Area')
        ax.legend(handles=[legend_patch], loc="upper right", frameon=True, facecolor="white")

        plt.tight_layout()
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved Debug Overlay: {output_png}")
        
    except Exception as e:
        import traceback
        print(f"[-] Failed to render debug overlay: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(image_path, config_path):
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    # Load config via pipeline_config
    cfg = load_pipeline_config(config_path)

    import json
    with open(config_path, "r") as f:
        raw_config_dict = json.load(f)

    vis_config = raw_config_dict.get("visualization", {})
        


    results_dir = os.path.join(REPO_ROOT, "Results")
    # Point to the two files generated by predict.py
    mask_tiff = os.path.join(results_dir, "unified_clean_mask.tif")
    heatmap_tiff = os.path.join(results_dir, "comfort_heatmap.tif")

    target_shape = None
    transform = None
    
    # Use the mask as the spatial reference for the RGB image
    if os.path.exists(mask_tiff):
        with rasterio.open(mask_tiff) as src:
            target_shape = (src.height, src.width)
            transform = src.transform

    rgb, transform = load_rgb(image_path, target_shape=target_shape)

    os.makedirs(results_dir, exist_ok=True)

    overlay_png    = os.path.join(results_dir, "overlay_combined.png")
    dino_overlay_png = os.path.join(results_dir, "overlay_dino.png")
    heatmap_png    = os.path.join(results_dir, "overlay_comfort_heatmap.png")
    land_cover_png = os.path.join(results_dir, "overlay_land_cover.png")
    clean_mask_png = os.path.join(results_dir, "overlay_cleaned_mask.png")
    nen8100_png    = os.path.join(results_dir, "overlay_nen8100.png")

    raw_models = json.loads(open(config_path, encoding="utf-8").read()).get("models", [])

    # ---------------------------------------------------------
    # 0: Render High-Res Roads Debug Overlay if enabled
    # ---------------------------------------------------------
    if vis_config.get("debug_high_res_roads", False):
        # Open the primary source TIF file to read its native spatial CRS structure
        with rasterio.open(image_path) as src:
            raster_crs = src.crs.to_string() if src.crs else "EPSG:3857"
        road_geojson = os.path.join(REPO_ROOT, "Results", "amenity_pipeline", "road_extraction.geojson")
        debug_png = os.path.join(REPO_ROOT, "Results", "debug_high_res_roads.png")
        
        road_color = vis_config.get("road_debug_color", [255, 0, 255])
        road_alpha = vis_config.get("road_debug_alpha", 0.65)
        
        render_debug_overlay(
                rgb=rgb, 
                geojson_path=road_geojson, 
                transform=transform, 
                raster_crs=raster_crs,  # Pass the raster projection fix here
                output_png=debug_png, 
                color=vis_config.get("road_debug_color", [255, 0, 255]), 
                alpha=vis_config.get("road_debug_alpha", 0.65),
                title="Debug: Esri High-Res Road Model Output"
            )

    # 1. Raw vector detections overlay
    render_combined_detections(rgb, transform, raw_models, overlay_png)
    render_dino_detections(rgb, transform, raw_models, dino_overlay_png)

    # 2. Raw land cover layer
    for mcfg in raw_models:
        if mcfg.get("name") == "land_cover":
            render_land_cover(rgb, transform, mcfg, land_cover_png)
            break

    # 3. Process Fused mask and pre-calculated Comfort Heatmap
    if os.path.exists(mask_tiff) and os.path.exists(heatmap_tiff):
        print(f"Found clean mask at: {mask_tiff} and heatmap at: {heatmap_tiff}")
        
        with rasterio.open(mask_tiff) as src:
            clean_map = src.read(1)
        with rasterio.open(heatmap_tiff) as src:
            heatmap = src.read(1)

        render_cleaned_mask(rgb, clean_map, clean_mask_png)
        render_nen8100(rgb, heatmap, nen8100_png)
        render_comfort_heatmap(rgb, heatmap, heatmap_png)

    else:
        print("Warning: Required rasters not found. Falling back to legacy legacy modes.")
        heatmap = generate_comfort_heatmap_legacy(rgb.shape, transform, raw_models)
        dummy_clean = np.zeros(rgb.shape[:2], dtype=np.uint8)
        render_nen8100(rgb, heatmap, nen8100_png)
        render_comfort_heatmap(rgb, heatmap, heatmap_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  default=os.path.join(REPO_ROOT, "Maps", "Tiles",
                                                          "Atlanta_split_google", "tile_003_002.tif"))
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.json"))
    args = parser.parse_args()
    run(args.image, args.config)