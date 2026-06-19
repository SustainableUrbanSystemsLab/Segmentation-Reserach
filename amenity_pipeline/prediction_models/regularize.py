"""
Post-process building footprints to regularize edges to right angles,
similar to ArcGIS Pro's "Regularize Building Footprint" tool.

Usage:
    uv run python 4_regularize.py
    uv run python 4_regularize.py --input output/footprints.geojson --tolerance 1.0
"""
import argparse, json, os
import numpy as np
from shapely.geometry import shape, mapping, Polygon
from shapely.affinity import rotate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PIPELINE_ROOT, ".."))


def dominant_angle(coords):
    """Angle of the longest edge, used as the building's principal direction."""
    edges = np.diff(np.vstack([coords, coords[0]]), axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    i = np.argmax(lengths)
    return np.degrees(np.arctan2(edges[i, 1], edges[i, 0]))


def snap_to_right_angles(coords):
    """
    Snap each vertex so that edges become axis-aligned.
    For each vertex: if the incoming edge is more horizontal, lock y to the
    previous vertex's y; if more vertical, lock x to the previous vertex's x.
    One pass is enough for convex shapes; two passes handles concave ones.
    """
    n = len(coords)
    c = coords.copy()
    for _ in range(2):
        for i in range(n):
            prev = c[(i - 1) % n]
            curr = c[i].copy()
            e = curr - prev
            if abs(e[0]) >= abs(e[1]):   # incoming edge is more horizontal
                curr[1] = prev[1]
            else:                          # incoming edge is more vertical
                curr[0] = prev[0]
            c[i] = curr
    return c


def regularize_polygon(poly, simplify_tol):
    poly = poly.simplify(simplify_tol, preserve_topology=True)
    if not poly.is_valid or poly.is_empty or poly.geom_type != "Polygon":
        return poly

    coords = np.array(poly.exterior.coords[:-1])
    if len(coords) < 3:
        return poly

    angle   = dominant_angle(coords)
    centroid = poly.centroid

    # Rotate so the dominant direction aligns with x-axis
    rotated  = rotate(poly, -angle, origin=centroid)
    rc       = np.array(rotated.exterior.coords[:-1], dtype=float)

    snapped  = snap_to_right_angles(rc)
    snapped  = np.vstack([snapped, snapped[0]])

    reg = Polygon(snapped)
    if not reg.is_valid:
        reg = reg.buffer(0)
    if reg.is_empty:
        return poly

    # Rotate back to original orientation
    return rotate(reg, angle, origin=centroid)


def run(input_path, output_path, simplify_tol):
    with open(input_path) as f:
        geojson = json.load(f)

    out_features = []
    skipped = 0
    for feat in geojson["features"]:
        geom = shape(feat["geometry"])
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        reg_polys = []
        for p in polys:
            r = regularize_polygon(p, simplify_tol)
            if r and not r.is_empty:
                reg_polys.append(r)

        if not reg_polys:
            skipped += 1
            continue

        result = reg_polys[0] if len(reg_polys) == 1 else reg_polys[0].union(*reg_polys[1:])
        out_features.append({
            "type": "Feature",
            "properties": feat["properties"],
            "geometry": mapping(result)
        })

    out = {"type": "FeatureCollection", "features": out_features}
    with open(output_path, "w") as f:
        json.dump(out, f)
    print(f"Done. {len(out_features)} regularized ({skipped} skipped) → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default=os.path.join(REPO_ROOT, "Results", "amenity_pipeline", "buildings.geojson"))
    parser.add_argument("--output",    default=os.path.join(REPO_ROOT, "Results", "amenity_pipeline", "buildings_reg.geojson"))
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Simplification tolerance in map units before snapping")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    run(args.input, args.output, args.tolerance)