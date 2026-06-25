import os
import pickle
import hashlib
import json
from pathlib import Path
import numpy as np
from models import config as cfg


def get_cache_dir() -> Path:
    """Return cache directory, creating it if needed."""
    cache_root = getattr(cfg, "pipeline_cache_root", None)
    if cache_root is not None:
        cache_dir = Path(cache_root) / ".segmentation_cache"
    else:
        cache_dir = cfg.results_dir / ".segmentation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key_for_image_dino(image_path: str, image_hash: str) -> str:
    """Generate cache key for full-image DINO results."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_dino_full.pkl"


def get_cache_key_for_image_masks(image_path: str, image_hash: str) -> str:
    """Generate cache key for full-image SAM masks."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_masks_full.pkl"


def get_cache_key_for_tile_dino(image_path: str, image_hash: str, tile_idx: int, tile_count: int) -> str:
    """Generate cache key for tiled DINO results."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_tile{tile_idx}-of-{tile_count}_dino.pkl"


def get_cache_key_for_tile_masks(image_path: str, image_hash: str, tile_idx: int, tile_count: int) -> str:
    """Generate cache key for tiled SAM masks."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_tile{tile_idx}-of-{tile_count}_masks.pkl"


def get_cache_key_for_region_context_scoring(image_path: str, image_hash: str, prompt_signature: str) -> str:
    """Generate cache key for region-context CLIP scoring output."""
    stem = Path(image_path).stem
    return f"{stem}_hash{image_hash}_regionctx_{prompt_signature}.pkl"


def compute_image_hash(img_array: np.ndarray) -> str:
    """Compute a quick hash of image array to detect if image changed."""
    # Only hash shape and first/last pixel values to keep it fast
    hash_input = f"{img_array.shape}_{img_array.dtype}_{img_array[0,0].tobytes()}_{img_array[-1,-1].tobytes()}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]


def build_region_context_prompt_signature(prompt_configs: list[dict]) -> str:
    """Build a stable signature for the prompt scoring recipe used by region-context mode."""
    signature_payload = []
    for prompt_config in prompt_configs:
        signature_payload.append(
            {
                "name": prompt_config.get("name"),
                "caption": prompt_config.get("caption"),
                "negative_captions": list(prompt_config.get("negative_captions", [])),
                "clip_negative_weight": float(prompt_config.get("clip_negative_weight", 0.0)),
                "clip_min_area_ratio": float(prompt_config.get("clip_min_area_ratio", 0.0)),
                "clip_max_area_ratio": float(prompt_config.get("clip_max_area_ratio", 1.0)),
                "max_saturation": float(prompt_config.get("max_saturation", 1.0)),
                "min_value": float(prompt_config.get("min_value", 0.0)),
            }
        )

    signature_text = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(signature_text.encode("utf-8")).hexdigest()[:12]


def save_region_context_scoring_cache(
    image_path: str,
    image_hash: str,
    file_context: str,
    region_context_masks: list[dict],
    total_scored_masks: int,
    prompt_signature: str,
) -> bool:
    """Save region-context CLIP scoring outputs to cache."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return False
    if not region_context_masks:
        return False

    # Avoid caching full segmentation arrays for region-context scoring. Instead
    # write a tiny metadata file so we can detect that scoring ran without
    # storing the potentially huge per-region `segmentation` arrays.
    if bool(getattr(cfg, "skip_mask_caching", True)):
        try:
            cache_dir = get_cache_dir()
            cache_file = cache_dir / file_context
            # Build minimal metadata (do NOT include 'segmentation' arrays)
            minimal_masks = []
            for m in region_context_masks:
                md = {
                    "tile_bounds": m.get("tile_bounds"),
                    "predicted_iou": float(m.get("predicted_iou", 0.0)),
                    "stability_score": float(m.get("stability_score", 0.0)),
                }
                minimal_masks.append(md)

            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "prompt_signature": prompt_signature,
                        "total_scored_masks": int(total_scored_masks),
                        "region_context_masks_meta": minimal_masks,
                    },
                    f,
                )
            print(f"[INFO] Cached region-context scoring metadata: {cache_file.name}")
            return True
        except Exception as exc:
            print(f"[WARN] Failed to save region-context scoring cache metadata: {exc}")
            return False

    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "prompt_signature": prompt_signature,
                    "total_scored_masks": int(total_scored_masks),
                    "region_context_masks": region_context_masks,
                },
                f,
            )
        print(f"[INFO] Cached region-context scoring results: {cache_file.name}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to save region-context scoring cache: {exc}")
        return False


def load_region_context_scoring_cache(
    image_path: str,
    image_hash: str,
    file_context: str,
    prompt_signature: str,
) -> list[dict] | None:
    """Load region-context CLIP scoring outputs from cache."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return None
    if bool(getattr(cfg, "overwrite_pipeline_cache", False)):
        return None

    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        if not cache_file.exists():
            return None

        # If masking caching is skipped we still wrote metadata; however the
        # full segmentation arrays are not stored and cannot be restored. To
        # avoid accidental misuse, return None so callers recompute masks.
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        if data.get("prompt_signature") != prompt_signature:
            print(f"[INFO] Ignoring region-context cache with mismatched prompt signature: {cache_file.name}")
            return None

        if bool(getattr(cfg, "skip_mask_caching", True)):
            print(f"[INFO] Region-context cache exists but contains metadata only: {cache_file.name}")
            return None

        region_context_masks = data.get("region_context_masks")
        if not isinstance(region_context_masks, list) or not region_context_masks:
            return None

        print(f"[INFO] Loaded region-context scoring cache: {cache_file.name}")
        return region_context_masks
    except Exception as exc:
        print(f"[WARN] Failed to load region-context scoring cache: {exc}")
        return None


def save_dino_cache(image_path: str, image_hash: str, dino_records: list | None, file_context: str) -> bool:
    """Save DINO results to cache. Returns True if saved, False otherwise."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return False
    if dino_records is None:
        return False
    cache_empty_dino = bool(getattr(cfg, "cache_empty_dino_results", False))
    if len(dino_records) == 0 and not cache_empty_dino:
        print("[INFO] Skipping DINO cache write for empty result set")
        return False

    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        with open(cache_file, "wb") as f:
            pickle.dump({"dino_records": dino_records}, f)
        print(f"[INFO] Cached DINO results: {cache_file.name}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to save DINO cache: {e}")
        return False


def load_dino_cache(image_path: str, image_hash: str, file_context: str) -> list | None:
    """Load DINO results from cache. Returns None if not found or caching disabled."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return None
    if bool(getattr(cfg, "overwrite_pipeline_cache", False)):
        return None

    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        if not cache_file.exists():
            return None

        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        cached_records = data.get("dino_records")
        cache_empty_dino = bool(getattr(cfg, "cache_empty_dino_results", False))
        if isinstance(cached_records, list) and len(cached_records) == 0 and not cache_empty_dino:
            print(f"[INFO] Ignoring empty DINO cache entry: {cache_file.name}")
            return None
        print(f"[INFO] Loaded DINO from cache: {cache_file.name}")
        return cached_records
    except Exception as e:
        print(f"[WARN] Failed to load DINO cache: {e}")
        return None


def save_masks_cache(image_path: str, image_hash: str, masks: list | None, file_context: str) -> bool:
    """Save SAM masks to cache. Returns True if saved, False otherwise."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return False
    if masks is None or not masks:
        return False

    # Optionally skip writing full mask arrays to disk. Instead write a
    # compact metadata-only cache entry so we still know masks were produced
    # without storing multi-GB segmentation arrays per trial.
    if bool(getattr(cfg, "skip_mask_caching", True)):
        try:
            cache_dir = get_cache_dir()
            cache_file = cache_dir / file_context
            masks_meta = []
            for m in masks:
                masks_meta.append(
                    {
                        "tile_bounds": m.get("tile_bounds"),
                        "predicted_iou": float(m.get("predicted_iou", 0.0)),
                        "stability_score": float(m.get("stability_score", 0.0)),
                        "dino_prompt_group": m.get("dino_prompt_group"),
                    }
                )

            with open(cache_file, "wb") as f:
                pickle.dump({"masks_meta": masks_meta, "num_masks": len(masks)}, f)
            print(f"[INFO] Cached SAM masks metadata (no segmentation arrays): {cache_file.name}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to save masks cache metadata: {e}")
            return False

    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        with open(cache_file, "wb") as f:
            pickle.dump({"masks": masks}, f)
        print(f"[INFO] Cached {len(masks)} SAM masks: {cache_file.name}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to save masks cache: {e}")
        return False


def load_masks_cache(image_path: str, image_hash: str, file_context: str) -> list | None:
    """Load SAM masks from cache. Returns None if not found or caching disabled."""
    if not bool(getattr(cfg, "enable_pipeline_caching", True)):
        return None
    if bool(getattr(cfg, "overwrite_pipeline_cache", False)):
        return None

    try:
        cache_dir = get_cache_dir()
        cache_file = cache_dir / file_context
        if not cache_file.exists():
            return None

        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        # If we are skipping mask caching then cache files only contain
        # metadata and cannot be used to restore segmentation arrays.
        if bool(getattr(cfg, "skip_mask_caching", True)):
            print(f"[INFO] Found masks cache metadata but skipping full restore: {cache_file.name}")
            return None

        masks = data.get("masks")
        print(f"[INFO] Loaded {len(masks) if masks else 0} masks from cache: {cache_file.name}")
        return masks
    except Exception as e:
        print(f"[WARN] Failed to load masks cache: {e}")
        return None
