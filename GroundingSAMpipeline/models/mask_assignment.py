import numpy as np
from scipy.ndimage import distance_transform_edt
from models import config as cfg


def _get_chunk_size_px() -> int:
    return int(getattr(cfg, "contrastive_assignment_chunk_px", getattr(cfg, "sam_auto_tile_size_px", 1200)))


def expand_mask_to_full_image(mask_dict: dict) -> np.ndarray:
    if "tile_bounds" not in mask_dict:
        return mask_dict["segmentation"]
    ty0, ty1, tx0, tx1 = mask_dict["tile_bounds"]
    full_h, full_w = mask_dict["full_shape"]
    full_seg = np.zeros((full_h, full_w), dtype=bool)
    full_seg[ty0:ty1, tx0:tx1] = mask_dict["segmentation"]
    return full_seg


def _update_score_map_from_mask(
    score_map: np.ndarray,
    mask_dict: dict,
    score: float,
    chunk_y0: int,
    chunk_x0: int,
) -> None:
    if "tile_bounds" in mask_dict:
        ty0, ty1, tx0, tx1 = mask_dict["tile_bounds"]
        y0 = max(chunk_y0, int(ty0))
        y1 = min(chunk_y0 + score_map.shape[0], int(ty1))
        x0 = max(chunk_x0, int(tx0))
        x1 = min(chunk_x0 + score_map.shape[1], int(tx1))
        if y0 >= y1 or x0 >= x1:
            return

        seg_y0 = y0 - int(ty0)
        seg_y1 = seg_y0 + (y1 - y0)
        seg_x0 = x0 - int(tx0)
        seg_x1 = seg_x0 + (x1 - x0)
        seg = mask_dict["segmentation"][seg_y0:seg_y1, seg_x0:seg_x1]
        local_y0 = y0 - chunk_y0
        local_y1 = y1 - chunk_y0
        local_x0 = x0 - chunk_x0
        local_x1 = x1 - chunk_x0
        local_view = score_map[local_y0:local_y1, local_x0:local_x1]
        update_pixels = seg & (score > local_view)
        if np.any(update_pixels):
            local_view[update_pixels] = score
        return

    seg = mask_dict["segmentation"]
    local_y0 = max(0, chunk_y0)
    local_x0 = max(0, chunk_x0)
    local_y1 = min(score_map.shape[0], seg.shape[0])
    local_x1 = min(score_map.shape[1], seg.shape[1])
    if local_y0 >= local_y1 or local_x0 >= local_x1:
        return
    seg_view = seg[local_y0:local_y1, local_x0:local_x1]
    local_view = score_map[local_y0:local_y1, local_x0:local_x1]
    update_pixels = seg_view & (score > local_view)
    if np.any(update_pixels):
        local_view[update_pixels] = score


def _assign_region_context_masks(
    scored_masks: list[dict],
    prompt_names_ordered: list[str],
    prompt_weights: dict[str, float] | None = None,
) -> dict[str, list[dict]]:
    prompt_weights = prompt_weights or {}
    assigned_masks: dict[str, list[dict]] = {prompt_name: [] for prompt_name in prompt_names_ordered}

    for mask_dict in scored_masks:
        best_prompt_name: str | None = None
        best_score = float("-inf")
        for prompt_name in prompt_names_ordered:
            score_key = f"clip_score_{prompt_name}"
            score = float(mask_dict.get(score_key, mask_dict.get("score", float("-inf"))))
            weighted_score = score * float(prompt_weights.get(prompt_name, 1.0))
            if weighted_score > best_score:
                best_score = weighted_score
                best_prompt_name = prompt_name

        if best_prompt_name is not None:
            assigned_masks[best_prompt_name].append(mask_dict)

    total_assigned = sum(len(v) for v in assigned_masks.values())
    print(f"[INFO] Region-context assignment complete: assigned {total_assigned} SAM regions")
    for prompt_name in prompt_names_ordered:
        print(f"[INFO]   {prompt_name}: {len(assigned_masks[prompt_name]):,} regions")

    return assigned_masks


def _build_region_context_full_image_masks(
    selected_masks_by_prompt: dict[str, list[dict]],
    image_shape: tuple[int, int],
    prompt_names_ordered: list[str],
) -> dict[str, np.ndarray]:
    """Build full-image masks from region-context prompt assignments in one pass."""
    h, w = image_shape
    full_image_masks: dict[str, np.ndarray] = {
        prompt_name: np.zeros((h, w), dtype=bool) for prompt_name in prompt_names_ordered
    }

    total_regions = 0
    for prompt_name in prompt_names_ordered:
        prompt_regions = selected_masks_by_prompt.get(prompt_name, [])
        prompt_mask = full_image_masks[prompt_name]
        prompt_region_count = 0
        for mask in prompt_regions:
            # More memory-efficient: if this mask is a tiled segmentation, OR the
            # small tile slice directly into the full-image mask instead of
            # allocating a full-size boolean array for every region.
            try:
                if "tile_bounds" in mask:
                    ty0, ty1, tx0, tx1 = mask["tile_bounds"]
                    seg = mask.get("segmentation")
                    if seg is None:
                        continue
                    # Validate shapes
                    th = int(ty1) - int(ty0)
                    tw = int(tx1) - int(tx0)
                    if seg.shape[0] == th and seg.shape[1] == tw:
                        if np.any(seg):
                            prompt_mask[ty0:ty1, tx0:tx1] |= seg
                        else:
                            # empty segment
                            pass
                    else:
                        # Fallback to expansion if shapes don't match
                        region_mask = expand_mask_to_full_image(mask)
                        if np.any(region_mask):
                            prompt_mask |= region_mask
                else:
                    seg = mask.get("segmentation")
                    if seg is None:
                        continue
                    if seg.shape == prompt_mask.shape:
                        if np.any(seg):
                            prompt_mask |= seg
                    else:
                        region_mask = expand_mask_to_full_image(mask)
                        if np.any(region_mask):
                            prompt_mask |= region_mask
            except Exception as exc:
                print(f"[WARN] Region-context mask merge failed for one region: {exc}")
                continue
            total_regions += 1
            prompt_region_count += 1
        print(f"[INFO] Region-context mask build: {prompt_name} -> {prompt_region_count:,} regions")

    print(f"[INFO] Region-context full-image masks built from {total_regions:,} regions")

    coverage_union = np.zeros((h, w), dtype=bool)
    prompt_index_map = np.zeros((h, w), dtype=np.uint8)
    for prompt_idx, prompt_name in enumerate(prompt_names_ordered):
        mask = full_image_masks[prompt_name]
        coverage_union |= mask
        prompt_index_map[mask] = np.uint8(prompt_idx)

    if not np.all(coverage_union) and np.any(coverage_union):
        uncovered_count = int((~coverage_union).sum())
        print(
            f"[INFO] Region-context fill: assigning {uncovered_count:,} uncovered pixels to nearest prompt region"
        )
        _, indices = distance_transform_edt(~coverage_union, return_indices=True)
        prompt_index_map = prompt_index_map[tuple(indices)]

        full_image_masks = {
            prompt_name: (prompt_index_map == np.uint8(prompt_idx))
            for prompt_idx, prompt_name in enumerate(prompt_names_ordered)
        }

    elif not np.any(coverage_union):
        print("[WARN] Region-context produced no coverage; leaving masks empty")

    return full_image_masks


def _fill_full_image_mask_gaps(
    combined_masks: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    prompt_names_ordered: list[str],
) -> dict[str, np.ndarray]:
    """Ensure every pixel is assigned to some prompt by filling gaps from the nearest label."""
    h, w = image_shape
    filled_masks: dict[str, np.ndarray] = {
        prompt_name: np.asarray(combined_masks.get(prompt_name, np.zeros((h, w), dtype=bool)), dtype=bool)
        for prompt_name in prompt_names_ordered
    }

    coverage_union = np.zeros((h, w), dtype=bool)
    prompt_index_map = np.zeros((h, w), dtype=np.uint8)
    for prompt_idx, prompt_name in enumerate(prompt_names_ordered):
        mask = filled_masks[prompt_name]
        coverage_union |= mask
        prompt_index_map[mask] = np.uint8(prompt_idx)

    if np.all(coverage_union):
        return filled_masks

    if not np.any(coverage_union):
        print("[WARN] No prompt masks cover any pixels; returning empty masks")
        return filled_masks

    uncovered_count = int((~coverage_union).sum())
    print(f"[INFO] Filling {uncovered_count:,} uncovered pixels from nearest prompt region")

    _, indices = distance_transform_edt(~coverage_union, return_indices=True)
    nearest_prompt_indices = prompt_index_map[tuple(indices)]

    return {
        prompt_name: (nearest_prompt_indices == np.uint8(prompt_idx))
        for prompt_idx, prompt_name in enumerate(prompt_names_ordered)
    }


def _apply_coarse_to_fine_pass(
    combined_masks: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    prompt_names_ordered: list[str],
    cell_px: int,
    majority_threshold: float = 0.65,
) -> dict[str, np.ndarray]:
    """Smooth the final label map by snapping confident coarse cells to one label."""
    if cell_px <= 1:
        return combined_masks

    h, w = image_shape
    if not prompt_names_ordered:
        return combined_masks

    label_map = np.zeros((h, w), dtype=np.uint8)
    for prompt_idx, prompt_name in enumerate(prompt_names_ordered):
        mask = np.asarray(combined_masks.get(prompt_name, np.zeros((h, w), dtype=bool)), dtype=bool)
        label_map[mask] = np.uint8(prompt_idx)

    smoothed_map = label_map.copy()
    confident_cells = 0
    total_cells = 0

    for y0 in range(0, h, cell_px):
        y1 = min(h, y0 + cell_px)
        for x0 in range(0, w, cell_px):
            x1 = min(w, x0 + cell_px)
            cell = label_map[y0:y1, x0:x1]
            total_cells += 1
            counts = np.bincount(cell.ravel(), minlength=len(prompt_names_ordered))
            winner_idx = int(np.argmax(counts))
            winner_fraction = float(counts[winner_idx]) / float(cell.size)
            if winner_fraction >= majority_threshold:
                smoothed_map[y0:y1, x0:x1] = np.uint8(winner_idx)
                confident_cells += 1

    print(
        f"[INFO] Coarse-to-fine smoothing: snapped {confident_cells:,}/{total_cells:,} cells "
        f"at cell size {cell_px}px"
    )

    return {
        prompt_name: (smoothed_map == np.uint8(prompt_idx))
        for prompt_idx, prompt_name in enumerate(prompt_names_ordered)
    }


def _apply_tier_thresholds(
    combined_masks: dict[str, np.ndarray],
    selected_masks_by_prompt: dict[str, list[dict]],
    image_shape: tuple[int, int],
    e_threshold: float = 0.25,
    c_threshold: float = 0.15,
    d_threshold: float = 0.0,
    b_threshold: float = 0.0,
) -> dict[str, np.ndarray]:
    """Apply confidence thresholds to tier assignments.
    
    Only assign E (uncomfortable) tier if score > e_threshold.
    Only assign C (pedestrian) tier if score > c_threshold.
    Otherwise default to A (comfortable) tier.
    """
    h, w = image_shape
    
    # Build full-image score maps for each tier (A/B/C/D/E)
    tier_scores = {
        "nen_cat_a": np.full((h, w), -np.inf, dtype=np.float32),
        "nen_cat_b": np.full((h, w), -np.inf, dtype=np.float32),
        "nen_cat_c": np.full((h, w), -np.inf, dtype=np.float32),
        "nen_cat_d": np.full((h, w), -np.inf, dtype=np.float32),
        "nen_cat_e": np.full((h, w), -np.inf, dtype=np.float32),
    }
    
    # Fill score maps from selected masks
    for prompt_name, masks in selected_masks_by_prompt.items():
        if prompt_name not in tier_scores:
            continue
        
        for mask_dict in masks:
            mask_bool = expand_mask_to_full_image(mask_dict)
            if not np.any(mask_bool):
                continue
            
            score_key = f"clip_score_{prompt_name}"
            score = float(mask_dict.get(score_key, mask_dict.get("score", 0.0)))
            
            # Update pixels where this score is better than previously seen
            update_pixels = mask_bool & (score > tier_scores[prompt_name])
            if np.any(update_pixels):
                tier_scores[prompt_name][update_pixels] = score
    
    # For each pixel, apply threshold-based tier selection: E > D > C > B > A
    e_scores = tier_scores["nen_cat_e"]
    d_scores = tier_scores["nen_cat_d"]
    c_scores = tier_scores["nen_cat_c"]
    b_scores = tier_scores["nen_cat_b"]

    # Build refined masks based on thresholds
    e_confident = e_scores > e_threshold
    d_confident = d_scores > d_threshold
    c_confident = c_scores > c_threshold
    b_confident = b_scores > b_threshold

    refined_masks = {
        "nen_cat_a": np.zeros((h, w), dtype=bool),
        "nen_cat_b": np.zeros((h, w), dtype=bool),
        "nen_cat_c": np.zeros((h, w), dtype=bool),
        "nen_cat_d": np.zeros((h, w), dtype=bool),
        "nen_cat_e": np.zeros((h, w), dtype=bool),
    }

    # Assign tiers in priority order
    refined_masks["nen_cat_e"] = e_confident.copy()
    refined_masks["nen_cat_d"] = d_confident.copy() & ~refined_masks["nen_cat_e"]
    refined_masks["nen_cat_c"] = c_confident.copy() & ~refined_masks["nen_cat_e"] & ~refined_masks["nen_cat_d"]
    refined_masks["nen_cat_b"] = b_confident.copy() & ~refined_masks["nen_cat_e"] & ~refined_masks["nen_cat_d"] & ~refined_masks["nen_cat_c"]

    # Fallback to A for any remaining pixels
    remaining = ~(refined_masks["nen_cat_e"] | refined_masks["nen_cat_d"] | refined_masks["nen_cat_c"] | refined_masks["nen_cat_b"])
    refined_masks["nen_cat_a"] = remaining.copy()
    
    # Ensure pixel-level winner-takes-all to avoid overlaps for the tiered classes.
    # Give priority: E > C > A.
    final_masks = {
        "nen_cat_e": refined_masks["nen_cat_e"].copy(),
        "nen_cat_c": refined_masks["nen_cat_c"].copy() & ~refined_masks["nen_cat_e"],
        "nen_cat_a": refined_masks["nen_cat_a"].copy() & ~refined_masks["nen_cat_e"] & ~refined_masks["nen_cat_c"],
    }

    # Preserve any non-tiered prompts (for split A-E runs this keeps B and D alive).
    for prompt_name, mask in combined_masks.items():
        if prompt_name not in final_masks:
            final_masks[prompt_name] = mask.copy()
    
    # Print statistics
    print("[INFO] Tier threshold refinement:")
    print(f"  E threshold: {e_threshold:.3f}, pixels assigned: {final_masks['nen_cat_e'].sum():,}")
    print(f"  D threshold: {d_threshold:.3f}, pixels assigned: {final_masks['nen_cat_d'].sum():,}")
    print(f"  C threshold: {c_threshold:.3f}, pixels assigned: {final_masks['nen_cat_c'].sum():,}")
    print(f"  B threshold: {b_threshold:.3f}, pixels assigned: {final_masks['nen_cat_b'].sum():,}")
    print(f"  A (fallback): pixels assigned: {final_masks['nen_cat_a'].sum():,}")
    for prompt_name, mask in final_masks.items():
        if prompt_name not in {"nen_cat_a", "nen_cat_c", "nen_cat_e"}:
            print(f"  {prompt_name}: pixels preserved: {mask.sum():,}")
    
    return final_masks


def _build_contrastive_full_image_masks(
    scored_masks_by_prompt: dict[str, list[dict]],
    image_shape: tuple[int, int],
    prompt_names_ordered: list[str],
    prompt_weights: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Assign each pixel by contrastive score across prompts.

    For each prompt, we build a per-pixel score map using the highest CLIP score from
    any selected mask covering that pixel. We then compare each prompt against the
    mean score of the other prompts and assign each pixel to the prompt with the
    highest contrastive score.
    """
    h, w = image_shape
    if not prompt_names_ordered:
        return {}

    prompt_weights = prompt_weights or {}
    weight_values = np.array(
        [float(prompt_weights.get(prompt_name, 1.0)) for prompt_name in prompt_names_ordered],
        dtype=np.float32,
    )

    chunk_size = max(256, _get_chunk_size_px())
    total_scored_masks = sum(len(v) for v in scored_masks_by_prompt.values())
    print(
        f"[INFO] Contrastive assignment scanning {total_scored_masks} scored masks in {chunk_size}px chunks"
    )
    if any(abs(weight - 1.0) > 1e-6 for weight in weight_values):
        weight_str = ", ".join(
            f"{prompt_name}={weight_values[idx]:.2f}" for idx, prompt_name in enumerate(prompt_names_ordered)
        )
        print(f"[INFO] Contrastive prompt weights: {weight_str}")

    global_covered = np.zeros((h, w), dtype=bool)
    global_winners = np.zeros((h, w), dtype=np.uint8)

    for y0 in range(0, h, chunk_size):
        y1 = min(h, y0 + chunk_size)
        chunk_h = y1 - y0
        for x0 in range(0, w, chunk_size):
            x1 = min(w, x0 + chunk_size)
            chunk_w = x1 - x0

            score_maps = []
            for prompt_name in prompt_names_ordered:
                # Initialize to -np.inf to preserve actual negative scores
                score_map = np.full((chunk_h, chunk_w), -np.inf, dtype=np.float32)
                score_key = f"clip_score_{prompt_name}"
                for mask_dict in scored_masks_by_prompt.get(prompt_name, []):
                    score = float(mask_dict.get(score_key, mask_dict.get("score", 0.0)))
                    _update_score_map_from_mask(score_map, mask_dict, score, y0, x0)
                score_maps.append(score_map)

            stack = np.stack(score_maps, axis=0)

            # Record pixel coverage: a pixel is covered if at least one prompt has a score > -inf
            chunk_covered = np.any(stack != -np.inf, axis=0)
            global_covered[y0:y1, x0:x1] = chunk_covered

            # Clean the stack: replace -np.inf values with a low neutral fallback (-1.0)
            stack_clean = np.where(stack == -np.inf, -1.0, stack)

            weighted_stack = stack_clean * weight_values[:, None, None]
            contrastive = np.zeros_like(weighted_stack)
            for idx in range(len(prompt_names_ordered)):
                others = np.delete(weighted_stack, idx, axis=0)
                mean_others = others.mean(axis=0)
                contrastive[idx] = weighted_stack[idx] - mean_others

            winners = np.argmax(contrastive, axis=0)
            global_winners[y0:y1, x0:x1] = winners

    # Propagate labels to any pixels that were never covered by any mask
    if not np.all(global_covered):
        uncovered_count = np.sum(~global_covered)
        if np.any(global_covered):
            print(
                f"[INFO] Propagating nearest category labels to {uncovered_count} uncovered pixels "
                "using scipy.ndimage.distance_transform_edt"
            )
            _, indices = distance_transform_edt(~global_covered, return_indices=True)
            global_winners = global_winners[tuple(indices)]
        else:
            print("[WARN] No masks generated across any prompt; keeping default category assignments.")

    full_image_masks = {}
    for idx, prompt_name in enumerate(prompt_names_ordered):
        full_image_masks[prompt_name] = (global_winners == idx)

    print(f"[INFO] Contrastive assignment complete: scored {total_scored_masks} masks")
    return full_image_masks


def _build_full_image_prompt_masks(
    selected_masks_by_prompt: dict[str, list[dict]],
    image_shape: tuple[int, int],
    prompt_names_ordered: list[str],
) -> dict[str, np.ndarray]:
    """Build a full-image, winner-takes-all mask for each prompt.

    Pixels covered by one or more prompt masks are assigned to the prompt with the highest
    score at that pixel. Uncovered pixels are filled by the nearest labeled region so the
    output covers the entire image.
    """
    h, w = image_shape
    if not prompt_names_ordered:
        return {}

    # Memory-safe winner-takes-all pass: avoid stacking per-prompt full-size score maps.
    best_score_map = np.full((h, w), -np.inf, dtype=np.float32)
    winner_indices = np.zeros((h, w), dtype=np.uint8)
    covered = np.zeros((h, w), dtype=bool)
    prompt_best_scores: dict[str, float] = {name: float("-inf") for name in prompt_names_ordered}

    for prompt_idx, prompt_name in enumerate(prompt_names_ordered):
        score_key = f"clip_score_{prompt_name}"
        for mask in selected_masks_by_prompt.get(prompt_name, []):
            mask_bool = expand_mask_to_full_image(mask)
            if not np.any(mask_bool):
                continue

            score = float(mask.get(score_key, mask.get("score", 0.0)))
            prompt_best_scores[prompt_name] = max(prompt_best_scores[prompt_name], score)

            update_pixels = mask_bool & (score > best_score_map)
            if not np.any(update_pixels):
                continue

            best_score_map[update_pixels] = score
            winner_indices[update_pixels] = np.uint8(prompt_idx)
            covered[update_pixels] = True

    # Ensure full-image assignment for pixels that were never covered by any selected mask.
    if not np.all(covered):
        best_prompt_idx = 0
        best_prompt_score = float("-inf")
        for idx, prompt_name in enumerate(prompt_names_ordered):
            score = prompt_best_scores.get(prompt_name, float("-inf"))
            if score > best_prompt_score:
                best_prompt_idx = idx
                best_prompt_score = score

        # If all prompts are empty, keep index 0 as deterministic fallback.
        winner_indices[~covered] = np.uint8(best_prompt_idx)

    full_image_masks = {}
    for idx, prompt_name in enumerate(prompt_names_ordered):
        full_image_masks[prompt_name] = winner_indices == idx

    return full_image_masks


def _build_prompt_strength_heatmap(
    masks_by_prompt: dict[str, list[dict]],
    image_shape: tuple[int, int],
    prompt_name: str,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Build a full-image CLIP strength map for one prompt.

    The map stores the strongest score seen for that prompt at each pixel.
    Any uncovered pixels are later filled from the nearest covered region so the
    output spans the full image even when prompt coverage is sparse.
    """
    h, w = image_shape
    score_map = np.full((h, w), -np.inf, dtype=np.float32)
    covered = np.zeros((h, w), dtype=bool)
    score_key = f"clip_score_{prompt_name}"

    for mask in masks_by_prompt.get(prompt_name, []):
        score = float(mask.get(score_key, mask.get("score", float("-inf"))))
        if not np.isfinite(score):
            continue

        if "tile_bounds" in mask:
            ty0, ty1, tx0, tx1 = mask["tile_bounds"]
            seg = mask.get("segmentation")
            if seg is not None:
                th = int(ty1) - int(ty0)
                tw = int(tx1) - int(tx0)
                if seg.shape[0] == th and seg.shape[1] == tw:
                    local_view = score_map[int(ty0):int(ty1), int(tx0):int(tx1)]
                    update_pixels = seg & (score > local_view)
                    if np.any(update_pixels):
                        local_view[update_pixels] = score
                        covered[int(ty0):int(ty1), int(tx0):int(tx1)] |= seg
                    continue

        region_mask = expand_mask_to_full_image(mask)
        if not np.any(region_mask):
            continue
        update_pixels = region_mask & (score > score_map)
        if np.any(update_pixels):
            score_map[update_pixels] = score
        covered |= region_mask

    if not np.any(covered):
        return None, None

    if not np.all(covered):
        print(f"[INFO] Prompt strength heatmap fill starting for {prompt_name}: uncovered pixels need nearest-neighbor fill")
        _, indices = distance_transform_edt(~covered, return_indices=True)
        score_map = score_map[tuple(indices)]
        print(f"[INFO] Prompt strength heatmap fill finished for {prompt_name}")

    return score_map, covered
