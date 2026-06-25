from time import perf_counter
import numpy as np
import torch
from matplotlib.colors import rgb_to_hsv
from PIL import Image
import open_clip
from models import config as cfg


def _extract_mask_geometry(mask_dict: dict) -> tuple | None:
    """Extract geometric coordinates and area safely from a mask dict. Returns None if invalid."""
    tile_seg = mask_dict["segmentation"]
    tile_bounds = mask_dict.get("tile_bounds")

    ys, xs = np.where(tile_seg)
    if ys.size == 0 or xs.size == 0:
        return None

    ly0, ly1 = int(ys.min()), int(ys.max()) + 1
    lx0, lx1 = int(xs.min()), int(xs.max()) + 1

    if tile_bounds is not None:
        ty0, _, tx0, _ = tile_bounds
        full_h, full_w = mask_dict["full_shape"]
        iy0, iy1 = ly0 + ty0, ly1 + ty0
        ix0, ix1 = lx0 + tx0, lx1 + tx0
    else:
        full_h, full_w = tile_seg.shape[:2]
        iy0, iy1, ix0, ix1 = ly0, ly1, lx0, lx1

    area_ratio = float(tile_seg.sum()) / float(full_h * full_w)
    h = max(1, iy1 - iy0)
    w = max(1, ix1 - ix0)
    bbox_fill = float(tile_seg.sum()) / float(h * w)

    return (iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, full_h, full_w)


class ClipScorer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        print(f"[INFO] Loading CLIP model (ViT-B-32) on device '{self.device}'")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def build_text_features(self, target_prompt: str, negative_prompts: list[str] | None = None):
        with torch.no_grad():
            pos_templates = [
                f"aerial image of {target_prompt}",
                f"overhead view of {target_prompt}",
                f"satellite photo of {target_prompt}",
                f"{target_prompt} in an urban area",
            ]

            pos_tokens = self.tokenizer(pos_templates).to(self.device)
            pos_features = self.model.encode_text(pos_tokens)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            pos_text_feature = pos_features.mean(dim=0, keepdim=True)
            pos_text_feature = pos_text_feature / pos_text_feature.norm(dim=-1, keepdim=True)

            neg_text_feature = None
            if negative_prompts:
                neg_templates: list[str] = []
                for neg in negative_prompts:
                    neg_templates.extend(
                        [
                            f"aerial image of {neg}",
                            f"overhead view of {neg}",
                            f"satellite photo of {neg}",
                        ]
                    )
                neg_tokens = self.tokenizer(neg_templates).to(self.device)
                neg_features = self.model.encode_text(neg_tokens)
                neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
                neg_text_feature = neg_features.mean(dim=0, keepdim=True)
                neg_text_feature = neg_text_feature / neg_text_feature.norm(dim=-1, keepdim=True)

        return pos_text_feature, neg_text_feature

    def score_mask(
        self,
        mask_dict: dict,
        pos_text_feature: torch.Tensor,
        neg_text_feature: torch.Tensor | None,
        prompt_config: dict,
        img_model: np.ndarray,
    ) -> float:
        """Score one mask with CLIP plus lightweight geometry and color heuristics."""
        geom = _extract_mask_geometry(mask_dict)
        if geom is None:
            return -1.0

        iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, _, _ = geom

        min_area = prompt_config.get("clip_min_area_ratio", 0.00015)
        max_area = prompt_config.get("clip_max_area_ratio", 0.25)
        if area_ratio < min_area or area_ratio > max_area:
            return -1.0

        tile_seg = mask_dict["segmentation"]
        crop = img_model[iy0:iy1, ix0:ix1]
        crop_mask = tile_seg[ly0:ly1, lx0:lx1][..., None]
        object_only = np.where(crop_mask, crop, 255).astype(np.uint8)
        pil_img = Image.fromarray(object_only)

        with torch.no_grad():
            image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            pos_sim = float((image_features @ pos_text_feature.T).item())
            score = pos_sim

            if neg_text_feature is not None:
                neg_sim = float((image_features @ neg_text_feature.T).item())
                neg_weight = float(prompt_config.get("clip_negative_weight", 0.35))
                score -= neg_weight * max(0.0, neg_sim)

            iou = float(mask_dict.get("predicted_iou", 0.0))
            stability = float(mask_dict.get("stability_score", 0.0))
            score += 0.05 * iou + 0.05 * stability

        aspect = max(h, w) / float(min(h, w))
        bbox_fill_score = np.clip((0.65 - bbox_fill) / 0.65, 0.0, 1.0)
        elongation_score = np.clip((aspect - 1.5) / 6.0, 0.0, 1.0)
        score += 0.12 * float(elongation_score)
        score += 0.06 * float(bbox_fill_score)

        mask_pixels = img_model[iy0:iy1, ix0:ix1][tile_seg[ly0:ly1, lx0:lx1]]
        if mask_pixels.size > 0:
            pixels_norm = mask_pixels.astype(np.float32) / 255.0
            hsv = rgb_to_hsv(pixels_norm.reshape(-1, 1, 3)).reshape(-1, 3)
            sat = float(hsv[:, 1].mean())
            val = float(hsv[:, 2].mean())

            max_sat = prompt_config.get("max_saturation", 0.28)
            min_val = prompt_config.get("min_value", 0.35)

            low_sat_score = np.clip((max_sat - sat) / max_sat, 0.0, 1.0)
            bright_enough_score = np.clip((val - min_val) / (1.0 - min_val), 0.0, 1.0)

            score += 0.05 * float(low_sat_score)
            score += 0.04 * float(bright_enough_score)

        return score

    def batch_score_masks_clip(
        self,
        masks_input: list,
        pos_text_feature: torch.Tensor,
        neg_text_feature: torch.Tensor | None,
        prompt_config: dict,
        img_model: np.ndarray,
    ) -> list[float]:
        """Batch-score masks using vectorized CLIP encoding. ~2-4x faster than serial."""
        scores = [0.0] * len(masks_input)

        # First pass: quick geometric filtering and crop preparation
        min_area = prompt_config.get("clip_min_area_ratio", 0.00015)
        max_area = prompt_config.get("clip_max_area_ratio", 0.25)

        candidates = []

        for idx, m in enumerate(masks_input):
            geom = _extract_mask_geometry(m)
            if geom is None:
                scores[idx] = -1.0
                continue

            iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, full_h, full_w = geom

            # Area filtering
            if area_ratio < min_area or area_ratio > max_area:
                scores[idx] = -1.0
                continue

            # Prepare crop
            tile_seg = m["segmentation"]
            candidates.append((idx, iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, tile_seg, full_h, full_w, m))

        if not candidates:
            return scores

        neg_weight = float(prompt_config.get("clip_negative_weight", 0.35))
        batch_size = max(1, int(prompt_config.get("clip_batch_size", getattr(cfg, "clip_batch_size", 16))))

        for start_idx in range(0, len(candidates), batch_size):
            batch_slice = candidates[start_idx : start_idx + batch_size]
            batch_tensors = []
            for (_, iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, tile_seg, full_h, full_w, m) in batch_slice:
                crop = img_model[iy0:iy1, ix0:ix1]
                crop_mask = tile_seg[ly0:ly1, lx0:lx1][..., None]
                object_only = np.where(crop_mask, crop, 255).astype(np.uint8)
                pil_img = Image.fromarray(object_only)
                batch_tensors.append(self.preprocess(pil_img))

            batch_tensor = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            for local_idx, (orig_idx, iy0, iy1, ix0, ix1, ly0, ly1, lx0, lx1, h, w, area_ratio, bbox_fill, tile_seg, full_h, full_w, m) in enumerate(batch_slice):
                image_features = batch_features[local_idx : local_idx + 1]

                pos_sim = float((image_features @ pos_text_feature.T).item())
                score = pos_sim

                if neg_text_feature is not None:
                    neg_sim = float((image_features @ neg_text_feature.T).item())
                    score -= neg_weight * max(0.0, neg_sim)

                iou = float(m.get("predicted_iou", 0.0))
                stability = float(m.get("stability_score", 0.0))
                score += 0.05 * iou + 0.05 * stability

                aspect = max(h, w) / float(min(h, w))
                elongation_score = np.clip((aspect - 1.5) / 6.0, 0.0, 1.0)
                sparse_fill_score = np.clip((0.65 - bbox_fill) / 0.65, 0.0, 1.0)
                score += 0.12 * float(elongation_score)
                score += 0.06 * float(sparse_fill_score)

                mask_pixels = img_model[iy0:iy1, ix0:ix1][tile_seg[ly0:ly1, lx0:lx1]]
                if mask_pixels.size > 0:
                    pixels_norm = mask_pixels.astype(np.float32) / 255.0
                    hsv = rgb_to_hsv(pixels_norm.reshape(-1, 1, 3)).reshape(-1, 3)
                    sat = float(hsv[:, 1].mean())
                    val = float(hsv[:, 2].mean())

                    max_sat = prompt_config.get("max_saturation", 0.28)
                    min_val = prompt_config.get("min_value", 0.35)

                    low_sat_score = np.clip((max_sat - sat) / max_sat, 0.0, 1.0)
                    bright_enough_score = np.clip((val - min_val) / (1.0 - min_val), 0.0, 1.0)

                    score += 0.05 * float(low_sat_score)
                    score += 0.04 * float(bright_enough_score)

                scores[orig_idx] = score

            del batch_tensor
            del batch_features

        return scores

    def serial_score_masks_clip(
        self,
        masks_input: list,
        pos_text_feature: torch.Tensor,
        neg_text_feature: torch.Tensor | None,
        prompt_config: dict,
        img_model: np.ndarray,
    ) -> list[float]:
        """Fallback CLIP scorer that processes one mask at a time."""
        scores: list[float] = []
        for mask_dict in masks_input:
            try:
                scores.append(self.score_mask(mask_dict, pos_text_feature, neg_text_feature, prompt_config, img_model))
            except Exception as exc:
                print(f"[ERROR] Serial CLIP scoring failed for one mask: {exc}")
                import traceback
                traceback.print_exc()
                scores.append(-1.0)
        return scores

    def score_masks_for_prompt(
        self,
        masks_input: list,
        score_key: str,
        pos_text_feature: torch.Tensor,
        neg_text_feature: torch.Tensor | None,
        prompt_name: str,
        prompt_config: dict,
        img_model: np.ndarray,
    ) -> list:
        stage_start = perf_counter()

        # Use batch CLIP scoring for ~2-4x speedup, but fall back to serial scoring if needed.
        try:
            scores = self.batch_score_masks_clip(masks_input, pos_text_feature, neg_text_feature, prompt_config, img_model)
        except Exception as exc:
            print(f"[ERROR] Batch CLIP scoring failed for '{prompt_name}': {exc}")
            import traceback
            traceback.print_exc()
            print(f"[INFO] Retrying CLIP scoring serially for '{prompt_name}'")
            try:
                scores = self.serial_score_masks_clip(masks_input, pos_text_feature, neg_text_feature, prompt_config, img_model)
            except Exception as retry_exc:
                print(f"[ERROR] Serial CLIP scoring also failed for '{prompt_name}': {retry_exc}")
                import traceback
                traceback.print_exc()
                print(f"[ERROR] No CLIP scoring method worked for '{prompt_name}'; skipping prompt")
                return []

        for idx, score in enumerate(scores):
            masks_input[idx][score_key] = score

        elapsed = perf_counter() - stage_start
        print(f"[INFO] Mask scoring complete for '{prompt_name}' ({elapsed:.2f}s)")

        return masks_input

    def select_masks_for_prompt(
        self,
        masks_input: list,
        score_key: str,
        pos_text_feature: torch.Tensor,
        neg_text_feature: torch.Tensor | None,
        prompt_name: str,
        prompt_config: dict,
        img_model: np.ndarray,
    ) -> tuple[list, float]:
        masks_input = self.score_masks_for_prompt(
            masks_input,
            score_key,
            pos_text_feature,
            neg_text_feature,
            prompt_name,
            prompt_config,
            img_model,
        )

        filtered = [m for m in masks_input if m[score_key] >= 0.0]
        filtered.sort(key=lambda m: m[score_key], reverse=True)

        if not filtered:
            print(f"[WARN] No non-negative CLIP scores for '{prompt_name}'. Selecting 0 masks.")
            return [], float(prompt_config.get("clip_score_threshold", -0.03))

        top_k = prompt_config.get("clip_top_k", 45)
        score_threshold = prompt_config.get("clip_score_threshold", -0.03)
        relative_margin = prompt_config.get("clip_relative_score_margin", 0.10)

        if filtered:
            best_score = float(filtered[0][score_key])
            dynamic_threshold = max(score_threshold, best_score - relative_margin)
        else:
            dynamic_threshold = score_threshold

        selected = [m for m in filtered if m[score_key] >= dynamic_threshold][:top_k]

        if not selected and filtered:
            selected = filtered[:1]

        return selected, dynamic_threshold

    def select_masks_for_prompt_contrastive(
        self,
        masks_input: list,
        score_key: str,
        pos_text_feature: torch.Tensor,
        neg_text_feature: torch.Tensor | None,
        prompt_name: str,
        prompt_config: dict,
        img_model: np.ndarray,
    ) -> list:
        scored_masks = self.score_masks_for_prompt(
            masks_input,
            score_key,
            pos_text_feature,
            neg_text_feature,
            prompt_name,
            prompt_config,
            img_model,
        )
        return scored_masks
