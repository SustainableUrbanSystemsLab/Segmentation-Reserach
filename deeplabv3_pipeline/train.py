"""
DeepLabV3+ training pipeline for NEN 8100 wind comfort segmentation.
 
Dependencies:
    pip install segmentation-models-pytorch albumentations
 
Key design decisions vs previous version:
    - freeze_encoder=True  : trains only ASPP + decoder (~2M params vs 59M).
                             Mandatory when training from <20 source tiles.
    - manual_class_weights : overrides auto-computed weights per class.
                             Frequency-based weighting actively harms NEN_B
                             (most common class → gets lowest weight → ignored).
    - albumentations       : elastic deformation + random scale/crop breaks
                             spatial memorisation from the small tile set.
    - ComboLoss            : Dice + weighted Focal, optimises region overlap
                             not just per-pixel accuracy.
"""
 
from __future__ import annotations
 
import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
 
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
 
# --------------------------------------------------
# Optional dependencies
# --------------------------------------------------
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print(
        "[WARN] albumentations not found — install with: pip install albumentations\n"
        "       Falling back to basic torchvision augmentation."
    )
 
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print(
        "[WARN] segmentation_models_pytorch not found.\n"
        "       Install with: pip install segmentation-models-pytorch\n"
        "       Falling back to torchvision DeepLabV3 (not V3+)."
    )
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        deeplabv3_resnet101,
        DeepLabV3_ResNet50_Weights,
        DeepLabV3_ResNet101_Weights,
    )
 
# --------------------------------------------------
# Environment
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
 
from yoloseg_pipeline.common import CLASS_NAMES
 
NUM_CLASSES  = len(CLASS_NAMES)
IGNORE_INDEX = 255
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
 
Image.MAX_IMAGE_PIXELS = None
 
 
# --------------------------------------------------
# Augmentation pipelines
# --------------------------------------------------
 
def _build_albumentations_train(imgsz: int, hsv_s: float, hsv_v: float) -> A.Compose:
    """
    Strong augmentation pipeline for aerial imagery with small tile counts.
 
    Elastic deformation + random scale/crop are the most important transforms
    here — they break the spatial memorisation pattern that develops when the
    model has seen the same 6 tiles hundreds of times.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(imgsz, imgsz),
            scale=(0.5, 1.0),   # random zoom 50-100% of the chip
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            p=0.4,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.ColorJitter(
            brightness=hsv_v,
            contrast=0.2,
            saturation=hsv_s,
            hue=0.01,
            p=0.7,
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
 
 
def _build_albumentations_val(imgsz: int) -> A.Compose:
    return A.Compose([
        A.Resize(imgsz, imgsz, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
 
 
# --------------------------------------------------
# Dataset
# --------------------------------------------------
 
class WindComfortDataset(Dataset):
    """
    Reads image + mask PNG pairs from split directories.
 
    Layout:
        data_dir/images/{split}/*.png
        data_dir/masks/{split}/*.png
 
    Mask values are class indices 0..NUM_CLASSES-1 or IGNORE_INDEX (255).
    """
 
    def __init__(
        self,
        data_dir: Path,
        split:    str,
        imgsz:    int   = 512,
        augment:  bool  = False,
        hsv_s:    float = 0.0,
        hsv_v:    float = 0.0,
    ) -> None:
        self.image_dir   = data_dir / "images" / split
        self.mask_dir    = data_dir / "masks"  / split
        self.augment     = augment
        self.use_albu    = ALBUMENTATIONS_AVAILABLE
 
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
 
        if self.use_albu:
            if augment:
                self.transform = _build_albumentations_train(imgsz, hsv_s, hsv_v)
            else:
                self.transform = _build_albumentations_val(imgsz)
        else:
            # Fallback: basic torchvision pipeline
            jitter_args: dict[str, float] = {}
            if hsv_s > 0:
                jitter_args["saturation"] = hsv_s
            if hsv_v > 0:
                jitter_args["brightness"] = hsv_v
            self._color_jitter = transforms.ColorJitter(**jitter_args) if jitter_args else None
            self._to_tensor    = transforms.ToTensor()
            self._normalize    = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            self._resize_img   = transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.BILINEAR)
            self._resize_mask  = transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.NEAREST)
 
    def __len__(self) -> int:
        return len(self.image_paths)
 
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path  = self.image_paths[index]
        mask_path = self.mask_dir / img_path.name
 
        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path))
 
        if self.use_albu:
            result       = self.transform(image=image, mask=mask)
            image_tensor = result["image"].float()
            mask_tensor  = result["mask"].long()
        else:
            # Fallback torchvision path
            img_pil  = Image.fromarray(image)
            mask_pil = Image.fromarray(mask)
            img_pil  = self._resize_img(img_pil)
            mask_pil = self._resize_mask(mask_pil)
 
            if self.augment:
                if torch.rand(1).item() > 0.5:
                    img_pil  = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
                if torch.rand(1).item() > 0.5:
                    img_pil  = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
                    mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)
                k = int(torch.randint(0, 4, (1,)).item())
                rot_map = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}
                if k in rot_map:
                    img_pil  = img_pil.transpose(rot_map[k])
                    mask_pil = mask_pil.transpose(rot_map[k])
                if self._color_jitter is not None:
                    img_pil = self._color_jitter(img_pil)
 
            image_tensor = self._normalize(self._to_tensor(img_pil))
            mask_tensor  = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
 
        return image_tensor, mask_tensor
 
 
# --------------------------------------------------
# Metrics
# --------------------------------------------------
 
def compute_metrics(
    pred:         torch.Tensor,
    target:       torch.Tensor,
    num_classes:  int,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, object]:
    """
    Returns mIoU, overall (micro) pixel accuracy, macro accuracy (mean of
    per-class recall), and per-class IoU / accuracy breakdowns.
 
    Macro accuracy is the mean of per-class recall — it weights every class
    equally regardless of pixel count, so a model that always predicts the
    majority class (NEN_C) scores poorly here even if micro pixel accuracy
    looks fine.
    """
    valid = target != ignore_index
    if not valid.any():
        return {
            "miou":           0.0,
            "pixel_accuracy": 0.0,
            "macro_accuracy": 0.0,
            "class_iou":      {n: 0.0 for n in CLASS_NAMES},
            "class_acc":      {n: 0.0 for n in CLASS_NAMES},
        }
 
    pred_valid   = pred[valid]
    target_valid = target[valid]
    pixel_accuracy = (pred_valid == target_valid).sum().item() / target_valid.numel()
 
    class_iou: dict[str, float] = {}
    class_acc: dict[str, float] = {}
    valid_ious: list[float]     = []
    valid_accs: list[float]     = []
 
    for cls in range(num_classes):
        pred_cls   = pred_valid   == cls
        target_cls = target_valid == cls
 
        intersection = (pred_cls & target_cls).sum().item()
        union        = (pred_cls | target_cls).sum().item()
        if union > 0:
            iou = intersection / union
            class_iou[CLASS_NAMES[cls]] = iou
            valid_ious.append(iou)
        else:
            class_iou[CLASS_NAMES[cls]] = float("nan")
 
        total_true = target_cls.sum().item()
        if total_true > 0:
            acc = (pred_valid[target_cls] == cls).sum().item() / total_true
            class_acc[CLASS_NAMES[cls]] = acc
            valid_accs.append(acc)
        else:
            class_acc[CLASS_NAMES[cls]] = float("nan")
 
    return {
        "miou":           float(np.nanmean(valid_ious)) if valid_ious else 0.0,
        "pixel_accuracy": pixel_accuracy,
        "macro_accuracy": float(np.nanmean(valid_accs)) if valid_accs else 0.0,
        "class_iou":      class_iou,
        "class_acc":      class_acc,
    }
 
 
def log_class_iou(class_iou: dict[str, float], class_acc: dict[str, float] | None = None) -> None:
    print("      Per-class IoU" + ("  /  Recall" if class_acc else "") + ":")
    for name, iou in class_iou.items():
        iou_str = "  n/a " if math.isnan(iou) else f"{iou:.4f}"
        bar     = "" if math.isnan(iou) else f"[{'█' * int(iou * 20)}{'░' * (20 - int(iou * 20))}]"
 
        if class_acc is not None:
            acc     = class_acc.get(name, float("nan"))
            acc_str = "  n/a " if math.isnan(acc) else f"{acc:.4f}"
            print(f"        {name:<18} IoU={iou_str}  Recall={acc_str}  {bar}")
        else:
            print(f"        {name:<18} {iou_str}  {bar}")
 
 
# --------------------------------------------------
# Class weights
# --------------------------------------------------
 
def compute_class_weights(
    data_dir:   Path,
    split:      str   = "train",
    min_weight: float = 0.1,
    max_weight: float = 5.0,
    manual_override: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Compute class weights for the loss function.
 
    If manual_override is provided (e.g. {"NEN_B": 1.5}), those values
    are used verbatim for the specified classes and auto-computed values
    are used for the rest. This is the recommended approach when auto
    weighting systematically ignores a class (as NEN_B was receiving ~0.16
    due to its high pixel frequency despite being hard to segment).
    """
    mask_dir = data_dir / "masks" / split
    counts   = np.zeros(NUM_CLASSES, dtype=np.float64)
 
    for mask_path in sorted(mask_dir.glob("*.png")):
        mask = np.array(Image.open(mask_path), dtype=np.int64)
        for cls in range(NUM_CLASSES):
            counts[cls] += (mask == cls).sum()
 
    total = counts.sum()
    if total == 0:
        print("[WARN] No mask pixels found — using uniform class weights.")
        return torch.ones(NUM_CLASSES, dtype=torch.float32)
 
    freq    = counts / total
    weights = 1.0 / np.log(1.02 + freq)   # ENet formula
    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.mean()
 
    # Apply manual overrides
    if manual_override:
        for cls_name, override_val in manual_override.items():
            if cls_name in CLASS_NAMES:
                idx = CLASS_NAMES.index(cls_name)
                weights[idx] = float(override_val)
                print(f"  [OVERRIDE] {cls_name}: {override_val:.4f}  (was auto={weights[idx]:.4f})")
        weights = weights / weights.mean()
 
    print("[INFO] Final class weights:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {weights[idx]:.4f}  (freq={freq[idx]:.4f})")
 
    return torch.tensor(weights, dtype=torch.float32)
 
 
# --------------------------------------------------
# Loss
# --------------------------------------------------
 
class WeightedFocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.weight       = weight
        self.gamma        = gamma
        self.ignore_index = ignore_index
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = torch.nn.functional.cross_entropy(
            logits, targets, weight=self.weight, ignore_index=self.ignore_index, reduction="none"
        )
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        mask = targets != self.ignore_index
        return loss[mask].mean()
 
 
class ComboLoss(nn.Module):
    """Dice + Focal combo loss. Optimises region overlap and hard boundary pixels."""
 
    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        if SMP_AVAILABLE:
            self.dice  = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
            self.focal = WeightedFocalLoss(weight=weight, gamma=2.0, ignore_index=ignore_index)
        else:
            self.dice  = None
            self.focal = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.dice is not None:
            return self.dice(logits, targets) + 0.5 * self.focal(logits, targets)
        return self.focal(logits, targets)
 
 
# --------------------------------------------------
# Model
# --------------------------------------------------
 
def build_model(
    backbone:       str   = "resnet101",
    pretrained:     bool  = True,
    dropout:        float = 0.3,
    freeze_encoder: bool  = True,
) -> nn.Module:
    """
    Build DeepLabV3+ and optionally freeze the encoder backbone.
 
    freeze_encoder=True is STRONGLY RECOMMENDED when training from fewer
    than ~20 source tiles. It reduces trainable parameters from ~59M to ~2M,
    preventing the encoder from memorizing spatial patterns in the training tiles.
 
    The encoder already contains powerful ImageNet features. Fine-tuning the
    decoder alone is sufficient — and far more stable — for small datasets.
    """
    if SMP_AVAILABLE:
        model = smp.DeepLabV3Plus(
            encoder_name    = backbone,
            encoder_weights = "imagenet" if pretrained else None,
            in_channels     = 3,
            classes         = NUM_CLASSES,
            activation      = None,
        )
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout),
            model.segmentation_head,
        )
 
        if freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"--> Model: smp.DeepLabV3Plus ({backbone})  dropout={dropout}")
            print(f"    Encoder FROZEN — trainable params: {trainable:,} / {total:,}")
        else:
            total = sum(p.numel() for p in model.parameters())
            print(f"--> Model: smp.DeepLabV3Plus ({backbone})  dropout={dropout}")
            print(f"    Encoder UNFROZEN — trainable params: {total:,} / {total:,}  [overfitting risk]")
        return model
 
    # Torchvision fallback
    if backbone == "resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model   = deeplabv3_resnet50(weights=weights)
    else:
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        model   = deeplabv3_resnet101(weights=weights)
 
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Sequential(nn.Dropout2d(p=dropout), nn.Conv2d(in_channels, NUM_CLASSES, 1))
    if model.aux_classifier is not None:
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Sequential(nn.Dropout2d(p=dropout), nn.Conv2d(aux_in, NUM_CLASSES, 1))
 
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name and "aux_classifier" not in name:
                param.requires_grad = False
 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"--> Model: torchvision DeepLabV3 ({backbone}, fallback)  dropout={dropout}")
    print(f"    Trainable: {trainable:,} / {total:,}  frozen_encoder={freeze_encoder}")
    return model
 
 
def _forward(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    out = model(images)
    return out["out"] if isinstance(out, dict) else out
 
 
def _forward_with_loss(
    model:     nn.Module,
    images:    torch.Tensor,
    masks:     torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    raw = model(images)
    if isinstance(raw, dict):
        loss = criterion(raw["out"], masks)
        if "aux" in raw:
            loss = loss + 0.4 * criterion(raw["aux"], masks)
        return loss
    return criterion(raw, masks)
 
 
# --------------------------------------------------
# Utilities
# --------------------------------------------------
 
def _default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return max(1, os.cpu_count() or 1)
 
 
def _cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, total: int, lr0: float, warmup: int) -> float:
    if epoch < warmup:
        lr = lr0 * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, total - warmup)
        lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr
 
 
# --------------------------------------------------
# CLI
# --------------------------------------------------
 
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",            default=str(PROJECT_ROOT / "data" / "deeplabv3_windcomfort_rgb_sliced"))
    p.add_argument("--backbone",            default="resnet101", choices=["resnet50", "resnet101"])
    p.add_argument("--pretrained",          action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-encoder",      action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs",              type=int,   default=200)
    p.add_argument("--batch",               type=int,   default=4)
    p.add_argument("--imgsz",               type=int,   default=512)
    p.add_argument("--device",              default="0")
    p.add_argument("--workers",             type=int,   default=_default_workers())
    p.add_argument("--name",                default="wind_comfort_deeplabv3")
    p.add_argument("--resume",              action="store_true")
    p.add_argument("--amp",                 action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lr0",                 type=float, default=1e-4)
    p.add_argument("--warmup-epochs",       type=int,   default=5)
    p.add_argument("--cos-lr",              action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--weight-decay",        type=float, default=5e-4)
    p.add_argument("--clip-grad",           type=float, default=10.0)
    p.add_argument("--dropout",             type=float, default=0.3)
    p.add_argument("--patience",            type=int,   default=40)
    p.add_argument("--min-weight",          type=float, default=0.1)
    p.add_argument("--max-weight",          type=float, default=5.0)
    p.add_argument("--manual-class-weights", default=None,
                   help='JSON string of per-class weight overrides e.g. \'{"NEN_B": 1.5, "NEN_A": 2.0}\'')
    p.add_argument("--auto-class-weights",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-class-iou-every", type=int,   default=10)
    p.add_argument("--hsv-s",               type=float, default=0.4)
    p.add_argument("--hsv-v",               type=float, default=0.3)
    return p.parse_args()
 
 
# --------------------------------------------------
# Main
# --------------------------------------------------
 
def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = str(PROJECT_ROOT / "deeplabv3_pipeline" / "train_config.json")
 
    args     = _parse_args()
    run_args = vars(args)
 
    config_file = Path(config_path)
    if config_file.exists():
        print(f"--> Loading config: {config_file}")
        with open(config_file) as f:
            for key, value in json.load(f).items():
                if not key.startswith("_"):
                    run_args[key.replace("-", "_")] = value
    else:
        print(f"[WARN] Config not found at {config_file} — using CLI defaults.")
 
    Image.MAX_IMAGE_PIXELS = None
 
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir = Path(run_args["data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
 
    run_name     = run_args.get("name", "wind_comfort_deeplabv3")
    project_dir  = (PROJECT_ROOT / "results" / "deeplabv3").resolve()
    run_dir      = project_dir / run_name
    weights_dir  = run_dir / "weights"
    last_ckpt    = weights_dir / "last.pt"
    best_ckpt    = weights_dir / "best.pt"
    best_macro_ckpt = weights_dir / "best_macro_acc.pt"
    archive_root = project_dir / "old_deeplabv3_results"
 
    project_dir.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
 
    print(f"\n{'='*60}")
    print(f"  Run      : {run_name}")
    print(f"  Data     : {data_dir}")
    print(f"  Run dir  : {run_dir}")
    print(f"{'='*60}\n")
 
    # ── Resume / archive ───────────────────────────────────────────────────
    resume_requested = bool(run_args.get("resume", False))
    start_epoch    = 0
    best_miou      = 0.0
    best_macro_acc = 0.0
 
    if resume_requested and not last_ckpt.exists():
        print(f"--> [WARN] Resume requested but {last_ckpt} not found — starting fresh.")
        resume_requested = False
 
    if run_dir.exists() and not resume_requested:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = archive_root / f"{run_name}_{ts}"
        print(f"--> [ARCHIVE] {run_dir.name} -> {archive_dir.name}")
        shutil.move(str(run_dir), str(archive_dir))
 
    weights_dir.mkdir(parents=True, exist_ok=True)
 
    # ── Device ─────────────────────────────────────────────────────────────
    ds = str(run_args.get("device", "0"))
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu") if ds in ("auto", "")
        else torch.device("cpu") if ds == "cpu"
        else torch.device(f"cuda:{ds}")
    )
    print(f"--> Device: {device}\n")
 
    # ── Hyperparameters ────────────────────────────────────────────────────
    imgsz          = int(run_args.get("imgsz",            512))
    batch_size     = int(run_args.get("batch",              4))
    num_workers    = int(run_args.get("workers",  _default_workers()))
    epochs         = int(run_args.get("epochs",           200))
    lr0            = float(run_args.get("lr0",            1e-4))
    weight_decay   = float(run_args.get("weight_decay",  5e-4))
    warmup_eps     = int(run_args.get("warmup_epochs",      5))
    use_cos_lr     = bool(run_args.get("cos_lr",          True))
    dropout        = float(run_args.get("dropout",         0.3))
    patience       = int(run_args.get("patience",           40))
    clip_grad      = float(run_args.get("clip_grad",       10.0))
    min_w          = float(run_args.get("min_weight",       0.1))
    max_w          = float(run_args.get("max_weight",       5.0))
    log_every      = int(run_args.get("log_class_iou_every", 10))
    hsv_s          = float(run_args.get("hsv_s",            0.4))
    hsv_v          = float(run_args.get("hsv_v",            0.3))
    backbone       = str(run_args.get("backbone",    "resnet101"))
    pretrained     = bool(run_args.get("pretrained",      True))
    freeze_encoder = bool(run_args.get("freeze_encoder",  True))
    use_amp        = bool(run_args.get("amp",             True)) and device.type == "cuda"
 
    # Parse manual class weight overrides from config or CLI
    manual_weights_raw = run_args.get("manual_class_weights", None)
    manual_weights: dict[str, float] | None = None
    if manual_weights_raw:
        if isinstance(manual_weights_raw, dict):
            manual_weights = {k: float(v) for k, v in manual_weights_raw.items()}
        elif isinstance(manual_weights_raw, str):
            manual_weights = {k: float(v) for k, v in json.loads(manual_weights_raw).items()}
 
    # ── Data ───────────────────────────────────────────────────────────────
    train_ds = WindComfortDataset(data_dir, "train", imgsz=imgsz, augment=True,  hsv_s=hsv_s, hsv_v=hsv_v)
    val_ds   = WindComfortDataset(data_dir, "val",   imgsz=imgsz, augment=False)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
 
    print(f"--> Dataset: {len(train_ds)} train / {len(val_ds)} val chips")
    print(f"--> Batch size: {batch_size}  |  Image size: {imgsz}")
    print(f"--> Albumentations augmentation: {ALBUMENTATIONS_AVAILABLE}")
 
    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(
        backbone=backbone, pretrained=pretrained, dropout=dropout, freeze_encoder=freeze_encoder
    ).to(device)
 
    # ── Loss ───────────────────────────────────────────────────────────────
    class_weight = None
    if run_args.get("auto_class_weights", True):
        class_weight = compute_class_weights(
            data_dir, split="train",
            min_weight=min_w, max_weight=max_w,
            manual_override=manual_weights,
        ).to(device)
 
    criterion = ComboLoss(weight=class_weight, ignore_index=IGNORE_INDEX)
 
    # ── Optimizer — only optimise trainable params ─────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr0, weight_decay=weight_decay)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
 
    # ── Resume ─────────────────────────────────────────────────────────────
    if resume_requested and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch    = ckpt.get("epoch", 0) + 1
        best_miou      = ckpt.get("best_miou", 0.0)
        best_macro_acc = ckpt.get("best_macro_acc", 0.0)
        print(f"--> Resumed epoch {start_epoch}  |  best mIoU: {best_miou:.4f}  |  best macro acc: {best_macro_acc:.4f}")
 
    print(f"\n--> Training for {epochs} epochs  |  patience={patience}  |  amp={use_amp}")
    print(f"    lr0={lr0}  warmup={warmup_eps}  freeze_encoder={freeze_encoder}  clip_grad={clip_grad}")
    print(f"    dropout={dropout}  weight_clip=[{min_w}, {max_w}]\n")
 
    # ── Training loop ──────────────────────────────────────────────────────
    history:               list[dict] = []
    epochs_no_improvement: int        = 0
 
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
 
        current_lr = _cosine_lr(optimizer, epoch, epochs, lr0, warmup_eps) if use_cos_lr else lr0
 
        # Train
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = _forward_with_loss(model, images, masks, criterion)
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()
            train_batches  += 1
 
        avg_train_loss = train_loss_sum / max(1, train_batches)
 
        # Validate
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = _forward(model, images)
                    loss   = criterion(logits, masks)
                val_loss_sum += loss.item()
                val_batches  += 1
                all_preds.append(logits.argmax(dim=1).cpu())
                all_targets.append(masks.cpu())
 
        avg_val_loss = val_loss_sum / max(1, val_batches)
        metrics      = compute_metrics(torch.cat(all_preds), torch.cat(all_targets), NUM_CLASSES)
        epoch_time   = time.time() - t0
 
        print(
            f"Epoch {epoch + 1:>3d}/{epochs}  |  "
            f"train_loss: {avg_train_loss:.4f}  |  "
            f"val_loss: {avg_val_loss:.4f}  |  "
            f"mIoU: {metrics['miou']:.4f}  |  "
            f"pxAcc: {metrics['pixel_accuracy']:.4f}  |  "
            f"MacroAcc: {metrics['macro_accuracy']:.4f}  |  "
            f"lr: {current_lr:.2e}  |  "
            f"{epoch_time:.1f}s"
        )
 
        if (epoch + 1) % log_every == 0:
            log_class_iou(metrics["class_iou"], metrics["class_acc"])
 
        # Checkpoint
        ckpt_payload = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou":            max(best_miou, metrics["miou"]),
            "best_macro_acc":       max(best_macro_acc, metrics["macro_accuracy"]),
            "run_args":             run_args,
        }
        if use_amp:
            ckpt_payload["scaler_state_dict"] = scaler.state_dict()
        torch.save(ckpt_payload, last_ckpt)
 
        # Track best macro accuracy separately (saved to its own file —
        # this is the metric that best reflects whether minority classes
        # like NEN_B / Uncomfortable are actually being learned)
        if metrics["macro_accuracy"] > best_macro_acc:
            best_macro_acc = metrics["macro_accuracy"]
            torch.save(ckpt_payload, best_macro_ckpt)
            print(f"  --> New best Macro Accuracy: {best_macro_acc:.4f} — saved {best_macro_ckpt.name}")
 
        # Best model (by mIoU) + early stopping
        if metrics["miou"] > best_miou:
            best_miou             = metrics["miou"]
            epochs_no_improvement = 0
            torch.save(ckpt_payload, best_ckpt)
            print(f"  --> New best mIoU: {best_miou:.4f} — saved {best_ckpt.name}")
            log_class_iou(metrics["class_iou"], metrics["class_acc"])
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= patience:
                print(f"\n--> Early stopping at epoch {epoch + 1} ({patience} epochs without improvement)")
                break
 
        history.append({
            "epoch":          epoch,
            "train_loss":     avg_train_loss,
            "val_loss":       avg_val_loss,
            "miou":           metrics["miou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "macro_accuracy": metrics["macro_accuracy"],
            "class_iou":      metrics["class_iou"],
            "class_acc":      metrics["class_acc"],
            "lr":             current_lr,
            "time_s":         round(epoch_time, 1),
        })
 
    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete  |  Best mIoU: {best_miou:.4f}  |  Best Macro Acc: {best_macro_acc:.4f}")
    print(f"  Weights : {weights_dir}")
    print(f"{'='*60}")
 
    history_path = run_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"--> History: {history_path}")
 
 
if __name__ == "__main__":
    main()
