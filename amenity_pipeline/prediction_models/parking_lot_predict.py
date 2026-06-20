import os
import sys
import json
import argparse
import math
from xml.parsers.expat import model
import numpy as np
from skimage.morphology import remove_small_objects
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.enums import Resampling
from affine import Affine
from shapely.geometry import shape, mapping, Polygon
from rasterio.features import shapes, rasterize

# =========================================================================
# 1. FIXED MODEL ARCHITECTURE (TRUE OUTPUT STRIDE 16 BACKBONE)
# =========================================================================

class ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise_conv = ConvModule(in_ch, in_ch, kernel_size, stride, padding, dilation, groups=in_ch)
        self.pointwise_conv = ConvModule(in_ch, out_ch, kernel_size=1)
        
    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))

class Bottleneck(nn.Module):
    def __init__(self, in_ch, bottle_ch, out_ch, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, bottle_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_ch)
        
        # Calculate padding dynamically to handle dilation changes cleanly
        pad = dilation if stride == 1 else 1
        self.conv2 = nn.Conv2d(bottle_ch, bottle_ch, kernel_size=3, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(bottle_ch)
        
        self.conv3 = nn.Conv2d(bottle_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class DeepLabV3PlusHeader(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(2048, 512, 1)
        )
        self.aspp_modules = nn.ModuleList([
            ConvModule(2048, 512, 1),
            DepthwiseSeparableConvModule(2048, 512, 3, padding=6, dilation=6),
            DepthwiseSeparableConvModule(2048, 512, 3, padding=12, dilation=12),
            DepthwiseSeparableConvModule(2048, 512, 3, padding=18, dilation=18)
        ])
        
        self.bottleneck = ConvModule(2560, 512, kernel_size=3, padding=1)
        self.c1_bottleneck = ConvModule(256, 48, 1)
        
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(560, 512, 3, padding=1),
            DepthwiseSeparableConvModule(512, 512, 3, padding=1)
        )
        self.conv_seg = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, c1, c4):
        pool = self.image_pool(c4)
        pool = F.interpolate(pool, size=c4.shape[2:], mode='bilinear', align_corners=False)
        aspp_outs = [pool] + [m(c4) for m in self.aspp_modules]
        x = self.bottleneck(torch.cat(aspp_outs, dim=1))
        x = F.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
        c1 = self.c1_bottleneck(c1)
        x = torch.cat([x, c1], dim=1)
        x = self.sep_bottleneck(x)
        return self.conv_seg(x)

class ParkingLotDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.ModuleDict({
            'stem': nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
            )
        })
        # CRITICAL FIX: The missing maxpool layer that drops spatial scale down to standard input requirements
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # CRITICAL FIX: True ResNet-101 D16 Output Stride configurations 
        self._build_layer('layer1', 64, 64, 256, blocks=3, stride=1, dilation=1)
        self._build_layer('layer2', 256, 128, 512, blocks=4, stride=2, dilation=1)
        self._build_layer('layer3', 512, 256, 1024, blocks=23, stride=2, dilation=1) 
        self._build_layer('layer4', 1024, 512, 2048, blocks=3, stride=1, dilation=2)  
        
        self.decode_head = DeepLabV3PlusHeader(num_classes)

    def _build_layer(self, name, in_ch, bottle_ch, out_ch, blocks, stride=1, dilation=1):
        layers = []
        # Re-engineered to properly handle dynamic downsampling path logic matching MMSegmentation configs
        ds = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        layers.append(Bottleneck(in_ch, bottle_ch, out_ch, stride, dilation, ds))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_ch, bottle_ch, out_ch, stride=1, dilation=dilation))
        self.backbone[name] = nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.backbone['stem'](x)
        x = self.maxpool(x) # Downsample to H/4
        c1 = self.backbone['layer1'](x) # Low-level skip connection point (H/4)
        x = self.backbone['layer2'](c1) # Downsample to H/8
        x = self.backbone['layer3'](x) # Downsample to H/16
        c4 = self.backbone['layer4'](x) # Deep high-level features (H/16)
        logits = self.decode_head(c1, c4)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)

# =========================================================================
# 2. EVALUATION EXECUTION ENGINE
# =========================================================================
def run(image_path, threshold, overlap, cell_size, batch_size, output_path, weights_path, device, target_class_idx=1, use_bgr=False):
    print(f"Loading Raster Target: {os.path.basename(image_path)}")
    bands = [3, 2, 1] if use_bgr else [1, 2, 3]
    
    with rasterio.open(image_path) as src:
        native_res = src.res[0]
        src_crs = src.crs
        if cell_size <= 0:
            cell_size = native_res
            w, h = src.width, src.height
            tx = src.transform
            data = src.read(bands)
        else:
            scale = native_res / cell_size
            w, h = round(src.width * scale), round(src.height * scale)
            tx = Affine(cell_size, 0, src.transform.c, 0, -cell_size, src.transform.f)
            data = src.read(bands, out_shape=(3, h, w), resampling=Resampling.bilinear)

    prob_accumulator = np.zeros((h, w), dtype=np.float32)
    weight_grid = np.zeros((h, w), dtype=np.float32)

    model = ParkingLotDeepLabV3Plus(num_classes=2).to(device)
    state = torch.load(weights_path, map_location=device)
    state = {k: v for k, v in state.items() if not k.startswith("auxiliary_head.")}
    
    # =========================================================================
    # DETECTOR DIAGNOSTIC INTERCEPT
    # =========================================================================
    import sys
    print("\n" + "="*60)
    print("RUNNING ARCHITECTURE COHERENCE CHECK...")
    print("="*60)
    
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)

    if missing_keys or unexpected_keys:
        print("\nWARNING: Checkpoint mismatch detected")

        if missing_keys:
            print("Missing Keys:")
            for k in missing_keys[:20]:
                print(f"  {k}")

        if unexpected_keys:
            print("Unexpected Keys:")
            for k in unexpected_keys[:20]:
                print(f"  {k}")

        raise RuntimeError("Model checkpoint incompatible with architecture")

    print("[OK] Checkpoint matches architecture.")

    model.load_state_dict(state, strict=True)
    # =========================================================================

    model.load_state_dict(state, strict=True) 
    model.eval()

    TILE_SIZE = 512
    stride = TILE_SIZE - overlap
    tile_positions = [(r, c) for r in range(0, h, stride) for c in range(0, w, stride)]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    for i in range(0, len(tile_positions), batch_size):
        batch = tile_positions[i:i+batch_size]
        tensor_list = []
        meta_list = []

        for r, c in batch:
            win_h, win_w = min(TILE_SIZE, h - r), min(TILE_SIZE, w - c)
            tile = data[:, r:r+win_h, c:c+win_w].astype(np.float32) / 255.0
            tile = (tile - mean) / std
            
            if win_h < TILE_SIZE or win_w < TILE_SIZE:
                pad = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                pad[:, :win_h, :win_w] = tile
                tile = pad
            
            tensor_list.append(torch.from_numpy(tile))
            meta_list.append((r, c, win_h, win_w))

        tensors = torch.stack(tensor_list).to(device)
        with torch.no_grad():
            outputs = model(tensors)
            probs = F.softmax(outputs, dim=1)[:, target_class_idx, :, :].cpu().numpy()

        for idx, (r, c, win_h, win_w) in enumerate(meta_list):
            tile_prob = probs[idx, :win_h, :win_w]
            prob_accumulator[r:r+win_h, c:c+win_w] += tile_prob
            weight_grid[r:r+win_h, c:c+win_w] += 1.0

    np.divide(prob_accumulator, weight_grid, out=prob_accumulator, where=weight_grid > 0)

    # Save probability raster for confidence-based fusion
    try:
        prob_output_path = output_path.replace(".geojson", ".prob.tif")
        with rasterio.open(
            prob_output_path, "w",
            driver="GTiff",
            height=h, width=w,
            count=1,
            dtype=rasterio.float32,
            crs=src_crs,
            transform=tx,
            compress="lzw"
        ) as dst:
            dst.write(prob_accumulator, 1)
        print(f"[Parking] Probability raster saved to: {prob_output_path}")
    except Exception as e:
        print(f"[Parking] WARNING: Could not save probability raster: {e}")

    binary_mask = (prob_accumulator >= threshold)

    from scipy.ndimage import binary_closing
    from skimage.morphology import remove_small_objects

    binary_mask = binary_closing(
        binary_mask,
        iterations=2
    )

    binary_mask = remove_small_objects(
        binary_mask,
        min_size=50
    )

    binary_mask = binary_mask.astype(np.uint8)

    features = []
    fid = 1
    for shape_geom, val in shapes(binary_mask, mask=(binary_mask == 1), transform=tx):
        poly_shape = shape(shape_geom)
        if poly_shape.area < 40: # Ignore isolated trace artifacts under 40 sq meters
            continue
            
        feature_footprint = rasterize([poly_shape], out_shape=binary_mask.shape, transform=tx, default_value=1, fill=0)
        local_pixels = prob_accumulator[feature_footprint == 1]
        localized_confidence = float(np.mean(local_pixels)) if local_pixels.size > 0 else 0.0
            
        features.append({
            "type": "Feature",
            "properties": {"FID": fid, "Class": "parking_lot", "Confidence": localized_confidence},
            "geometry": mapping(poly_shape)
        })
        fid += 1

    geojson = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    print(f"Successfully exported {len(features)} structured polygons to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--cell_size", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--class_idx", type=int, default=1)
    parser.add_argument("--bgr", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS = os.path.join(PIPELINE_ROOT, "extracted_models", "ParkingLotClassification_USA.pth")

    run(args.image, args.threshold, args.overlap, args.cell_size, args.batch_size, args.output, WEIGHTS, device, 
        target_class_idx=args.class_idx, use_bgr=args.bgr)