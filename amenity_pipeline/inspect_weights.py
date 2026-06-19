"""
Inspect the .pth checkpoint to understand its exact format.
"""
import torch

PATH = r"C:\Users\Ian\Documents\GitHub\Segmentation-Reserach\amenity_pipeline\extracted_models\ParkingLotClassification_USA.pth"

ckpt = torch.load(PATH, map_location="cpu", weights_only=False)

print("Type:", type(ckpt))

if isinstance(ckpt, dict):
    print("Top-level keys:", list(ckpt.keys()))
    for k, v in ckpt.items():
        if isinstance(v, dict):
            keys = list(v.keys())
            print(f"\n  [{k}] -> dict with {len(keys)} keys")
            print("  First 10:", keys[:10])
        elif hasattr(v, 'shape'):
            print(f"  [{k}] -> tensor {v.shape}")
        else:
            print(f"  [{k}] -> {type(v)}")
elif hasattr(ckpt, 'state_dict'):
    print("Has state_dict(). Keys sample:")
    keys = list(ckpt.state_dict().keys())
    print(keys[:15])
    print(f"  ... {len(keys)} total keys")
else:
    print("Unknown format:", dir(ckpt))