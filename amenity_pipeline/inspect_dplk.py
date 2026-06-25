"""
Inspect the .dlpk to understand model requirements before running inference.
"""
import zipfile, json, os

DLPK_PATH = r"C:\Users\Ian\Documents\GitHub\Segmentation-Reserach\amenity_pipeline\models\PoolSegmentation_USA.dlpk"
EXTRACT_DIR = r"C:\Users\Ian\Documents\GitHub\Segmentation-Reserach\amenity_pipeline\extracted_models"

# Extract the dlpk (it's just a zip)
with zipfile.ZipFile(DLPK_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)
    print("Files inside .dlpk:")
    for name in z.namelist():
        print(f"  {name}")

# Read the EMD (Esri Model Definition) - contains all model metadata
emd_files = [f for f in os.listdir(EXTRACT_DIR) if f.endswith('.emd')]
if emd_files:
    with open(os.path.join(EXTRACT_DIR, emd_files[0])) as f:
        emd = json.load(f)
    print("\n--- Model Metadata (EMD) ---")
    for key in ['ModelType', 'Framework', 'ModelFile', 'ImageHeight', 'ImageWidth',
                'ExtractBands', 'CellSize', 'MinCellSize', 'MaxCellSize']:
        if key in emd:
            print(f"  {key}: {emd[key]}")
    print(f"\nFull EMD saved. Required resolution (CellSize): {emd.get('CellSize', 'not specified')} meters/pixel")