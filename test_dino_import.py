import sys
sys.path.insert(0, r"c:\Users\Ian\Documents\GitHub\Segmentation-Reserach")

# Install the finder first
from models.dino_processing import _install_groundingdino_bytecode_finder
from pathlib import Path
local_checkout = Path(r"c:\Users\Ian\Documents\GitHub\Segmentation-Reserach\GroundingDINO-main\groundingdino")
_install_groundingdino_bytecode_finder(local_checkout)

# Now try the import
try:
    from groundingdino.util.inference import load_model, predict
    print("SUCCESS: imported predict and load_model")
    print("predict:", predict)
    print("load_model:", load_model)
except Exception as e:
    print("FAILED:", type(e).__name__, str(e))
    import traceback
    traceback.print_exc()
