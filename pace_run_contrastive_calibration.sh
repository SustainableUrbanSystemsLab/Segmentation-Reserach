#!/bin/bash
#SBATCH -J contrastive_calibration        # Job name
#SBATCH -A gts-pkastner3                     # Your PACE charge account
#SBATCH -N 1                              # Request 1 node
#SBATCH --ntasks-per-node=1               # One task
#SBATCH --cpus-per-task=4                 # CPU cores for data loading
#SBATCH --mem=32G                         # Memory
#SBATCH --gres=gpu:V100:1                 # Request 1 V100 GPU
#SBATCH -t 04:00:00                       # Walltime limit (hh:mm:ss)
#SBATCH -q inferno                        # Queue
#SBATCH -o logs/job_%j.out                # Slurm standard output log
#SBATCH -e logs/job_%j.err                # Slurm standard error log

# ==========================================
# 1. ENVIRONMENT & CLUSTER MODULE SETUP
# ==========================================
module purge
module load gcc/11.3.1 || true

# Keep module load best-effort only. Some PACE shell initializations emit
# harmless conda warnings from Lmod when conda is unavailable.
module load anaconda3 || true

# ==========================================
# 2. SEPARATE CODE PATH FROM OUTPUT PATHS
# ==========================================
# Securely fallback to current working directory if not run by Slurm scheduler
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

# Target Output Directory (Leverages the secure PACE scratch symlink)
OUTPUT_BASE_DIR="$HOME/scratch/segmentation_outputs"

# Ensure target directories exist before execution
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "${OUTPUT_BASE_DIR}/results"
mkdir -p "${OUTPUT_BASE_DIR}/Results"

# Prefer an explicit env python path to avoid depending on `conda activate` in batch.
DEFAULT_ENV_PYTHON="$HOME/.conda/envs/seg_env/bin/python"
PYTHON_EXE="${PYTHON_EXE:-$DEFAULT_ENV_PYTHON}"
if [ ! -x "$PYTHON_EXE" ]; then
    PYTHON_EXE="$(command -v python || true)"
fi

# GPU / CUDA settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

# Calibration Settings -> Exported so Python code inherits the Scratch path
CALIBRATION_CACHE_KEY="split_3tiles_steps5_v2"
export CALIBRATION_OUTPUT_DIR="${OUTPUT_BASE_DIR}/results/contrastive_calibration/${CALIBRATION_CACHE_KEY}"
mkdir -p "$CALIBRATION_OUTPUT_DIR"

# Pipeline runtime environment flags
export REQUIRE_CUDA=1
export PREFER_CUDA=1
export OVERWRITE_PIPELINE_CACHE=0
export SKIP_MASK_CACHING=1
export SKIP_IF_VISUALIZATIONS_EXIST=1
export SAVE_INPUT_IMAGES=0

echo "[INFO] Code Repository Root: ${PROJECT_ROOT}"
echo "[INFO] High-Capacity Scratch Output: ${OUTPUT_BASE_DIR}"
echo "[INFO] Calibration Output Target: ${CALIBRATION_OUTPUT_DIR}"

if [ ! -x "$PYTHON_EXE" ]; then
    echo "[ERROR] Python executable not found at ${PYTHON_EXE}"
    echo "[HINT] Export PYTHON_EXE to your env python path, e.g. $HOME/.conda/envs/seg_env/bin/python"
    exit 1
fi

echo "[INFO] Python executable: ${PYTHON_EXE}"

# Tile location roots for single-tile runs.
# Priority:
# 1) explicit TILE_BASE_DIR env var
# 2) local repo path
# 3) PACE shared project path
LOCAL_TILE_BASE="${PROJECT_ROOT}/Maps/Tiles/Atlanta_split_google"
PACE_TILE_BASE="/storage/project/r-pkastner3-0/ibaracskay3/Segmentation-Reserach-Manual/Maps/Tiles/Atlanta_split_google"

if [ -n "${TILE_BASE_DIR:-}" ]; then
    RESOLVED_TILE_BASE="$TILE_BASE_DIR"
elif [ -d "$LOCAL_TILE_BASE" ]; then
    RESOLVED_TILE_BASE="$LOCAL_TILE_BASE"
elif [ -d "$PACE_TILE_BASE" ]; then
    RESOLVED_TILE_BASE="$PACE_TILE_BASE"
else
    RESOLVED_TILE_BASE="$LOCAL_TILE_BASE"
fi

echo "[INFO] Tile base directory: ${RESOLVED_TILE_BASE}"

# ==========================================
# 3. RUN MODES (SINGLE TILE VS FULL RUN)
# ==========================================

# Mode A: Single-Tile Prediction Mode
if [ "$1" = "single" ]; then
    TILE_INPUT="${2:-tile_002_003}"

    # Accept either a full tif path or a tile stem resolved against TILE_BASE_DIR.
    if [[ "$TILE_INPUT" == *.tif ]] || [[ "$TILE_INPUT" == */* ]]; then
        SOURCE_TIF="$TILE_INPUT"
    else
        TILE_STEM="$TILE_INPUT"
        SOURCE_TIF="${RESOLVED_TILE_BASE}/${TILE_STEM}.tif"
    fi

    if [ ! -f "$SOURCE_TIF" ]; then
        echo "[ERROR] Could not locate source tif for input: ${TILE_INPUT}"
        echo "[INFO] Expected path: ${SOURCE_TIF}"
        echo "[INFO] Current tile base: ${RESOLVED_TILE_BASE}"
        echo "[HINT] Pass a full tif path as the second argument, e.g.:"
        echo "[HINT] sbatch pace_run_contrastive_calibration.sh single /path/to/tile_002_003.tif"
        echo "[HINT] Or set TILE_BASE_DIR to your tiles directory before running."
        exit 2
    fi

    TILE_STEM="$(basename "$SOURCE_TIF" .tif)"

    # Route the heavy visualization outputs directly to Scratch space
    OUT_DIR="${OUTPUT_BASE_DIR}/Results/dino_sam_preview_${TILE_STEM}"
    mkdir -p "$OUT_DIR"

    echo "[INFO] Running single-tile DINO+SAM pipeline for ${TILE_STEM}"
    echo "[INFO] Source Map: ${SOURCE_TIF}"
    echo "[INFO] Output Destination: ${OUT_DIR}"

    # Export the per-process override so get_satellite.py can detect the chosen tif
    export SEGMENTATION_TIF_FILE="$SOURCE_TIF"

    # Run the runs/get_satellite.py inside a small Python wrapper so we can set
    # models.config.results_dir to our scratch OUT_DIR before executing the module.
    "$PYTHON_EXE" - <<PYTHON
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from models import config as cfg
cfg.results_dir = Path("$OUT_DIR")
import runpy
runpy.run_path(str(Path("$PROJECT_ROOT") / "runs" / "get_satellite.py"), run_name="__main__")
PYTHON

    EXIT_CODE=$?
    echo "[INFO] Single DINO+SAM run finished with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Mode B: Default Full Contrastive Calibration Run
echo "[INFO] Running full contrastive calibration..."
"$PYTHON_EXE" "${PROJECT_ROOT}/runs/contrastive_calibration.py"
EXIT_CODE=$?

echo "[INFO] Full pipeline finished with exit code $EXIT_CODE"
exit $EXIT_CODE