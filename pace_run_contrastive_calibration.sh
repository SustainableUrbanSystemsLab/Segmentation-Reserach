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

# Try loading anaconda module if available (harmless if absent)
module load anaconda3 || true

# Initialize Conda for headless shell execution safely. We try multiple
# locations so the script works on different cluster images and user setups.
if command -v conda >/dev/null 2>&1; then
    conda activate seg_env || true
else
    if [ -f "/usr/local/pace-apps/spack/packages/linux-rhel8-zen2/gcc-11.3.1/anaconda3-2022.05-tw3uiww7g7sc7wunw7atshwkyfxtw7gn/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "/usr/local/pace-apps/spack/packages/linux-rhel8-zen2/gcc-11.3.1/anaconda3-2022.05-tw3uiww7g7sc7wunw7atshwkyfxtw7gn/etc/profile.d/conda.sh" && conda activate seg_env || true
    elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.conda/etc/profile.d/conda.sh" && conda activate seg_env || true
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate seg_env || true
    else
        echo "[WARN] No conda activation script found; continuing with PATH python if available"
    fi
fi

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

# Prefer the active environment's python, otherwise fall back to the hardcoded path.
PYTHON_EXE="$(command -v python || true)"
if [ -z "$PYTHON_EXE" ]; then
    PYTHON_EXE="/storage/home/hcoda1/3/ibaracskay3/.conda/envs/seg_env/bin/python"
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

if [ ! -f "$PYTHON_EXE" ]; then
    echo "[ERROR] Python executable not found at ${PYTHON_EXE}"
    exit 1
fi

# ==========================================
# 3. RUN MODES (SINGLE TILE VS FULL RUN)
# ==========================================

# Mode A: Single-Tile Prediction Mode
if [ "$1" = "single" ]; then
    if [ -z "$2" ]; then
        TILE_STEM="tile_002_003"
    else
        TILE_STEM="$2"
    fi

    # Read the raw source TIF directly from your repository location
    SOURCE_TIF="${PROJECT_ROOT}/Maps/Tiles/Atlanta_split_google/${TILE_STEM}.tif"

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