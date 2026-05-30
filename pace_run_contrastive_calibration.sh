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
module load gcc/11.3.1
module load anaconda3                     # Exposes conda commands inside Slurm cleanly

# Initialize Conda for headless shell execution safely
source /usr/local/pace-apps/spack/packages/linux-rhel8-zen2/gcc-11.3.1/anaconda3-2022.05-tw3uiww7g7sc7wunw7atshwkyfxtw7gn/etc/profile.d/conda.sh || source ~/.conda/etc/profile.d/conda.sh
conda activate seg_env

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

# Absolute path to your active environment's Python binary
PYTHON_EXE="/storage/home/hcoda1/3/ibaracskay3/.conda/envs/seg_env/bin/python"

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
    OUT_DIR="${OUTPUT_BASE_DIR}/Results/yoloseg_preview_${TILE_STEM}"
    mkdir -p "$OUT_DIR"

    echo "[INFO] Running single-tile prediction for ${TILE_STEM}"
    echo "[INFO] Source Map: ${SOURCE_TIF}"
    echo "[INFO] Output Destination: ${OUT_DIR}"

    # Runs python module tracking context relative to your workspace root
    "$PYTHON_EXE" -m yoloseg_pipeline.predict \
        --weights Results/yoloseg/wind_comfort_seg-2/weights/best.pt \
        --source "$SOURCE_TIF" \
        --output-dir "$OUT_DIR" \
        --imgsz 1024 \
        --conf 0.25 \
        --tile-size 1024 \
        --tile-overlap 128

    EXIT_CODE=$?
    echo "[INFO] Single-tile run finished with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Mode B: Default Full Contrastive Calibration Run
echo "[INFO] Running full contrastive calibration..."
"$PYTHON_EXE" "${PROJECT_ROOT}/runs/contrastive_calibration.py"
EXIT_CODE=$?

echo "[INFO] Full pipeline finished with exit code $EXIT_CODE"
exit $EXIT_CODE