#!/usr/bin/env bash
set -euo pipefail

# PACE Phoenix launcher for contrastive calibration.
# Use this with the PACE job composer on a GPU node.
#
# Suggested composer settings for this workload:
# - Node type: GPU node, not CPU-only
# - CPUs: 4 for a smoke test, 8 if available for full runs
# - Memory: 16 GB minimum, 32 GB safer for long sweeps
# - Hours: 2 for a smoke test, ~16 hours for the current 55-config x 3-tile sweep
# - QOS: whatever your allocation permits; your screenshot showed "inferno"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"

# If your PACE environment uses modules, load Python/CUDA here.
# Example:
# module purge
# module load python
# module load cuda

# If you have a conda env on PACE, activate it here instead of using the local Windows venv.
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate segmentation-research

# Calibration controls.
export CALIBRATION_CACHE_KEY="${CALIBRATION_CACHE_KEY:-split_3tiles_steps5_v2}"
export CALIBRATION_RESUME_FROM_CACHE="${CALIBRATION_RESUME_FROM_CACHE:-1}"
export CALIBRATION_FORCE_RERUN="${CALIBRATION_FORCE_RERUN:-0}"
export STEPS_EACH_SIDE="${STEPS_EACH_SIDE:-5}"
export PREFER_CUDA="${PREFER_CUDA:-1}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OVERWRITE_PIPELINE_CACHE="${OVERWRITE_PIPELINE_CACHE:-0}"
export SKIP_MASK_CACHING="${SKIP_MASK_CACHING:-1}"
export SKIP_IF_VISUALIZATIONS_EXIST="${SKIP_IF_VISUALIZATIONS_EXIST:-1}"
export SAVE_INPUT_IMAGES="${SAVE_INPUT_IMAGES:-0}"
export ESTIMATED_MINUTES_PER_TILE="${ESTIMATED_MINUTES_PER_TILE:-5.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

# Use the first Python on PATH. On PACE, that should come from the loaded module or conda env.
PYTHON_BIN="${PYTHON_BIN:-python}"

cat <<EOF
[INFO] Project root: $PROJECT_ROOT
[INFO] Python: $PYTHON_BIN
[INFO] Cache key: $CALIBRATION_CACHE_KEY
[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
[INFO] SKIP_MASK_CACHING: $SKIP_MASK_CACHING
[INFO] SKIP_IF_VISUALIZATIONS_EXIST: $SKIP_IF_VISUALIZATIONS_EXIST
[INFO] SAVE_INPUT_IMAGES: $SAVE_INPUT_IMAGES
EOF

exec "$PYTHON_BIN" runs/contrastive_calibration.py
