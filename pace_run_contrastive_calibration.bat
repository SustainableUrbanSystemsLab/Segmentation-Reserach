@echo off
setlocal enabledelayedexpansion

rem PACE Phoenix launcher for contrastive calibration
rem Adjust the variables below if you want to change GPUs, output location, or the calibration key.

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

rem Use the local virtual environment Python directly.
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"

rem GPU / CUDA settings.
set "CUDA_VISIBLE_DEVICES=0"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256"

rem Calibration settings.
set "CALIBRATION_CACHE_KEY=split_3tiles_steps5_v2"
set "CALIBRATION_OUTPUT_DIR=%PROJECT_ROOT%results\contrastive_calibration\%CALIBRATION_CACHE_KEY%"

rem Pipeline behavior.
set "REQUIRE_CUDA=1"
set "PREFER_CUDA=1"
set "OVERWRITE_PIPELINE_CACHE=0"
set "SKIP_MASK_CACHING=1"
set "SKIP_IF_VISUALIZATIONS_EXIST=1"
set "SAVE_INPUT_IMAGES=0"

rem Optional: pass a specific tile list or let runs\contrastive_calibration.py use its built-in tile list.
rem To keep this batch simple, we call the calibration driver directly.

echo [INFO] Project root: %PROJECT_ROOT%
echo [INFO] Python: %PYTHON_EXE%
echo [INFO] Output dir: %CALIBRATION_OUTPUT_DIR%
echo [INFO] CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found at "%PYTHON_EXE%"
    exit /b 1
)

"%PYTHON_EXE%" "%PROJECT_ROOT%runs\contrastive_calibration.py"
set "EXIT_CODE=%ERRORLEVEL%"

echo [INFO] Finished with exit code %EXIT_CODE%
exit /b %EXIT_CODE%
