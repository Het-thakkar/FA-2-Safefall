@echo off
REM Setup script for SafeFall project

echo.
echo ========================================
echo SafeFall Fall Detection - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version
echo OK!

echo.
echo [2/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo OK!

echo.
echo [3/4] Checking for MediaPipe pose model...
if not exist "pose_landmarker_lite.task" (
    echo.
    echo WARNING: pose_landmarker_lite.task not found!
    echo.
    echo Please download it from:
    echo https://storage.googleapis.com/mediapipe-tasks/python/vision/20221208/pose_landmarker_lite.task
    echo.
    echo Save it to: %CD%\pose_landmarker_lite.task
    echo.
    pause
) else (
    echo OK! Model found
)

echo.
echo [4/4] Verifying project structure...
if exist "train_model.py" (
    echo OK! All files present
) else (
    echo ERROR: Missing required files
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download pose_landmarker_lite.task if not present
echo 2. Run: python train_model.py
echo 3. Run: streamlit run Safefall.py
echo.
pause
