@echo off
title HATI Mission Dashboard
cd /d "%~dp0"
echo.
echo   H A T I   -  Hazard Assessment and Terrain Intelligence
echo   starting mission dashboard ...
echo.

REM prefer the project venv if one exists, otherwise fall back to system python
if exist ".venv\Scripts\python.exe" (
  set "HATI_PY=.venv\Scripts\python.exe"
  echo   using .venv
) else (
  set "HATI_PY=python"
  echo   using system python ^(no .venv found^)
)
echo.

"%HATI_PY%" "dashboard\hati_dashboard.py" %*
if errorlevel 1 (
  echo.
  echo   [!] failed to start.
  echo       - is Python on PATH?   python --version
  echo       - are deps installed?  python -m pip install numpy scipy scikit-image matplotlib rasterio
  echo       - port busy?           HATI_Dashboard.bat --port 8800
  pause
)
