@echo off
title HATI Mission Dashboard
cd /d "%~dp0"
echo.
echo   H A T I   -  Hazard Assessment and Terrain Intelligence
echo   starting mission dashboard ...
echo.
python "dashboard\hati_dashboard.py"
if errorlevel 1 (
  echo.
  echo   [!] failed to start. Is Python on PATH?  ^(python --version^)
  pause
)
