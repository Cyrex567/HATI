@echo off
REM Build a single-file HATI.exe -- run this ON the machine that will run it
REM (PyInstaller output is platform-specific: a Windows build runs on Windows only).
cd /d "%~dp0"
pip install pyinstaller pillow || goto :err
python dashboard\make_icon.py
pyinstaller --onefile --name HATI ^
  --icon dashboard\static\assets\hati.ico ^
  --add-data "dashboard\static;static" ^
  dashboard\hati_dashboard.py || goto :err
echo.
echo   built: dist\HATI.exe   (place it in the project root next to scripts\ and data\)
goto :eof
:err
echo build failed & pause
