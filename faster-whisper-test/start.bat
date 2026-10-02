@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    echo.
    echo Python 3.9 or newer is required.
    echo Please install Python from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating local Python environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
  )
)

echo [2/3] Installing/refreshing required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed.
  pause
  exit /b 1
)

echo [3/3] Starting faster-whisper local server...
start "" /min cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:7860/"
".venv\Scripts\python.exe" server.py
pause
