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
  echo.
  echo Package installation failed.
  echo The local server was NOT started.
  echo.
  pause
  exit /b 1
)

echo [3/3] Starting faster-whisper local server...
echo.
echo The server window will stay open while you use the tool.
echo Do not close that window.
echo.

start "faster-whisper server" cmd /k ""%~dp0.venv\Scripts\python.exe" "%~dp0server.py""

echo Waiting for the local server...
for /L %%i in (1,1,30) do (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:7860/api/health' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {} ; exit 1" >nul 2>&1
  if not errorlevel 1 (
    echo.
    echo Server is ready.
    start "" "http://127.0.0.1:7860/"
    exit /b 0
  )
  timeout /t 1 /nobreak >nul
)

echo.
echo The local server did not respond within 30 seconds.
echo Please check the "faster-whisper server" window for the error message.
echo.
pause
