@echo off
echo Installing dependencies...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Download it from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

pip install -r requirements.txt

echo.
echo Running checks...
python check.py

echo.
echo Done. Run "python run.py" to start recording.
pause
