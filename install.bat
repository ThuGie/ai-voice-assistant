@echo off
echo ======================================
echo   AI Voice Assistant Installer
echo ======================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found! Please install Python 3.8 or higher.
    echo You can download Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check Python version
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo Python 3.8 or higher is required. Please update your Python installation.
    pause
    exit /b 1
)

echo Python check passed.
echo.

REM Run the installation script
echo Starting installation process...
echo This may take a few minutes depending on your internet connection.
echo.
python install.py

if %ERRORLEVEL% NEQ 0 (
    echo Installation failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ======================================
echo  Installation completed successfully!
echo ======================================
echo.
echo You can now run the AI Voice Assistant using:
echo   1. Double-click on the desktop shortcut
echo   2. Or run "python main.py" in this directory
echo.
echo Enjoy your AI Voice Assistant!
echo.
pause
