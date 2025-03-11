#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}${BOLD}======================================${NC}"
echo -e "${BLUE}${BOLD}   AI Voice Assistant Installer${NC}"
echo -e "${BLUE}${BOLD}======================================${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found! Please install Python 3.8 or higher.${NC}"
    echo "You can install it using your package manager:"
    echo "  - Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  - Fedora: sudo dnf install python3 python3-pip"
    echo "  - macOS: brew install python3"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_VERSION_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_VERSION_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_VERSION_MAJOR" -lt 3 ] || ([ "$PYTHON_VERSION_MAJOR" -eq 3 ] && [ "$PYTHON_VERSION_MINOR" -lt 8 ]); then
    echo -e "${RED}Python 3.8 or higher is required. Found Python $PYTHON_VERSION${NC}"
    echo "Please update your Python installation."
    exit 1
fi

echo -e "${GREEN}Python check passed. Found Python $PYTHON_VERSION${NC}"
echo ""

# Check if running with root privileges
if [ "$(id -u)" -eq 0 ]; then
    echo -e "${YELLOW}Warning: You are running this script as root.${NC}"
    echo "It's generally not recommended to install Python packages as root."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

# Run the installation script
echo -e "${BLUE}Starting installation process...${NC}"
echo "This may take a few minutes depending on your internet connection."
echo ""

# Make the script executable first (in case it was downloaded without executable permission)
chmod +x install.py

# Run the installation script
python3 install.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Installation failed. Please check the error messages above.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}${BOLD}======================================${NC}"
echo -e "${GREEN}${BOLD} Installation completed successfully!${NC}"
echo -e "${GREEN}${BOLD}======================================${NC}"
echo ""
echo "You can now run the AI Voice Assistant using:"
echo "  1. Double-click on the desktop shortcut"
echo "  2. Or run \"python3 main.py\" in this directory"
echo ""
echo "Enjoy your AI Voice Assistant!"
echo ""
