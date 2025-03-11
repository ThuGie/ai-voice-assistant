#!/usr/bin/env python3
"""
Installation script for AI Voice Assistant

This script automates the installation process for the AI Voice Assistant,
including checking prerequisites, installing dependencies, and setting up
required components.
"""

import os
import sys
import platform
import subprocess
import shutil
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_step(step_num: int, total_steps: int, message: str):
    """Print a formatted step message"""
    print(f"{Colors.BOLD}[{step_num}/{total_steps}] {Colors.BLUE}{message}{Colors.ENDC}")

def print_success(message: str):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message: str):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def check_python_version() -> bool:
    """Check if Python version meets requirements"""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print_error(f"Python {required_version[0]}.{required_version[1]} or higher is required")
        print(f"Current version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    print_success(f"Python version {current_version[0]}.{current_version[1]}.{current_version[2]} meets requirements")
    return True

def check_command(command: str) -> bool:
    """Check if a command is available"""
    try:
        # Use 'where' on Windows, 'which' on Unix-like systems
        check_cmd = 'where' if platform.system() == 'Windows' else 'which'
        subprocess.run([check_cmd, command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def run_command(command: List[str], capture_output: bool = False, check: bool = True) -> Tuple[int, Optional[str]]:
    """Run a command and return the return code and output"""
    try:
        if capture_output:
            result = subprocess.run(command, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return result.returncode, result.stdout
        else:
            result = subprocess.run(command, check=check)
            return result.returncode, None
    except subprocess.SubprocessError as e:
        print_error(f"Command failed: {' '.join(command)}")
        print(f"Error: {e}")
        return 1, str(e) if capture_output else None

def check_gpu() -> bool:
    """Check if a CUDA-compatible GPU is available"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                print_success(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print_warning("No CUDA-compatible GPU detected. The assistant will run on CPU, which may be slower.")
            return False
    except ImportError:
        print_warning("PyTorch not installed yet. GPU check will be performed after installation.")
        return False

def create_virtual_environment(venv_path: str) -> bool:
    """Create a virtual environment"""
    print_step(3, 8, "Creating virtual environment...")
    
    if os.path.exists(venv_path):
        response = input(f"Virtual environment already exists at {venv_path}. Recreate? (y/n): ")
        if response.lower() != 'y':
            print_success("Using existing virtual environment")
            return True
        
        # Remove existing venv
        try:
            shutil.rmtree(venv_path)
        except Exception as e:
            print_error(f"Failed to remove existing virtual environment: {e}")
            return False
    
    # Create new venv
    returncode, _ = run_command([sys.executable, "-m", "venv", venv_path], capture_output=True, check=False)
    
    if returncode != 0:
        print_error("Failed to create virtual environment")
        return False
    
    print_success(f"Virtual environment created at {venv_path}")
    return True

def get_pip_path(venv_path: str) -> str:
    """Get the path to pip in the virtual environment"""
    if platform.system() == 'Windows':
        return os.path.join(venv_path, 'Scripts', 'pip.exe')
    else:
        return os.path.join(venv_path, 'bin', 'pip')

def install_dependencies(venv_path: str, requirements_path: str) -> bool:
    """Install dependencies from requirements.txt"""
    print_step(4, 8, "Installing dependencies...")
    
    pip_path = get_pip_path(venv_path)
    
    # Upgrade pip first
    print("Upgrading pip...")
    returncode, _ = run_command([pip_path, "install", "--upgrade", "pip"], check=False)
    
    if returncode != 0:
        print_warning("Failed to upgrade pip. Continuing with existing version.")
    
    # Install main dependencies
    print("Installing dependencies from requirements.txt...")
    returncode, output = run_command([pip_path, "install", "-r", requirements_path], capture_output=True, check=False)
    
    if returncode != 0:
        print_error("Failed to install dependencies")
        if output:
            print("Error details:")
            print(output)
        return False
    
    print_success("Dependencies installed successfully")
    return True

def setup_ollama() -> bool:
    """Check or install Ollama"""
    print_step(5, 8, "Setting up Ollama...")
    
    # Check if Ollama is already installed
    if check_command('ollama'):
        print_success("Ollama is already installed")
        return True
    
    # Offer to install Ollama
    print_warning("Ollama is not installed")
    print("Ollama is required to run the AI models locally.")
    install_choice = input("Do you want to install Ollama now? (y/n): ")
    
    if install_choice.lower() != 'y':
        print_warning("Ollama installation skipped. You will need to install it manually before using the assistant.")
        print("See installation instructions at: https://github.com/ollama/ollama")
        return False
    
    # Install Ollama based on platform
    system = platform.system()
    
    if system == 'Linux':
        print("Installing Ollama for Linux...")
        returncode, _ = run_command(['bash', '-c', 'curl -fsSL https://ollama.ai/install.sh | sh'], check=False)
        
        if returncode != 0:
            print_error("Failed to install Ollama")
            print("Please install manually from: https://github.com/ollama/ollama")
            return False
    
    elif system == 'Darwin':  # macOS
        if shutil.which('brew'):
            print("Installing Ollama using Homebrew...")
            returncode, _ = run_command(['brew', 'install', 'ollama'], check=False)
            
            if returncode != 0:
                print_error("Failed to install Ollama via Homebrew")
                print("Please install manually from: https://ollama.ai/download/mac")
                return False
        else:
            print_warning("Homebrew not found. Please install Ollama manually:")
            print("Download from: https://ollama.ai/download/mac")
            return False
    
    elif system == 'Windows':
        print_warning("Automated Ollama installation for Windows is not supported")
        print("Please download and install from: https://ollama.ai/download/windows")
        return False
    
    else:
        print_error(f"Unsupported platform: {system}")
        print("Please install Ollama manually from: https://github.com/ollama/ollama")
        return False
    
    print_success("Ollama installed successfully")
    return True

def download_ai_model() -> bool:
    """Download the default AI model from Ollama"""
    print_step(6, 8, "Setting up AI model...")
    
    # Check if Ollama is available
    if not check_command('ollama'):
        print_error("Ollama is not installed. Cannot download AI model.")
        return False
    
    # Get the default model from config.json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            model = config.get('ai', {}).get('model', 'llama3')
    except (FileNotFoundError, json.JSONDecodeError):
        model = 'llama3'  # Fallback to default
    
    print(f"Checking for model: {model}")
    
    # Check if the model is already available
    returncode, output = run_command(['ollama', 'list'], capture_output=True, check=False)
    
    if returncode != 0:
        print_error("Failed to check available models")
        return False
    
    if output and model in output:
        print_success(f"Model {model} is already available")
        return True
    
    # Ask if we should download the model
    download_choice = input(f"Model {model} not found. Download it now? (y/n): ")
    
    if download_choice.lower() != 'y':
        print_warning(f"Model {model} download skipped. You will need to download it manually before using the assistant.")
        print("You can download it later using: ollama pull " + model)
        return False
    
    # Start Ollama server if it's not running (in background)
    if platform.system() == 'Windows':
        # For Windows, we can't easily run the server in the background
        print_warning("Please ensure Ollama server is running before downloading the model")
        print("You can start it manually by running 'ollama serve' in another terminal.")
        server_running = input("Is Ollama server running? (y/n): ")
        if server_running.lower() != 'y':
            print_error("Please start Ollama server and try again")
            return False
    else:
        # Try to start the server in the background
        print("Starting Ollama server...")
        
        # Use a temporary file to capture output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            temp_filename = tmp.name
        
        # Start the server process
        server_process = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True  # Detach the process
        )
        
        # Give the server some time to start
        print("Waiting for Ollama server to start...")
        time.sleep(5)
    
    # Download the model
    print(f"Downloading model {model}. This may take a while...")
    returncode, _ = run_command(['ollama', 'pull', model], check=False)
    
    if returncode != 0:
        print_error(f"Failed to download model {model}")
        print(f"You can download it later using: ollama pull {model}")
        return False
    
    print_success(f"Model {model} downloaded successfully")
    return True

def create_desktop_shortcut() -> bool:
    """Create a desktop shortcut to launch the assistant"""
    print_step(7, 8, "Creating desktop shortcut...")
    
    system = platform.system()
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    if not os.path.exists(desktop_path):
        print_warning("Desktop directory not found. Skipping shortcut creation.")
        return False
    
    if system == 'Windows':
        # Create a Windows .bat file
        shortcut_path = os.path.join(desktop_path, "AI-Voice-Assistant.bat")
        
        script_path = os.path.abspath("main.py")
        venv_activate = os.path.abspath(os.path.join("venv", "Scripts", "activate.bat"))
        
        with open(shortcut_path, 'w') as f:
            f.write('@echo off\n')
            f.write(f'cd /d "{os.path.dirname(script_path)}"\n')
            f.write(f'call "{venv_activate}"\n')
            f.write(f'python "{script_path}"\n')
            f.write('pause\n')
    
    elif system == 'Linux' or system == 'Darwin':
        # Create a shell script
        shortcut_path = os.path.join(desktop_path, "AI-Voice-Assistant.sh")
        
        script_path = os.path.abspath("main.py")
        venv_activate = os.path.abspath(os.path.join("venv", "bin", "activate"))
        
        with open(shortcut_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'cd "{os.path.dirname(script_path)}"\n')
            f.write(f'source "{venv_activate}"\n')
            f.write(f'python "{script_path}"\n')
        
        # Make the script executable
        os.chmod(shortcut_path, 0o755)
    
    else:
        print_warning(f"Desktop shortcut creation not supported on {system}")
        return False
    
    print_success(f"Desktop shortcut created at: {shortcut_path}")
    return True

def final_setup() -> bool:
    """Perform any final setup steps"""
    print_step(8, 8, "Finalizing installation...")
    
    # Create a basic .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# Environment variables for AI Voice Assistant\n")
            f.write("# Uncomment and modify as needed\n\n")
            f.write("# TTS_VOICE=english_male_1\n")
            f.write("# AI_MODEL=llama3\n")
            f.write("# LOG_LEVEL=INFO\n")
    
    print_success("Installation completed successfully!")
    
    # Provide instructions to start the assistant
    print("\n" + Colors.BOLD + "To start the AI Voice Assistant:" + Colors.ENDC)
    
    if platform.system() == 'Windows':
        print("1. Activate the virtual environment:")
        print("   venv\\Scripts\\activate")
        print("2. Run the assistant:")
        print("   python main.py")
    else:
        print("1. Activate the virtual environment:")
        print("   source venv/bin/activate")
        print("2. Run the assistant:")
        print("   python main.py")
    
    print("\nOr use the desktop shortcut created during installation.")
    
    return True

def main():
    """Main installation function"""
    print(Colors.HEADER + "\nAI Voice Assistant Installer" + Colors.ENDC)
    print(Colors.BOLD + "==========================" + Colors.ENDC)
    print("This script will install the AI Voice Assistant and its dependencies.\n")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check prerequisites
    print_step(1, 8, "Checking prerequisites...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check for FFmpeg
    if check_command('ffmpeg'):
        print_success("FFmpeg is installed")
    else:
        print_warning("FFmpeg is not installed. Audio processing may not work correctly.")
        print("Please install FFmpeg from: https://ffmpeg.org/download.html")
    
    # Check for GPU
    print_step(2, 8, "Checking for GPU...")
    check_gpu()
    
    # Create virtual environment
    venv_path = os.path.join(script_dir, "venv")
    if not create_virtual_environment(venv_path):
        sys.exit(1)
    
    # Install dependencies
    requirements_path = os.path.join(script_dir, "requirements.txt")
    if not install_dependencies(venv_path, requirements_path):
        sys.exit(1)
    
    # Setup Ollama
    if not setup_ollama():
        print_warning("Ollama setup incomplete. Some features may not work correctly.")
    
    # Download AI model
    if not download_ai_model():
        print_warning("AI model setup incomplete. You will need to download it manually.")
    
    # Create desktop shortcut
    create_desktop_shortcut()
    
    # Final setup
    final_setup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Installation failed: {e}")
        sys.exit(1)
