#!/usr/bin/env python3
"""
Setup script for AI Voice Assistant.

This script handles installation of the AI Voice Assistant package,
including all its dependencies.
"""

import os
import sys
import platform
from setuptools import setup, find_packages

# Check Python version
if sys.version_info < (3, 8):
    sys.exit('Python >= 3.8 is required')

# Package metadata
NAME = 'ai-voice-assistant'
DESCRIPTION = 'An AI voice assistant that integrates MeloTTS, Faster-Whisper, Ollama, and PyTorch Vision'
URL = 'https://github.com/ThuGie/ai-voice-assistant'
EMAIL = 'author@example.com'
AUTHOR = 'AI Voice Assistant Team'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.1.0'

# Required packages
REQUIRED = [
    # Core dependencies
    'numpy>=1.20.0',
    'pillow>=8.3.1',
    'requests>=2.25.0',
    'tqdm>=4.62.0',
    'pydub>=0.25.1',
    'sounddevice>=0.4.4',
    'python-dotenv>=0.19.0',
    'typer>=0.4.0',
    'rich>=10.9.0',
    
    # AI and ML dependencies
    'torch>=2.0.0',
    'torchaudio>=2.0.0',
    'torchvision>=0.15.0',
    'transformers>=4.25.1',
    'accelerate>=0.19.0',
    
    # MeloTTS dependencies
    'melotts>=0.1.8',
    'librosa>=0.9.2',
    
    # Faster-Whisper dependencies
    'faster-whisper>=0.8.0',
    'ctranslate2>=3.10.0',
    'ffmpeg-python>=0.2.0',
    
    # Ollama dependencies
    'ollama>=0.1.2',
    
    # Computer Vision dependencies
    'opencv-python>=4.6.0',
    'mss>=6.1.0',  # For screen capture
    
    # Additional utilities
    'pynput>=1.7.6',  # For keyboard and mouse control
    'pyaudio>=0.2.11',  # Audio recording
    'webrtcvad>=2.0.10',  # Voice activity detection
]

# Platform-specific dependencies
if platform.system() == 'Windows':
    REQUIRED.append('d3dshot>=0.14.4')  # Fast screen capture for Windows

# Development dependencies
EXTRAS = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'flake8>=3.8.0',
        'black>=21.5b2',
        'isort>=5.0.0',
        'mypy>=0.800',
    ],
    'gui': [
        'customtkinter>=4.6.3',  # Modern looking tkinter
        'PyQt5>=5.15.6',  # Alternative GUI option
    ],
}

# Read the README for the long description
try:
    with open('README.md', encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Multimedia :: Video',
    ],
    entry_points={
        'console_scripts': [
            'ai-assistant=main:main',
        ],
    },
)
