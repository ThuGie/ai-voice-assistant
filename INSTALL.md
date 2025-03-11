# Installation Guide

This guide will help you set up the AI Voice Assistant on your system.

## Prerequisites

Before installation, make sure you have the following prerequisites:

- **Python 3.8 or higher**
- **CUDA-compatible GPU** (recommended, but not required)
- **FFmpeg** (required for audio processing)
- **Microphone** (for voice input)
- **Speakers** (for voice output)

## Automatic Installation

We provide automated installation scripts for easy setup:

### Windows

1. Double-click `install.bat` in the project directory
2. Follow the on-screen instructions

### macOS/Linux

1. Open a terminal in the project directory
2. Make the script executable: `chmod +x install.sh`  
3. Run the installation script: `./install.sh`
4. Follow the on-screen instructions

## Manual Installation

If you prefer to install manually or the automatic installation doesn't work for you, follow these steps:

## Step 1: Install Ollama

[Ollama](https://github.com/ollama/ollama) is required to run large language models locally. Follow the official installation instructions for your platform:

### Windows

Download and install from [Ollama's website](https://ollama.ai/download/windows).

### macOS

Download and install from [Ollama's website](https://ollama.ai/download/mac) or use Homebrew:

```bash
brew install ollama
```

### Linux

Follow the official Linux installation instructions:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## Step 2: Clone the Repository

Clone the AI Voice Assistant repository:

```bash
git clone https://github.com/ThuGie/ai-voice-assistant.git
cd ai-voice-assistant
```

## Step 3: Create a Virtual Environment

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\\Scripts\\activate
# On macOS/Linux
source venv/bin/activate
```

## Step 4: Install Dependencies

Install the required dependencies:

```bash
# Update pip to the latest version
pip install --upgrade pip

# Install using pip
pip install -r requirements.txt
```

## Step 5: Download Required Models

Pull the necessary large language model with Ollama:

```bash
# Start the Ollama service
ollama serve  # Run this in a separate terminal

# Pull the Llama 3 model (this will download several gigabytes)
ollama pull llama3
```

## Step 6: Set Up Configuration Files

The assistant requires several configuration files to function properly. The default templates are included in the repository:

- `config.json` - Main configuration file
- `assistant_profile.json` - Assistant identity information
- `assistant_personality.json` - Personality traits
- `assistant_context.json` - Context awareness and topics
- `emotions_memory.json` - Emotional memory storage

These files will be created with default values if they don't exist.

## Step 7: Run the Assistant

Start the AI Voice Assistant:

```bash
# Run with default settings
python main.py

# Run with a specific voice
python main.py --voice english_female_1

# Run with a specific TTS model
python main.py --tts-model en_US/vctk_medium

# Run with a specific AI model
python main.py --model llama3

# Run with a better speech recognition model
python main.py --stt-model medium

# Run with a custom configuration file
python main.py --config custom_config.json
```

## Command-Line Options

The assistant supports various command-line options:

- `--config PATH` - Path to a custom configuration file
- `--voice NAME` - Voice to use for text-to-speech
- `--tts-model NAME` - Text-to-speech model to use
- `--model NAME` - AI model to use
- `--stt-model SIZE` - Speech-to-text model size
- `--emotion NAME` - Initial emotion for the assistant
- `--name NAME` - Assistant's name
- `--user-name NAME` - User's name
- `--debug` - Enable debug logging

## Troubleshooting

### Common Issues

1. **Missing dependencies**: If you encounter errors about missing dependencies, try:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **CUDA issues**: If you experience GPU-related errors, try running in CPU mode by setting:
   ```
   # In config.json, modify the "vision" section:
   "vision": {
     "device": "cpu",
     ...
   }
   ```

3. **Audio issues**: If you have problems with audio input/output:
   - Make sure your microphone and speakers are correctly configured
   - Check the audio device settings on your system
   - Verify that PyAudio is correctly installed
   - Try installing PortAudio if it's missing (`apt-get install portaudio19-dev` on Ubuntu/Debian)

4. **MeloTTS model loading issues**: If you encounter errors loading the TTS models:
   - Make sure you have a good internet connection for downloading models
   - Try using a smaller model (e.g., `en_US/vctk_low` instead of `en_US/vctk_medium`)
   - Check disk space, as models can require several hundred MB

5. **Faster-Whisper issues**: For speech recognition problems:
   - Try using a smaller model size if you have limited memory
   - Check your microphone input levels and ensure it's the default device
   - Increase the VAD aggressiveness in the config file if the assistant doesn't detect your speech

6. **Ollama connectivity**: If the assistant can't connect to Ollama:
   - Make sure Ollama is running (`ollama serve` in a separate terminal)
   - Check that the API endpoint in config.json is correct (usually `http://localhost:11434/api`)
   - Verify that the model is downloaded (`ollama list` to check)

### Getting Help

If you encounter any issues not covered here, please:

1. Check the [GitHub issues](https://github.com/ThuGie/ai-voice-assistant/issues) to see if it's a known problem
2. Create a new issue with details about your problem and system configuration

## Optional Components

### VTube Studio Support (Future)

Support for VTube Studio will be added in a future release.

### Custom RAG (Future)

Retrieval-Augmented Generation capabilities will be added in a future release.

### PC Control Capabilities (Future)

More advanced PC control features will be available in future versions.
