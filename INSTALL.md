# Installation Guide

This guide will help you set up the AI Voice Assistant on your system.

## Prerequisites

Before installation, make sure you have the following prerequisites:

- **Python 3.8 or higher**
- **CUDA-compatible GPU** (recommended, but not required)
- **FFmpeg** (required for audio processing)
- **Microphone** (for voice input)
- **Speakers** (for voice output)

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
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

## Step 4: Install Dependencies

Install the required dependencies:

```bash
# Install using pip
pip install -r requirements.txt

# Or install as a package (development mode)
pip install -e .
```

## Step 5: Download Required Models

Pull the necessary large language model with Ollama:

```bash
# Start the Ollama service
ollama serve  # Run this in a separate terminal

# Pull the Llama 3 model (this will download several gigabytes)
ollama pull llama3
```

## Step 6: Configure the Assistant

The assistant comes with a default configuration in `config.json`. You can modify this file to customize your experience:

- Choose a different voice model
- Select an alternative AI model
- Change speech recognition settings
- Adjust memory and conversation settings

## Step 7: Run the Assistant

Start the AI Voice Assistant:

```bash
# Run with default settings
python main.py

# Run with a specific voice
python main.py --voice english_female_1

# Run with a specific AI model
python main.py --model gemma:7b

# Run with a custom configuration file
python main.py --config custom_config.json
```

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
   - Try changing the sample rate in the config file

4. **Ollama connectivity**: If the assistant can't connect to Ollama:
   - Make sure Ollama is running (`ollama serve`)
   - Check that the API endpoint in config.json is correct

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

## Advanced Installation

For advanced installation options, additional features, or development setup, see the [Advanced Setup](https://github.com/ThuGie/ai-voice-assistant/wiki/Advanced-Setup) wiki page.
