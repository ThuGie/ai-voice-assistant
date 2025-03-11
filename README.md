# AI Voice Assistant

An advanced voice-based AI assistant that integrates multiple open-source technologies to create a comprehensive interactive experience with your AI.

## Features

- **Voice Interaction**: Speak naturally to your AI using Faster-Whisper for accurate speech-to-text conversion
- **Voice Response**: AI responses are converted to natural-sounding speech with MeloTTS
- **Multiple Voice Options**: Select from various voice models in MeloTTS to customize your experience
- **Emotional Awareness**: Assistant can express emotions through voice modulation and response style
- **Personality Development**: Assistant develops a unique personality that evolves through interactions
- **Local AI Model**: Uses Ollama to run powerful AI models locally on your machine
- **Computer Vision**: Integrated PyTorch Vision capabilities allow the AI to "see" your screen or camera
- **Conversation Memory**: Stores and references past conversations for more contextual interactions
- **Context Awareness**: Maintains session context and can proactively engage based on interests
- **Extensible Architecture**: Designed for future extensions like VTube Studio support and custom RAG

## Technologies

This project seamlessly integrates several powerful open-source technologies:

- [MeloTTS](https://github.com/myshell-ai/MeloTTS) - High-quality multilingual text-to-speech library
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - Efficient speech-to-text with CTranslate2
- [Ollama](https://github.com/ollama/ollama) - Run large language models locally
- [PyTorch Vision](https://github.com/pytorch/vision) - Computer vision capabilities using PyTorch

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended for optimal performance
- Microphone for voice input
- Speakers for voice output
- Webcam (optional, for vision capabilities)

## Quick Installation

We provide easy installation scripts for different platforms:

### Windows

1. Download or clone this repository
2. Double-click `install.bat`
3. Follow the on-screen instructions

### Linux/macOS

1. Download or clone this repository
2. Open a terminal in the project directory
3. Make the script executable: `chmod +x install.sh`
4. Run the installation script: `./install.sh`
5. Follow the on-screen instructions

### Manual Installation

For more control over the installation process:

1. Clone this repository:
   ```bash
   git clone https://github.com/ThuGie/ai-voice-assistant.git
   cd ai-voice-assistant
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\\Scripts\\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Ollama following the instructions at [Ollama Installation](https://github.com/ollama/ollama#installation)

5. Download and install a compatible model for Ollama (e.g., Llama 3, Mistral, Gemma)
   ```bash
   ollama pull llama3
   ```

## Usage

1. Start the assistant:
   ```bash
   python main.py
   ```

2. Customize your assistant with optional parameters:
   ```bash
   python main.py --voice english_female_1 --model llama3 --stt-model medium
   ```

3. Start speaking to interact with the assistant

4. Use special commands prefixed with `!` for additional functionality

## Command Interface

In addition to voice commands, you can use text commands prefixed with `!` during your conversation:

- `!exit` or `!quit` - Exit the assistant
- `!voice [name]` - Change voice or list available voices
- `!voice_model [name]` - Change TTS model or list available models
- `!model [name]` - Change AI model or list available models
- `!name [new_name]` - Change assistant name
- `!user [name]` - Set or show user name
- `!personality` - Show current personality
- `!personality set <trait> <value>` - Set personality trait (0-1)
- `!screenshot` - Take and analyze a screenshot
- `!webcam` - Capture and analyze from webcam
- `!memory` - Show current conversation info
- `!memory list` - List recent conversations
- `!memory switch <id>` - Switch to a different conversation
- `!memory new [title]` - Create a new conversation
- `!emotion` - Show current emotion status
- `!emotion set <emotion> [intensity]` - Set emotion manually
- `!emotion reset` - Reset to neutral emotion
- `!emotion triggers` - List emotion triggers
- `!help` - Show help information

## Voice and Model Selection

### MeloTTS Voice Selection

MeloTTS comes with several voice options. You can select your preferred voice:

Available voices include:
- english_male_1
- english_female_1
- english_female_2
- chinese_male_1
- korean_female_1
- japanese_female_1
- french_male_1
- spanish_female_1

### MeloTTS Model Selection

You can select different TTS models for varying quality and language support:

```bash
python main.py --voice english_male_1 --tts-model en_US/vctk_medium
```

Available TTS models include:
- en_US/vctk_low
- en_US/vctk_medium
- en_US/ljspeech_low
- en_US/ljspeech_medium
- en_GB/vctk_low
- en_GB/vctk_medium
- fr_FR/css10_low
- fr_FR/css10_medium
- de_DE/css10_low
- de_DE/css10_medium
- es_ES/css10_low
- es_ES/css10_medium
- and more...

### Speech-to-Text Models

Choose from various Faster-Whisper model sizes:

```bash
python main.py --stt-model medium
```

Available STT models:
- tiny
- base
- small
- medium
- large-v2
- large-v3

## Configuration

The assistant can be configured through the `config.json` file. You can modify:

```json
{
  "tts": {
    "voice": "english_male_1",
    "model": "en_US/vctk_low"
  },
  "stt": {
    "model_size": "base",
    "language": "en",
    "vad_aggressiveness": 3,
    "silence_threshold_sec": 1.0,
    "max_recording_sec": 30.0
  },
  "ai": {
    "model": "llama3",
    "api_base": "http://localhost:11434/api",
    "system_prompt": ""
  },
  "emotions": {
    "memory_path": "emotions_memory.json",
    "initial_emotion": "neutral",
    "initial_intensity": 0.6
  },
  "profile": {
    "profile_path": "assistant_profile.json"
  },
  "personality": {
    "personality_path": "assistant_personality.json"
  },
  "context": {
    "context_path": "assistant_context.json",
    "idle_initiative": true,
    "idle_interval_minutes": 20
  },
  "memory": {
    "db_path": "conversations.db"
  },
  "vision": {
    "device": "cuda"
  }
}
```

## Personality and Emotional Framework

The assistant features an advanced personality and emotional framework:

- **Emotional States**: The assistant can express various emotions (happy, excited, neutral, concerned, confused, sad, angry) that are influenced by your interactions
- **Voice Modulation**: Emotions affect speech parameters like rate, pitch, volume, and emphasis
- **Personality Traits**: The assistant develops traits like friendliness, formality, curiosity, and enthusiasm
- **Contextual Awareness**: Remembers topics of interest and can proactively engage

You can ask questions like "How are you feeling?" or use commands like `!emotion` to interact with the emotional framework.

## Future Enhancements

The following features are planned for future releases:

- **VTube Studio Integration**: Connect with VTube Studio for avatar animations
- **Custom RAG**: Integrate Retrieval-Augmented Generation for enhanced knowledge
- **PC Control**: Allow the AI to perform system operations with proper safeguards
- **Web Browsing**: Enable the AI to search and browse the internet
- **Text-Generation-WebUI Integration**: Support for additional text generation interfaces

## Troubleshooting

For common issues and solutions, see the [INSTALL.md](INSTALL.md) file.

### Common Issues

#### MeloTTS Issues

- **Error loading voice model**: Make sure you have sufficient disk space and a good internet connection for downloading models
- **CUDA out of memory**: Try using a smaller model (e.g., switch from medium to low quality)

#### Faster-Whisper Issues

- **Speech recognition not working**: Check your microphone settings and ensure it's properly connected
- **Poor recognition quality**: Try a larger model size (`--stt-model medium` or `--stt-model large-v2`)

#### Ollama Issues

- **Connection refused**: Make sure Ollama is running (`ollama serve`)
- **Model not found**: Ensure you've pulled the model (`ollama pull llama3`)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the incredible open-source projects that made this possible
- Special thanks to the creators of MeloTTS, Faster-Whisper, Ollama, and PyTorch Vision
