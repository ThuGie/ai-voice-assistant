# AI Voice Assistant

An advanced voice-based AI assistant that integrates multiple open-source technologies to create a comprehensive interactive experience with your AI.

## Features

- **Voice Interaction**: Speak naturally to your AI using Faster-Whisper for accurate speech-to-text conversion
- **Voice Response**: AI responses are converted to natural-sounding speech with MeloTTS
- **Multiple Voice Options**: Select from various voice models in MeloTTS to customize your experience
- **Local AI Model**: Uses Ollama to run powerful AI models locally on your machine
- **Computer Vision**: Integrated PyTorch Vision capabilities allow the AI to "see" your screen or camera
- **Extensible Architecture**: Designed for future extensions like VTube Studio support and custom RAG
- **Conversation Memory**: Stores and references past conversations for more contextual interactions

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

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-voice-assistant.git
   cd ai-voice-assistant
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
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

2. Select your preferred voice model when prompted

3. Start speaking to interact with the assistant

4. Use special commands for additional functionality:
   - "Take a screenshot" - Captures and analyzes your screen
   - "Share my window" - Allows the AI to see a specific window
   - "Remember this conversation" - Explicitly saves the current conversation for future reference

## Voice Model Selection

MeloTTS comes with several voice models. You can select your preferred voice by:

1. Using the command line argument:
   ```bash
   python main.py --voice english_male_1
   ```

2. Selecting from the menu when the application starts

3. Changing the voice during a conversation with:
   ```
   "Change voice to [voice_name]"
   ```

Available voices include:
- english_male_1
- english_female_1
- english_female_2
- chinese_male_1
- korean_female_1
- japanese_female_1
- french_male_1
- spanish_female_1

## Future Enhancements

The following features are planned for future releases:

- **VTube Studio Integration**: Connect with VTube Studio for avatar animations
- **Custom RAG**: Integrate Retrieval-Augmented Generation for enhanced knowledge
- **PC Control**: Allow the AI to perform system operations with proper safeguards
- **Web Browsing**: Enable the AI to search and browse the internet
- **Text-Generation-WebUI Integration**: Support for additional text generation interfaces

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
