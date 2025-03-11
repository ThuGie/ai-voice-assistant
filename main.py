#!/usr/bin/env python3
"""
AI Voice Assistant

This is the main entry point for the AI Voice Assistant application.
It integrates MeloTTS, Faster-Whisper, Ollama, and PyTorch Vision
to create a comprehensive interactive AI experience.
"""

import os
import sys
import time
import logging
import argparse
from typing import Optional, Dict, List, Any, Union
import threading
import queue
import json
import signal
import tempfile
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('assistant.log')
    ]
)
logger = logging.getLogger(__name__)

# Import assistant modules
try:
    from src.tts import TTSEngine
    from src.stt import STTEngine
    from src.ai import AIEngine
    from src.vision import VisionEngine
    from src.memory import Memory
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you have installed all dependencies with 'pip install -r requirements.txt'")
    sys.exit(1)

class AIAssistant:
    """Main AI Assistant class that integrates all components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AI Assistant with the given configuration.
        
        Args:
            config: Configuration dictionary with settings for each component
        """
        self.config = config or {}
        self.running = False
        self.processing_input = False
        
        # Command queue for handling special commands
        self.command_queue = queue.Queue()
        
        # Initialize components
        self._initialize_components()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("AI Assistant initialized")
    
    def _initialize_components(self):
        """Initialize all assistant components"""
        try:
            # Initialize Text-to-Speech engine
            logger.info("Initializing Text-to-Speech engine...")
            tts_config = self.config.get("tts", {})
            self.tts = TTSEngine(
                voice=tts_config.get("voice", "english_male_1")
            )
            
            # Initialize Speech-to-Text engine
            logger.info("Initializing Speech-to-Text engine...")
            stt_config = self.config.get("stt", {})
            self.stt = STTEngine(
                model_size=stt_config.get("model_size", "base"),
                language=stt_config.get("language", "en"),
                vad_aggressiveness=stt_config.get("vad_aggressiveness", 3)
            )
            
            # Initialize AI engine
            logger.info("Initializing AI engine...")
            ai_config = self.config.get("ai", {})
            self.ai = AIEngine(
                model=ai_config.get("model", "llama3"),
                api_base=ai_config.get("api_base", "http://localhost:11434/api"),
                system_prompt=ai_config.get("system_prompt", None)
            )
            
            # Initialize Vision engine (only if needed for performance)
            self.vision = None
            
            # Initialize Memory
            logger.info("Initializing Memory...")
            memory_config = self.config.get("memory", {})
            self.memory = Memory(
                db_path=memory_config.get("db_path", "conversations.db")
            )
            
            # Create or load conversation
            self.conversation_id = self._setup_conversation()
            
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_vision(self):
        """Initialize Vision engine on demand"""
        if self.vision is None:
            logger.info("Initializing Vision engine...")
            vision_config = self.config.get("vision", {})
            self.vision = VisionEngine(
                device=vision_config.get("device", None)
            )
    
    def _setup_conversation(self) -> int:
        """
        Set up a new or existing conversation.
        
        Returns:
            ID of the conversation
        """
        memory_config = self.config.get("memory", {})
        conversation_id = memory_config.get("conversation_id")
        
        if conversation_id:
            # Check if conversation exists
            conversation = self.memory.get_conversation(conversation_id)
            if conversation:
                logger.info(f"Loaded existing conversation: {conversation.get('title')}")
                return conversation_id
        
        # Create a new conversation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_id = self.memory.create_conversation(f"Conversation {timestamp}")
        
        # Add system message
        system_prompt = self.ai.system_prompt
        self.memory.add_message(new_id, "system", system_prompt)
        
        logger.info(f"Created new conversation with ID {new_id}")
        return new_id
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self):
        """Start the assistant"""
        if self.running:
            logger.warning("Assistant is already running")
            return
            
        self.running = True
        
        try:
            # Welcome message
            logger.info("Starting AI Assistant...")
            welcome_message = "Hello! I'm your AI assistant. You can speak to me or type commands. Say 'help' for assistance."
            print(welcome_message)
            self.tts.speak(welcome_message)
            
            # Start command processing thread
            command_thread = threading.Thread(target=self._command_processor, daemon=True)
            command_thread.start()
            
            # Start speech recognition
            self._start_listening()
            
            # Main loop - handle text input while also processing speech
            while self.running:
                try:
                    # Get text input (non-blocking)
                    user_input = input("> ")
                    if user_input.strip():
                        self._process_input(user_input)
                except EOFError:
                    # Handle EOF (Ctrl+D)
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    break
        finally:
            self.stop()
    
    def stop(self):
        """Stop the assistant and clean up resources"""
        if not self.running:
            return
            
        self.running = False
        
        # Stop listening
        self.stt.stop_recording()
        
        # Clean up vision resources if initialized
        if self.vision:
            self.vision.release_webcam()
        
        # Final message
        print("\nShutting down assistant. Goodbye!")
        
        logger.info("Assistant stopped")
    
    def _start_listening(self):
        """Start listening for voice input"""
        logger.info("Starting speech recognition...")
        
        def speech_callback(text):
            """Callback function for speech recognition"""
            if text and not self.processing_input:
                print(f"\nHeard: {text}")
                self._process_input(text)
        
        # Start recording with callback
        self.stt.start_recording(
            callback=speech_callback,
            silence_threshold_sec=1.0,
            max_recording_sec=30.0
        )
    
    def _process_input(self, user_input: str):
        """
        Process user input (text or transcribed speech).
        
        Args:
            user_input: User input text
        """
        # Set flag to prevent processing multiple inputs simultaneously
        self.processing_input = True
        
        try:
            # Check for special commands
            if user_input.startswith("!"):
                self._handle_command(user_input[1:])
                return
                
            # Add user message to memory
            self.memory.add_message(self.conversation_id, "user", user_input)
            
            # Process with AI and get response
            print("Assistant is thinking...")
            
            def streaming_callback(chunk):
                """Callback for streaming response"""
                print(chunk, end="", flush=True)
            
            # Generate streaming response
            response = self.ai.generate_response(
                user_input,
                stream=True,
                stream_callback=streaming_callback
            )
            
            print()  # Add newline after streaming response
            
            # Add assistant response to memory
            self.memory.add_message(self.conversation_id, "assistant", response)
            
            # Speak the response
            self.tts.speak(response)
        finally:
            self.processing_input = False
    
    def _handle_command(self, command: str):
        """
        Handle special commands.
        
        Args:
            command: Command string (without the ! prefix)
        """
        # Split command and arguments
        parts = command.strip().split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        # Put command in queue for processing
        self.command_queue.put((cmd, args))
    
    def _command_processor(self):
        """Process commands from the command queue"""
        while self.running:
            try:
                # Get command from queue (with timeout to allow clean shutdown)
                cmd, args = self.command_queue.get(timeout=1.0)
                
                # Process based on command
                if cmd == "exit" or cmd == "quit":
                    logger.info("Exit command received")
                    self.running = False
                
                elif cmd == "voice":
                    # Change voice
                    if args:
                        voice_name = args[0]
                        logger.info(f"Changing voice to {voice_name}")
                        if self.tts.change_voice(voice_name):
                            self.tts.speak(f"Voice changed to {voice_name}")
                        else:
                            print(f"Voice '{voice_name}' not available")
                    else:
                        # List available voices
                        voices = self.tts.list_available_voices()
                        print("Available voices:")
                        for voice in voices:
                            print(f"- {voice}")
                
                elif cmd == "model":
                    # Change AI model
                    if args:
                        model_name = args[0]
                        logger.info(f"Changing AI model to {model_name}")
                        if self.ai.change_model(model_name):
                            self.tts.speak(f"Model changed to {model_name}")
                        else:
                            print(f"Model '{model_name}' not available")
                    else:
                        # List available models
                        models = self.ai.list_available_models()
                        print("Available models:")
                        for model in models:
                            print(f"- {model['name']}")
                
                elif cmd == "screenshot":
                    # Take a screenshot and analyze it
                    logger.info("Taking screenshot")
                    self._initialize_vision()
                    
                    # Capture screen
                    screen = self.vision.capture_screen()
                    
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.png"
                    self.vision.save_image(screen, filename)
                    
                    # Analyze and describe
                    description = self.vision.describe_image(screen)
                    print(f"Screen description: {description}")
                    self.tts.speak(f"I took a screenshot. {description}")
                
                elif cmd == "webcam":
                    # Capture from webcam and analyze
                    logger.info("Capturing from webcam")
                    self._initialize_vision()
                    
                    # Capture frame
                    frame = self.vision.capture_webcam()
                    if frame is None:
                        print("Failed to capture from webcam")
                        self.tts.speak("I couldn't access your webcam")
                        continue
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"webcam_{timestamp}.png"
                    self.vision.save_image(frame, filename)
                    
                    # Analyze and describe
                    description = self.vision.describe_image(frame)
                    print(f"Webcam description: {description}")
                    self.tts.speak(f"I captured from your webcam. {description}")
                
                elif cmd == "memory":
                    # Memory management
                    if not args:
                        # Show current conversation info
                        conversation = self.memory.get_conversation(self.conversation_id)
                        messages = conversation.get("messages", [])
                        print(f"Current conversation: {conversation.get('title')}")
                        print(f"Messages: {len(messages)}")
                    elif args[0] == "list":
                        # List recent conversations
                        conversations = self.memory.list_conversations()
                        print("Recent conversations:")
                        for conv in conversations:
                            print(f"- {conv['id']}: {conv['title']} ({conv['message_count']} messages)")
                    elif args[0] == "switch" and len(args) > 1:
                        # Switch to a different conversation
                        try:
                            new_id = int(args[1])
                            conversation = self.memory.get_conversation(new_id)
                            if conversation:
                                self.conversation_id = new_id
                                print(f"Switched to conversation: {conversation.get('title')}")
                                self.tts.speak(f"Switched to conversation: {conversation.get('title')}")
                            else:
                                print(f"Conversation {new_id} not found")
                        except ValueError:
                            print("Invalid conversation ID")
                    elif args[0] == "new":
                        # Create a new conversation
                        title = " ".join(args[1:]) if len(args) > 1 else None
                        self.conversation_id = self._setup_conversation()
                        print(f"Created new conversation with ID {self.conversation_id}")
                        self.tts.speak("Created a new conversation")
                
                elif cmd == "help":
                    # Show help information
                    print("Available commands:")
                    print("  !exit, !quit - Exit the assistant")
                    print("  !voice [name] - Change voice or list available voices")
                    print("  !model [name] - Change AI model or list available models")
                    print("  !screenshot - Take and analyze a screenshot")
                    print("  !webcam - Capture and analyze from webcam")
                    print("  !memory - Conversation memory management")
                    print("  !memory list - List recent conversations")
                    print("  !memory switch <id> - Switch to a different conversation")
                    print("  !memory new [title] - Create a new conversation")
                    print("  !help - Show this help information")
                
                else:
                    print(f"Unknown command: {cmd}")
                
                # Mark command as processed
                self.command_queue.task_done()
            
            except queue.Empty:
                # No commands in queue, continue
                pass
            except Exception as e:
                logger.error(f"Error processing command: {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI Voice Assistant")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (JSON)"
    )
    
    parser.add_argument(
        "--voice", 
        type=str, 
        default="english_male_1",
        help="Voice to use for text-to-speech"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3",
        help="AI model to use"
    )
    
    parser.add_argument(
        "--stt-model", 
        type=str, 
        default="base",
        help="Speech-to-text model size (tiny, base, small, medium, large-v2, large-v3)"
    )
    
    return parser.parse_args()

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or create default configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    # Default configuration
    return {
        "tts": {
            "voice": "english_male_1"
        },
        "stt": {
            "model_size": "base",
            "language": "en",
            "vad_aggressiveness": 3
        },
        "ai": {
            "model": "llama3",
            "api_base": "http://localhost:11434/api"
        },
        "memory": {
            "db_path": "conversations.db"
        }
    }

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.voice:
        config["tts"]["voice"] = args.voice
    if args.model:
        config["ai"]["model"] = args.model
    if args.stt_model:
        config["stt"]["model_size"] = args.stt_model
    
    try:
        # Create and start assistant
        assistant = AIAssistant(config)
        assistant.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error running assistant: {e}", exc_info=True)
    
    logger.info("Exiting")

if __name__ == "__main__":
    main()
