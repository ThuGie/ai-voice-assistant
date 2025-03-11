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
from typing import Optional, Dict, Any
import json
import signal

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

# Import the AIAssistant class from src.assistant
try:
    from src.assistant import AIAssistant
except ImportError as e:
    logger.error(f"Failed to import AIAssistant module: {e}")
    logger.error("Make sure you have installed all dependencies with 'pip install -r requirements.txt'")
    sys.exit(1)

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
        help="Voice to use for text-to-speech"
    )
    
    parser.add_argument(
        "--tts-model", 
        type=str, 
        help="Text-to-speech model to use (e.g., en_US/vctk_low)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="AI model to use"
    )
    
    parser.add_argument(
        "--stt-model", 
        type=str, 
        help="Speech-to-text model size (tiny, base, small, medium, large-v2, large-v3)"
    )
    
    parser.add_argument(
        "--emotion",
        type=str,
        help="Initial emotion for the assistant (happy, excited, neutral, concerned, confused, sad, angry)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Assistant's name"
    )
    
    parser.add_argument(
        "--user-name",
        type=str,
        help="User's name"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
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
    
    # Default configuration file in the current directory
    default_config_path = "config.json"
    if os.path.exists(default_config_path):
        try:
            with open(default_config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {default_config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load default configuration: {e}")
    
    # Minimal default configuration if no file is found
    logger.warning("No configuration file found. Using minimal default configuration.")
    return {
        "tts": {
            "voice": "english_male_1",
            "model": "en_US/vctk_low"
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
            "idle_initiative": True,
            "idle_interval_minutes": 20
        },
        "memory": {
            "db_path": "conversations.db"
        }
    }

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.voice:
        config["tts"]["voice"] = args.voice
    if args.tts_model:
        config["tts"]["model"] = args.tts_model
    if args.model:
        config["ai"]["model"] = args.model
    if args.stt_model:
        config["stt"]["model_size"] = args.stt_model
    if args.emotion:
        config["emotions"]["initial_emotion"] = args.emotion
    if args.name:
        if "profile" not in config:
            config["profile"] = {}
        config["profile"]["name"] = args.name
    if args.user_name:
        if "profile" not in config:
            config["profile"] = {}
        config["profile"]["user_name"] = args.user_name
    
    # Create and start the assistant
    try:
        assistant = AIAssistant(config)
        assistant.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error running assistant: {e}", exc_info=True)
    
    logger.info("Exiting")

if __name__ == "__main__":
    main()
