"""
Text-to-Speech Module using MeloTTS

This module handles converting text to speech using the MeloTTS library.
It provides functionality to select different voice models and synthesize speech.
"""

import os
import sys
import logging
import tempfile
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import torch
from pydub import AudioSegment
from pydub.playback import play

# Set up logging
logger = logging.getLogger(__name__)

class TTSEngine:
    """Text-to-Speech engine using MeloTTS"""
    
    # Dictionary mapping voice names to model names
    AVAILABLE_VOICES = {
        "english_male_1": "en_US_ryan",
        "english_female_1": "en_US_amy", 
        "english_female_2": "en_US_emma",
        "chinese_male_1": "zh_CN_jason",
        "korean_female_1": "ko_KR_hyejin",
        "japanese_female_1": "ja_JP_hina",
        "french_male_1": "fr_FR_pierre",
        "spanish_female_1": "es_ES_lucia"
    }
    
    def __init__(self, voice: str = "english_male_1"):
        """
        Initialize the TTS engine with the specified voice.
        
        Args:
            voice: Name of the voice to use (from AVAILABLE_VOICES)
        """
        self.voice = voice
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self._load_model()
            logger.info(f"Initialized TTS engine with voice '{voice}' on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise
    
    def _load_model(self):
        """Load the MeloTTS model for the selected voice"""
        try:
            from melotts import MeloTTS
            
            if self.voice not in self.AVAILABLE_VOICES:
                logger.warning(f"Voice '{self.voice}' not recognized. Defaulting to english_male_1.")
                self.voice = "english_male_1"
                
            model_name = self.AVAILABLE_VOICES[self.voice]
            self.model = MeloTTS(model_name=model_name, device=self.device)
            logger.info(f"Loaded MeloTTS model '{model_name}'")
        except ImportError:
            logger.error("Failed to import MeloTTS. Make sure it's installed correctly.")
            raise
        except Exception as e:
            logger.error(f"Failed to load MeloTTS model: {e}")
            raise
    
    def change_voice(self, voice: str) -> bool:
        """
        Change the current voice.
        
        Args:
            voice: New voice to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        if voice not in self.AVAILABLE_VOICES:
            logger.warning(f"Voice '{voice}' not recognized.")
            return False
        
        if voice == self.voice:
            return True  # No change needed
            
        self.voice = voice
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.error(f"Failed to change voice: {e}")
            return False
    
    def list_available_voices(self) -> List[str]:
        """
        Get a list of available voice names.
        
        Returns:
            List of available voice names
        """
        return list(self.AVAILABLE_VOICES.keys())
    
    def synthesize(self, text: str, output_file: Optional[str] = None, play_audio: bool = True) -> Optional[str]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_file: Path to save the audio file (optional)
            play_audio: Whether to play the audio immediately
            
        Returns:
            Path to the output file if output_file is specified, otherwise None
        """
        if not text.strip():
            logger.warning("Empty text provided for synthesis.")
            return None
            
        try:
            # Generate audio using MeloTTS
            audio_array = self.model.tts(text)
            
            # Create a temporary file if no output path is provided
            if output_file is None and play_audio:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    output_file = temp_file.name
            
            if output_file:
                # Convert audio array to appropriate format
                # MeloTTS returns ndarray of float32 values in the range [-1, 1]
                # Convert to int16 for WAV file
                audio_int16 = (audio_array * 32767).astype(np.int16)
                
                # Save the audio to a WAV file
                from scipy.io import wavfile
                wavfile.write(output_file, self.model.sample_rate, audio_int16)
                logger.info(f"Audio saved to {output_file}")
            
            if play_audio:
                # Play the audio directly from memory
                sd.play(audio_array, samplerate=self.model.sample_rate)
                sd.wait()  # Wait until audio playback is finished
            
            return output_file
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return None
    
    def speak(self, text: str) -> bool:
        """
        Synthesize speech from text and play it immediately.
        
        Args:
            text: Text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.synthesize(text, play_audio=True)
            return result is not None
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Get voice from command line if provided
    voice = sys.argv[1] if len(sys.argv) > 1 else "english_male_1"
    
    # Initialize the TTS engine
    tts = TTSEngine(voice=voice)
    
    # List available voices
    print("Available voices:")
    for voice_name in tts.list_available_voices():
        print(f"- {voice_name}")
    
    # Test speech synthesis
    text = "Hello! I am your AI voice assistant. I can help you with various tasks and answer your questions."
    tts.speak(text)
