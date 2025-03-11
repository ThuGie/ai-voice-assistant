"""
Text-to-Speech Module using MeloTTS

This module handles converting text to speech using the MeloTTS library.
It provides functionality to select different voice models and synthesize speech.
"""

import os
import sys
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Any

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
    
    # Default voice parameters
    DEFAULT_VOICE_PARAMS = {
        "rate": 1.0,      # Speech rate multiplier
        "pitch": 1.0,     # Pitch multiplier
        "volume": 1.0,    # Volume multiplier
        "emphasis": 1.0   # Emphasis/stress multiplier
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
        
        # Current voice parameters
        self.voice_params = self.DEFAULT_VOICE_PARAMS.copy()
        
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
    
    def set_voice_parameters(self, params: Dict[str, float]) -> None:
        """
        Set voice parameters for emotional expression.
        
        Args:
            params: Dictionary of voice parameters (rate, pitch, volume, emphasis)
        """
        for param, value in params.items():
            if param in self.voice_params:
                # Ensure values are within reasonable limits
                if param == "rate":
                    # Rate between 0.5 (half speed) and 2.0 (double speed)
                    self.voice_params[param] = max(0.5, min(2.0, value))
                elif param == "pitch":
                    # Pitch between 0.7 (lower) and 1.3 (higher)
                    self.voice_params[param] = max(0.7, min(1.3, value))
                elif param == "volume":
                    # Volume between 0.5 (quieter) and 1.5 (louder)
                    self.voice_params[param] = max(0.5, min(1.5, value))
                elif param == "emphasis":
                    # Emphasis between 0.7 (less) and 1.5 (more)
                    self.voice_params[param] = max(0.7, min(1.5, value))
                else:
                    self.voice_params[param] = value
        
        logger.debug(f"Voice parameters set to: {self.voice_params}")
    
    def reset_voice_parameters(self) -> None:
        """Reset voice parameters to default values"""
        self.voice_params = self.DEFAULT_VOICE_PARAMS.copy()
        logger.debug("Voice parameters reset to defaults")
    
    def _apply_voice_parameters(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply voice parameters to the audio array.
        
        Args:
            audio_array: Original audio as numpy array
            
        Returns:
            Modified audio array
        """
        try:
            import librosa
            import pyrubberband as pyrb
            from scipy.signal import lfilter
            
            # Make a copy of the original audio
            modified_audio = audio_array.copy()
            
            # Apply volume adjustment
            if self.voice_params["volume"] != 1.0:
                modified_audio *= self.voice_params["volume"]
            
            # Get sample rate from model
            sample_rate = self.model.sample_rate
            
            # Apply pitch and speed changes using rubberband
            if self.voice_params["pitch"] != 1.0 or self.voice_params["rate"] != 1.0:
                # Convert to the format expected by pyrubberband
                if modified_audio.dtype != np.float32:
                    modified_audio = modified_audio.astype(np.float32)
                
                # Apply the pitch and rate changes
                modified_audio = pyrb.pitch_shift(
                    modified_audio, 
                    sample_rate, 
                    self.voice_params["pitch"] - 1.0
                )
                
                modified_audio = pyrb.time_stretch(
                    modified_audio, 
                    sample_rate, 
                    1.0 / self.voice_params["rate"]
                )
            
            # Apply emphasis (simplified approach using a basic filter)
            if self.voice_params["emphasis"] != 1.0:
                # Create a simple emphasis filter
                # Higher emphasis boosts the mid-frequencies
                if self.voice_params["emphasis"] > 1.0:
                    # Boosting mid-range frequencies
                    b, a = [1.0, -0.97], [1.0, -0.9]  # Simple 1-pole filter
                    modified_audio = lfilter(b, a, modified_audio)
                elif self.voice_params["emphasis"] < 1.0:
                    # Reducing mid-range frequencies 
                    b, a = [1.0, -0.9], [1.0, -0.97]  # Inverse of the above
                    modified_audio = lfilter(b, a, modified_audio)
            
            # Normalize audio if it's clipping
            max_val = np.max(np.abs(modified_audio))
            if max_val > 1.0:
                modified_audio = modified_audio / max_val * 0.95  # Leave a little headroom
            
            return modified_audio
            
        except ImportError as e:
            logger.warning(f"Could not apply all voice parameters due to missing dependencies: {e}")
            logger.warning("Install librosa and pyrubberband for full voice parameter support")
            # Return original audio if we can't modify it
            return audio_array
        except Exception as e:
            logger.error(f"Error applying voice parameters: {e}")
            # Return original audio if there was an error
            return audio_array
    
    def synthesize(self, 
                 text: str, 
                 output_file: Optional[str] = None, 
                 play_audio: bool = True,
                 voice_params: Optional[Dict[str, float]] = None) -> Optional[str]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_file: Path to save the audio file (optional)
            play_audio: Whether to play the audio immediately
            voice_params: Voice parameters to use for this synthesis only
            
        Returns:
            Path to the output file if output_file is specified, otherwise None
        """
        if not text.strip():
            logger.warning("Empty text provided for synthesis.")
            return None
        
        # Temporarily set voice parameters if provided
        original_params = None
        if voice_params:
            original_params = self.voice_params.copy()
            self.set_voice_parameters(voice_params)
            
        try:
            # Generate audio using MeloTTS
            audio_array = self.model.tts(text)
            
            # Apply voice parameters to modify the audio
            modified_audio = self._apply_voice_parameters(audio_array)
            
            # Create a temporary file if no output path is provided
            if output_file is None and play_audio:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    output_file = temp_file.name
            
            if output_file:
                # Convert audio array to appropriate format
                # MeloTTS returns ndarray of float32 values in the range [-1, 1]
                # Convert to int16 for WAV file
                audio_int16 = (modified_audio * 32767).astype(np.int16)
                
                # Save the audio to a WAV file
                from scipy.io import wavfile
                wavfile.write(output_file, self.model.sample_rate, audio_int16)
                logger.info(f"Audio saved to {output_file}")
            
            if play_audio:
                # Play the audio directly from memory
                sd.play(modified_audio, samplerate=self.model.sample_rate)
                sd.wait()  # Wait until audio playback is finished
            
            return output_file
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return None
        finally:
            # Restore original voice parameters if they were changed
            if original_params:
                self.voice_params = original_params
    
    def speak(self, 
             text: str, 
             voice_params: Optional[Dict[str, float]] = None) -> bool:
        """
        Synthesize speech from text and play it immediately.
        
        Args:
            text: Text to speak
            voice_params: Voice parameters to use for this synthesis only
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.synthesize(text, play_audio=True, voice_params=voice_params)
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
    
    # Test speech synthesis with different voice parameters
    text = "Hello! I am your AI voice assistant. I can help you with various tasks and answer your questions."
    
    print("\nNormal voice:")
    tts.reset_voice_parameters()
    tts.speak(text)
    
    print("\nHappy voice:")
    tts.speak(text, voice_params={"rate": 1.1, "pitch": 1.1, "volume": 1.1, "emphasis": 1.1})
    
    print("\nSad voice:")
    tts.speak(text, voice_params={"rate": 0.85, "pitch": 0.9, "volume": 0.9, "emphasis": 0.8})
    
    print("\nAngry voice:")
    tts.speak(text, voice_params={"rate": 1.1, "pitch": 0.95, "volume": 1.2, "emphasis": 1.4})
