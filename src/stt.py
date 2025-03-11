"""
Speech-to-Text Module using Faster-Whisper

This module provides speech recognition capabilities using the Faster-Whisper 
library, which offers high-quality, efficient transcription with low latency.
"""

import os
import sys
import time
import logging
import threading
import queue
import tempfile
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
import numpy as np
import webrtcvad
import sounddevice as sd
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Set up logging
logger = logging.getLogger(__name__)

class STTEngine:
    """Speech-to-Text engine using Faster-Whisper"""
    
    # Available model sizes for Faster-Whisper
    AVAILABLE_MODELS = [
        "tiny", "base", "small", "medium", "large-v2", "large-v3"
    ]
    
    # Default VAD (Voice Activity Detection) parameters
    DEFAULT_VAD_FRAME_DURATION_MS = 30  # Frame duration in milliseconds
    DEFAULT_VAD_SAMPLE_RATE = 16000     # Sample rate in Hz
    DEFAULT_VAD_AGGRESSIVENESS = 3      # Aggressiveness level (0-3)
    
    def __init__(
        self, 
        model_size: str = "base",
        language: str = "en",
        device: Optional[str] = None,
        compute_type: str = "float16",
        vad_aggressiveness: int = 3
    ):
        """
        Initialize the STT engine with the specified configuration.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large-v2, large-v3)
            language: Language code for transcription (e.g., "en", "fr", "de")
            device: Computation device ("cpu", "cuda", "auto")
            compute_type: Compute type for the model (float16, float32, int8)
            vad_aggressiveness: Voice Activity Detection aggressiveness (0-3)
        """
        if model_size not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown model size '{model_size}', using 'base' instead")
            model_size = "base"
        
        self.model_size = model_size
        self.language = language
        self.compute_type = compute_type
        self.vad_aggressiveness = max(0, min(3, vad_aggressiveness))  # Ensure 0-3 range
        
        # Set device
        if device is None:
            # Auto-detect device
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        
        # Recording state
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.processing_thread = None
        
        # VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"STT Engine initialized with {model_size} model on {self.device}")
    
    def _initialize_model(self):
        """Initialize the Whisper model"""
        try:
            # Load the Whisper model
            logger.info(f"Loading Faster-Whisper {self.model_size} model on {self.device}...")
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def start_recording(
        self,
        callback: Callable[[str], None],
        silence_threshold_sec: float = 1.0,
        max_recording_sec: float = 30.0,
        sample_rate: int = DEFAULT_VAD_SAMPLE_RATE
    ):
        """
        Start recording audio for transcription.
        
        Args:
            callback: Function to call with transcription result
            silence_threshold_sec: Silence duration (seconds) to consider speech segment complete
            max_recording_sec: Maximum recording duration (seconds)
            sample_rate: Audio sample rate (Hz)
        """
        if self.recording:
            logger.warning("Recording is already in progress")
            return
        
        self.recording = True
        self.callback = callback
        self.silence_threshold_sec = silence_threshold_sec
        self.max_recording_sec = max_recording_sec
        self.sample_rate = sample_rate
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._record_audio_thread,
            daemon=True
        )
        self.recording_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_audio_thread,
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Started audio recording and processing")
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Wait for threads to terminate
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Stopped audio recording")
    
    def _record_audio_thread(self):
        """Thread for recording audio from microphone"""
        try:
            # Set up audio capture
            channels = 1  # Mono audio
            chunk_duration_ms = self.DEFAULT_VAD_FRAME_DURATION_MS
            chunk_samples = int(self.sample_rate * chunk_duration_ms / 1000)
            
            # Create a buffer for the current speech segment
            speech_buffer = []
            silence_counter = 0
            is_speaking = False
            total_duration = 0
            
            # Set up stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=channels,
                dtype='int16',
                blocksize=chunk_samples,
                callback=None
            ) as stream:
                logger.info("Audio stream started")
                
                while self.recording:
                    # Read audio chunk
                    chunk, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        logger.warning("Audio buffer overflow detected")
                    
                    # Convert to the right format for VAD
                    chunk_mono = chunk.reshape(-1).astype(np.int16)
                    
                    # Check if this chunk contains speech
                    try:
                        is_speech = self.vad.is_speech(
                            chunk_mono.tobytes(),
                            self.sample_rate
                        )
                    except Exception as e:
                        logger.error(f"VAD error: {e}")
                        is_speech = False
                    
                    # Update state based on VAD result
                    if is_speech and not is_speaking:
                        # Start of speech detected
                        is_speaking = True
                        silence_counter = 0
                        speech_buffer = [chunk_mono]
                        logger.debug("Speech started")
                    
                    elif is_speech and is_speaking:
                        # Continuing speech
                        speech_buffer.append(chunk_mono)
                        silence_counter = 0
                    
                    elif not is_speech and is_speaking:
                        # Possible silence during speech
                        speech_buffer.append(chunk_mono)
                        silence_counter += chunk_duration_ms / 1000
                        
                        # If silence exceeds threshold, consider speech segment complete
                        if silence_counter >= self.silence_threshold_sec:
                            is_speaking = False
                            
                            # Combine all chunks into one audio segment
                            combined_audio = np.concatenate(speech_buffer)
                            
                            # Put in queue for processing
                            self.audio_queue.put(combined_audio)
                            
                            logger.debug(f"Speech segment complete: {len(combined_audio) / self.sample_rate:.2f}s")
                            speech_buffer = []
                    
                    # Check if max duration exceeded while speaking
                    if is_speaking:
                        total_duration += chunk_duration_ms / 1000
                        if total_duration >= self.max_recording_sec:
                            # Max duration reached, force end of segment
                            is_speaking = False
                            
                            # Combine all chunks into one audio segment
                            combined_audio = np.concatenate(speech_buffer)
                            
                            # Put in queue for processing
                            self.audio_queue.put(combined_audio)
                            
                            logger.debug(f"Max duration reached: {total_duration:.2f}s")
                            speech_buffer = []
                            total_duration = 0
                    
                    # Small delay to prevent CPU hogging
                    time.sleep(0.001)
                
                # Final check for any remaining audio
                if speech_buffer:
                    combined_audio = np.concatenate(speech_buffer)
                    self.audio_queue.put(combined_audio)
                    logger.debug(f"Final speech segment: {len(combined_audio) / self.sample_rate:.2f}s")
        
        except Exception as e:
            logger.error(f"Error in recording thread: {e}")
        
        logger.info("Recording thread terminated")
    
    def _process_audio_thread(self):
        """Thread for processing audio segments and transcribing"""
        try:
            while self.recording or not self.audio_queue.empty():
                try:
                    # Get audio segment from queue (with timeout to check recording flag)
                    audio_segment = self.audio_queue.get(timeout=0.5)
                    
                    # Skip very short segments (likely noise)
                    duration_sec = len(audio_segment) / self.sample_rate
                    if duration_sec < 0.3:  # Less than 300ms
                        logger.debug(f"Skipping short segment: {duration_sec:.2f}s")
                        continue
                    
                    # Save audio to temporary file for processing
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        # Create WAV file from numpy array
                        audio_segment_pydub = AudioSegment(
                            data=audio_segment.tobytes(),
                            sample_width=2,  # 16-bit
                            frame_rate=self.sample_rate,
                            channels=1
                        )
                        audio_segment_pydub.export(temp_file.name, format='wav')
                        
                        # Use Whisper to transcribe
                        logger.debug(f"Transcribing audio segment: {duration_sec:.2f}s")
                        result = self._transcribe(temp_file.name)
                        
                        # Clean up temporary file
                        temp_file_name = temp_file.name
                    
                    os.unlink(temp_file_name)
                    
                    # If result is not empty, call callback with transcription
                    if result and result.strip():
                        logger.info(f"Transcription: {result}")
                        if self.callback:
                            self.callback(result)
                
                except queue.Empty:
                    # Queue empty, continue to check recording flag
                    continue
                except Exception as e:
                    logger.error(f"Error processing audio segment: {e}")
        
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
        
        logger.info("Processing thread terminated")
    
    def _transcribe(self, audio_file: str) -> str:
        """
        Transcribe an audio file using Faster-Whisper.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Use Whisper to transcribe
            segments, info = self.model.transcribe(
                audio=audio_file,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum silence duration (ms)
                    speech_pad_ms=400             # Padding around speech (ms)
                )
            )
            
            # Combine all segments into one string
            transcription = " ".join(segment.text for segment in segments)
            
            # Clean up transcription
            transcription = transcription.strip()
            
            return transcription
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def transcribe_file(self, audio_file: str) -> str:
        """
        Transcribe an audio file directly.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        return self._transcribe(audio_file)
    
    def change_model(self, model_size: str) -> bool:
        """
        Change the Whisper model size.
        
        Args:
            model_size: New model size
            
        Returns:
            True if successful, False otherwise
        """
        if model_size not in self.AVAILABLE_MODELS:
            logger.error(f"Unknown model size: {model_size}")
            return False
        
        if model_size == self.model_size:
            logger.info(f"Already using {model_size} model")
            return True
        
        # Update model size
        self.model_size = model_size
        
        # Re-initialize model
        try:
            self._initialize_model()
            return True
        except Exception as e:
            logger.error(f"Failed to change model: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """
        Get list of available Whisper models.
        
        Returns:
            List of available model sizes
        """
        return self.AVAILABLE_MODELS.copy()
