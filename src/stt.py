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
import weakref
import atexit
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
import numpy as np
import webrtcvad
import sounddevice as sd
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Set up logging
logger = logging.getLogger(__name__)

class STTEngineError(Exception):
    """Base class for STT engine errors."""
    pass

class ModelLoadError(STTEngineError):
    """Error raised when model loading fails."""
    pass

class AudioCaptureError(STTEngineError):
    """Error raised when audio capture fails."""
    pass

class TranscriptionError(STTEngineError):
    """Error raised during transcription."""
    pass

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
    
    # Keep track of all instances for proper cleanup
    _instances = set()
    
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
        self.stream = None
        
        # VAD (Voice Activity Detection)
        self.vad = None
        
        # Model
        self.model = None
        
        # Initialization locks
        self._vad_lock = threading.Lock()
        self._model_lock = threading.Lock()
        
        # Register instance for cleanup
        self._instances.add(weakref.ref(self))
        atexit.register(self._cleanup_at_exit)
        
        # Initialize VAD
        self._initialize_vad()
        
        # Initialize model (can be deferred until needed)
        try:
            self._initialize_model()
            logger.info(f"STT Engine initialized with {model_size} model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize model (will retry when needed): {e}")
    
    def __del__(self):
        """Clean up resources when the object is garbage collected"""
        self.stop_recording()
        
        # Clean up recording resources
        if self.stream is not None and self.stream.active:
            try:
                self.stream.close()
            except Exception:
                pass
        
        # Clear references to larger objects
        self.model = None
        self.vad = None
    
    @classmethod
    def _cleanup_at_exit(cls):
        """Clean up all instances at exit"""
        instances = list(cls._instances)
        for ref in instances:
            instance = ref()
            if instance is not None:
                try:
                    instance.stop_recording()
                except Exception:
                    pass  # Ignore errors during shutdown
    
    def _initialize_vad(self):
        """Initialize the Voice Activity Detection module"""
        with self._vad_lock:
            try:
                if self.vad is None:
                    self.vad = webrtcvad.Vad(self.vad_aggressiveness)
                    logger.debug(f"Initialized VAD with aggressiveness {self.vad_aggressiveness}")
            except Exception as e:
                logger.error(f"Failed to initialize VAD: {e}")
                raise STTEngineError(f"Failed to initialize Voice Activity Detection: {e}")
    
    def _initialize_model(self):
        """Initialize the Whisper model"""
        with self._model_lock:
            try:
                if self.model is None:
                    # Load the Whisper model
                    logger.info(f"Loading Faster-Whisper {self.model_size} model on {self.device}...")
                    self.model = WhisperModel(
                        model_size_or_path=self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=os.path.join(os.path.expanduser("~"), ".cache", "faster-whisper")
                    )
                    logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise ModelLoadError(f"Failed to load speech recognition model: {e}")
    
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
        
        # Make sure model is initialized
        try:
            self._initialize_model()
        except ModelLoadError as e:
            logger.error(f"Cannot start recording: {e}")
            raise
        
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
        """Stop recording audio and clean up resources"""
        if not self.recording:
            return
            
        # Signal threads to stop
        self.recording = False
        
        # Wait for threads to terminate
        try:
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)
        except Exception as e:
            logger.error(f"Error joining recording thread: {e}")
        
        try:
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
        except Exception as e:
            logger.error(f"Error joining processing thread: {e}")
        
        # Close audio stream if open
        try:
            if self.stream is not None and hasattr(self.stream, 'close'):
                self.stream.close()
                self.stream = None
        except Exception as e:
            logger.error(f"Error closing audio stream: {e}")
        
        # Clear audio queue
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except Exception:
            pass
        
        # Reset threads
        self.recording_thread = None
        self.processing_thread = None
        
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
            
            # Create stream inside the thread
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels,
                    dtype='int16',
                    blocksize=chunk_samples,
                    callback=None
                )
                self.stream.start()
                logger.info("Audio stream started")
            except Exception as e:
                logger.error(f"Failed to open audio stream: {e}")
                raise AudioCaptureError(f"Failed to access microphone: {e}")
            
            # Initialize VAD if not already done
            if self.vad is None:
                self._initialize_vad()
            
            while self.recording:
                # Read audio chunk
                try:
                    chunk, overflowed = self.stream.read(chunk_samples)
                    if overflowed:
                        logger.warning("Audio buffer overflow detected")
                except Exception as e:
                    logger.error(f"Failed to read from audio stream: {e}")
                    if self.recording:  # Only raise if we're still supposed to be recording
                        raise AudioCaptureError(f"Failed to read from microphone: {e}")
                    break
                
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
            # If this is a fatal error in the recording thread, we should stop the entire process
            self.recording = False
        finally:
            # Clean up stream if it exists
            try:
                if self.stream is not None and hasattr(self.stream, 'close'):
                    self.stream.close()
                    self.stream = None
            except Exception as e:
                logger.error(f"Error closing audio stream during cleanup: {e}")
        
        logger.info("Recording thread terminated")
    
    def _process_audio_thread(self):
        """Thread for processing audio segments and transcribing"""
        temp_files = []  # Keep track of temp files for cleanup
        
        try:
            while self.recording or not self.audio_queue.empty():
                try:
                    # Get audio segment from queue (with timeout to check recording flag)
                    try:
                        audio_segment = self.audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        # Queue empty, continue to check recording flag
                        continue
                    
                    # Skip very short segments (likely noise)
                    duration_sec = len(audio_segment) / self.sample_rate
                    if duration_sec < 0.3:  # Less than 300ms
                        logger.debug(f"Skipping short segment: {duration_sec:.2f}s")
                        continue
                    
                    # Save audio to temporary file for processing
                    temp_file = None
                    try:
                        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        temp_files.append(temp_file.name)
                        
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
                    finally:
                        # Close temp file handle
                        if temp_file:
                            temp_file.close()
                    
                    # Clean up temporary file
                    self._safe_delete_file(temp_file.name)
                    if temp_file.name in temp_files:
                        temp_files.remove(temp_file.name)
                    
                    # If result is not empty, call callback with transcription
                    if result and result.strip():
                        logger.info(f"Transcription: {result}")
                        if self.callback and self.recording:  # Only call if still recording
                            self.callback(result)
                
                except Exception as e:
                    logger.error(f"Error processing audio segment: {e}")
        
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
        finally:
            # Clean up any remaining temp files
            for file_path in temp_files:
                self._safe_delete_file(file_path)
        
        logger.info("Processing thread terminated")
    
    def _safe_delete_file(self, file_path: str):
        """Safely delete a file, ignoring errors"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")
    
    def _transcribe(self, audio_file: str) -> str:
        """
        Transcribe an audio file using Faster-Whisper.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Ensure model is initialized
            self._initialize_model()
            
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
            raise TranscriptionError(f"Failed to transcribe audio: {e}")
    
    def transcribe_file(self, audio_file: str) -> str:
        """
        Transcribe an audio file directly.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
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
        
        # Clear current model
        self.model = None
        
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
