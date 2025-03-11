"""
AI Assistant Core Module

This module contains the core AIAssistant class that integrates all components
and provides the main functionality for the voice assistant.
"""

import os
import sys
import time
import logging
import threading
import queue
import json
import signal
import tempfile
import re
import traceback
from typing import Optional, Dict, List, Any, Union, Callable
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class ComponentInitError(Exception):
    """Exception raised when a component fails to initialize."""
    pass

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
        self.components_initialized = False
        
        # Command queue for handling special commands
        self.command_queue = queue.Queue()
        
        # Component instances (will be set during initialization)
        self.profile = None
        self.personality = None
        self.context = None
        self.emotions = None
        self.tts = None
        self.stt = None
        self.ai = None
        self.vision = None
        self.memory = None
        
        # Track which components were successfully initialized
        self.initialized_components = set()
        
        # Initialize components with error handling
        try:
            self._initialize_components()
            self.components_initialized = True
        except ComponentInitError as e:
            logger.error(f"Failed to initialize components: {e}")
            # Don't raise here - we'll operate in degraded mode if possible
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("AI Assistant initialized")
    
    def _initialize_components(self):
        """Initialize all assistant components with proper error handling"""
        initialization_errors = []
        
        # Define the initialization order with dependencies considered
        initialization_order = [
            ('profile', self._init_profile),
            ('emotions', self._init_emotions), 
            ('personality', self._init_personality),
            ('context', self._init_context), 
            ('memory', self._init_memory),
            ('tts', self._init_tts),
            ('stt', self._init_stt),
            ('ai', self._init_ai)
            # Vision is initialized on demand
        ]
        
        # Initialize components in order
        for component_name, init_func in initialization_order:
            try:
                init_func()
                self.initialized_components.add(component_name)
                logger.info(f"Successfully initialized {component_name}")
            except Exception as e:
                error_msg = f"Failed to initialize {component_name}: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                initialization_errors.append(error_msg)
                
                # For critical components, raise ComponentInitError
                if component_name in ('ai', 'tts'):
                    raise ComponentInitError(f"Critical component {component_name} failed to initialize: {e}")
        
        # If there were non-critical errors, log them but continue
        if initialization_errors and self.components_initialized:
            logger.warning("Some components failed to initialize. Assistant will run in degraded mode.")
            for error in initialization_errors:
                logger.warning(f"- {error}")
    
    def _init_profile(self):
        """Initialize the Profile component"""
        # Import here to avoid circular imports
        from src.profile import Profile
        
        profile_config = self.config.get("profile", {})
        self.profile = Profile(
            profile_path=profile_config.get("profile_path", "assistant_profile.json")
        )
        
        # If name was specified in config, set it
        if "name" in profile_config:
            self.profile.set_name(profile_config["name"])
            
        # If user_name was specified in config, set it
        if "user_name" in profile_config:
            self.profile.set_user_name(profile_config["user_name"])
    
    def _init_personality(self):
        """Initialize the Personality component"""
        # Import here to avoid circular imports
        from src.personality import Personality
        
        personality_config = self.config.get("personality", {})
        self.personality = Personality(
            personality_path=personality_config.get("personality_path", "assistant_personality.json")
        )
        
        # Set traits if specified in config
        for trait, value in personality_config.items():
            if trait != "personality_path" and isinstance(value, (int, float)):
                self.personality.set_trait(trait, float(value))
    
    def _init_context(self):
        """Initialize the Context Manager component"""
        # Import here to avoid circular imports
        from src.context import ContextManager
        
        context_config = self.config.get("context", {})
        self.context = ContextManager(
            context_path=context_config.get("context_path", "assistant_context.json"),
            idle_initiative=context_config.get("idle_initiative", True),
            idle_interval_minutes=context_config.get("idle_interval_minutes", 20)
        )
    
    def _init_emotions(self):
        """Initialize the Emotion Manager component"""
        # Import here to avoid circular imports
        from src.emotions import EmotionManager, EmotionType
        
        emotion_config = self.config.get("emotions", {})
        self.emotions = EmotionManager(
            memory_path=emotion_config.get("memory_path", "emotions_memory.json"),
            initial_emotion=emotion_config.get("initial_emotion", EmotionType.NEUTRAL.value),
            initial_intensity=emotion_config.get("initial_intensity", 0.6)
        )
    
    def _init_tts(self):
        """Initialize the Text-to-Speech component"""
        # Import here to avoid circular imports
        from src.tts import TTSEngine
        
        tts_config = self.config.get("tts", {})
        self.tts = TTSEngine(
            voice=tts_config.get("voice", "english_male_1"),
            model=tts_config.get("model", "en_US/vctk_low")
        )
        
        # Test the TTS engine with a short phrase to verify it works
        try:
            # Use a non-blocking test (no actual audio playback)
            self.tts.synthesize("Test", play_audio=False)
        except Exception as e:
            logger.error(f"TTS engine test failed: {e}")
            raise
    
    def _init_stt(self):
        """Initialize the Speech-to-Text component"""
        # Import here to avoid circular imports
        from src.stt import STTEngine
        
        stt_config = self.config.get("stt", {})
        self.stt = STTEngine(
            model_size=stt_config.get("model_size", "base"),
            language=stt_config.get("language", "en"),
            vad_aggressiveness=stt_config.get("vad_aggressiveness", 3)
        )
    
    def _init_ai(self):
        """Initialize the AI Engine component"""
        # Import here to avoid circular imports
        from src.ai import AIEngine
        
        ai_config = self.config.get("ai", {})
        
        # Create a comprehensive system prompt
        system_prompt = self._create_system_prompt(ai_config.get("system_prompt", ""))
        
        # Try to connect to Ollama with proper error handling
        try:
            self.ai = AIEngine(
                model=ai_config.get("model", "llama3"),
                api_base=ai_config.get("api_base", "http://localhost:11434/api"),
                system_prompt=system_prompt
            )
        except Exception as e:
            # Improve error message for common Ollama issues
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise ComponentInitError(
                    f"Could not connect to Ollama API. "
                    f"Make sure Ollama is running with 'ollama serve' in a separate terminal. Error: {e}"
                )
            elif "not found" in str(e).lower() or "pull" in str(e).lower():
                model = ai_config.get("model", "llama3")
                raise ComponentInitError(
                    f"The model '{model}' was not found in Ollama. "
                    f"Try pulling it with 'ollama pull {model}' first. Error: {e}"
                )
            else:
                raise
    
    def _init_memory(self):
        """Initialize the Memory component"""
        # Import here to avoid circular imports
        from src.memory import Memory
        
        memory_config = self.config.get("memory", {})
        self.memory = Memory(
            db_path=memory_config.get("db_path", "conversations.db")
        )
    
    def _initialize_vision(self):
        """Initialize Vision engine on demand"""
        # Skip if vision is already initialized
        if self.vision is not None:
            return
            
        try:
            # Import here to avoid circular imports
            from src.vision import VisionEngine
            
            logger.info("Initializing Vision engine...")
            vision_config = self.config.get("vision", {})
            self.vision = VisionEngine(
                device=vision_config.get("device", None)
            )
            self.initialized_components.add("vision")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vision engine: {e}")
            logger.debug(traceback.format_exc())
            # We don't raise here since vision is optional
            # Just log the error and return
    
    def _create_system_prompt(self, base_prompt: str) -> str:
        """
        Create a comprehensive system prompt incorporating all components.
        
        Args:
            base_prompt: Base system prompt from config
            
        Returns:
            Complete system prompt
        """
        # Start with the base prompt or a default one
        if not base_prompt:
            base_prompt = (
                "You are a helpful AI assistant with voice capabilities, emotional awareness, "
                "and a unique personality. You can hear the user through their microphone, see through "
                "their camera if they allow it, and respond with your voice."
            )
        
        # Build complete prompt depending on which components are available
        system_prompt = base_prompt + "\n\n"
        
        # Add profile information if available
        if 'profile' in self.initialized_components and self.profile:
            system_prompt += self.profile.get_system_prompt_addition() + "\n"
        
        # Add personality traits if available
        if 'personality' in self.initialized_components and self.personality:
            system_prompt += self.personality.get_system_prompt_addition() + "\n"
        
        # Add emotional framework
        if 'emotions' in self.initialized_components and self.emotions:
            system_prompt += (
                "You have emotional states that can be affected by interactions. "
                "Your emotions include: happy, excited, neutral, concerned, confused, sad, and angry. "
                "These emotions affect how you respond and the tone of your voice. "
                "You should acknowledge your emotions when appropriate and can explain why you feel a certain way.\n"
            )
        
        # Add contextual awareness if available
        if 'context' in self.initialized_components and self.context:
            system_prompt += self.context.get_system_prompt_addition() + "\n"
        
        # Add instructions for handling special commands
        system_prompt += (
            "The user may use special commands prefixed with '!'. These are handled separately "
            "and are not part of the conversation. However, the user can also make requests "
            "using natural language, which you should respond to appropriately.\n"
            
            "When the user asks you to change your name or personality aspects, "
            "you should acknowledge and adapt to these changes.\n"
        )
        
        logger.debug(f"Created system prompt of length {len(system_prompt)}")
        return system_prompt
    
    def _setup_conversation(self) -> int:
        """
        Set up a new or existing conversation.
        
        Returns:
            ID of the conversation
        """
        if 'memory' not in self.initialized_components or not self.memory:
            logger.warning("Memory component not initialized. Conversation history won't be saved.")
            return 0  # Return dummy ID
            
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
        title = f"Conversation with {self.profile.get_name() if self.profile else 'Assistant'} - {timestamp}"
        new_id = self.memory.create_conversation(title)
        
        # Add system message if AI is initialized
        if 'ai' in self.initialized_components and self.ai:
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
        
        # Initialize conversation if memory is available
        if 'memory' in self.initialized_components and self.memory:
            self.conversation_id = self._setup_conversation()
        else:
            self.conversation_id = 0
        
        try:
            # Welcome message
            logger.info("Starting AI Assistant...")
            
            # Get name from profile or use default
            assistant_name = self.profile.get_name() if self.profile else "Assistant"
            user_name = self.profile.get_user_name() if self.profile else "there"
            
            # Create personalized welcome message
            welcome_message = f"Hello {user_name}! I'm {assistant_name}. "
            welcome_message += "You can speak to me or type commands. Say 'help' for assistance."
            
            # Customize welcome message based on emotional state and personality
            if 'emotions' in self.initialized_components and self.emotions:
                if self.emotions.current_emotion != "neutral":
                    welcome_message += f" By the way, I'm feeling {self.emotions.current_emotion} right now."
            
            # Apply personality style if available
            if 'personality' in self.initialized_components and self.personality:
                welcome_message = self.personality.adjust_text_style(welcome_message)
            
            print(welcome_message)
            
            # Speak welcome message if TTS is available
            if 'tts' in self.initialized_components and self.tts:
                # Apply emotional voice parameters if emotions are available
                voice_params = None
                if 'emotions' in self.initialized_components and self.emotions:
                    voice_params = self.emotions.get_voice_parameters()
                
                self.tts.speak(welcome_message, voice_params=voice_params)
            else:
                logger.warning("TTS not available. Text output only.")
            
            # Start context manager with idle callback if available
            if 'context' in self.initialized_components and self.context:
                self.context.start(idle_callback=self._handle_idle_initiative)
            
            # Start command processing thread
            command_thread = threading.Thread(target=self._command_processor, daemon=True)
            command_thread.start()
            
            # Start speech recognition if available
            if 'stt' in self.initialized_components and self.stt:
                self._start_listening()
            else:
                logger.warning("STT not available. Text input only.")
            
            # Main loop - handle text input while also processing speech
            while self.running:
                try:
                    # Get text input (non-blocking)
                    user_input = input("> ")
                    if user_input.strip():
                        self._process_input(user_input)
                        # Record activity in context manager if available
                        if 'context' in self.initialized_components and self.context:
                            self.context.record_activity()
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
        logger.info("Stopping AI Assistant...")
        
        # Stop context manager if initialized
        if 'context' in self.initialized_components and self.context:
            self.context.stop()
        
        # Stop listening if STT was initialized
        if 'stt' in self.initialized_components and self.stt:
            try:
                self.stt.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping STT: {e}")
        
        # Clean up vision resources if initialized
        if 'vision' in self.initialized_components and self.vision:
            try:
                self.vision.release_webcam()
            except Exception as e:
                logger.error(f"Error releasing webcam: {e}")
        
        # Final message
        farewell = f"Shutting down. Goodbye!"
        
        # Customize farewell based on personality if available
        if 'personality' in self.initialized_components and self.personality:
            farewell = self.personality.adjust_text_style(farewell)
        
        print(f"\n{farewell}")
        
        # Speak farewell if TTS is available
        if 'tts' in self.initialized_components and self.tts:
            voice_params = None
            if 'emotions' in self.initialized_components and self.emotions:
                voice_params = self.emotions.get_voice_parameters()
                
            try:
                self.tts.speak(farewell, voice_params=voice_params)
            except Exception as e:
                logger.error(f"Error in final TTS: {e}")
        
        logger.info("Assistant stopped")
    
    def _start_listening(self):
        """Start listening for voice input"""
        logger.info("Starting speech recognition...")
        
        def speech_callback(text):
            """Callback function for speech recognition"""
            if text and not self.processing_input:
                print(f"\nHeard: {text}")
                self._process_input(text)
                # Record activity in context manager if available
                if 'context' in self.initialized_components and self.context:
                    self.context.record_activity()
        
        # Start recording with callback
        silence_threshold = self.config.get("stt", {}).get("silence_threshold_sec", 1.0)
        max_recording = self.config.get("stt", {}).get("max_recording_sec", 30.0)
        
        try:
            self.stt.start_recording(
                callback=speech_callback,
                silence_threshold_sec=silence_threshold,
                max_recording_sec=max_recording
            )
        except Exception as e:
            logger.error(f"Failed to start speech recognition: {e}")
            logger.debug(traceback.format_exc())
            print("Speech recognition could not be started. Please use text input.")
    
    def _handle_idle_initiative(self, initiative_text: str):
        """
        Handle an idle initiative from the context manager.
        
        Args:
            initiative_text: Proactive message from context manager
        """
        if self.processing_input:
            # Don't interrupt if already processing input
            return
            
        logger.info(f"Idle initiative triggered: {initiative_text}")
        
        # Apply personality style if available
        styled_text = initiative_text
        if 'personality' in self.initialized_components and self.personality:
            styled_text = self.personality.adjust_text_style(initiative_text)
        
        # Display the initiative
        assistant_name = self.profile.get_name() if 'profile' in self.initialized_components and self.profile else "Assistant"
        print(f"\n{assistant_name}: {styled_text}")
        
        # Speak the initiative if TTS is available
        if 'tts' in self.initialized_components and self.tts:
            voice_params = None
            if 'emotions' in self.initialized_components and self.emotions:
                voice_params = self.emotions.get_voice_parameters()
                
            try:
                self.tts.speak(styled_text, voice_params=voice_params)
            except Exception as e:
                logger.error(f"Error in initiative TTS: {e}")
    
    def _extract_name_change(self, text: str) -> Optional[str]:
        """
        Extract a name change request from the message.
        
        Args:
            text: User message
            
        Returns:
            New name or None if no valid name change request found
        """
        # Patterns for name change requests
        patterns = [
            r"call\s+you(?:rself)?\s+(\w+)",
            r"your\s+name\s+is\s+(\w+)",
            r"change\s+your\s+name\s+to\s+(\w+)",
            r"set\s+your\s+name\s+(?:as|to)\s+(\w+)",
            r"rename\s+you(?:rself)?\s+(?:as|to)\s+(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                new_name = match.group(1)
                # Verify it's a reasonable name (not too short or too long)
                if 2 <= len(new_name) <= 20:
                    return new_name
        
        return None
    
    def _extract_user_name(self, text: str) -> Optional[str]:
        """
        Extract the user's name from the message.
        
        Args:
            text: User message
            
        Returns:
            User name or None if no valid user name found
        """
        # Patterns for user name mentions
        patterns = [
            r"my\s+name\s+is\s+(\w+)",
            r"call\s+me\s+(\w+)",
            r"i\s+am\s+(\w+)",
            r"i'm\s+(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Verify it's a reasonable name
                if 2 <= len(name) <= 20:
                    return name
        
        return None
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract potential topics from the message.
        
        Args:
            text: Message text
            
        Returns:
            List of extracted topics
        """
        # Simple topic extraction based on nouns in the text
        topics = []
        
        # Split text into words and clean them
        words = [word.strip().lower() for word in re.split(r'\W+', text) if word.strip()]
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'their', 'this', 'that', 'these', 'those'}
        potential_topics = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Add most frequent words as topics
        word_counts = {}
        for word in potential_topics:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        topics = [word for word, count in sorted_words[:5]]  # Take top 5 words as topics
        
        return topics
    
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
            
            # Check for name change requests if profile is available
            if 'profile' in self.initialized_components and self.profile:
                new_name = self._extract_name_change(user_input)
                if new_name:
                    old_name = self.profile.get_name()
                    if self.profile.set_name(new_name):
                        response = f"I'll respond to {new_name} from now on."
                        
                        # Apply personality style if available
                        if 'personality' in self.initialized_components and self.personality:
                            response = self.personality.adjust_text_style(response)
                            
                        print(f"{new_name}: {response}")
                        
                        # Speak response if TTS is available
                        if 'tts' in self.initialized_components and self.tts:
                            voice_params = None
                            if 'emotions' in self.initialized_components and self.emotions:
                                voice_params = self.emotions.get_voice_parameters()
                                
                            self.tts.speak(response, voice_params=voice_params)
                        
                        # Update conversation history if memory is available
                        if 'memory' in self.initialized_components and self.memory:
                            self.memory.add_message(self.conversation_id, "user", user_input)
                            self.memory.add_message(self.conversation_id, "assistant", response)
                        
                        # Update AI system prompt if AI is available
                        if 'ai' in self.initialized_components and self.ai:
                            system_prompt = self._create_system_prompt("")
                            self.ai.system_prompt = system_prompt
                            
                        return
            
            # Check for user name mentions if profile is available
            if 'profile' in self.initialized_components and self.profile:
                user_name = self._extract_user_name(user_input)
                if user_name:
                    self.profile.set_user_name(user_name)
                    # This will be handled by the AI model, so we let it continue processing
            
            # Extract potential topics for context if context is available
            if 'context' in self.initialized_components and self.context:
                topics = self._extract_topics(user_input)
                if topics:
                    self.context.update_current_session(1, topics)
                    
                    # Add user interests based on frequent topics
                    for topic in topics[:2]:  # Only consider top 2 topics
                        if topic in user_input.lower().split():
                            self.context.add_user_interest(topic)
            
            # Update emotional state based on user input if emotions are available
            if 'emotions' in self.initialized_components and self.emotions:
                self.emotions.process_message(user_input, source="user")
            
            # Update personality based on user input if personality is available
            if 'personality' in self.initialized_components and self.personality:
                self.personality.process_message(user_input, source="user")
            
            # Check for emotion-specific commands if emotions are available
            if 'emotions' in self.initialized_components and self.emotions:
                if user_input.lower() in ["how are you feeling", "what's your mood", "how do you feel"]:
                    emotion_explanation = self.emotions.get_emotion_reason()
                    
                    # Apply personality style if available
                    styled_explanation = emotion_explanation
                    if 'personality' in self.initialized_components and self.personality:
                        styled_explanation = self.personality.adjust_text_style(emotion_explanation)
                    
                    # Get assistant name if profile is available
                    assistant_name = self.profile.get_name() if 'profile' in self.initialized_components and self.profile else "Assistant"
                    
                    print(f"{assistant_name}: {styled_explanation}")
                    
                    # Speak response if TTS is available
                    if 'tts' in self.initialized_components and self.tts:
                        voice_params = None
                        if 'emotions' in self.initialized_components and self.emotions:
                            voice_params = self.emotions.get_voice_parameters()
                            
                        self.tts.speak(styled_explanation, voice_params=voice_params)
                    
                    # Add to conversation history if memory is available
                    if 'memory' in self.initialized_components and self.memory:
                        self.memory.add_message(self.conversation_id, "user", user_input)
                        self.memory.add_message(self.conversation_id, "assistant", emotion_explanation)
                        
                    return
            
            # Look for relevant context if context is available
            context_addition = ""
            if 'context' in self.initialized_components and self.context:
                relevant_topics = self.context.get_relevant_topics(user_input)
                
                # Add context to the AI request if relevant topics found
                if relevant_topics:
                    context_addition = "Based on our previous conversations: "
                    for topic in relevant_topics:
                        context_addition += f"{topic['topic']}: {topic['content']} "
            
            # Add user message to memory if memory is available
            if 'memory' in self.initialized_components and self.memory:
                self.memory.add_message(self.conversation_id, "user", user_input)
            
            # Generate AI response if AI is available
            if 'ai' in self.initialized_components and self.ai:
                # Get assistant name if profile is available
                assistant_name = self.profile.get_name() if 'profile' in self.initialized_components and self.profile else "Assistant"
                print(f"{assistant_name} is thinking...")
                
                def streaming_callback(chunk):
                    """Callback for streaming response"""
                    print(chunk, end="", flush=True)
                
                try:
                    # Generate streaming response
                    prompt = user_input
                    if context_addition:
                        prompt += f"\n\nRecall from previous conversations: {context_addition}"
                        
                    response = self.ai.generate_response(
                        prompt,
                        stream=True,
                        stream_callback=streaming_callback
                    )
                    
                    print()  # Add newline after streaming response
                    
                    # Apply personality styling to the response if personality is available
                    styled_response = response
                    if 'personality' in self.initialized_components and self.personality:
                        styled_response = self.personality.adjust_text_style(response)
                        
                        # If the styling changed the response significantly, display it differently
                        if styled_response != response:
                            print(f"{assistant_name}: {styled_response}")
                            response = styled_response
                    
                    # Update emotional state based on AI response if emotions are available
                    if 'emotions' in self.initialized_components and self.emotions:
                        self.emotions.process_message(response, source="assistant")
                    
                    # Add assistant response to memory if memory is available
                    if 'memory' in self.initialized_components and self.memory:
                        self.memory.add_message(self.conversation_id, "assistant", response)
                    
                    # Speak response if TTS is available
                    if 'tts' in self.initialized_components and self.tts:
                        voice_params = None
                        if 'emotions' in self.initialized_components and self.emotions:
                            voice_params = self.emotions.get_voice_parameters()
                            
                        self.tts.speak(response, voice_params=voice_params)
                    
                except Exception as e:
                    error_msg = f"Error generating AI response: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    print(f"\nSorry, I encountered an error: {e}")
                    
                    # Speak error message if TTS is available
                    if 'tts' in self.initialized_components and self.tts:
                        self.tts.speak(f"Sorry, I encountered an error: {e}")
            else:
                # If AI is not available, provide a simple response
                print("I'm currently running in limited mode without AI capabilities.")
                
                # Speak response if TTS is available
                if 'tts' in self.initialized_components and self.tts:
                    self.tts.speak("I'm currently running in limited mode without AI capabilities.")
            
            # Update context with the message if context is available
            if 'context' in self.initialized_components and self.context:
                self.context.record_activity()
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            logger.debug(traceback.format_exc())
            print(f"Error processing your input: {e}")
            
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
                try:
                    cmd, args = self.command_queue.get(timeout=1.0)
                except queue.Empty:
                    # No commands in queue, continue
                    continue
                
                # Process based on command
                if cmd == "exit" or cmd == "quit":
                    logger.info("Exit command received")
                    self.running = False
                    continue
                
                elif cmd == "voice":
                    # Change voice if TTS is available
                    if 'tts' not in self.initialized_components or not self.tts:
                        print("TTS is not available. Voice commands are disabled.")
                        continue
                        
                    if args:
                        voice_name = args[0]
                        logger.info(f"Changing voice to {voice_name}")
                        try:
                            if self.tts.change_voice(voice_name):
                                response = f"Voice changed to {voice_name}"
                                voice_params = None
                                if 'emotions' in self.initialized_components and self.emotions:
                                    voice_params = self.emotions.get_voice_parameters()
                                    
                                self.tts.speak(response, voice_params=voice_params)
                            else:
                                print(f"Voice '{voice_name}' not available")
                        except Exception as e:
                            logger.error(f"Error changing voice: {e}")
                            print(f"Error changing voice: {e}")
                    else:
                        # List available voices
                        try:
                            voices = self.tts.list_available_voices()
                            print("Available voices:")
                            for voice in voices:
                                print(f"- {voice}")
                        except Exception as e:
                            logger.error(f"Error listing voices: {e}")
                            print(f"Error listing voices: {e}")
                
                elif cmd == "voice_model":
                    # Change TTS model if TTS is available
                    if 'tts' not in self.initialized_components or not self.tts:
                        print("TTS is not available. Voice model commands are disabled.")
                        continue
                        
                    if args:
                        model_name = args[0]
                        logger.info(f"Changing TTS model to {model_name}")
                        try:
                            if self.tts.change_model(model_name):
                                response = f"Voice model changed to {model_name}"
                                voice_params = None
                                if 'emotions' in self.initialized_components and self.emotions:
                                    voice_params = self.emotions.get_voice_parameters()
                                    
                                self.tts.speak(response, voice_params=voice_params)
                            else:
                                print(f"Voice model '{model_name}' not available")
                        except Exception as e:
                            logger.error(f"Error changing TTS model: {e}")
                            print(f"Error changing TTS model: {e}")
                    else:
                        # List available models
                        try:
                            models = self.tts.list_available_models()
                            print("Available TTS models:")
                            for model in models:
                                print(f"- {model}")
                        except Exception as e:
                            logger.error(f"Error listing TTS models: {e}")
                            print(f"Error listing TTS models: {e}")
                
                elif cmd == "model":
                    # Change AI model if AI is available
                    if 'ai' not in self.initialized_components or not self.ai:
                        print("AI is not available. Model commands are disabled.")
                        continue
                        
                    if args:
                        model_name = args[0]
                        logger.info(f"Changing AI model to {model_name}")
                        try:
                            if self.ai.change_model(model_name):
                                response = f"Model changed to {model_name}"
                                
                                # Speak response if TTS is available
                                if 'tts' in self.initialized_components and self.tts:
                                    voice_params = None
                                    if 'emotions' in self.initialized_components and self.emotions:
                                        voice_params = self.emotions.get_voice_parameters()
                                        
                                    self.tts.speak(response, voice_params=voice_params)
                                    
                                print(response)
                            else:
                                print(f"Model '{model_name}' not available")
                        except Exception as e:
                            logger.error(f"Error changing AI model: {e}")
                            print(f"Error changing AI model: {e}")
                    else:
                        # List available models
                        try:
                            models = self.ai.list_available_models()
                            print("Available AI models:")
                            for model in models:
                                print(f"- {model['name']}")
                        except Exception as e:
                            logger.error(f"Error listing AI models: {e}")
                            print(f"Error listing AI models: {e}")
                
                elif cmd == "name":
                    # Change assistant name if profile is available
                    if 'profile' not in self.initialized_components or not self.profile:
                        print("Profile is not available. Name commands are disabled.")
                        continue
                        
                    if args:
                        new_name = args[0]
                        old_name = self.profile.get_name()
                        logger.info(f"Changing name from {old_name} to {new_name}")
                        try:
                            if self.profile.set_name(new_name):
                                response = f"My name is now {new_name}."
                                
                                # Apply personality style if available
                                if 'personality' in self.initialized_components and self.personality:
                                    response = self.personality.adjust_text_style(response)
                                    
                                print(f"{new_name}: {response}")
                                
                                # Speak response if TTS is available
                                if 'tts' in self.initialized_components and self.tts:
                                    voice_params = None
                                    if 'emotions' in self.initialized_components and self.emotions:
                                        voice_params = self.emotions.get_voice_parameters()
                                        
                                    self.tts.speak(response, voice_params=voice_params)
                                
                                # Update AI system prompt if AI is available
                                if 'ai' in self.initialized_components and self.ai:
                                    system_prompt = self._create_system_prompt("")
                                    self.ai.system_prompt = system_prompt
                            else:
                                print(f"Failed to change name to '{new_name}'")
                        except Exception as e:
                            logger.error(f"Error changing name: {e}")
                            print(f"Error changing name: {e}")
                    else:
                        # Show current name
                        try:
                            name = self.profile.get_name()
                            print(f"My current name is {name}")
                        except Exception as e:
                            logger.error(f"Error getting name: {e}")
                            print(f"Error getting name: {e}")
                
                elif cmd == "user":
                    # Set user name if profile is available
                    if 'profile' not in self.initialized_components or not self.profile:
                        print("Profile is not available. User commands are disabled.")
                        continue
                        
                    if args:
                        user_name = args[0]
                        logger.info(f"Setting user name to {user_name}")
                        try:
                            if self.profile.set_user_name(user_name):
                                response = f"I'll call you {user_name} from now on."
                                
                                # Apply personality style if available
                                if 'personality' in self.initialized_components and self.personality:
                                    response = self.personality.adjust_text_style(response)
                                    
                                # Get assistant name
                                assistant_name = self.profile.get_name()
                                print(f"{assistant_name}: {response}")
                                
                                # Speak response if TTS is available
                                if 'tts' in self.initialized_components and self.tts:
                                    voice_params = None
                                    if 'emotions' in self.initialized_components and self.emotions:
                                        voice_params = self.emotions.get_voice_parameters()
                                        
                                    self.tts.speak(response, voice_params=voice_params)
                                
                                # Update AI system prompt if AI is available
                                if 'ai' in self.initialized_components and self.ai:
                                    system_prompt = self._create_system_prompt("")
                                    self.ai.system_prompt = system_prompt
                            else:
                                print(f"Failed to set user name to '{user_name}'")
                        except Exception as e:
                            logger.error(f"Error setting user name: {e}")
                            print(f"Error setting user name: {e}")
                    else:
                        # Show current user name
                        try:
                            user_name = self.profile.get_user_name()
                            if user_name:
                                print(f"I'm currently calling you {user_name}")
                            else:
                                print("I don't know your name yet")
                        except Exception as e:
                            logger.error(f"Error getting user name: {e}")
                            print(f"Error getting user name: {e}")
                
                elif cmd == "personality":
                    # Personality management if personality is available
                    if 'personality' not in self.initialized_components or not self.personality:
                        print("Personality is not available. Personality commands are disabled.")
                        continue
                        
                    if not args:
                        # Show current personality
                        try:
                            print(self.personality.get_personality_description())
                        except Exception as e:
                            logger.error(f"Error getting personality description: {e}")
                            print(f"Error getting personality description: {e}")
                    elif args[0] == "set" and len(args) >= 3:
                        # Set personality trait
                        trait = args[1]
                        try:
                            value = float(args[2])
                            value = max(0.0, min(1.0, value))  # Ensure value is between 0 and 1
                            
                            if self.personality.set_trait(trait, value):
                                response = f"Personality trait '{trait}' set to {value:.2f}."
                                print(response)
                                
                                # Speak response if TTS is available
                                if 'tts' in self.initialized_components and self.tts:
                                    voice_params = None
                                    if 'emotions' in self.initialized_components and self.emotions:
                                        voice_params = self.emotions.get_voice_parameters()
                                        
                                    self.tts.speak(response, voice_params=voice_params)
                                
                                # Update AI system prompt if AI is available
                                if 'ai' in self.initialized_components and self.ai:
                                    system_prompt = self._create_system_prompt("")
                                    self.ai.system_prompt = system_prompt
                            else:
                                print(f"Failed to set personality trait '{trait}'")
                        except ValueError:
                            print(f"Invalid value for personality trait. Must be a number between 0 and 1")
                        except Exception as e:
                            logger.error(f"Error setting personality trait: {e}")
                            print(f"Error setting personality trait: {e}")
                
                elif cmd == "emotion":
                    # Emotion management if emotions are available
                    if 'emotions' not in self.initialized_components or not self.emotions:
                        print("Emotions are not available. Emotion commands are disabled.")
                        continue
                        
                    if not args:
                        # Show current emotion info
                        try:
                            current_emotion = self.emotions.current_emotion
                            current_intensity = self.emotions.current_intensity
                            print(f"Current emotion: {current_emotion} (intensity: {current_intensity:.2f})")
                            print(self.emotions.get_emotion_reason())
                        except Exception as e:
                            logger.error(f"Error getting emotion info: {e}")
                            print(f"Error getting emotion info: {e}")
                    elif args[0] == "set" and len(args) > 1:
                        # Set emotion manually
                        emotion = args[1]
                        intensity = float(args[2]) if len(args) > 2 else 0.7
                        try:
                            self.emotions.set_emotion(emotion, intensity)
                            print(f"Emotion set to {emotion} (intensity: {intensity:.2f})")
                            response = f"I'm now feeling {emotion}."
                            
                            # Apply personality style if available
                            if 'personality' in self.initialized_components and self.personality:
                                response = self.personality.adjust_text_style(response)
                                
                            # Speak response if TTS is available
                            if 'tts' in self.initialized_components and self.tts:
                                voice_params = self.emotions.get_voice_parameters()
                                self.tts.speak(response, voice_params=voice_params)
                        except Exception as e:
                            logger.error(f"Error setting emotion: {e}")
                            print(f"Error setting emotion: {e}")
                    elif args[0] == "reset":
                        # Reset to neutral
                        try:
                            self.emotions.reset_emotion()
                            print("Emotion reset to neutral")
                            response = "I've reset my emotional state to neutral."
                            
                            # Apply personality style if available
                            if 'personality' in self.initialized_components and self.personality:
                                response = self.personality.adjust_text_style(response)
                                
                            # Speak response if TTS is available
                            if 'tts' in self.initialized_components and self.tts:
                                voice_params = self.emotions.get_voice_parameters()
                                self.tts.speak(response, voice_params=voice_params)
                        except Exception as e:
                            logger.error(f"Error resetting emotion: {e}")
                            print(f"Error resetting emotion: {e}")
                    elif args[0] == "triggers":
                        # List emotion triggers
                        try:
                            triggers = self.emotions.get_all_triggers()
                            print("Emotion triggers:")
                            for emotion, trigger_list in triggers.items():
                                print(f"- {emotion}: {', '.join(trigger_list[:5])}{'...' if len(trigger_list) > 5 else ''}")
                        except Exception as e:
                            logger.error(f"Error getting emotion triggers: {e}")
                            print(f"Error getting emotion triggers: {e}")
                
                elif cmd == "screenshot":
                    # Take a screenshot and analyze it - initialize vision if needed
                    self._initialize_vision()
                    
                    if 'vision' not in self.initialized_components or not self.vision:
                        print("Vision is not available. Screenshot command is disabled.")
                        continue
                        
                    logger.info("Taking screenshot")
                    try:
                        # Capture screen
                        screen = self.vision.capture_screen()
                        
                        # Save screenshot
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"screenshot_{timestamp}.png"
                        self.vision.save_image(screen, filename)
                        
                        # Analyze and describe
                        description = self.vision.describe_image(screen)
                        print(f"Screen description: {description}")
                        response = f"I took a screenshot. {description}"
                        
                        # Apply personality style if available
                        if 'personality' in self.initialized_components and self.personality:
                            response = self.personality.adjust_text_style(response)
                            
                        # Speak response if TTS is available
                        if 'tts' in self.initialized_components and self.tts:
                            voice_params = None
                            if 'emotions' in self.initialized_components and self.emotions:
                                voice_params = self.emotions.get_voice_parameters()
                                
                            self.tts.speak(response, voice_params=voice_params)
                    except Exception as e:
                        logger.error(f"Error taking screenshot: {e}")
                        print(f"Error taking screenshot: {e}")
                
                elif cmd == "webcam":
                    # Capture from webcam and analyze - initialize vision if needed
                    self._initialize_vision()
                    
                    if 'vision' not in self.initialized_components or not self.vision:
                        print("Vision is not available. Webcam command is disabled.")
                        continue
                        
                    logger.info("Capturing from webcam")
                    try:
                        # Capture frame
                        frame = self.vision.capture_webcam()
                        if frame is None:
                            print("Failed to capture from webcam")
                            
                            # Speak response if TTS is available
                            if 'tts' in self.initialized_components and self.tts:
                                self.tts.speak("I couldn't access your webcam")
                                
                            continue
                        
                        # Save image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"webcam_{timestamp}.png"
                        self.vision.save_image(frame, filename)
                        
                        # Analyze and describe
                        description = self.vision.describe_image(frame)
                        print(f"Webcam description: {description}")
                        
                        # Speak response if TTS is available
                        if 'tts' in self.initialized_components and self.tts:
                            self.tts.speak(f"I captured from your webcam. {description}")
                    except Exception as e:
                        logger.error(f"Error capturing from webcam: {e}")
                        print(f"Error capturing from webcam: {e}")
                
                elif cmd == "memory":
                    # Memory management if memory is available
                    if 'memory' not in self.initialized_components or not self.memory:
                        print("Memory is not available. Memory commands are disabled.")
                        continue
                        
                    try:
                        if not args:
                            # Show current conversation info
                            conversation = self.memory.get_conversation(self.conversation_id)
                            messages = self.memory.get_messages(self.conversation_id)
                            print(f"Current conversation: {conversation.get('title')}")
                            print(f"Messages: {len(messages)}")
                        elif args[0] == "list":
                            # List recent conversations
                            conversations = self.memory.list_conversations()
                            print("Recent conversations:")
                            for conv in conversations:
                                print(f"- {conv['id']}: {conv['title']} ({len(self.memory.get_messages(conv['id']))} messages)")
                        elif args[0] == "switch" and len(args) > 1:
                            # Switch to a different conversation
                            try:
                                new_id = int(args[1])
                                conversation = self.memory.get_conversation(new_id)
                                if conversation:
                                    self.conversation_id = new_id
                                    print(f"Switched to conversation: {conversation.get('title')}")
                                    
                                    # Speak response if TTS is available
                                    if 'tts' in self.initialized_components and self.tts:
                                        response = f"Switched to conversation: {conversation.get('title')}"
                                        voice_params = None
                                        if 'emotions' in self.initialized_components and self.emotions:
                                            voice_params = self.emotions.get_voice_parameters()
                                            
                                        self.tts.speak(response, voice_params=voice_params)
                                else:
                                    print(f"Conversation {new_id} not found")
                            except ValueError:
                                print("Invalid conversation ID")
                        elif args[0] == "new":
                            # Create a new conversation
                            title = " ".join(args[1:]) if len(args) > 1 else None
                            self.conversation_id = self._setup_conversation()
                            print(f"Created new conversation with ID {self.conversation_id}")
                            
                            # Speak response if TTS is available
                            if 'tts' in self.initialized_components and self.tts:
                                response = "Created a new conversation"
                                voice_params = None
                                if 'emotions' in self.initialized_components and self.emotions:
                                    voice_params = self.emotions.get_voice_parameters()
                                    
                                self.tts.speak(response, voice_params=voice_params)
                    except Exception as e:
                        logger.error(f"Error executing memory command: {e}")
                        print(f"Error executing memory command: {e}")
                
                elif cmd == "help":
                    # Show help information
                    self._show_help()
                
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type !help for a list of available commands")
                
                # Mark command as processed
                self.command_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                logger.debug(traceback.format_exc())
                print(f"Error processing command: {e}")
    
    def _show_help(self):
        """Show help information for available commands"""
        available_components = {
            'voice': 'tts' in self.initialized_components,
            'voice_model': 'tts' in self.initialized_components,
            'model': 'ai' in self.initialized_components,
            'name': 'profile' in self.initialized_components,
            'user': 'profile' in self.initialized_components,
            'personality': 'personality' in self.initialized_components,
            'emotion': 'emotions' in self.initialized_components,
            'screenshot': 'vision' in self.initialized_components,
            'webcam': 'vision' in self.initialized_components,
            'memory': 'memory' in self.initialized_components
        }
        
        print("Available commands:")
        print("  !exit, !quit - Exit the assistant")
        
        if available_components['voice']:
            print("  !voice [name] - Change voice or list available voices")
            
        if available_components['voice_model']:
            print("  !voice_model [name] - Change TTS model or list available models")
            
        if available_components['model']:
            print("  !model [name] - Change AI model or list available models")
            
        if available_components['name']:
            print("  !name [new_name] - Change assistant name")
            
        if available_components['user']:
            print("  !user [name] - Set or show user name")
            
        if available_components['personality']:
            print("  !personality - Show current personality")
            print("  !personality set <trait> <value> - Set personality trait (0-1)")
            
        if available_components['screenshot']:
            print("  !screenshot - Take and analyze a screenshot")
            
        if available_components['webcam']:
            print("  !webcam - Capture and analyze from webcam")
            
        if available_components['memory']:
            print("  !memory - Show current conversation info")
            print("  !memory list - List recent conversations")
            print("  !memory switch <id> - Switch to a different conversation")
            print("  !memory new [title] - Create a new conversation")
            
        if available_components['emotion']:
            print("  !emotion - Show current emotion status")
            print("  !emotion set <emotion> [intensity] - Set emotion manually")
            print("  !emotion reset - Reset to neutral emotion")
            print("  !emotion triggers - List emotion triggers")
            
        print("  !help - Show this help information")
