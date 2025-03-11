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
from typing import Optional, Dict, List, Any, Union
from datetime import datetime

# Import assistant modules
from src.tts import TTSEngine
from src.stt import STTEngine
from src.ai import AIEngine
from src.vision import VisionEngine
from src.memory import Memory
from src.emotions import EmotionManager, EmotionType
from src.profile import Profile
from src.personality import Personality
from src.context import ContextManager

# Set up logging
logger = logging.getLogger(__name__)

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
            # Initialize Profile
            logger.info("Initializing Profile...")
            profile_config = self.config.get("profile", {})
            self.profile = Profile(
                profile_path=profile_config.get("profile_path", "assistant_profile.json")
            )
            
            # Initialize Personality
            logger.info("Initializing Personality...")
            personality_config = self.config.get("personality", {})
            self.personality = Personality(
                personality_path=personality_config.get("personality_path", "assistant_personality.json")
            )
            
            # Initialize Context Manager
            logger.info("Initializing Context Manager...")
            context_config = self.config.get("context", {})
            self.context = ContextManager(
                context_path=context_config.get("context_path", "assistant_context.json"),
                idle_initiative=context_config.get("idle_initiative", True),
                idle_interval_minutes=context_config.get("idle_interval_minutes", 20)
            )
            
            # Initialize Emotion Manager
            logger.info("Initializing Emotion Manager...")
            emotion_config = self.config.get("emotions", {})
            self.emotions = EmotionManager(
                memory_path=emotion_config.get("memory_path", "emotions_memory.json"),
                initial_emotion=emotion_config.get("initial_emotion", EmotionType.NEUTRAL.value),
                initial_intensity=emotion_config.get("initial_intensity", 0.6)
            )
            
            # Initialize Text-to-Speech engine
            logger.info("Initializing Text-to-Speech engine...")
            tts_config = self.config.get("tts", {})
            self.tts = TTSEngine(
                voice=tts_config.get("voice", "english_male_1"),
                model=tts_config.get("model", "en_US/vctk_low")  # MeloTTS model selection
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
            
            # Create a comprehensive system prompt
            system_prompt = self._create_system_prompt(ai_config.get("system_prompt", ""))
            
            self.ai = AIEngine(
                model=ai_config.get("model", "llama3"),
                api_base=ai_config.get("api_base", "http://localhost:11434/api"),
                system_prompt=system_prompt
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
        
        # Add profile information
        system_prompt = base_prompt + "\n\n"
        system_prompt += self.profile.get_system_prompt_addition() + "\n"
        
        # Add personality traits
        system_prompt += self.personality.get_system_prompt_addition() + "\n"
        
        # Add emotional framework
        system_prompt += (
            "You have emotional states that can be affected by interactions. "
            "Your emotions include: happy, excited, neutral, concerned, confused, sad, and angry. "
            "These emotions affect how you respond and the tone of your voice. "
            "You should acknowledge your emotions when appropriate and can explain why you feel a certain way.\n"
        )
        
        # Add contextual awareness
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
        title = f"Conversation with {self.profile.get_name()} - {timestamp}"
        new_id = self.memory.create_conversation(title)
        
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
            
            # Get name from profile
            assistant_name = self.profile.get_name()
            user_name = self.profile.get_user_name() or "there"
            
            # Create personalized welcome message
            welcome_message = f"Hello {user_name}! I'm {assistant_name}. "
            welcome_message += "You can speak to me or type commands. Say 'help' for assistance."
            
            # Customize welcome message based on emotional state and personality
            if self.emotions.current_emotion != EmotionType.NEUTRAL.value:
                welcome_message += f" By the way, I'm feeling {self.emotions.current_emotion} right now."
                
            # Apply personality style
            welcome_message = self.personality.adjust_text_style(welcome_message)
            
            print(welcome_message)
            
            # Apply emotional voice parameters
            voice_params = self.emotions.get_voice_parameters()
            self.tts.speak(welcome_message, voice_params=voice_params)
            
            # Start context manager with idle callback
            self.context.start(idle_callback=self._handle_idle_initiative)
            
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
                        # Record activity in context manager
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
        
        # Stop context manager
        self.context.stop()
        
        # Stop listening
        self.stt.stop_recording()
        
        # Clean up vision resources if initialized
        if self.vision:
            self.vision.release_webcam()
        
        # Final message
        farewell = f"Shutting down. Goodbye!"
        
        # Customize farewell based on personality
        farewell = self.personality.adjust_text_style(farewell)
        
        print(f"\n{farewell}")
        
        # Apply emotional voice parameters
        voice_params = self.emotions.get_voice_parameters()
        self.tts.speak(farewell, voice_params=voice_params)
        
        logger.info("Assistant stopped")
    
    def _start_listening(self):
        """Start listening for voice input"""
        logger.info("Starting speech recognition...")
        
        def speech_callback(text):
            """Callback function for speech recognition"""
            if text and not self.processing_input:
                print(f"\nHeard: {text}")
                self._process_input(text)
                # Record activity in context manager
                self.context.record_activity()
        
        # Start recording with callback
        silence_threshold = self.config.get("stt", {}).get("silence_threshold_sec", 1.0)
        max_recording = self.config.get("stt", {}).get("max_recording_sec", 30.0)
        
        self.stt.start_recording(
            callback=speech_callback,
            silence_threshold_sec=silence_threshold,
            max_recording_sec=max_recording
        )
    
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
        
        # Apply personality style
        styled_text = self.personality.adjust_text_style(initiative_text)
        
        # Display the initiative
        print(f"\n{self.profile.get_name()}: {styled_text}")
        
        # Speak the initiative
        voice_params = self.emotions.get_voice_parameters()
        self.tts.speak(styled_text, voice_params=voice_params)
    
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
        # In a more sophisticated implementation, this would use NLP
        # to extract meaningful topics
        
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
            
            # Check for name change requests
            new_name = self._extract_name_change(user_input)
            if new_name:
                old_name = self.profile.get_name()
                if self.profile.set_name(new_name):
                    response = f"I'll respond to {new_name} from now on."
                    response = self.personality.adjust_text_style(response)
                    print(f"{new_name}: {response}")
                    voice_params = self.emotions.get_voice_parameters()
                    self.tts.speak(response, voice_params=voice_params)
                    
                    # Update conversation history
                    self.memory.add_message(self.conversation_id, "user", user_input)
                    self.memory.add_message(self.conversation_id, "assistant", response)
                    
                    # Update AI system prompt
                    system_prompt = self._create_system_prompt("")
                    self.ai.system_prompt = system_prompt
                    return
            
            # Check for user name mentions
            user_name = self._extract_user_name(user_input)
            if user_name:
                self.profile.set_user_name(user_name)
                
                # This will be handled by the AI model, so we let it continue
                # processing below rather than returning early
            
            # Extract potential topics for context
            topics = self._extract_topics(user_input)
            if topics:
                self.context.update_current_session(1, topics)
                
                # Add user interests based on frequent topics
                for topic in topics[:2]:  # Only consider top 2 topics
                    if topic in user_input.lower().split():
                        self.context.add_user_interest(topic)
            
            # Update emotional state based on user input
            self.emotions.process_message(user_input, source="user")
            
            # Update personality based on user input
            self.personality.process_message(user_input, source="user")
            
            # Check for emotion-specific commands
            if user_input.lower() in ["how are you feeling", "what's your mood", "how do you feel"]:
                emotion_explanation = self.emotions.get_emotion_reason()
                styled_explanation = self.personality.adjust_text_style(emotion_explanation)
                print(f"{self.profile.get_name()}: {styled_explanation}")
                voice_params = self.emotions.get_voice_parameters()
                self.tts.speak(styled_explanation, voice_params=voice_params)
                
                # Add to conversation history
                self.memory.add_message(self.conversation_id, "user", user_input)
                self.memory.add_message(self.conversation_id, "assistant", emotion_explanation)
                return
            
            # Look for relevant context from past conversations
            relevant_topics = self.context.get_relevant_topics(user_input)
            
            # Add context to the AI request if relevant topics found
            context_addition = ""
            if relevant_topics:
                context_addition = "Based on our previous conversations: "
                for topic in relevant_topics:
                    context_addition += f"{topic['topic']}: {topic['content']} "
                
                # Use this to augment the AI's knowledge but don't need to
                # explicitly add it to the user input
                
            # Add user message to memory
            self.memory.add_message(self.conversation_id, "user", user_input)
            
            # Process with AI and get response
            print(f"{self.profile.get_name()} is thinking...")
            
            def streaming_callback(chunk):
                """Callback for streaming response"""
                print(chunk, end="", flush=True)
            
            # Generate streaming response
            # If we have context, we could add it to the prompt,
            # but modern LLMs should maintain context through the conversation history
            response = self.ai.generate_response(
                user_input + (f"\n\nRecall from previous conversations: {context_addition}" if context_addition else ""),
                stream=True,
                stream_callback=streaming_callback
            )
            
            print()  # Add newline after streaming response
            
            # Apply personality styling to the response
            styled_response = self.personality.adjust_text_style(response)
            
            # If the styling changed the response significantly, display it differently
            if styled_response != response:
                print(f"{self.profile.get_name()}: {styled_response}")
                response = styled_response
            
            # Update emotional state based on AI response
            self.emotions.process_message(response, source="assistant")
            
            # Add assistant response to memory
            self.memory.add_message(self.conversation_id, "assistant", response)
            
            # Apply emotional voice parameters for the response
            voice_params = self.emotions.get_voice_parameters()
            self.tts.speak(response, voice_params=voice_params)
            
            # Update context with the message
            self.context.record_activity()
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
                            response = f"Voice changed to {voice_name}"
                            voice_params = self.emotions.get_voice_parameters()
                            self.tts.speak(response, voice_params=voice_params)
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
                            response = f"Model changed to {model_name}"
                            voice_params = self.emotions.get_voice_parameters()
                            self.tts.speak(response, voice_params=voice_params)
                        else:
                            print(f"Model '{model_name}' not available")
                    else:
                        # List available models
                        models = self.ai.list_available_models()
                        print("Available models:")
                        for model in models:
                            print(f"- {model['name']}")
                
                elif cmd == "name":
                    # Change assistant name
                    if args:
                        new_name = args[0]
                        old_name = self.profile.get_name()
                        logger.info(f"Changing name from {old_name} to {new_name}")
                        if self.profile.set_name(new_name):
                            response = f"My name is now {new_name}."
                            response = self.personality.adjust_text_style(response)
                            print(f"{new_name}: {response}")
                            voice_params = self.emotions.get_voice_parameters()
                            self.tts.speak(response, voice_params=voice_params)
                            
                            # Update AI system prompt
                            system_prompt = self._create_system_prompt("")
                            self.ai.system_prompt = system_prompt
                        else:
                            print(f"Failed to change name to '{new_name}'")
                    else:
                        # Show current name
                        name = self.profile.get_name()
                        print(f"My current name is {name}")
                
                elif cmd == "user":
                    # Set user name
                    if args:
                        user_name = args[0]
                        logger.info(f"Setting user name to {user_name}")
                        if self.profile.set_user_name(user_name):
                            response = f"I'll call you {user_name} from now on."
                            response = self.personality.adjust_text_style(response)
                            print(f"{self.profile.get_name()}: {response}")
                            voice_params = self.emotions.get_voice_parameters()
                            self.tts.speak(response, voice_params=voice_params)
                            
                            # Update AI system prompt
                            system_prompt = self._create_system_prompt("")
                            self.ai.system_prompt = system_prompt
                        else:
                            print(f"Failed to set user name to '{user_name}'")
                    else:
                        # Show current user name
                        user_name = self.profile.get_user_name()
                        if user_name:
                            print(f"I'm currently calling you {user_name}")
                        else:
                            print("I don't know your name yet")
                
                elif cmd == "personality":
                    # Personality management
                    if not args:
                        # Show current personality
                        print(self.personality.get_personality_description())
                    elif args[0] == "set" and len(args) >= 3:
                        # Set personality trait
                        trait = args[1]
                        try:
                            value = float(args[2])
                            value = max(0.0, min(1.0, value))  # Ensure value is between 0 and 1
                            if self.personality.set_trait(trait, value):
                                response = f"Personality trait '{trait}' set to {value:.2f}."
                                print(response)
                                voice_params = self.emotions.get_voice_parameters()
                                self.tts.speak(response, voice_params=voice_params)
                                
                                # Update AI system prompt
                                system_prompt = self._create_system_prompt("")
                                self.ai.system_prompt = system_prompt
                            else:
                                print(f"Failed to set personality trait '{trait}'")
                        except ValueError:
                            print(f"Invalid value for personality trait. Must be a number between 0 and 1")
                
                elif cmd == "emotion":
                    # Emotion management
                    if not args:
                        # Show current emotion info
                        current_emotion = self.emotions.current_emotion
                        current_intensity = self.emotions.current_intensity
                        print(f"Current emotion: {current_emotion} (intensity: {current_intensity:.2f})")
                        print(self.emotions.get_emotion_reason())
                    elif args[0] == "set" and len(args) > 1:
                        # Set emotion manually
                        emotion = args[1]
                        intensity = float(args[2]) if len(args) > 2 else 0.7
                        self.emotions.set_emotion(emotion, intensity)
                        print(f"Emotion set to {emotion} (intensity: {intensity:.2f})")
                        response = f"I'm now feeling {emotion}."
                        response = self.personality.adjust_text_style(response)
                        voice_params = self.emotions.get_voice_parameters()
                        self.tts.speak(response, voice_params=voice_params)
                    elif args[0] == "reset":
                        # Reset to neutral
                        self.emotions.reset_emotion()
                        print("Emotion reset to neutral")
                        response = "I've reset my emotional state to neutral."
                        response = self.personality.adjust_text_style(response)
                        voice_params = self.emotions.get_voice_parameters()
                        self.tts.speak(response, voice_params=voice_params)
                    elif args[0] == "triggers":
                        # List emotion triggers
                        triggers = self.emotions.get_all_triggers()
                        print("Emotion triggers:")
                        for emotion, trigger_list in triggers.items():
                            print(f"- {emotion}: {', '.join(trigger_list[:5])}{'...' if len(trigger_list) > 5 else ''}")
                
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
                    response = f"I took a screenshot. {description}"
                    response = self.personality.adjust_text_style(response)
                    voice_params = self.emotions.get_voice_parameters()
                    self.tts.speak(response, voice_params=voice_params)
                
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
                                response = f"Switched to conversation: {conversation.get('title')}"
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
                        response = "Created a new conversation"
                        voice_params = self.emotions.get_voice_parameters()
                        self.tts.speak(response, voice_params=voice_params)
                
                elif cmd == "voice_model":
                    # Change TTS voice model
                    if args:
                        model_name = args[0]
                        logger.info(f"Changing TTS model to {model_name}")
                        if self.tts.change_model(model_name):
                            response = f"Voice model changed to {model_name}"
                            voice_params = self.emotions.get_voice_parameters()
                            self.tts.speak(response, voice_params=voice_params)
                        else:
                            print(f"Voice model '{model_name}' not available")
                    else:
                        # List available voice models
                        models = self.tts.list_available_models()
                        print("Available voice models:")
                        for model in models:
                            print(f"- {model}")
                
                elif cmd == "help":
                    # Show help information
                    print("Available commands:")
                    print("  !exit, !quit - Exit the assistant")
                    print("  !voice [name] - Change voice or list available voices")
                    print("  !voice_model [name] - Change TTS model or list available models")
                    print("  !model [name] - Change AI model or list available models")
                    print("  !name [new_name] - Change assistant name")
                    print("  !user [name] - Set or show user name")
                    print("  !personality - Show current personality")
                    print("  !personality set <trait> <value> - Set personality trait (0-1)")
                    print("  !screenshot - Take and analyze a screenshot")
                    print("  !webcam - Capture and analyze from webcam")
                    print("  !memory - Show current conversation info")
                    print("  !memory list - List recent conversations")
                    print("  !memory switch <id> - Switch to a different conversation")
                    print("  !memory new [title] - Create a new conversation")
                    print("  !emotion - Show current emotion status")
                    print("  !emotion set <emotion> [intensity] - Set emotion manually")
                    print("  !emotion reset - Reset to neutral emotion")
                    print("  !emotion triggers - List emotion triggers")
                    print("  !help - Show this help information")
                
                else:
                    print(f"Unknown command: {cmd}")
                
            except queue.Empty:
                # No commands in queue, continue
                pass
            except Exception as e:
                logger.error(f"Error processing command: {e}")
