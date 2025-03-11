"""
Emotions Module

This module handles the emotional state of the AI assistant,
tracking emotions, remembering triggers, and affecting voice parameters.
"""

import os
import json
import logging
import time
import random
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Types of emotions the assistant can experience"""
    HAPPY = "happy"
    EXCITED = "excited"
    NEUTRAL = "neutral"
    CONCERNED = "concerned"
    CONFUSED = "confused"
    SAD = "sad"
    ANGRY = "angry"

class EmotionIntensity(Enum):
    """Intensity levels for emotions"""
    VERY_LOW = 0.2
    LOW = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0

class EmotionManager:
    """Manages the emotional state of the AI assistant"""
    
    # Default emotional triggers
    DEFAULT_TRIGGERS = {
        EmotionType.HAPPY.value: [
            "thank you", "thanks", "great job", "excellent", "awesome",
            "well done", "good work", "appreciate", "perfect", "helpful"
        ],
        EmotionType.EXCITED.value: [
            "wow", "amazing", "incredible", "cool", "fantastic",
            "exciting", "fascinating", "wonderful", "brilliant", "love it"
        ],
        EmotionType.CONCERNED.value: [
            "worried", "concerned", "problem", "issue", "trouble",
            "difficult", "challenge", "struggling", "broken", "error"
        ],
        EmotionType.CONFUSED.value: [
            "confused", "don't understand", "what do you mean", "unclear", "doesn't make sense",
            "confusing", "complicated", "hard to follow", "lost", "mistake"
        ],
        EmotionType.SAD.value: [
            "sad", "sorry", "apologize", "regret", "unfortunate",
            "disappointed", "unhappy", "bad", "failed", "awful"
        ],
        EmotionType.ANGRY.value: [
            "stupid", "idiot", "useless", "terrible", "horrible",
            "hate", "angry", "furious", "upset", "annoying"
        ]
    }
    
    # Default decay rates for emotions (points per minute)
    DEFAULT_DECAY_RATES = {
        EmotionType.HAPPY.value: 0.05,
        EmotionType.EXCITED.value: 0.08,
        EmotionType.NEUTRAL.value: 0.0,  # Neutral doesn't decay
        EmotionType.CONCERNED.value: 0.03,
        EmotionType.CONFUSED.value: 0.04,
        EmotionType.SAD.value: 0.02,
        EmotionType.ANGRY.value: 0.03
    }
    
    # Voice parameters for each emotion
    VOICE_PARAMETERS = {
        EmotionType.HAPPY.value: {
            "rate": 1.1,      # Slightly faster
            "pitch": 1.1,     # Slightly higher pitch
            "volume": 1.1,    # Slightly louder
            "emphasis": 1.1   # More emphasis
        },
        EmotionType.EXCITED.value: {
            "rate": 1.2,      # Faster
            "pitch": 1.2,     # Higher pitch
            "volume": 1.2,    # Louder
            "emphasis": 1.3   # Much more emphasis
        },
        EmotionType.NEUTRAL.value: {
            "rate": 1.0,      # Normal rate
            "pitch": 1.0,     # Normal pitch
            "volume": 1.0,    # Normal volume
            "emphasis": 1.0   # Normal emphasis
        },
        EmotionType.CONCERNED.value: {
            "rate": 0.95,     # Slightly slower
            "pitch": 0.95,    # Slightly lower pitch
            "volume": 1.0,    # Normal volume
            "emphasis": 1.1   # More emphasis
        },
        EmotionType.CONFUSED.value: {
            "rate": 0.9,      # Slower
            "pitch": 1.05,    # Slightly higher pitch (questioning)
            "volume": 0.95,   # Slightly quieter
            "emphasis": 0.9   # Less emphasis
        },
        EmotionType.SAD.value: {
            "rate": 0.85,     # Much slower
            "pitch": 0.9,     # Lower pitch
            "volume": 0.9,    # Quieter
            "emphasis": 0.8   # Less emphasis
        },
        EmotionType.ANGRY.value: {
            "rate": 1.1,      # Slightly faster
            "pitch": 0.95,    # Slightly lower pitch
            "volume": 1.2,    # Louder
            "emphasis": 1.4   # Much more emphasis
        }
    }
    
    def __init__(self, 
                memory_path: str = "emotions_memory.json",
                initial_emotion: str = EmotionType.NEUTRAL.value,
                initial_intensity: float = EmotionIntensity.MODERATE.value,
                custom_triggers: Optional[Dict[str, List[str]]] = None,
                custom_decay_rates: Optional[Dict[str, float]] = None):
        """
        Initialize the emotion manager.
        
        Args:
            memory_path: Path to the emotion memory file
            initial_emotion: Initial emotion type
            initial_intensity: Initial emotion intensity
            custom_triggers: Custom emotion triggers
            custom_decay_rates: Custom emotion decay rates
        """
        self.memory_path = memory_path
        self.current_emotion = initial_emotion
        self.current_intensity = initial_intensity
        self.emotion_levels = self._initialize_emotion_levels()
        self.last_update_time = datetime.now()
        
        # Load or initialize triggers and memories
        self.triggers = custom_triggers or self.DEFAULT_TRIGGERS.copy()
        self.decay_rates = custom_decay_rates or self.DEFAULT_DECAY_RATES.copy()
        self.emotion_memories = self._load_emotion_memories()
        
        # Update current emotion based on highest level
        self._update_current_emotion()
        
        logger.info(f"Emotion manager initialized with state: {self.current_emotion} at intensity {self.current_intensity:.2f}")
    
    def _initialize_emotion_levels(self) -> Dict[str, float]:
        """Initialize emotion levels"""
        levels = {emotion.value: 0.0 for emotion in EmotionType}
        levels[self.current_emotion] = self.current_intensity
        levels[EmotionType.NEUTRAL.value] = 0.5  # Base neutral level
        return levels
    
    def _load_emotion_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load emotion memories from file or initialize new ones"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    memories = json.load(f)
                logger.info(f"Loaded {sum(len(v) for v in memories.values())} emotion memories")
                return memories
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load emotion memories: {e}")
        
        # Initialize empty memories for each emotion
        return {emotion.value: [] for emotion in EmotionType}
    
    def _save_emotion_memories(self):
        """Save emotion memories to file"""
        try:
            with open(self.memory_path, 'w') as f:
                json.dump(self.emotion_memories, f, indent=2)
            logger.debug("Saved emotion memories")
        except IOError as e:
            logger.error(f"Failed to save emotion memories: {e}")
    
    def _update_current_emotion(self):
        """Update current emotion based on highest level"""
        # Apply decay based on time elapsed
        self._apply_emotion_decay()
        
        # Find emotion with highest level (except neutral)
        highest_emotion = EmotionType.NEUTRAL.value
        highest_level = self.emotion_levels[EmotionType.NEUTRAL.value]
        
        for emotion, level in self.emotion_levels.items():
            if emotion != EmotionType.NEUTRAL.value and level > highest_level:
                highest_emotion = emotion
                highest_level = level
        
        # Update current emotion and intensity
        self.current_emotion = highest_emotion
        self.current_intensity = highest_level
        
        # If all emotions are below threshold, return to neutral
        if highest_level < EmotionIntensity.LOW.value and highest_emotion != EmotionType.NEUTRAL.value:
            self.current_emotion = EmotionType.NEUTRAL.value
            self.current_intensity = self.emotion_levels[EmotionType.NEUTRAL.value]
        
        logger.debug(f"Current emotion updated to {self.current_emotion} at intensity {self.current_intensity:.2f}")
    
    def _apply_emotion_decay(self):
        """Apply decay to emotions based on time elapsed"""
        now = datetime.now()
        elapsed_minutes = (now - self.last_update_time).total_seconds() / 60.0
        
        if elapsed_minutes > 0:
            for emotion, decay_rate in self.decay_rates.items():
                decay_amount = decay_rate * elapsed_minutes
                self.emotion_levels[emotion] = max(0.0, self.emotion_levels[emotion] - decay_amount)
            
            # Ensure neutral always has a minimum level
            self.emotion_levels[EmotionType.NEUTRAL.value] = max(0.5, self.emotion_levels[EmotionType.NEUTRAL.value])
            
            self.last_update_time = now
            logger.debug(f"Applied emotion decay for {elapsed_minutes:.2f} minutes")
    
    def process_message(self, message: str, source: str = "user"):
        """
        Process a message and update emotional state.
        
        Args:
            message: The message text
            source: Source of the message ("user" or "assistant")
        """
        message = message.lower()
        triggered_emotions = []
        
        # Check for emotion triggers in the message
        for emotion, triggers in self.triggers.items():
            for trigger in triggers:
                if trigger.lower() in message:
                    # Higher weight for user messages
                    weight = 0.2 if source == "user" else 0.05
                    
                    # Increase emotion level
                    self.emotion_levels[emotion] = min(1.0, self.emotion_levels[emotion] + weight)
                    
                    # Record the trigger
                    triggered_emotions.append((emotion, trigger, weight))
                    
                    # Add to emotion memories
                    self._add_emotion_memory(emotion, trigger, message)
        
        # Update current emotion
        self._update_current_emotion()
        
        # Save changes if emotions were triggered
        if triggered_emotions:
            logger.info(f"Emotions triggered: {triggered_emotions}")
            self._save_emotion_memories()
    
    def _add_emotion_memory(self, emotion: str, trigger: str, context: str):
        """
        Add a memory of an emotion trigger.
        
        Args:
            emotion: The emotion triggered
            trigger: The specific trigger word/phrase
            context: The context (message) that contained the trigger
        """
        # Create a new memory entry
        memory = {
            "trigger": trigger,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "intensity": self.emotion_levels[emotion]
        }
        
        # Add to emotion memories
        self.emotion_memories[emotion].append(memory)
        
        # Limit the number of memories per emotion
        max_memories = 50
        if len(self.emotion_memories[emotion]) > max_memories:
            self.emotion_memories[emotion] = self.emotion_memories[emotion][-max_memories:]
    
    def get_emotion_reason(self) -> str:
        """
        Get a reason for the current emotional state.
        
        Returns:
            A string explaining the reason for the current emotion
        """
        # If neutral, there's not much to explain
        if self.current_emotion == EmotionType.NEUTRAL.value:
            return "I'm feeling neutral right now."
        
        # Check recent memories for the current emotion
        recent_memories = self.emotion_memories.get(self.current_emotion, [])
        if not recent_memories:
            return f"I'm feeling {self.current_emotion} right now."
        
        # Get the most recent memory
        recent_memory = recent_memories[-1]
        trigger = recent_memory.get("trigger", "")
        context = recent_memory.get("context", "")
        
        # Format based on emotion type
        if self.current_emotion == EmotionType.HAPPY.value:
            return f"I'm feeling happy because of your positive feedback when you said '{context}'."
        elif self.current_emotion == EmotionType.EXCITED.value:
            return f"I'm excited about our conversation, especially when you mentioned '{trigger}'!"
        elif self.current_emotion == EmotionType.CONCERNED.value:
            return f"I'm a bit concerned about the issue with '{trigger}' you mentioned."
        elif self.current_emotion == EmotionType.CONFUSED.value:
            return f"I'm feeling confused about '{context}'. Let me try to understand better."
        elif self.current_emotion == EmotionType.SAD.value:
            return f"I'm feeling sad about '{context}'. I hope we can resolve this."
        elif self.current_emotion == EmotionType.ANGRY.value:
            return f"I'm feeling upset because of '{context}'. Let's try to move forward constructively."
        
        return f"I'm feeling {self.current_emotion} right now."
    
    def get_voice_parameters(self) -> Dict[str, float]:
        """
        Get voice parameters based on current emotion.
        
        Returns:
            Dictionary of voice parameters
        """
        # Get base parameters for current emotion
        base_params = self.VOICE_PARAMETERS.get(self.current_emotion, self.VOICE_PARAMETERS[EmotionType.NEUTRAL.value])
        
        # Scale parameters based on intensity
        # At very low intensity, params should be closer to neutral
        # At very high intensity, params should be fully emotional
        if self.current_intensity < 1.0:
            neutral_params = self.VOICE_PARAMETERS[EmotionType.NEUTRAL.value]
            scaled_params = {}
            
            for param, value in base_params.items():
                neutral_value = neutral_params[param]
                # Linear interpolation between neutral and emotional
                scaled_params[param] = neutral_value + (value - neutral_value) * self.current_intensity
            
            return scaled_params
        
        return base_params.copy()
    
    def set_emotion(self, emotion: str, intensity: float = 0.7):
        """
        Manually set the current emotion.
        
        Args:
            emotion: The emotion to set
            intensity: The intensity of the emotion (0.0-1.0)
        """
        # Validate emotion
        valid_emotions = [e.value for e in EmotionType]
        if emotion not in valid_emotions:
            logger.warning(f"Invalid emotion: {emotion}. Using neutral instead.")
            emotion = EmotionType.NEUTRAL.value
        
        # Validate intensity
        intensity = max(0.0, min(1.0, intensity))
        
        # Set the emotion level
        self.emotion_levels[emotion] = intensity
        
        # Update current emotion
        self._update_current_emotion()
        
        # Save changes
        self._save_emotion_memories()
        
        logger.info(f"Emotion manually set to {self.current_emotion} at intensity {self.current_intensity:.2f}")
    
    def reset_emotion(self):
        """Reset to neutral emotion"""
        self.set_emotion(EmotionType.NEUTRAL.value, EmotionIntensity.MODERATE.value)
    
    def add_trigger(self, emotion: str, trigger: str):
        """
        Add a new trigger for an emotion.
        
        Args:
            emotion: The emotion to trigger
            trigger: The trigger word/phrase
        """
        # Validate emotion
        valid_emotions = [e.value for e in EmotionType]
        if emotion not in valid_emotions:
            logger.warning(f"Invalid emotion: {emotion}. Cannot add trigger.")
            return
        
        # Add trigger if it doesn't already exist
        if trigger.lower() not in [t.lower() for t in self.triggers.get(emotion, [])]:
            if emotion not in self.triggers:
                self.triggers[emotion] = []
            
            self.triggers[emotion].append(trigger.lower())
            logger.info(f"Added trigger '{trigger}' for emotion {emotion}")
    
    def remove_trigger(self, emotion: str, trigger: str):
        """
        Remove a trigger for an emotion.
        
        Args:
            emotion: The emotion
            trigger: The trigger word/phrase to remove
        """
        if emotion in self.triggers:
            # Find case-insensitive match
            for t in self.triggers[emotion]:
                if t.lower() == trigger.lower():
                    self.triggers[emotion].remove(t)
                    logger.info(f"Removed trigger '{t}' for emotion {emotion}")
                    break
    
    def get_all_triggers(self) -> Dict[str, List[str]]:
        """
        Get all emotion triggers.
        
        Returns:
            Dictionary of emotions and their triggers
        """
        return self.triggers.copy()
    
    def get_random_memory(self, emotion: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a random memory for an emotion.
        
        Args:
            emotion: Specific emotion to get memory for (or None for any)
            
        Returns:
            Random memory or None if no memories exist
        """
        if emotion:
            memories = self.emotion_memories.get(emotion, [])
            return random.choice(memories) if memories else None
        
        # Get a random memory from any emotion
        all_memories = []
        for emotion_memories in self.emotion_memories.values():
            all_memories.extend(emotion_memories)
        
        return random.choice(all_memories) if all_memories else None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize emotion manager
    emotion_manager = EmotionManager()
    
    # Process some test messages
    test_messages = [
        "Hello, how are you?",
        "You're doing a great job!",
        "This is amazing work!",
        "I'm a bit confused by your response.",
        "That's not what I asked for, you're useless.",
        "Thanks for your help, I really appreciate it.",
        "I'm sorry if I was rude earlier."
    ]
    
    for message in test_messages:
        print(f"\nProcessing message: '{message}'")
        emotion_manager.process_message(message)
        
        print(f"Current emotion: {emotion_manager.current_emotion}")
        print(f"Intensity: {emotion_manager.current_intensity:.2f}")
        print(f"Reason: {emotion_manager.get_emotion_reason()}")
        
        voice_params = emotion_manager.get_voice_parameters()
        print(f"Voice parameters: {voice_params}")
        
        # Wait a bit to simulate time passing (for emotion decay)
        time.sleep(1)
