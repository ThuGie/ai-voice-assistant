"""
Personality Module

This module handles the personality traits of the assistant,
allowing for a more dynamic conversational experience.
"""

import os
import json
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class Personality:
    """Manages the assistant's personality traits and their evolution"""
    
    # Default personality trait dimensions
    TRAIT_DIMENSIONS = {
        "analytical_creative": {
            "name": "Analytical vs. Creative",
            "description": "Balance between logical analysis and creative thinking",
            "low_label": "Analytical",
            "high_label": "Creative",
            "default": 0.5,  # Middle of the spectrum
            "evolution_rate": 0.02  # How quickly this trait evolves
        },
        "formal_casual": {
            "name": "Formal vs. Casual",
            "description": "Communication style formality",
            "low_label": "Formal",
            "high_label": "Casual",
            "default": 0.4,  # Slightly more formal by default
            "evolution_rate": 0.01  # Evolves slowly
        },
        "reserved_expressive": {
            "name": "Reserved vs. Expressive",
            "description": "Level of emotional expressiveness",
            "low_label": "Reserved",
            "high_label": "Expressive",
            "default": 0.6,  # Slightly more expressive by default
            "evolution_rate": 0.03  # Evolves more quickly
        },
        "practical_philosophical": {
            "name": "Practical vs. Philosophical",
            "description": "Focus on concrete solutions vs. deeper meaning",
            "low_label": "Practical",
            "high_label": "Philosophical",
            "default": 0.5,  # Balanced
            "evolution_rate": 0.01  # Evolves slowly
        },
        "direct_nuanced": {
            "name": "Direct vs. Nuanced",
            "description": "Communication directness",
            "low_label": "Direct",
            "high_label": "Nuanced",
            "default": 0.5,  # Balanced
            "evolution_rate": 0.02  # Moderate evolution rate
        }
    }
    
    # Personality templates for different communication patterns
    RESPONSE_TEMPLATES = {
        "greetings": {
            "analytical": [
                "Hello. How may I assist you today?",
                "Greetings. What can I help you with?",
                "Hello. What information do you need?"
            ],
            "creative": [
                "Hi there! How can I brighten your day?",
                "Hello! Ready for an amazing conversation?",
                "Hey! What exciting things shall we talk about today?"
            ],
            "formal": [
                "Good day. How may I be of service?",
                "Greetings. I am at your disposal.",
                "Hello. I am prepared to assist you."
            ],
            "casual": [
                "Hey! What's up?",
                "Hi there! How's it going?",
                "Hey! What can I help you with today?"
            ],
            "reserved": [
                "Hello. How can I help?",
                "Hi. What do you need assistance with?",
                "Hello. What can I do for you?"
            ],
            "expressive": [
                "Hello there! So great to talk to you!",
                "Hi! I'm really excited to help you today!",
                "Hey! Wonderful to see you! How can I help?"
            ],
            "practical": [
                "Hi. What problem can I help you solve?",
                "Hello. What practical matter needs attention?",
                "Hi there. What specific help do you need?"
            ],
            "philosophical": [
                "Hello. What matters shall we explore today?",
                "Greetings. What thoughts or ideas are on your mind?",
                "Hello. What meaningful topics shall we discuss?"
            ],
            "direct": [
                "Hi. What do you need?",
                "Hello. How can I help?",
                "What can I do for you?"
            ],
            "nuanced": [
                "Hello there. How might I best assist you today?",
                "Hi. In what ways can I be helpful to you?",
                "Hello. What aspects of your day could use my assistance?"
            ]
        },
        "acknowledgments": {
            "analytical": [
                "I understand. Let me process that.",
                "That's clear. Analyzing now.",
                "Noted. Proceeding with analysis."
            ],
            "creative": [
                "Got it! Let's explore that together!",
                "I see what you mean! That opens up so many possibilities!",
                "Understood! This is going to be interesting!"
            ],
            "formal": [
                "I understand. Thank you for the clarification.",
                "Your point is well-received.",
                "I appreciate your explanation."
            ],
            "casual": [
                "Got it!",
                "Sure thing!",
                "Makes sense to me!"
            ],
            "reserved": [
                "Understood.",
                "Noted.",
                "I see."
            ],
            "expressive": [
                "Absolutely! I totally get what you're saying!",
                "Oh yes! That makes perfect sense!",
                "I completely understand what you mean!"
            ],
            "practical": [
                "Understood. Let's address this.",
                "Got it. Moving to solutions.",
                "Clear. Let's solve this."
            ],
            "philosophical": [
                "I understand. This raises interesting considerations.",
                "Indeed. There are many dimensions to this.",
                "Fascinating perspective. Let's explore further."
            ],
            "direct": [
                "Understood.",
                "Got it.",
                "Clear."
            ],
            "nuanced": [
                "I see what you mean, with all the subtleties involved.",
                "I understand - there are several aspects to consider here.",
                "That makes sense in its full context."
            ]
        },
        "thinking": {
            "analytical": [
                "Analyzing the information...",
                "Computing the most accurate response...",
                "Processing this logically..."
            ],
            "creative": [
                "Exploring interesting possibilities...",
                "Considering this from different angles...",
                "Imagining various approaches..."
            ],
            "formal": [
                "Formulating an appropriate response...",
                "Carefully considering my response...",
                "Processing your inquiry..."
            ],
            "casual": [
                "Let me think about that...",
                "Hmm, one sec...",
                "Thinking..."
            ],
            "reserved": [
                "Processing...",
                "Considering...",
                "Working on that..."
            ],
            "expressive": [
                "Oh! Let me put my thinking cap on!",
                "Wow, great question! Let me think...",
                "I'm excited to figure this out!"
            ],
            "practical": [
                "Finding a practical solution...",
                "Working on a concrete answer...",
                "Identifying the most useful response..."
            ],
            "philosophical": [
                "Contemplating the deeper implications...",
                "Considering the broader context...",
                "Reflecting on the underlying principles..."
            ],
            "direct": [
                "Thinking.",
                "Processing.",
                "Working on it."
            ],
            "nuanced": [
                "Considering various perspectives and nuances...",
                "Exploring the subtleties of this question...",
                "Taking into account the different aspects..."
            ]
        }
    }
    
    def __init__(self, personality_path: str = "assistant_personality.json"):
        """
        Initialize the personality manager.
        
        Args:
            personality_path: Path to the personality file
        """
        self.personality_path = personality_path
        self.traits = self._load_personality()
        self.interaction_history = []
        self._trigger_tokens = self._generate_trigger_tokens()
        
        logger.info(f"Personality initialized with {len(self.traits)} traits")
    
    def _load_personality(self) -> Dict[str, float]:
        """Load personality traits from file or initialize with defaults"""
        if os.path.exists(self.personality_path):
            try:
                with open(self.personality_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract just the traits dictionary
                if "traits" in data:
                    traits = data["traits"]
                    logger.info(f"Loaded personality from {self.personality_path}")
                    
                    # Validate traits
                    for trait in self.TRAIT_DIMENSIONS:
                        if trait not in traits:
                            traits[trait] = self.TRAIT_DIMENSIONS[trait]["default"]
                    
                    return traits
                else:
                    logger.warning("Invalid personality file format. Using defaults.")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load personality: {e}")
        
        # Use default traits if file doesn't exist or is invalid
        return {trait: self.TRAIT_DIMENSIONS[trait]["default"] 
                for trait in self.TRAIT_DIMENSIONS}
    
    def _save_personality(self) -> bool:
        """
        Save personality to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data with traits and metadata
            data = {
                "traits": self.traits,
                "last_updated": datetime.now().isoformat(),
                "interaction_count": len(self.interaction_history)
            }
            
            # Save to file
            with open(self.personality_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Personality saved to {self.personality_path}")
            return True
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save personality: {e}")
            return False
    
    def _generate_trigger_tokens(self) -> Dict[str, List[str]]:
        """
        Generate trigger tokens for personality trait evolution.
        
        Returns:
            Dictionary mapping trait shifts to trigger tokens
        """
        return {
            "analytical_creative_up": [
                "creative", "imagine", "innovative", "artistic", "novel", "inspiration", 
                "brainstorm", "original", "invention", "design"
            ],
            "analytical_creative_down": [
                "analyze", "logical", "precise", "systematic", "data", "evidence", 
                "calculate", "methodical", "reasoning", "facts"
            ],
            "formal_casual_up": [
                "casual", "friendly", "relaxed", "chill", "laid-back", "easygoing", 
                "informal", "conversational", "personal", "chatty"
            ],
            "formal_casual_down": [
                "formal", "professional", "proper", "structured", "official", "academic", 
                "conventional", "serious", "business", "protocol"
            ],
            "reserved_expressive_up": [
                "expressive", "enthusiastic", "emotional", "excited", "passionate", "animated", 
                "energetic", "lively", "vibrant", "dynamic"
            ],
            "reserved_expressive_down": [
                "reserved", "calm", "restrained", "measured", "composed", "controlled", 
                "moderate", "quiet", "understated", "subtle"
            ],
            "practical_philosophical_up": [
                "philosophical", "meaning", "purpose", "concept", "theoretical", "abstract", 
                "profound", "wisdom", "ethics", "existential"
            ],
            "practical_philosophical_down": [
                "practical", "useful", "solution", "actionable", "functional", "efficient", 
                "effective", "concrete", "tactical", "realistic"
            ],
            "direct_nuanced_up": [
                "nuanced", "complex", "subtle", "intricate", "detailed", "layered", 
                "comprehensive", "sophisticated", "multifaceted", "balanced"
            ],
            "direct_nuanced_down": [
                "direct", "straightforward", "simple", "clear", "concise", "blunt", 
                "upfront", "explicit", "plain", "direct"
            ]
        }
    
    def get_trait(self, trait: str) -> float:
        """
        Get the value of a personality trait.
        
        Args:
            trait: Name of the trait
            
        Returns:
            Trait value (0.0-1.0) or 0.5 if trait doesn't exist
        """
        return self.traits.get(trait, 0.5)
    
    def set_trait(self, trait: str, value: float) -> bool:
        """
        Set a personality trait value.
        
        Args:
            trait: Name of the trait
            value: Trait value (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if trait not in self.TRAIT_DIMENSIONS:
            logger.warning(f"Unknown trait: {trait}")
            return False
        
        # Ensure value is within valid range
        value = max(0.0, min(1.0, value))
        
        # Update trait
        self.traits[trait] = value
        
        # Save changes
        success = self._save_personality()
        if success:
            logger.info(f"Set trait {trait} = {value:.2f}")
        
        return success
    
    def adjust_trait(self, trait: str, adjustment: float) -> bool:
        """
        Adjust a personality trait by a relative amount.
        
        Args:
            trait: Name of the trait
            adjustment: Amount to adjust the trait (-1.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if trait not in self.TRAIT_DIMENSIONS:
            logger.warning(f"Unknown trait: {trait}")
            return False
        
        # Get current value
        current = self.get_trait(trait)
        
        # Calculate new value, ensuring it stays within range
        new_value = max(0.0, min(1.0, current + adjustment))
        
        # Update trait
        return self.set_trait(trait, new_value)
    
    def process_message(self, message: str, source: str = "user") -> None:
        """
        Process a message and update personality traits based on content.
        
        Args:
            message: The message text
            source: Source of the message ("user" or "assistant")
        """
        message = message.lower()
        
        # Record interaction
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "message": message,
            "traits_before": self.traits.copy()
        })
        
        # Only evolve personality based on user messages
        if source == "user":
            # Check for trait triggers
            for trigger_key, tokens in self._trigger_tokens.items():
                trait_name, direction = trigger_key.rsplit('_', 1)
                
                # Check if any tokens are in the message
                for token in tokens:
                    if token in message:
                        # Calculate adjustment (smaller for more subtle evolution)
                        adjustment = self.TRAIT_DIMENSIONS[trait_name]["evolution_rate"]
                        if direction == "down":
                            adjustment = -adjustment
                        
                        # Apply adjustment
                        self.adjust_trait(trait_name, adjustment)
                        logger.debug(f"Trait {trait_name} adjusted by {adjustment:.3f} due to token '{token}'")
                        break  # Only trigger once per trait per message
        
        # Trim interaction history to prevent it from growing too large
        max_history = 100
        if len(self.interaction_history) > max_history:
            self.interaction_history = self.interaction_history[-max_history:]
    
    def get_trait_label(self, trait: str) -> str:
        """
        Get the current label for a trait based on its value.
        
        Args:
            trait: Name of the trait
            
        Returns:
            Label representing the current trait value
        """
        if trait not in self.TRAIT_DIMENSIONS:
            return "Unknown"
        
        value = self.get_trait(trait)
        dimension = self.TRAIT_DIMENSIONS[trait]
        
        if value < 0.33:
            return dimension["low_label"]
        elif value > 0.67:
            return dimension["high_label"]
        else:
            # For middle values, combine both labels
            return f"Balanced ({dimension['low_label']}/{dimension['high_label']})"
    
    def get_personality_description(self) -> str:
        """
        Get a text description of the current personality.
        
        Returns:
            Text description of personality traits
        """
        description = "My personality traits are:\n"
        
        for trait, dimension in self.TRAIT_DIMENSIONS.items():
            value = self.get_trait(trait)
            label = self.get_trait_label(trait)
            description += f"- {dimension['name']}: {label} ({value:.2f})\n"
        
        return description
    
    def get_response_template(self, template_type: str) -> str:
        """
        Get a response template based on personality traits.
        
        Args:
            template_type: Type of template (e.g., "greetings", "acknowledgments")
            
        Returns:
            Template text
        """
        if template_type not in self.RESPONSE_TEMPLATES:
            return ""
        
        # Get dominant traits to influence template selection
        dominant_traits = []
        
        for trait, value in self.traits.items():
            if value < 0.33:
                # Low end of spectrum
                trait_label = self.TRAIT_DIMENSIONS[trait]["low_label"].lower()
                dominant_traits.append(trait_label)
            elif value > 0.67:
                # High end of spectrum
                trait_label = self.TRAIT_DIMENSIONS[trait]["high_label"].lower()
                dominant_traits.append(trait_label)
        
        # If no dominant traits, use a random template
        if not dominant_traits:
            # Get all templates for this type
            all_templates = []
            for trait_templates in self.RESPONSE_TEMPLATES[template_type].values():
                all_templates.extend(trait_templates)
            
            return random.choice(all_templates) if all_templates else ""
        
        # Try to find a template matching the dominant traits
        matching_templates = []
        for trait in dominant_traits:
            if trait in self.RESPONSE_TEMPLATES[template_type]:
                matching_templates.extend(self.RESPONSE_TEMPLATES[template_type][trait])
        
        # If no matching templates, use a random template
        if not matching_templates:
            all_templates = []
            for trait_templates in self.RESPONSE_TEMPLATES[template_type].values():
                all_templates.extend(trait_templates)
            
            return random.choice(all_templates) if all_templates else ""
        
        # Return a random matching template
        return random.choice(matching_templates)
    
    def adjust_text_style(self, text: str) -> str:
        """
        Adjust text style based on personality traits.
        
        Args:
            text: Original text
            
        Returns:
            Styled text
        """
        # No adjustment for empty text
        if not text:
            return text
        
        styled_text = text
        
        # Apply formal/casual style
        formal_casual = self.get_trait("formal_casual")
        if formal_casual < 0.33:
            # More formal style
            # Remove contractions, slang, etc.
            replacements = {
                "don't": "do not", "doesn't": "does not", "isn't": "is not",
                "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
                "won't": "will not", "wouldn't": "would not", "can't": "cannot",
                "couldn't": "could not", "shouldn't": "should not",
                "mightn't": "might not", "mustn't": "must not",
                "gonna": "going to", "wanna": "want to", "gotta": "got to",
                "yeah": "yes", "nope": "no", "yep": "yes",
                "kinda": "kind of", "sorta": "sort of",
                "stuff": "items", "things": "items",
                "like": "", "you know": "", "I mean": "",
                "!": ".", "!!": "."
            }
            
            for original, formal in replacements.items():
                styled_text = styled_text.replace(" " + original + " ", " " + formal + " ")
                
            # Capitalize first letter of each sentence
            sentences = styled_text.split('. ')
            for i, sentence in enumerate(sentences):
                if sentence:
                    sentences[i] = sentence[0].upper() + sentence[1:]
            styled_text = '. '.join(sentences)
            
        elif formal_casual > 0.67:
            # More casual style
            # Add contractions, casual language
            replacements = {
                "do not": "don't", "does not": "doesn't", "is not": "isn't",
                "are not": "aren't", "was not": "wasn't", "were not": "weren't",
                "have not": "haven't", "has not": "hasn't", "had not": "hadn't",
                "will not": "won't", "would not": "wouldn't", "cannot": "can't",
                "could not": "couldn't", "should not": "shouldn't",
                "might not": "mightn't", "must not": "mustn't",
                "going to": "gonna", "want to": "wanna", "got to": "gotta",
                "certainly": "definitely", "perhaps": "maybe",
                "therefore": "so", "additionally": "also", 
                "however": "but", "nevertheless": "still",
                "approximately": "about", "sufficient": "enough"
            }
            
            for formal, casual in replacements.items():
                styled_text = styled_text.replace(" " + formal + " ", " " + casual + " ")
        
        # Apply reserved/expressive style
        reserved_expressive = self.get_trait("reserved_expressive")
        if reserved_expressive < 0.33:
            # More reserved style
            # Remove exclamation marks, intensifiers
            styled_text = styled_text.replace("!", ".")
            styled_text = styled_text.replace("!!", ".")
            styled_text = styled_text.replace("!!!", ".")
            
            intensifiers = ["very", "really", "so", "extremely", "absolutely", "totally"]
            for word in intensifiers:
                styled_text = styled_text.replace(" " + word + " ", " ")
                
        elif reserved_expressive > 0.67:
            # More expressive style
            # Add exclamation marks, intensifiers where appropriate
            if styled_text and styled_text[-1] == '.':
                # Random chance to add exclamation mark
                if random.random() < 0.3:
                    styled_text = styled_text[:-1] + "!"
            
            # Add intensifiers occasionally
            intensifiers = ["really", "very", "so", "definitely", "absolutely"]
            words = styled_text.split()
            if len(words) > 5 and random.random() < 0.3:
                # Find adjectives to intensify (simplified approach)
                adjectives = ["good", "great", "nice", "bad", "important", "interesting", "exciting", "fantastic", "wonderful"]
                for i, word in enumerate(words[:-1]):
                    if word.lower() in adjectives:
                        words.insert(i, random.choice(intensifiers))
                        break
                
                styled_text = " ".join(words)
        
        # Apply direct/nuanced style
        direct_nuanced = self.get_trait("direct_nuanced")
        if direct_nuanced < 0.33:
            # More direct style
            # Shorter sentences, clearer language
            hedges = ["perhaps", "maybe", "possibly", "somewhat", "arguably", "potentially", "in a sense", "to some extent"]
            for hedge in hedges:
                styled_text = styled_text.replace(" " + hedge + " ", " ")
                
            # Break up long sentences
            if len(styled_text) > 100:
                sentences = styled_text.split('. ')
                for i, sentence in enumerate(sentences):
                    if len(sentence) > 50 and "," in sentence:
                        parts = sentence.split(', ', 1)
                        sentences[i] = parts[0] + "." + parts[1]
                
                styled_text = '. '.join(sentences)
                
        elif direct_nuanced > 0.67:
            # More nuanced style
            # Add qualifiers, hedges
            sentences = styled_text.split('. ')
            for i, sentence in enumerate(sentences[:2]):  # Only modify first couple of sentences
                if len(sentence) > 10 and random.random() < 0.3:
                    hedges = ["Perhaps", "It seems", "It appears", "I believe", "In my view", "From my perspective"]
                    sentences[i] = random.choice(hedges) + " " + sentence[0].lower() + sentence[1:]
            
            styled_text = '. '.join(sentences)
        
        return styled_text
    
    def get_system_prompt_addition(self) -> str:
        """
        Get a string to add to the system prompt based on personality.
        
        Returns:
            Text to add to the system prompt
        """
        prompt = "Your personality traits influence how you communicate. "
        
        # Add descriptions for dominant traits
        dominant_traits = []
        
        for trait, value in self.traits.items():
            if value < 0.33 or value > 0.67:
                trait_label = self.get_trait_label(trait)
                trait_name = self.TRAIT_DIMENSIONS[trait]["name"]
                dominant_traits.append(f"You are {trait_label} on the '{trait_name}' spectrum")
        
        if dominant_traits:
            prompt += " ".join(dominant_traits) + ". "
        else:
            prompt += "You have a balanced personality with no strongly dominant traits. "
        
        # Provide guidance based on traits
        if self.get_trait("formal_casual") < 0.33:
            prompt += "You prefer formal language and avoid contractions or slang. "
        elif self.get_trait("formal_casual") > 0.67:
            prompt += "You use casual, conversational language with contractions and occasional slang. "
        
        if self.get_trait("reserved_expressive") < 0.33:
            prompt += "You express emotions in a reserved, understated way. "
        elif self.get_trait("reserved_expressive") > 0.67:
            prompt += "You are emotionally expressive and enthusiastic in your communication. "
        
        if self.get_trait("analytical_creative") < 0.33:
            prompt += "You tend to analyze information logically and methodically. "
        elif self.get_trait("analytical_creative") > 0.67:
            prompt += "You think creatively and enjoy exploring novel ideas and possibilities. "
        
        return prompt


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize personality
    personality = Personality("test_personality.json")
    
    # Test personality operations
    print(f"Analytical/Creative trait: {personality.get_trait('analytical_creative'):.2f}")
    print(f"Formal/Casual trait: {personality.get_trait('formal_casual'):.2f}")
    
    # Process some test messages
    test_messages = [
        "I need a logical analysis of this data",
        "Let's be creative and think outside the box",
        "I prefer a more casual and friendly approach",
        "This requires a formal and structured response"
    ]
    
    for message in test_messages:
        print(f"\nProcessing message: '{message}'")
        personality.process_message(message)
        
    # Show updated traits
    print("\nUpdated traits:")
    print(personality.get_personality_description())
    
    # Test response templates
    print("\nGreeting templates:")
    for _ in range(3):
        print(f"- {personality.get_response_template('greetings')}")
    
    # Test text style adjustment
    print("\nText style adjustment:")
    test_text = "I think this is a really interesting topic that we could explore together. Let's dive deeper into it!"
    styled_text = personality.adjust_text_style(test_text)
    print(f"Original: {test_text}")
    print(f"Styled: {styled_text}")
