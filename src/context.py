"""
Context Module

This module provides contextual awareness for the assistant, including
time, date, conversation history recall, and initiative.
"""

import os
import json
import logging
import random
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import calendar

# Set up logging
logger = logging.getLogger(__name__)

class ContextManager:
    """Manages contextual awareness for the assistant"""
    
    # Holidays and special days
    HOLIDAYS = {
        # Format: "MM-DD": "Holiday Name"
        "01-01": "New Year's Day",
        "02-14": "Valentine's Day",
        "03-17": "St. Patrick's Day",
        "04-01": "April Fools' Day",
        "04-22": "Earth Day",
        "05-05": "Cinco de Mayo",
        "07-04": "Independence Day",
        "09-05": "Labor Day",  # First Monday in September
        "10-31": "Halloween",
        "11-11": "Veterans Day",
        "11-24": "Thanksgiving Day",  # Fourth Thursday in November (approximation)
        "12-24": "Christmas Eve",
        "12-25": "Christmas Day",
        "12-31": "New Year's Eve"
    }
    
    # Seasons in Northern Hemisphere (approximate dates)
    SEASONS = {
        (3, 20): "Spring",  # March 20
        (6, 21): "Summer",  # June 21
        (9, 22): "Fall",    # September 22
        (12, 21): "Winter"  # December 21
    }
    
    # Topics of potential interest for proactive conversation
    PROACTIVE_TOPICS = [
        "how the day is going",
        "recent interests or hobbies",
        "plans for the day",
        "if they need help with anything specific",
        "if they want information about a topic",
        "how they're feeling today",
        "a suggestion based on the weather",
        "a recent interesting topic in tech news",
        "a fun fact about the current date"
    ]
    
    def __init__(self, 
                context_path: str = "assistant_context.json",
                idle_initiative: bool = True,
                idle_interval_minutes: int = 20):
        """
        Initialize the context manager.
        
        Args:
            context_path: Path to the context file
            idle_initiative: Whether to enable idle initiative
            idle_interval_minutes: Interval between idle initiatives in minutes
        """
        self.context_path = context_path
        self.context_data = self._load_context()
        self.last_activity = datetime.now()
        self.idle_initiative = idle_initiative
        self.idle_interval = idle_interval_minutes * 60  # Convert to seconds
        self.idle_timer = None
        self.idle_callback = None
        self.running = False
        
        # Important topics from conversations (for recall)
        self.important_topics = self._get_important_topics()
        
        # Topics that would potentially interest the user
        self.user_interests = self._get_user_interests()
        
        # Initialize session data
        self._init_session_data()
        
        logger.info("Context manager initialized")
    
    def _load_context(self) -> Dict[str, Any]:
        """Load context data from file or initialize with defaults"""
        if os.path.exists(self.context_path):
            try:
                with open(self.context_path, 'r') as f:
                    context_data = json.load(f)
                logger.info(f"Loaded context from {self.context_path}")
                return context_data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load context: {e}")
        
        # Default context data
        return {
            "important_topics": [],
            "user_interests": [],
            "session_history": [],
            "idle_initiative_count": 0,
            "last_initiative_time": None,
            "conversation_statistics": {
                "total_conversations": 0,
                "total_messages": 0,
                "average_response_length": 0
            }
        }
    
    def _save_context(self) -> bool:
        """
        Save context data to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update timestamp
            self.context_data["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.context_path, 'w') as f:
                json.dump(self.context_data, f, indent=2)
            
            logger.debug(f"Context saved to {self.context_path}")
            return True
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save context: {e}")
            return False
    
    def _init_session_data(self):
        """Initialize session-specific data"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create new session entry
        session_data = {
            "id": session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "messages": 0,
            "topics": []
        }
        
        # Add to session history
        if "session_history" not in self.context_data:
            self.context_data["session_history"] = []
        
        self.context_data["session_history"].append(session_data)
        self.current_session_id = session_id
        
        # Save changes
        self._save_context()
    
    def _get_important_topics(self) -> List[Dict[str, Any]]:
        """Get important topics from context data"""
        return self.context_data.get("important_topics", [])
    
    def _get_user_interests(self) -> List[str]:
        """Get user interests from context data"""
        return self.context_data.get("user_interests", [])
    
    def start(self, idle_callback: callable = None):
        """
        Start context monitoring and idle initiative.
        
        Args:
            idle_callback: Function to call when idle initiative is triggered
        """
        if self.running:
            return
        
        self.running = True
        self.idle_callback = idle_callback
        
        # Start idle timer if enabled
        if self.idle_initiative and idle_callback:
            self._start_idle_timer()
            
        logger.info("Context manager started")
    
    def stop(self):
        """Stop context monitoring and idle initiative"""
        self.running = False
        
        # Stop idle timer
        if self.idle_timer:
            self.idle_timer.cancel()
            self.idle_timer = None
        
        # Update session end time
        self._end_current_session()
        
        logger.info("Context manager stopped")
    
    def _start_idle_timer(self):
        """Start or restart the idle timer"""
        # Cancel existing timer if any
        if self.idle_timer:
            self.idle_timer.cancel()
        
        # Create new timer
        self.idle_timer = threading.Timer(self.idle_interval, self._idle_initiative_triggered)
        self.idle_timer.daemon = True  # Timer won't block program exit
        self.idle_timer.start()
        
        logger.debug(f"Idle timer started for {self.idle_interval} seconds")
    
    def _idle_initiative_triggered(self):
        """Called when idle timer expires"""
        # Check if still running
        if not self.running:
            return
        
        # Check if enough time has passed since last activity
        time_since_activity = (datetime.now() - self.last_activity).total_seconds()
        if time_since_activity < self.idle_interval:
            # Restart timer
            self._start_idle_timer()
            return
        
        # Generate initiative
        initiative_text = self._generate_initiative()
        
        # Call the callback
        if self.idle_callback and initiative_text:
            self.idle_callback(initiative_text)
        
        # Update stats
        if "idle_initiative_count" not in self.context_data:
            self.context_data["idle_initiative_count"] = 0
        self.context_data["idle_initiative_count"] += 1
        self.context_data["last_initiative_time"] = datetime.now().isoformat()
        self._save_context()
        
        # Restart timer if still running
        if self.running:
            self._start_idle_timer()
    
    def record_activity(self):
        """Record user activity to reset idle timer"""
        self.last_activity = datetime.now()
    
    def _end_current_session(self):
        """Update the end time of the current session"""
        if "session_history" in self.context_data and self.context_data["session_history"]:
            for session in self.context_data["session_history"]:
                if session["id"] == self.current_session_id:
                    session["end_time"] = datetime.now().isoformat()
                    break
            
            self._save_context()
    
    def update_current_session(self, message_count: int = 1, topics: List[str] = None):
        """
        Update the current session with message count and topics.
        
        Args:
            message_count: Number of messages to add
            topics: List of topics discussed
        """
        if "session_history" in self.context_data:
            for session in self.context_data["session_history"]:
                if session["id"] == self.current_session_id:
                    session["messages"] += message_count
                    
                    if topics:
                        if "topics" not in session:
                            session["topics"] = []
                        
                        # Add new topics
                        for topic in topics:
                            if topic not in session["topics"]:
                                session["topics"].append(topic)
                    
                    break
            
            self._save_context()
        
        # Update statistics
        if "conversation_statistics" not in self.context_data:
            self.context_data["conversation_statistics"] = {
                "total_conversations": 1,
                "total_messages": 0,
                "average_response_length": 0
            }
        
        stats = self.context_data["conversation_statistics"]
        stats["total_messages"] += message_count
        
        self._save_context()
    
    def add_important_topic(self, topic: str, content: str, importance: float = 0.7) -> bool:
        """
        Add an important topic for future recall.
        
        Args:
            topic: Topic name or keyword
            content: Content related to the topic
            importance: Importance score (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if not topic or not content:
            return False
        
        # Ensure important_topics exists
        if "important_topics" not in self.context_data:
            self.context_data["important_topics"] = []
        
        # Check if topic already exists
        for existing_topic in self.context_data["important_topics"]:
            if existing_topic.get("topic") == topic:
                # Update existing topic
                existing_topic["content"] = content
                existing_topic["importance"] = importance
                existing_topic["last_updated"] = datetime.now().isoformat()
                
                # Update local cache
                self.important_topics = self._get_important_topics()
                
                # Save changes
                self._save_context()
                return True
        
        # Add new topic
        topic_data = {
            "topic": topic,
            "content": content,
            "importance": importance,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "recall_count": 0
        }
        
        self.context_data["important_topics"].append(topic_data)
        
        # Update local cache
        self.important_topics = self._get_important_topics()
        
        # Save changes
        logger.info(f"Added important topic: {topic}")
        return self._save_context()
    
    def add_user_interest(self, interest: str) -> bool:
        """
        Add a user interest.
        
        Args:
            interest: Interest to add
            
        Returns:
            True if successful, False otherwise
        """
        if not interest:
            return False
        
        # Ensure user_interests exists
        if "user_interests" not in self.context_data:
            self.context_data["user_interests"] = []
        
        # Add interest if not already present
        interest = interest.strip().lower()
        if interest not in self.context_data["user_interests"]:
            self.context_data["user_interests"].append(interest)
            
            # Update local cache
            self.user_interests = self._get_user_interests()
            
            # Save changes
            logger.info(f"Added user interest: {interest}")
            return self._save_context()
        
        return True  # Interest already exists
    
    def recall_topic(self, topic_keyword: str) -> Optional[Dict[str, Any]]:
        """
        Recall information about a topic.
        
        Args:
            topic_keyword: Keyword to search for
            
        Returns:
            Topic data if found, None otherwise
        """
        if not self.important_topics:
            return None
        
        topic_keyword = topic_keyword.lower()
        
        # Look for exact match first
        for topic in self.important_topics:
            if topic.get("topic", "").lower() == topic_keyword:
                # Update recall count
                topic["recall_count"] = topic.get("recall_count", 0) + 1
                topic["last_recalled"] = datetime.now().isoformat()
                self._save_context()
                return topic
        
        # Look for partial matches
        for topic in self.important_topics:
            if topic_keyword in topic.get("topic", "").lower() or topic_keyword in topic.get("content", "").lower():
                # Update recall count
                topic["recall_count"] = topic.get("recall_count", 0) + 1
                topic["last_recalled"] = datetime.now().isoformat()
                self._save_context()
                return topic
        
        return None
    
    def get_relevant_topics(self, message: str, max_topics: int = 3) -> List[Dict[str, Any]]:
        """
        Get topics relevant to the current message.
        
        Args:
            message: Current message
            max_topics: Maximum number of topics to return
            
        Returns:
            List of relevant topics
        """
        if not self.important_topics or not message:
            return []
        
        message = message.lower()
        relevant_topics = []
        
        # Find topics with keywords in the message
        for topic in self.important_topics:
            topic_name = topic.get("topic", "").lower()
            if topic_name in message:
                relevant_topics.append(topic)
                continue
            
            # Check for important words from the topic content
            content = topic.get("content", "").lower()
            important_words = [word for word in content.split() if len(word) > 4]
            
            for word in important_words[:10]:  # Check only the first 10 important words
                if word in message:
                    relevant_topics.append(topic)
                    break
        
        # Sort by importance and limit the number
        relevant_topics.sort(key=lambda x: x.get("importance", 0), reverse=True)
        return relevant_topics[:max_topics]
    
    def _generate_initiative(self) -> str:
        """
        Generate a proactive initiative based on context.
        
        Returns:
            Initiative text or empty string if none is appropriate
        """
        # Get current time context
        now = datetime.now()
        time_context = self._get_time_context(now)
        
        # Different initiative types
        initiative_types = [
            "time_based",
            "topic_recall",
            "interest_based",
            "conversation_starter"
        ]
        
        # Select an initiative type with some randomness
        weights = [0.3, 0.2, 0.3, 0.2]  # Weights for each type
        initiative_type = random.choices(initiative_types, weights=weights, k=1)[0]
        
        if initiative_type == "time_based":
            # Initiative based on time of day, special day, etc.
            if time_context.get("special_day"):
                return f"I noticed it's {time_context['special_day']}. Any special plans for today?"
            
            if time_context.get("time_of_day") == "morning":
                return "Good morning! How is your day starting off?"
            elif time_context.get("time_of_day") == "afternoon":
                return "Hope your afternoon is going well. Is there anything I can help you with?"
            elif time_context.get("time_of_day") == "evening":
                return "Good evening! How was your day today?"
            elif time_context.get("time_of_day") == "night":
                return "It's getting late. Is there anything you need help with before the day ends?"
        
        elif initiative_type == "topic_recall" and self.important_topics:
            # Recall an important topic from past conversations
            # Prioritize topics that haven't been recalled recently
            topics_by_recall = sorted(self.important_topics, key=lambda x: x.get("recall_count", 0))
            topic = topics_by_recall[0] if topics_by_recall else None
            
            if topic:
                return f"I remember we talked about {topic['topic']} before. Would you like to continue that conversation?"
        
        elif initiative_type == "interest_based" and self.user_interests:
            # Initiative based on user interests
            interest = random.choice(self.user_interests)
            return f"I recall you're interested in {interest}. Would you like to talk more about that?"
        
        elif initiative_type == "conversation_starter":
            # General conversation starter
            topic = random.choice(self.PROACTIVE_TOPICS)
            
            if topic == "how the day is going":
                return "How is your day going so far?"
            elif topic == "recent interests or hobbies":
                return "Have you picked up any new interests or hobbies recently?"
            elif topic == "plans for the day":
                return "Do you have any interesting plans for today?"
            elif topic == "if they need help with anything specific":
                return "Is there anything specific I can help you with today?"
            elif topic == "if they want information about a topic":
                return "Is there any topic you'd like me to provide information about?"
            elif topic == "how they're feeling today":
                return "How are you feeling today?"
            elif topic == "a fun fact about the current date":
                # Get a random fact about today's date
                month = now.month
                day = now.day
                return f"Did you know that today, {calendar.month_name[month]} {day}, is an interesting date in history? Would you like to know more?"
        
        # Fallback initiative
        return "It's been a while since we last talked. How can I assist you today?"
    
    def _get_time_context(self, current_time: datetime = None) -> Dict[str, Any]:
        """
        Get context information about the current time.
        
        Args:
            current_time: Current datetime or None to use now
            
        Returns:
            Dictionary with time context information
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Basic time information
        hour = current_time.hour
        minute = current_time.minute
        month = current_time.month
        day = current_time.day
        weekday = current_time.weekday()  # 0 is Monday, 6 is Sunday
        
        # Time of day
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Check for special day
        date_str = f"{month:02d}-{day:02d}"
        special_day = self.HOLIDAYS.get(date_str)
        
        # Check for season
        season = None
        for (season_month, season_day), season_name in self.SEASONS.items():
            if (month > season_month or (month == season_month and day >= season_day)):
                season = season_name
        
        # If we've gone past winter start but haven't hit spring, it's still winter
        if not season and (month < 3 or (month == 3 and day < 20)):
            season = "Winter"
            
        # Weekend vs. weekday
        is_weekend = weekday >= 5  # Saturday or Sunday
        
        # Assemble context
        context = {
            "time": f"{hour:02d}:{minute:02d}",
            "hour": hour,
            "minute": minute,
            "date": f"{current_time.year}-{month:02d}-{day:02d}",
            "month": month,
            "month_name": calendar.month_name[month],
            "day": day,
            "weekday": weekday,
            "weekday_name": calendar.day_name[weekday],
            "time_of_day": time_of_day,
            "is_weekend": is_weekend,
            "special_day": special_day,
            "season": season
        }
        
        return context
    
    def get_system_prompt_addition(self) -> str:
        """
        Get a string to add to the system prompt based on context.
        
        Returns:
            Text to add to the system prompt
        """
        # Get current time context
        time_context = self._get_time_context()
        
        prompt = f"The current time is {time_context['time']} ({time_context['time_of_day']}). "
        prompt += f"Today is {time_context['weekday_name']}, {time_context['month_name']} {time_context['day']}. "
        
        if time_context.get("special_day"):
            prompt += f"Today is {time_context['special_day']}. "
            
        if time_context.get("season"):
            prompt += f"The current season is {time_context['season']}. "
        
        # Add user interests if available
        if self.user_interests:
            prompt += "The user has expressed interest in " + ", ".join(self.user_interests[:5])
            if len(self.user_interests) > 5:
                prompt += f", and {len(self.user_interests) - 5} other topics"
            prompt += ". "
        
        # Add session information
        if "session_history" in self.context_data and self.context_data["session_history"]:
            session_count = len(self.context_data["session_history"])
            if session_count > 1:
                prompt += f"This is session #{session_count} with this user. "
            
            # Add conversation statistics if available
            if "conversation_statistics" in self.context_data:
                stats = self.context_data["conversation_statistics"]
                if stats.get("total_messages", 0) > 0:
                    prompt += f"You've exchanged {stats['total_messages']} messages with the user in total. "
        
        return prompt


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Define a simple idle callback for testing
    def idle_callback(message):
        print(f"\nAssistant (idle initiative): {message}")
    
    # Initialize context manager
    context = ContextManager("test_context.json", idle_interval_minutes=1)
    
    # Add some test data
    context.add_important_topic("Python programming", "The user is learning Python and asked about functions and classes.", 0.8)
    context.add_important_topic("Vacation plans", "The user mentioned planning a trip to Japan next spring.", 0.7)
    context.add_user_interest("artificial intelligence")
    context.add_user_interest("photography")
    
    # Start context manager with idle callback
    context.start(idle_callback=idle_callback)
    
    # Test session updates
    context.update_current_session(2, ["Python", "programming"])
    
    # Test topic recall
    topic = context.recall_topic("Python")
    if topic:
        print(f"Recalled topic: {topic['topic']}")
        print(f"Content: {topic['content']}")
    
    # Test time context
    time_context = context._get_time_context()
    print(f"\nCurrent time context:")
    for key, value in time_context.items():
        print(f"- {key}: {value}")
    
    # Test system prompt addition
    print(f"\nSystem prompt addition:")
    print(context.get_system_prompt_addition())
    
    # Let the idle initiative trigger once for demonstration
    print("\nWaiting for idle initiative (should trigger in 1 minute)...")
    try:
        # Keep running for a minute to allow idle initiative to trigger
        time.sleep(70)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop context manager
        context.stop()
        print("Context manager stopped")
