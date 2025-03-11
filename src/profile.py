"""
Profile Module

This module handles the assistant's identity information including name,
backstory, and other customizable traits.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class Profile:
    """Manages the assistant's identity and customization settings"""
    
    # Default profile values
    DEFAULT_PROFILE = {
        "name": "Assistant",
        "backstory": "I am an AI assistant designed to be helpful, friendly, and supportive.",
        "interests": ["helping users", "learning new information", "having interesting conversations"],
        "preferences": {},
        "creation_date": None,
        "last_modified": None,
        "user_info": {
            "name": None,
            "preferences": {}
        }
    }
    
    def __init__(self, profile_path: str = "assistant_profile.json"):
        """
        Initialize the profile manager.
        
        Args:
            profile_path: Path to the profile file
        """
        self.profile_path = profile_path
        self.profile = self._load_profile()
        
        # Update timestamps if creating a new profile
        if self.profile.get("creation_date") is None:
            current_time = datetime.now().isoformat()
            self.profile["creation_date"] = current_time
            self.profile["last_modified"] = current_time
            self._save_profile()
            
        logger.info(f"Profile initialized for '{self.get_name()}'")
    
    def _load_profile(self) -> Dict[str, Any]:
        """Load profile from file or initialize with defaults"""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r') as f:
                    profile = json.load(f)
                logger.info(f"Loaded profile from {self.profile_path}")
                return profile
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load profile: {e}")
        
        # Return default profile if file doesn't exist or can't be read
        logger.info("Creating new profile with default values")
        return self.DEFAULT_PROFILE.copy()
    
    def _save_profile(self) -> bool:
        """
        Save profile to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update last modified timestamp
            self.profile["last_modified"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.profile_path, 'w') as f:
                json.dump(self.profile, f, indent=2)
            
            logger.debug(f"Profile saved to {self.profile_path}")
            return True
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save profile: {e}")
            return False
    
    def get_name(self) -> str:
        """Get the assistant's name"""
        return self.profile.get("name", "Assistant")
    
    def set_name(self, name: str) -> bool:
        """
        Set the assistant's name.
        
        Args:
            name: New name for the assistant
            
        Returns:
            True if successful, False otherwise
        """
        if not name or not isinstance(name, str):
            logger.warning(f"Invalid name provided: {name}")
            return False
        
        # Store old name for logging
        old_name = self.profile.get("name", "Assistant")
        
        # Update name
        self.profile["name"] = name.strip()
        
        # Save changes
        success = self._save_profile()
        if success:
            logger.info(f"Assistant name changed from '{old_name}' to '{name}'")
        
        return success
    
    def get_backstory(self) -> str:
        """Get the assistant's backstory"""
        return self.profile.get("backstory", self.DEFAULT_PROFILE["backstory"])
    
    def set_backstory(self, backstory: str) -> bool:
        """
        Set the assistant's backstory.
        
        Args:
            backstory: New backstory text
            
        Returns:
            True if successful, False otherwise
        """
        if not backstory or not isinstance(backstory, str):
            logger.warning(f"Invalid backstory provided")
            return False
        
        # Update backstory
        self.profile["backstory"] = backstory.strip()
        
        # Save changes
        success = self._save_profile()
        if success:
            logger.info(f"Assistant backstory updated")
        
        return success
    
    def get_interests(self) -> List[str]:
        """Get the assistant's interests"""
        return self.profile.get("interests", self.DEFAULT_PROFILE["interests"])
    
    def add_interest(self, interest: str) -> bool:
        """
        Add an interest to the assistant's profile.
        
        Args:
            interest: New interest to add
            
        Returns:
            True if successful, False otherwise
        """
        if not interest or not isinstance(interest, str):
            logger.warning(f"Invalid interest provided: {interest}")
            return False
        
        # Ensure interests list exists
        if "interests" not in self.profile:
            self.profile["interests"] = []
        
        # Add interest if not already present
        interest = interest.strip().lower()
        if interest not in [i.lower() for i in self.profile["interests"]]:
            self.profile["interests"].append(interest)
            
            # Save changes
            success = self._save_profile()
            if success:
                logger.info(f"Added interest: {interest}")
            return success
        
        return True  # Interest already exists
    
    def remove_interest(self, interest: str) -> bool:
        """
        Remove an interest from the assistant's profile.
        
        Args:
            interest: Interest to remove
            
        Returns:
            True if successful, False otherwise
        """
        if "interests" not in self.profile:
            return False
        
        # Find the interest (case-insensitive)
        interest_lower = interest.strip().lower()
        for i in self.profile["interests"][:]:  # Create a copy to iterate
            if i.lower() == interest_lower:
                self.profile["interests"].remove(i)
                
                # Save changes
                success = self._save_profile()
                if success:
                    logger.info(f"Removed interest: {i}")
                return success
        
        logger.warning(f"Interest not found: {interest}")
        return False
    
    def set_user_name(self, name: str) -> bool:
        """
        Set the user's name.
        
        Args:
            name: User's name
            
        Returns:
            True if successful, False otherwise
        """
        if not name or not isinstance(name, str):
            logger.warning(f"Invalid user name provided: {name}")
            return False
        
        # Ensure user_info dictionary exists
        if "user_info" not in self.profile:
            self.profile["user_info"] = {}
        
        # Update user name
        self.profile["user_info"]["name"] = name.strip()
        
        # Save changes
        success = self._save_profile()
        if success:
            logger.info(f"User name set to '{name}'")
        
        return success
    
    def get_user_name(self) -> Optional[str]:
        """Get the user's name if set"""
        if "user_info" in self.profile and "name" in self.profile["user_info"]:
            return self.profile["user_info"]["name"]
        return None
    
    def set_preference(self, key: str, value: Any) -> bool:
        """
        Set a preference value.
        
        Args:
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        if not key or not isinstance(key, str):
            logger.warning(f"Invalid preference key: {key}")
            return False
        
        # Ensure preferences dictionary exists
        if "preferences" not in self.profile:
            self.profile["preferences"] = {}
        
        # Update preference
        self.profile["preferences"][key] = value
        
        # Save changes
        success = self._save_profile()
        if success:
            logger.info(f"Set preference {key} = {value}")
        
        return success
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a preference value.
        
        Args:
            key: Preference key
            default: Default value if not found
            
        Returns:
            Preference value or default
        """
        if "preferences" in self.profile:
            return self.profile["preferences"].get(key, default)
        return default
    
    def set_user_preference(self, key: str, value: Any) -> bool:
        """
        Set a user preference value.
        
        Args:
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        if not key or not isinstance(key, str):
            logger.warning(f"Invalid user preference key: {key}")
            return False
        
        # Ensure user_info and preferences dictionaries exist
        if "user_info" not in self.profile:
            self.profile["user_info"] = {}
        if "preferences" not in self.profile["user_info"]:
            self.profile["user_info"]["preferences"] = {}
        
        # Update user preference
        self.profile["user_info"]["preferences"][key] = value
        
        # Save changes
        success = self._save_profile()
        if success:
            logger.info(f"Set user preference {key} = {value}")
        
        return success
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference value.
        
        Args:
            key: Preference key
            default: Default value if not found
            
        Returns:
            User preference value or default
        """
        if "user_info" in self.profile and "preferences" in self.profile["user_info"]:
            return self.profile["user_info"]["preferences"].get(key, default)
        return default
    
    def get_system_prompt_addition(self) -> str:
        """
        Get a string to add to the system prompt based on the profile.
        
        Returns:
            Text to add to the system prompt
        """
        name = self.get_name()
        backstory = self.get_backstory()
        interests = self.get_interests()
        user_name = self.get_user_name()
        
        # Build the prompt addition
        prompt = f"Your name is {name}. "
        prompt += f"{backstory} "
        
        if interests:
            prompt += f"You have interests in {', '.join(interests)}. "
        
        if user_name:
            prompt += f"The user's name is {user_name}, and you should address them by name occasionally. "
        
        return prompt
    
    def export_profile(self) -> Dict[str, Any]:
        """
        Export the complete profile.
        
        Returns:
            Copy of the profile dictionary
        """
        return self.profile.copy()
    
    def import_profile(self, profile_data: Dict[str, Any]) -> bool:
        """
        Import a profile from dictionary data.
        
        Args:
            profile_data: Profile data to import
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate required fields
            required_fields = ["name", "backstory"]
            for field in required_fields:
                if field not in profile_data:
                    logger.error(f"Missing required field in profile data: {field}")
                    return False
            
            # Update profile with new data
            self.profile = profile_data
            
            # Update timestamps
            current_time = datetime.now().isoformat()
            if "creation_date" not in self.profile:
                self.profile["creation_date"] = current_time
            self.profile["last_modified"] = current_time
            
            # Save changes
            success = self._save_profile()
            if success:
                logger.info(f"Imported profile for '{self.get_name()}'")
            
            return success
        except Exception as e:
            logger.error(f"Failed to import profile: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize profile
    profile = Profile("test_profile.json")
    
    # Test profile operations
    print(f"Current name: {profile.get_name()}")
    
    profile.set_name("Jarvis")
    profile.set_backstory("I am an advanced AI assistant created to help with various tasks.")
    profile.add_interest("technology")
    profile.add_interest("science")
    profile.set_user_name("User")
    
    print(f"Updated name: {profile.get_name()}")
    print(f"Backstory: {profile.get_backstory()}")
    print(f"Interests: {profile.get_interests()}")
    print(f"User name: {profile.get_user_name()}")
    
    print("\nSystem prompt addition:")
    print(profile.get_system_prompt_addition())
