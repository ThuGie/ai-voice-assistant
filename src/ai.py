"""
AI Module using Ollama

This module handles interactions with the Ollama API to access large language models.
It provides functionality to generate responses from the AI model.
"""

import os
import sys
import json
import logging
import requests
from typing import Dict, List, Optional, Union, Any

# Set up logging
logger = logging.getLogger(__name__)

class AIEngine:
    """AI engine using Ollama"""
    
    # Default API settings
    DEFAULT_API_BASE = "http://localhost:11434/api"
    DEFAULT_MODEL = "llama3"
    
    def __init__(self, 
                model: str = DEFAULT_MODEL, 
                api_base: str = DEFAULT_API_BASE,
                context_window: int = 4096,
                system_prompt: Optional[str] = None):
        """
        Initialize the AI engine with the specified model.
        
        Args:
            model: Name of the Ollama model to use
            api_base: Base URL for the Ollama API
            context_window: Size of the context window in tokens
            system_prompt: System prompt to use for the conversation
        """
        self.model = model
        self.api_base = api_base.rstrip("/")  # Remove trailing slash if present
        self.context_window = context_window
        
        # Default system prompt if none is provided
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant with voice capabilities. "
            "You can hear the user through their microphone and respond with your voice. "
            "Keep your responses conversational and concise. "
            "If the user asks you to perform actions on their computer, explain what you would do "
            "but note that you need additional permissions to actually execute them."
        )
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Add system prompt to conversation history
        self._add_to_history("system", self.system_prompt)
        
        try:
            # Check if Ollama is available
            self._check_ollama_availability()
            logger.info(f"Initialized AI engine with model '{model}'")
        except Exception as e:
            logger.error(f"Failed to initialize AI engine: {e}")
            raise
    
    def _check_ollama_availability(self):
        """Check if Ollama API is available and the model exists"""
        try:
            # Check if Ollama server is running
            response = requests.get(f"{self.api_base}/tags", timeout=5)
            response.raise_for_status()
            
            # Check if the model exists
            available_models = [model["name"] for model in response.json()["models"]]
            
            if self.model not in available_models:
                logger.warning(
                    f"Model '{self.model}' not found. Available models: {', '.join(available_models)}. "
                    f"Pulling model '{self.model}'..."
                )
                self._pull_model()
            else:
                logger.info(f"Model '{self.model}' is available.")
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Could not connect to Ollama API at {self.api_base}. "
                f"Make sure Ollama is running and accessible."
            )
            raise
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            raise
    
    def _pull_model(self):
        """Pull the model from Ollama repository"""
        try:
            # Pull the model
            response = requests.post(
                f"{self.api_base}/pull",
                json={"name": self.model},
                stream=True
            )
            
            # Stream the response for progress updates
            for line in response.iter_lines():
                if line:
                    status = json.loads(line)
                    if "status" in status:
                        logger.info(f"Pulling model: {status.get('status')}")
                    if "error" in status:
                        logger.error(f"Error pulling model: {status.get('error')}")
                        raise Exception(status.get("error"))
                        
            logger.info(f"Successfully pulled model '{self.model}'")
        except Exception as e:
            logger.error(f"Failed to pull model '{self.model}': {e}")
            raise
    
    def _add_to_history(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message ("user", "assistant", or "system")
            content: Content of the message
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Ensure conversation history doesn't grow too large
        # This is a simple approach; a more sophisticated approach would consider token counts
        if len(self.conversation_history) > 20:  # Arbitrary limit
            # Keep system message and last N messages
            self.conversation_history = [
                self.conversation_history[0],  # System message
                *self.conversation_history[-19:]  # Last 19 messages
            ]
    
    def generate_response(self, 
                         prompt: str, 
                         temperature: float = 0.7, 
                         max_tokens: int = 500,
                         stream: bool = False,
                         stream_callback = None) -> str:
        """
        Generate a response from the AI model.
        
        Args:
            prompt: User input prompt
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            stream_callback: Callback function for streaming response
            
        Returns:
            Generated response as string
        """
        # Add user prompt to history
        self._add_to_history("user", prompt)
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": self.conversation_history,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Send request to Ollama API
            if stream:
                return self._generate_streaming(payload, stream_callback)
            else:
                return self._generate_complete(payload)
        except Exception as e:
            error_msg = f"Failed to generate response: {e}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def _generate_complete(self, payload: Dict[str, Any]) -> str:
        """
        Generate a complete response (non-streaming).
        
        Args:
            payload: Request payload
            
        Returns:
            Complete generated response
        """
        try:
            response = requests.post(
                f"{self.api_base}/chat",
                json=payload,
                timeout=60  # Longer timeout for complete responses
            )
            response.raise_for_status()
            
            # Extract response content
            result = response.json()
            content = result.get("message", {}).get("content", "")
            
            # Add response to history
            self._add_to_history("assistant", content)
            
            return content
        except Exception as e:
            logger.error(f"Error generating complete response: {e}")
            raise
    
    def _generate_streaming(self, payload: Dict[str, Any], callback) -> str:
        """
        Generate a streaming response.
        
        Args:
            payload: Request payload
            callback: Callback function for streaming response
            
        Returns:
            Complete generated response (after streaming)
        """
        full_response = ""
        
        try:
            response = requests.post(
                f"{self.api_base}/chat",
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        full_response += content
                        
                        # Call callback with the chunk of text
                        if callback:
                            callback(content)
            
            # Add full response to history
            self._add_to_history("assistant", full_response)
            
            return full_response
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            raise
    
    def clear_history(self, keep_system_prompt: bool = True):
        """
        Clear the conversation history.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt
        """
        if keep_system_prompt and self.conversation_history and self.conversation_history[0]["role"] == "system":
            # Keep just the system prompt
            system_prompt = self.conversation_history[0]
            self.conversation_history = [system_prompt]
        else:
            # Clear everything and re-add system prompt
            self.conversation_history = []
            self._add_to_history("system", self.system_prompt)
    
    def list_available_models(self) -> List[Dict]:
        """
        Get a list of available models.
        
        Returns:
            List of available models with metadata
        """
        try:
            response = requests.get(f"{self.api_base}/tags", timeout=5)
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            return []
    
    def change_model(self, model: str) -> bool:
        """
        Change the current model.
        
        Args:
            model: New model to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if model exists
            available_models = [m["name"] for m in self.list_available_models()]
            
            if model not in available_models:
                logger.warning(f"Model '{model}' not found. Attempting to pull it...")
                
                # Try to pull the model
                old_model = self.model
                self.model = model
                try:
                    self._pull_model()
                except Exception as pull_error:
                    logger.error(f"Failed to pull model '{model}': {pull_error}")
                    self.model = old_model  # Revert to old model
                    return False
            
            # Change the model
            self.model = model
            logger.info(f"Changed model to '{model}'")
            
            # Clear conversation history but keep system prompt
            self.clear_history(keep_system_prompt=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to change model: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Get model name from command line if provided
    model = sys.argv[1] if len(sys.argv) > 1 else AIEngine.DEFAULT_MODEL
    
    # Initialize the AI engine
    ai = AIEngine(model=model)
    
    # List available models
    print("Available models:")
    for model_info in ai.list_available_models():
        print(f"- {model_info['name']}")
    
    # Test response generation
    prompt = "Hello! Can you tell me about yourself?"
    print(f"\nPrompt: {prompt}")
    
    # Streaming response example
    def print_chunk(chunk):
        print(chunk, end="", flush=True)
    
    print("\nResponse (streaming):")
    response = ai.generate_response(prompt, stream=True, stream_callback=print_chunk)
    print("\n")
