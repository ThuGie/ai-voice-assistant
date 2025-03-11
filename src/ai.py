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
import time
import platform
import subprocess
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class OllamaNotFoundError(Exception):
    """Exception raised when Ollama is not installed or not found."""
    pass

class OllamaConnectionError(Exception):
    """Exception raised when unable to connect to Ollama API."""
    pass

class OllamaModelError(Exception):
    """Exception raised when there are issues with Ollama models."""
    pass

class AIEngine:
    """AI engine using Ollama"""
    
    # Default API settings
    DEFAULT_API_BASE = "http://localhost:11434/api"
    DEFAULT_MODEL = "llama3"
    
    def __init__(self, 
                model: str = DEFAULT_MODEL, 
                api_base: str = DEFAULT_API_BASE,
                context_window: int = 4096,
                system_prompt: Optional[str] = None,
                auto_start_ollama: bool = True):
        """
        Initialize the AI engine with the specified model.
        
        Args:
            model: Name of the Ollama model to use
            api_base: Base URL for the Ollama API
            context_window: Size of the context window in tokens
            system_prompt: System prompt to use for the conversation
            auto_start_ollama: Whether to attempt to start Ollama if not running
        """
        self.model = model
        self.api_base = api_base.rstrip("/")  # Remove trailing slash if present
        self.context_window = context_window
        self.auto_start_ollama = auto_start_ollama
        
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
        except OllamaConnectionError as e:
            if self.auto_start_ollama:
                self._start_ollama_server()
                # Try again after starting
                try:
                    self._check_ollama_availability()
                    logger.info(f"Started Ollama server and initialized AI engine with model '{model}'")
                except Exception as e:
                    logger.error(f"Still failed to connect to Ollama after starting server: {e}")
                    raise
            else:
                logger.error(f"Failed to connect to Ollama: {e}")
                logger.error("Make sure Ollama is running (`ollama serve`) and accessible")
                raise
        except OllamaModelError as e:
            logger.error(f"Model error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AI engine: {e}")
            raise
    
    def _is_ollama_installed(self) -> bool:
        """Check if Ollama is installed on the system"""
        try:
            # Use 'where' on Windows, 'which' on Unix-like systems
            check_cmd = 'where' if platform.system() == 'Windows' else 'which'
            subprocess.run([check_cmd, 'ollama'], check=True, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _start_ollama_server(self) -> bool:
        """Try to start the Ollama server if it's not running"""
        if not self._is_ollama_installed():
            raise OllamaNotFoundError("Ollama is not installed. Please install it from https://ollama.ai")
            
        logger.info("Attempting to start Ollama server...")
        
        try:
            if platform.system() == 'Windows':
                # On Windows, we can't easily start as daemon
                logger.warning("Automatic Ollama startup on Windows is not supported.")
                logger.warning("Please start Ollama server manually by running 'ollama serve' in another terminal.")
                return False
            else:
                # Start server as a background process
                subprocess.Popen(['ollama', 'serve'], 
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               start_new_session=True)
                
                # Give it time to start
                logger.info("Waiting for Ollama server to start...")
                time.sleep(5)
                return True
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    
    def _check_ollama_availability(self):
        """Check if Ollama API is available and the model exists"""
        try:
            # Check if Ollama server is running
            response = requests.get(f"{self.api_base}/tags", timeout=5)
            
            # Handle HTTP errors
            if response.status_code != 200:
                raise OllamaConnectionError(
                    f"Ollama API returned status code {response.status_code}. " 
                    f"Response: {response.text}"
                )
            
            # Check if the model exists
            available_models = [model["name"] for model in response.json().get("models", [])]
            
            if self.model not in available_models:
                logger.warning(
                    f"Model '{self.model}' not found. Available models: {', '.join(available_models)}. "
                    f"Attempting to pull model '{self.model}'..."
                )
                self._pull_model()
            else:
                logger.info(f"Model '{self.model}' is available.")
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(
                f"Could not connect to Ollama API at {self.api_base}. "
                f"Make sure Ollama is running and accessible."
            )
        except requests.exceptions.Timeout:
            raise OllamaConnectionError(
                f"Connection to Ollama API at {self.api_base} timed out. "
                f"The server might be overloaded or experiencing issues."
            )
        except json.JSONDecodeError:
            raise OllamaConnectionError(
                f"Received invalid JSON from Ollama API at {self.api_base}. "
                f"The server might be returning an unexpected response format."
            )
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
                        raise OllamaModelError(status.get("error"))
                        
            logger.info(f"Successfully pulled model '{self.model}'")
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(
                f"Lost connection to Ollama API while pulling model. "
                f"Check your network connection and ensure Ollama is still running."
            )
        except Exception as e:
            logger.error(f"Failed to pull model '{self.model}': {e}")
            raise OllamaModelError(f"Failed to pull model '{self.model}': {e}")
    
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
                         stream_callback: Optional[Callable[[str], None]] = None) -> str:
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
        except OllamaConnectionError as e:
            error_msg = f"Lost connection to Ollama: {e}"
            logger.error(error_msg)
            
            # If auto-start is enabled, try to restart Ollama
            if self.auto_start_ollama:
                logger.info("Attempting to restart Ollama server...")
                if self._start_ollama_server():
                    logger.info("Ollama restarted, retrying request...")
                    try:
                        # Retry the request with the same parameters (without recursion to avoid loops)
                        payload = {
                            "model": self.model,
                            "messages": self.conversation_history,
                            "stream": stream,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens
                            }
                        }
                        
                        if stream:
                            return self._generate_streaming(payload, stream_callback)
                        else:
                            return self._generate_complete(payload)
                    except Exception as retry_e:
                        logger.error(f"Retry failed: {retry_e}")
                        return f"I'm having trouble connecting to my AI engine. Please check if Ollama is running properly. Error: {retry_e}"
                else:
                    return "I'm having trouble connecting to my AI engine. Please make sure Ollama is running by executing 'ollama serve' in a terminal."
            
            return f"I'm having trouble connecting to my AI engine. Please make sure Ollama is running. Error: {e}"
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
            
            # Handle HTTP errors
            if response.status_code != 200:
                error_message = f"Ollama API returned status code {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_message += f": {error_data['error']}"
                except:
                    error_message += f": {response.text}"
                
                raise OllamaConnectionError(error_message)
            
            # Extract response content
            result = response.json()
            content = result.get("message", {}).get("content", "")
            
            # Add response to history
            self._add_to_history("assistant", content)
            
            return content
        except requests.exceptions.Timeout:
            raise OllamaConnectionError("Request to Ollama API timed out. The model may be taking too long to respond.")
        except json.JSONDecodeError:
            raise OllamaConnectionError("Received invalid JSON from Ollama API.")
        except Exception as e:
            logger.error(f"Error generating complete response: {e}")
            raise
    
    def _generate_streaming(self, payload: Dict[str, Any], callback: Optional[Callable[[str], None]]) -> str:
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
            
            # Handle HTTP errors
            if response.status_code != 200:
                error_message = f"Ollama API returned status code {response.status_code}"
                # Try to extract error message
                try:
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if "error" in chunk:
                                error_message += f": {chunk['error']}"
                                break
                except:
                    error_message += f": {response.text}"
                
                raise OllamaConnectionError(error_message)
            
            # Process streaming response
            line_buffer = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk:
                            content = chunk["message"].get("content", "")
                            full_response += content
                            
                            # Call callback with the chunk of text
                            if callback:
                                callback(content)
                    except json.JSONDecodeError as e:
                        # Handle partial JSON lines by buffering
                        line_buffer += line.decode('utf-8')
                        try:
                            chunk = json.loads(line_buffer)
                            line_buffer = ""
                            if "message" in chunk:
                                content = chunk["message"].get("content", "")
                                full_response += content
                                
                                # Call callback with the chunk of text
                                if callback:
                                    callback(content)
                        except json.JSONDecodeError:
                            # Continue buffering
                            pass
            
            # Add full response to history
            self._add_to_history("assistant", full_response)
            
            return full_response
        except requests.exceptions.ChunkedEncodingError:
            # This can happen if the connection is interrupted mid-stream
            raise OllamaConnectionError("The streaming connection was interrupted.")
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
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list available models: {e}")
            return []
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
    try:
        ai = AIEngine(model=model, auto_start_ollama=True)
        
        # List available models
        print("Available models:")
        models = ai.list_available_models()
        if models:
            for model_info in models:
                print(f"- {model_info['name']}")
        else:
            print("Could not retrieve available models.")
        
        # Test response generation
        prompt = "Hello! Can you tell me about yourself?"
        print(f"\nPrompt: {prompt}")
        
        # Streaming response example
        def print_chunk(chunk):
            print(chunk, end="", flush=True)
        
        print("\nResponse (streaming):")
        response = ai.generate_response(prompt, stream=True, stream_callback=print_chunk)
        print("\n")
        
    except OllamaNotFoundError:
        print("ERROR: Ollama is not installed. Please install it from https://ollama.ai")
    except OllamaConnectionError as e:
        print(f"ERROR: Connection to Ollama failed: {e}")
        print("Make sure Ollama is running with 'ollama serve' command.")
    except Exception as e:
        print(f"ERROR: {e}")
