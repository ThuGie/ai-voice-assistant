"""
VTube Studio Integration Module

This module provides the VTube Studio integration for the AI assistant.
It uses pyvts to communicate with the VTube Studio API.
"""

import logging
import asyncio

logger = logging.getLogger(__name__)

try:
    import pyvts
    VTS_AVAILABLE = True
except ImportError:
    logger.warning("pyvts is not installed. VTube Studio functionality will be disabled.")
    logger.info("To enable VTube Studio integration, please install the required package: pip install pyvts")
    VTS_AVAILABLE = False
    # Define a dummy class for type hinting
    class pyvts:
        class VTS:
            def __init__(self, **kwargs): pass
            async def connect(self): pass
            async def close(self): pass
            async def request_authentication_token(self, **kwargs): pass
            async def verify_authentication_token(self, **kwargs): pass
            async def request(self, payload): pass

class VTubeStudio:
    """A class to handle VTube Studio integration."""

    def __init__(self, plugin_name="AI_Assistant", plugin_developer="Jules", auth_token_path="./vts_token.txt"):
        """
        Initialize the VTubeStudio integration.
        """
        self.is_available = VTS_AVAILABLE
        if not self.is_available:
            self.vts = None
            return

        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.auth_token_path = auth_token_path
        self.vts = pyvts.VTS(plugin_name=self.plugin_name, plugin_developer=self.plugin_developer)
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.connect())

        # Mapping from assistant emotions to VTube Studio hotkey IDs
        self.emotion_hotkey_map = {
            "happy": "TriggerHappy",
            "sad": "TriggerSad",
            "angry": "TriggerAngry",
            "excited": "TriggerExcited",
            "concerned": "TriggerConcerned",
            "confused": "TriggerConfused",
            "neutral": "TriggerNeutral",
        }

    async def connect(self):
        """Connect to the VTube Studio API and authenticate."""
        if not self.is_available:
            return

        try:
            await self.vts.connect()

            # Check for saved token
            auth_token = None
            try:
                with open(self.auth_token_path, 'r') as f:
                    auth_token = f.read()
            except FileNotFoundError:
                pass

            if auth_token:
                await self.vts.verify_authentication_token(auth_token)
                if not self.vts.is_authenticated:
                    auth_token = None # Token is invalid, request a new one

            if not auth_token:
                token_response = await self.vts.request_authentication_token(plugin_name=self.plugin_name, plugin_developer=self.plugin_developer)
                auth_token = token_response['authenticationToken']
                with open(self.auth_token_path, 'w') as f:
                    f.write(auth_token)

                logger.info("VTube Studio authentication required. Please allow the plugin in the VTube Studio app.")
                await self.vts.verify_authentication_token(auth_token)

            if self.vts.is_authenticated:
                logger.info("Successfully connected and authenticated with VTube Studio.")
            else:
                logger.error("Failed to authenticate with VTube Studio.")

        except Exception as e:
            logger.error(f"Failed to connect to VTube Studio: {e}")
            self.is_available = False

    async def trigger_hotkey(self, hotkey_id: str):
        """Trigger a hotkey in VTube Studio."""
        if not self.is_available or not self.vts.is_authenticated:
            return

        request = self.vts.vts_request.requestTriggerHotkey(hotkeyID=hotkey_id)
        try:
            await self.vts.request(request)
            logger.info(f"Triggered VTube Studio hotkey: {hotkey_id}")
        except Exception as e:
            logger.error(f"Failed to trigger VTube Studio hotkey: {e}")

    def set_emotion(self, emotion: str):
        """Set the avatar's emotion by triggering a hotkey."""
        if not self.is_available:
            return

        hotkey_id = self.emotion_hotkey_map.get(emotion.lower())
        if hotkey_id:
            asyncio.run_coroutine_threadsafe(self.trigger_hotkey(hotkey_id), self.loop)
        else:
            logger.warning(f"No VTube Studio hotkey defined for emotion: {emotion}")

    async def close(self):
        """Close the connection to VTube Studio."""
        if self.vts:
            await self.vts.close()
