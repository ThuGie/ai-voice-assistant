"""
PC Control Module

This module provides the PC control capabilities for the AI assistant.
It uses pynput to control the keyboard and mouse.
"""

import logging
import time

logger = logging.getLogger(__name__)

try:
    from pynput.mouse import Controller as MouseController, Button
    from pynput.keyboard import Controller as KeyboardController
    PYNPUT_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"pynput not available: {e}. PC control will be disabled.")
    PYNPUT_AVAILABLE = False
    # Define dummy classes for type hinting and to avoid further errors
    class MouseController: pass
    class KeyboardController: pass
    class Button:
        left = 'left'
        right = 'right'
        middle = 'middle'


class PCController:
    """A controller for the PC's mouse and keyboard."""

    def __init__(self):
        """Initialize the PCController."""
        if not PYNPUT_AVAILABLE:
            self.mouse = None
            self.keyboard = None
            logger.warning("PC control is not available because pynput could not be imported.")
            return

        try:
            self.mouse = MouseController()
            self.keyboard = KeyboardController()
            logger.info("PCController initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pynput controllers: {e}")
            logger.error("PC control may not work. This can happen if you are in an environment without a display server (e.g., a headless server).")
            self.mouse = None
            self.keyboard = None

    def type_text(self, text: str, delay: float = 0.05):
        """
        Type the given text using the keyboard.

        Args:
            text: The text to type.
            delay: The delay between keystrokes in seconds.
        """
        if not self.keyboard:
            logger.error("Keyboard controller not available.")
            return "Error: Keyboard control is not available."

        logger.info(f"Typing text: {text}")
        try:
            for char in text:
                self.keyboard.press(char)
                self.keyboard.release(char)
                time.sleep(delay)
            return f"Successfully typed: {text}"
        except Exception as e:
            logger.error(f"Error typing text: {e}")
            return f"Error: Could not type the text. {e}"

    def click(self, x: int, y: int, button: str = 'left', count: int = 1):
        """
        Move the mouse and click at the specified coordinates.

        Args:
            x: The x-coordinate.
            y: The y-coordinate.
            button: The mouse button to click ('left', 'right', 'middle').
            count: The number of clicks.
        """
        if not self.mouse:
            logger.error("Mouse controller not available.")
            return "Error: Mouse control is not available."

        logger.info(f"Clicking at ({x}, {y}) with {button} button, {count} times.")
        try:
            self.mouse.position = (x, y)

            button_to_click = Button.left
            if button.lower() == 'right':
                button_to_click = Button.right
            elif button.lower() == 'middle':
                button_to_click = Button.middle

            self.mouse.click(button_to_click, count)
            return f"Successfully clicked at ({x}, {y})."
        except Exception as e:
            logger.error(f"Error clicking mouse: {e}")
            return f"Error: Could not perform mouse click. {e}"

    def scroll(self, dx: int, dy: int):
        """
        Scroll the mouse wheel.

        Args:
            dx: The horizontal scroll amount.
            dy: The vertical scroll amount.
        """
        if not self.mouse:
            logger.error("Mouse controller not available.")
            return "Error: Mouse control is not available."

        logger.info(f"Scrolling by ({dx}, {dy}).")
        try:
            self.mouse.scroll(dx, dy)
            return f"Successfully scrolled by ({dx}, {dy})."
        except Exception as e:
            logger.error(f"Error scrolling mouse: {e}")
            return f"Error: Could not perform mouse scroll. {e}"
