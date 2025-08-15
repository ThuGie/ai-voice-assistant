import unittest
from unittest.mock import patch, MagicMock
import src.control

class TestPCController(unittest.TestCase):

    def setUp(self):
        self.pynput_patcher = patch('src.control.PYNPUT_AVAILABLE', True)
        self.mouse_patcher = patch('src.control.MouseController')
        self.keyboard_patcher = patch('src.control.KeyboardController')

        self.mock_pynput_available = self.pynput_patcher.start()
        self.mock_mouse_controller = self.mouse_patcher.start()
        self.mock_keyboard_controller = self.keyboard_patcher.start()

        from src.control import PCController
        self.controller = PCController()

    def tearDown(self):
        self.pynput_patcher.stop()
        self.mouse_patcher.stop()
        self.keyboard_patcher.stop()

    def test_type_text(self):
        self.controller.type_text("hello")
        self.assertEqual(self.controller.keyboard.press.call_count, 5)
        self.assertEqual(self.controller.keyboard.release.call_count, 5)

    def test_click(self):
        with patch('src.control.Button') as mock_button:
            self.controller.click(100, 200)
            self.controller.mouse.click.assert_called_once()

    def test_scroll(self):
        self.controller.scroll(0, 10)
        self.controller.mouse.scroll.assert_called_once_with(0, 10)

if __name__ == '__main__':
    unittest.main()
