import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from src.vtube import VTubeStudio

class TestVTubeStudio(unittest.TestCase):

    def setUp(self):
        self.vts_available_patcher = patch('src.vtube.VTS_AVAILABLE', True)
        self.pyvts_patcher = patch('src.vtube.pyvts')

        self.mock_vts_available = self.vts_available_patcher.start()
        self.mock_pyvts = self.pyvts_patcher.start()

        # Mock the VTS class and its async methods
        self.mock_vts_instance = MagicMock()
        self.mock_vts_instance.connect = AsyncMock()
        self.mock_vts_instance.verify_authentication_token = AsyncMock()
        self.mock_vts_instance.request_authentication_token = AsyncMock(return_value={'authenticationToken': 'test_token'})
        self.mock_vts_instance.request = AsyncMock()
        self.mock_vts_instance.close = AsyncMock()
        self.mock_pyvts.VTS.return_value = self.mock_vts_instance

    def tearDown(self):
        self.vts_available_patcher.stop()
        self.pyvts_patcher.stop()

    def test_vtube_init_success(self):
        vtube = VTubeStudio()
        # To test the async connect method, we need to run the event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.sleep(0.01)) # Allow time for the connect task to run

        self.assertTrue(vtube.is_available)
        self.mock_pyvts.VTS.assert_called_once()
        self.mock_vts_instance.connect.assert_called_once()

    def test_set_emotion(self):
        vtube = VTubeStudio()
        vtube.trigger_hotkey = AsyncMock() # Mock the async method

        vtube.set_emotion("happy")

        # To test the async trigger_hotkey method, we need to run the event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.sleep(0.01))

        vtube.trigger_hotkey.assert_called_once_with("TriggerHappy")

class TestVTubeStudioNoDependencies(unittest.TestCase):

    def test_vtube_init_no_dependencies(self):
        with patch('src.vtube.VTS_AVAILABLE', False):
            from src.vtube import VTubeStudio
            vtube = VTubeStudio()
            self.assertFalse(vtube.is_available)

if __name__ == '__main__':
    unittest.main()
