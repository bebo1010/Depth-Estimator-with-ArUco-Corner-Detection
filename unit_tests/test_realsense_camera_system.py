import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import logging
from camera_objects.realsense_camera_system import RealsenseCameraSystem

class TestRealsenseCameraSystem(unittest.TestCase):
    @patch('camera_objects.realsense_camera_system.rs.pipeline')
    def setUp(self, mock_pipeline):
        logging.disable(logging.CRITICAL)  # Suppress log messages below CRITICAL level
        self.mock_pipeline = mock_pipeline.return_value
        self.mock_pipeline.wait_for_frames = MagicMock()
        self.camera_system = RealsenseCameraSystem(640, 480)

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_get_grayscale_images_success(self):
        mock_frames = MagicMock()
        mock_ir_frame_left = MagicMock()
        mock_ir_frame_right = MagicMock()
        mock_ir_frame_left.get_data.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_ir_frame_right.get_data.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_frames.get_infrared_frame.side_effect = [mock_ir_frame_left, mock_ir_frame_right]
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, left_image, right_image = self.camera_system.get_grayscale_images()
        self.assertTrue(success)
        self.assertIsNotNone(left_image)
        self.assertIsNotNone(right_image)

    def test_get_grayscale_images_failure(self):
        mock_frames = MagicMock()
        mock_frames.get_infrared_frame.side_effect = [None, None]
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, left_image, right_image = self.camera_system.get_grayscale_images()
        self.assertFalse(success)
        self.assertIsNone(left_image)
        self.assertIsNone(right_image)

    def test_get_depth_image_success(self):
        mock_frames = MagicMock()
        mock_depth_frame = MagicMock()
        mock_depth_frame.get_data.return_value = np.zeros((480, 640), dtype=np.uint16)
        mock_frames.get_depth_frame.return_value = mock_depth_frame
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, depth_image, _ = self.camera_system.get_depth_image()
        self.assertTrue(success)
        self.assertIsNotNone(depth_image)

    def test_get_depth_image_failure(self):
        mock_frames = MagicMock()
        mock_frames.get_depth_frame.return_value = None
        self.mock_pipeline.wait_for_frames.return_value = mock_frames

        success, depth_image, _ = self.camera_system.get_depth_image()
        self.assertFalse(success)
        self.assertIsNone(depth_image)

    def test_get_width(self):
        self.assertEqual(self.camera_system.get_width(), 640)

    def test_get_height(self):
        self.assertEqual(self.camera_system.get_height(), 480)

    def test_release(self):
        self.assertTrue(self.camera_system.release())
        self.mock_pipeline.stop.assert_called_once()

if __name__ == '__main__':
    unittest.main()
