import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import logging
from camera_objects.dual_realsense_system import DualRealsenseSystem
from camera_objects.realsense_camera_system import RealsenseCameraSystem

class TestDualRealsenseSystem(unittest.TestCase):
    @patch('camera_objects.realsense_camera_system.rs.pipeline')
    def setUp(self, mock_pipeline):
        logging.disable(logging.CRITICAL)  # Suppress log messages below CRITICAL level

        # Mock the RealsenseCameraSystem instances
        self.mock_camera1 = MagicMock(spec=RealsenseCameraSystem)
        self.mock_camera2 = MagicMock(spec=RealsenseCameraSystem)

        # Set return values for get_width and get_height methods
        self.mock_camera1.get_width.return_value = 640
        self.mock_camera1.get_height.return_value = 480
        self.mock_camera2.get_width.return_value = 640
        self.mock_camera2.get_height.return_value = 480

        # Set up the DualRealsenseSystem with the mocked cameras
        self.camera_system = DualRealsenseSystem(self.mock_camera1, self.mock_camera2)

    def tearDown(self):
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_get_grayscale_images_success(self):
        self.mock_camera1.get_grayscale_images.return_value = (True, np.zeros((480, 640), dtype=np.uint8), None)
        self.mock_camera2.get_grayscale_images.return_value = (True, np.zeros((480, 640), dtype=np.uint8), None)

        success, left_image, right_image = self.camera_system.get_grayscale_images()
        self.assertTrue(success)
        self.assertIsNotNone(left_image)
        self.assertIsNotNone(right_image)

    def test_get_grayscale_images_failure(self):
        self.mock_camera1.get_grayscale_images.return_value = (False, None, None)
        self.mock_camera2.get_grayscale_images.return_value = (False, None, None)

        success, left_image, right_image = self.camera_system.get_grayscale_images()
        self.assertFalse(success)
        self.assertIsNone(left_image)
        self.assertIsNone(right_image)

    def test_get_depth_image_success(self):
        self.mock_camera1.get_depth_image.return_value = (True, np.zeros((480, 640), dtype=np.uint16), None)
        self.mock_camera2.get_depth_image.return_value = (True, np.zeros((480, 640), dtype=np.uint16), None)

        success, depth_image1, depth_image2 = self.camera_system.get_depth_image()
        self.assertTrue(success)
        self.assertIsNotNone(depth_image1)
        self.assertIsNotNone(depth_image2)

    def test_get_depth_image_failure(self):
        self.mock_camera1.get_depth_image.return_value = (False, None, None)
        self.mock_camera2.get_depth_image.return_value = (False, None, None)

        success, depth_image1, depth_image2 = self.camera_system.get_depth_image()
        self.assertFalse(success)
        self.assertIsNone(depth_image1)
        self.assertIsNone(depth_image2)

    def test_get_width(self):
        self.assertEqual(self.camera_system.get_width(), 640)

    def test_get_height(self):
        self.assertEqual(self.camera_system.get_height(), 480)

    def test_release(self):
        self.mock_camera1.release.return_value = True
        self.mock_camera2.release.return_value = True

        self.assertTrue(self.camera_system.release())
        self.mock_camera1.release.assert_called_once()
        self.mock_camera2.release.assert_called_once()

if __name__ == '__main__':
    unittest.main()
