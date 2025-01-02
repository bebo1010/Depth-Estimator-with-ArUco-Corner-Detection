"""
Unit tests for the ChessboardCalibrator class.
"""

from typing import List, Tuple
import unittest
import numpy as np
import cv2
from src.opencv_objects import ChessboardCalibrator

class TestChessboardCalibrator(unittest.TestCase):
    """
    Test cases for the ChessboardCalibrator class.
    """

    def setUp(self):
        """
        Set up the test case environment.
        """
        self.calibrator = ChessboardCalibrator()

    def test_initialization(self):
        """
        Test the initialization of the ChessboardCalibrator.
        """
        self.assertEqual(self.calibrator.pattern_size, (11, 7))
        self.assertEqual(self.calibrator.square_size_mm, 30.0)

    def test_set_pattern_size(self):
        """
        Test setting a new pattern size for the chessboard.
        """
        new_pattern_size = (9, 6)
        self.calibrator.pattern_size = new_pattern_size
        self.assertEqual(self.calibrator.pattern_size, new_pattern_size)

    def test_set_square_size_mm(self):
        """
        Test setting a new square size for the chessboard in millimeters.
        """
        new_square_size_mm = 25.0
        self.calibrator.square_size_mm = new_square_size_mm
        self.assertEqual(self.calibrator.square_size_mm, new_square_size_mm)

    def test_detect_chessboard_corners_success(self):
        """
        Test successful detection of chessboard corners in an image.
        """
        # Create a synthetic chessboard image with white squares on a black background
        pattern_size = (8, 8)
        square_size = 50
        chessboard_image = np.zeros((pattern_size[0] * square_size, pattern_size[1] * square_size), dtype=np.uint8)

        for i in range(pattern_size[0]):
            for j in range(pattern_size[1]):
                if (i + j) % 2 == 0:
                    cv2.rectangle(chessboard_image,
                        (j * square_size, i * square_size),
                        ((j + 1) * square_size, (i + 1) * square_size),
                        (255, 255, 255),
                        -1)

        self.calibrator.pattern_size = (pattern_size[0] - 1, pattern_size[1] - 1)
        ret, corners = self.calibrator.detect_chessboard_corners(chessboard_image)
        self.assertTrue(ret)
        self.assertIsNotNone(corners)

    def test_detect_chessboard_corners_failure(self):
        """
        Test failure to detect chessboard corners in an image without a chessboard.
        """
        # Create an image without a chessboard
        image = np.zeros((500, 500), dtype=np.uint8)

        ret, corners = self.calibrator.detect_chessboard_corners(image)
        self.assertFalse(ret)
        self.assertIsNone(corners)

    def test_calibrate_single_camera_success(self):
        """
        Test successful calibration of a single camera.
        """
        pattern_size = (7, 7)
        self.calibrator.pattern_size = pattern_size
        image_points = self._generate_image_points(pattern_size, 20)
        image_size = (640, 480)
        ret = self.calibrator.calibrate_single_camera(image_points, image_size, camera_index=0)
        self.assertTrue(ret)
        self.assertIn("camera_matrix", self.calibrator.left_camera_parameters)

    def test_calibrate_single_camera_failure(self):
        """
        Test failure to calibrate a single camera with no image points.
        """
        image_points = []  # No image points
        image_size = (640, 480)
        with self.assertRaises(AssertionError):
            self.calibrator.calibrate_single_camera(image_points, image_size, camera_index=0)

    def test_calibrate_stereo_camera_success(self):
        """
        Test successful calibration of a stereo camera setup.
        """
        pattern_size = (7, 7)
        self.calibrator.pattern_size = pattern_size
        left_image_points = self._generate_image_points(pattern_size, 20)
        right_image_points = self._generate_image_points(pattern_size, 20)
        image_size = (640, 480)
        ret = self.calibrator.calibrate_stereo_camera(left_image_points, right_image_points, image_size)
        self.assertTrue(ret)
        self.assertIn("camera_matrix_left", self.calibrator.stereo_camera_parameters)
        self.assertIn("camera_matrix_right", self.calibrator.stereo_camera_parameters)

    def test_calibrate_stereo_camera_failure(self):
        """
        Test failure to calibrate a stereo camera setup with mismatched number of points.
        """
        left_image_points = [np.random.rand(49, 1, 2).astype(np.float32) for _ in range(10)]
        right_image_points = []  # Mismatched number of points
        image_size = (640, 480)
        with self.assertRaises(AssertionError):
            self.calibrator.calibrate_stereo_camera(left_image_points, right_image_points, image_size)

    def test_rectify_images_success(self):
        """
        Test successful rectification of stereo images.
        """
        left_image = np.random.rand(480, 640, 3).astype(np.uint8)
        right_image = np.random.rand(480, 640, 3).astype(np.uint8)
        self.calibrator.stereo_camera_parameters = {
            "camera_matrix_left": np.eye(3),
            "distortion_coefficients_left": np.zeros(5),
            "camera_matrix_right": np.eye(3),
            "distortion_coefficients_right": np.zeros(5),
            "left_rectified_rotation_matrix": np.eye(3),
            "right_rectified_rotation_matrix": np.eye(3),
            "left_projection_matrix": np.eye(3),
            "right_projection_matrix": np.eye(3)
        }
        left_rectified, right_rectified = self.calibrator.rectify_images(left_image, right_image)
        self.assertEqual(left_rectified.shape, left_image.shape)
        self.assertEqual(right_rectified.shape, right_image.shape)

    def test_rectify_images_failure(self):
        """
        Test failure to rectify stereo images with missing calibration parameters.
        """
        left_image = np.random.rand(480, 640, 3).astype(np.uint8)
        right_image = np.random.rand(480, 640, 3).astype(np.uint8)
        with self.assertRaises(KeyError):
            self.calibrator.rectify_images(left_image, right_image)

    def _generate_image_points(self, pattern_size: Tuple[int, int], num_images: int) -> List[np.ndarray]:
        """
        Helper method to generate synthetic image points for testing.

        Args:
            pattern_size (Tuple[int, int]): Size of the chessboard pattern.
            num_images (int): Number of images.

        Returns:
            List[np.ndarray]: List of synthetic image points.
        """
        total_squares = pattern_size[0] * pattern_size[1]
        object_points = np.zeros((total_squares, 2), np.float32)
        object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        image_points = []
        for _ in range(num_images):
            # Apply random rotation and scaling
            angle = np.random.uniform(-30, 30)
            scale = np.random.uniform(0.8, 1.2)
            rotation_matrix = cv2.getRotationMatrix2D((pattern_size[0] / 2, pattern_size[1] / 2), angle, scale)
            transformed_points = cv2.transform(np.array([object_points[:, :2]]), rotation_matrix)[0]

            # Add some noise
            noise = np.random.normal(0, 0.5, transformed_points.shape)
            transformed_points += noise

            image_points.append(transformed_points.astype(np.float32).reshape(-1, 1, 2))

        return image_points

if __name__ == '__main__':
    unittest.main()
