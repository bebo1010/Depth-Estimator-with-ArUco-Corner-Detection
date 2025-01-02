"""
This module contains unit tests for the EpipolarLineDetector class,
which is responsible for detecting and computing epipolar lines in stereo images.
The tests cover the following functionalities:
- Setting feature detectors
- Switching between different detectors
- Computing epilines from the entire scene
- Computing epilines from specific corner points
Classes
TestEpipolarLineDetector:

"""
import unittest
import cv2
import numpy as np

from src.opencv_objects import EpipolarLineDetector

class TestEpipolarLineDetector(unittest.TestCase):
    """
    Unit tests for the EpipolarLineDetector class.

    This class contains unit tests for the methods of the EpipolarLineDetector class.
    It tests the functionality of setting feature detectors, switching detectors,
    computing epilines, and computing epilines from corners.

    Methods
    -------
    setUp():
        Sets up the test environment before each test method.
    test_set_feature_detector():
        Tests the set_feature_detector method of EpipolarLineDetector.
    test_switch_detector():
        Tests the switch_detector method of EpipolarLineDetector.
    test_compute_epilines():
        Tests the compute_epilines method of EpipolarLineDetector.
    test_compute_epilines_from_corners():
        Tests the compute_epilines_from_corners method of EpipolarLineDetector.
    """

    def setUp(self):
        """
        Sets up the test environment before each test method.

        This method initializes an instance of EpipolarLineDetector, sets a fundamental matrix,
        creates blank left and right images, and draws basic shapes on them. It also initializes
        corner points for testing.
        """
        self.detector = EpipolarLineDetector()
        self.detector.fundamental_matrix = np.array([[1.0, 0.0, -0.5],
                                 [0.0, 1.0, -0.5],
                                 [0.0, 0.0, 1.0]])
        self.left_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.right_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw basic shapes on left_image
        cv2.rectangle(self.left_image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(self.left_image, (300, 300), 50, (0, 255, 0), -1)  # Green circle
        cv2.line(self.left_image, (400, 400), (500, 500), (0, 0, 255), 5)  # Red line

        # Draw basic shapes on right_image
        cv2.rectangle(self.right_image, (60, 60), (160, 160), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(self.right_image, (310, 310), 50, (0, 255, 0), -1)  # Green circle
        cv2.line(self.right_image, (410, 410), (510, 510), (0, 0, 255), 5)  # Red line
        self.corners_left = np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]], dtype=np.float32)
        self.corners_right = np.array([[[110, 110], [210, 110], [210, 210], [110, 210]]], dtype=np.float32)

    def test_set_feature_detector(self):
        """
        Tests the set_feature_detector method of EpipolarLineDetector.

        This method verifies that the feature detector can be set correctly.
        """
        sift = cv2.SIFT_create()
        self.detector.set_feature_detector(sift)
        self.assertEqual(self.detector.detector, sift)

    def test_switch_detector(self):
        """
        Tests the switch_detector method of EpipolarLineDetector.

        This method verifies that the detector can be switched and restored correctly.
        """
        initial_detector = self.detector.detector
        self.detector.switch_detector('n')
        self.assertNotEqual(self.detector.detector, initial_detector)
        self.detector.switch_detector('p')
        self.assertEqual(self.detector.detector, initial_detector)

    def test_compute_epilines(self):
        """
        Tests the compute_epilines method of EpipolarLineDetector.

        This method verifies that epilines can be computed and drawn on the images correctly.
        """
        self.detector.set_feature_detector(cv2.ORB_create())
        left_image_with_lines, right_image_with_lines = \
            self.detector.compute_epilines_from_scene(self.left_image, self.right_image)
        self.assertEqual(left_image_with_lines.shape, self.left_image.shape)
        self.assertEqual(right_image_with_lines.shape, self.right_image.shape)

    def test_compute_epilines_from_corners(self):
        """
        Tests the compute_epilines_from_corners method of EpipolarLineDetector.

        This method verifies that epilines can be computed from corner points and drawn on the images correctly.
        """
        left_image_with_lines, right_image_with_lines = self.detector.compute_epilines_from_corners(
            self.left_image, self.right_image, self.corners_left, self.corners_right)
        self.assertEqual(left_image_with_lines.shape, self.left_image.shape)
        self.assertEqual(right_image_with_lines.shape, self.right_image.shape)


if __name__ == '__main__':
    unittest.main()
