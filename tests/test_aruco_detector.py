"""
Unit tests for the ArUcoDetector class.
"""
import logging

import unittest
import coverage

import numpy as np
import cv2

from src.opencv_objects import ArUcoDetector

class TestArUcoDetector(unittest.TestCase):
    """
    Test suite for the ArUcoDetector class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        """
        logging.disable(logging.CRITICAL)  # Suppress log messages below CRITICAL level

        self.detector = ArUcoDetector()

        # Create a dummy image with ArUco markers for testing
        self.image = np.full((200, 200), fill_value=255, dtype=np.uint8)
        marker = cv2.aruco.generateImageMarker(self.detector.aruco_dict, 0, 100)
        self.image[50:150, 50:150] = marker  # Place the marker in the center with a white border

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        logging.disable(logging.NOTSET)  # Re-enable logging after tests

    def test_detect_aruco(self):
        """
        Test detection of a single ArUco marker.
        """
        _, ids = self.detector.detect_aruco(self.image)
        self.assertIsNotNone(ids)
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 0)

    def test_detect_aruco_two_images(self):
        """
        Test detection of ArUco markers in two images.
        """
        # Create another dummy image with the same ArUco marker
        image_right = np.full((200, 200), fill_value=255, dtype=np.uint8)
        marker_right = cv2.aruco.generateImageMarker(self.detector.aruco_dict, 0, 100)
        image_right[50:150, 50:150] = marker_right  # Place the marker in the center with a white border

        matching_ids, _, _ = self.detector.detect_aruco_two_images(self.image, image_right)
        self.assertIsNotNone(matching_ids)
        self.assertEqual(len(matching_ids), 1)
        self.assertEqual(matching_ids[0], 0)

    def test_detect_aruco_no_markers(self):
        """
        Test detection when no ArUco markers are present in the image.
        """
        empty_image = np.full((200, 200), fill_value=255, dtype=np.uint8)  # Completely white image
        _, ids = self.detector.detect_aruco(empty_image)
        self.assertIsNone(ids)

    def test_detect_aruco_two_images_no_markers(self):
        """
        Test detection when no ArUco markers are present in both images.
        """
        empty_image_left = np.full((200, 200), fill_value=255, dtype=np.uint8)  # Completely white image
        empty_image_right = np.full((200, 200), fill_value=255, dtype=np.uint8)  # Completely white image

        matching_ids, _, _ = self.detector.detect_aruco_two_images(empty_image_left, empty_image_right)
        self.assertEqual(len(matching_ids), 0)

if __name__ == '__main__':
    cov = coverage.Coverage()
    cov.start()

    unittest.main()

    cov.stop()
    cov.save()

    cov.html_report()
    print("Done.")
