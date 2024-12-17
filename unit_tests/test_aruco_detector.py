import unittest
import numpy as np
import cv2
from aruco_detector.aruco_detector import ArUcoDetector

class TestArUcoDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ArUcoDetector()
        # Create a dummy image with ArUco markers for testing
        self.image = np.full((200, 200), fill_value=255, dtype=np.uint8)
        marker = cv2.aruco.generateImageMarker(self.detector.aruco_dict, 0, 100)
        self.image[50:150, 50:150] = marker  # Place the marker in the center with a white border

    def test_detect_aruco(self):
        _, ids = self.detector.detect_aruco(self.image)
        self.assertIsNotNone(ids)
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 0)

    def test_detect_aruco_two_images(self):
        # Create another dummy image with the same ArUco marker
        image_right = np.full((200, 200), fill_value=255, dtype=np.uint8)
        marker_right = cv2.aruco.generateImageMarker(self.detector.aruco_dict, 0, 100)
        image_right[50:150, 50:150] = marker_right  # Place the marker in the center with a white border

        matching_ids, _, _ = self.detector.detect_aruco_two_images(self.image, image_right)
        self.assertIsNotNone(matching_ids)
        self.assertEqual(len(matching_ids), 1)
        self.assertEqual(matching_ids[0], 0)

if __name__ == '__main__':
    unittest.main()