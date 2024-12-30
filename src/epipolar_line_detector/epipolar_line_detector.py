"""
Module: epipolar_line_detector
Description: This module contains the EpipolarLineDetector class for detecting epipolar lines using OpenCV.
"""
from typing import Tuple

import cv2
import numpy as np

class EpipolarLineDetector:
    """
    A class to detect epipolar lines using OpenCV.

    Methods
    -------
    set_feature_detector(detector):
        Sets the feature detector to be used.

    compute_epilines(left_image, right_image, points, fundamental_matrix):
        Detects features and computes the epipolar lines for the given points.
    """

    def __init__(self):
        """
        Initializes the EpipolarLineDetector class.
        """
        self.detector = None
        self.detector_index = 0
        self.detectors = [
            ("ORB", cv2.ORB_create()),
            ("SIFT", cv2.SIFT_create()),
            # ("SURF", cv2.xfeatures2d.SURF_create()),
            # ("FAST", cv2.FastFeatureDetector_create()),
            # ("BRIEF", cv2.xfeatures2d.BriefDescriptorExtractor_create()),
            # ("KAZE", cv2.KAZE_create())
        ]
        self.set_feature_detector(self.detectors[self.detector_index][1])
        self.fundamental_matrix = None

    def set_feature_detector(self, detector: cv2.Feature2D) -> None:
        """
        Sets the feature detector to be used.

        For example, you can pass in an instance of cv2.SIFT() or cv2.ORB().

        Example
        -------
        >>> detector = cv2.SIFT_create()
        >>> obj.set_feature_detector(detector)

        Parameters
        ----------
        detector : cv2.Feature2D
            The feature detector to be used.
        """
        self.detector = detector

    def switch_detector(self, direction: str) -> None:
        """
        Switches the feature detector cyclically.

        Parameters
        ----------
        direction : str
            Direction to switch the detector ('n' for next, 'p' for previous).
        """
        if direction == 'n':
            self.detector_index = (self.detector_index + 1) % len(self.detectors)
        elif direction == 'p':
            self.detector_index = (self.detector_index - 1) % len(self.detectors)
        self.set_feature_detector(self.detectors[self.detector_index][1])

    def compute_epilines_from_scene(self,
                         left_image: np.ndarray,
                         right_image: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects features and computes the epipolar lines for the detected points.

        Parameters
        ----------
        left_image : numpy.ndarray
            The left input image in which to detect features.
        right_image : numpy.ndarray
            The right input image in which to detect features.

        Returns
        -------
        left_image_with_lines : numpy.ndarray
            The left image with epipolar lines drawn.
        right_image_with_lines : numpy.ndarray
            The right image with epipolar lines drawn.
        """
        if self.detector is None:
            raise ValueError("Feature detector is not set. Use set_feature_detector method to set it.")

        keypoints_left, _ = self.detector.detectAndCompute(left_image, None)
        keypoints_right, _ = self.detector.detectAndCompute(right_image, None)

        points_left = cv2.KeyPoint.convert(keypoints_left)
        points_right = cv2.KeyPoint.convert(keypoints_right)

        if self.fundamental_matrix is None:
            # Match features between the left and right images
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(points_left, points_right)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract the matched points
            points_left = np.float32([points_left[m.queryIdx] for m in matches])
            points_right = np.float32([points_right[m.trainIdx] for m in matches])
            self.fundamental_matrix, _ = cv2.findFundamentalMat(points_left, points_right,
                                                       method=cv2.FM_RANSAC,
                                                       ransacReprojThreshold=3,
                                                       confidence=0.99,
                                                       maxIters=100)

        epilines_left = cv2.computeCorrespondEpilines(points_right, 2, self.fundamental_matrix).reshape(-1, 3)
        epilines_right = cv2.computeCorrespondEpilines(points_left, 1, self.fundamental_matrix).reshape(-1, 3)

        # Limit the number of epipolar lines to 10
        num_lines = min(100, len(epilines_left), len(epilines_right))

        left_image_with_lines = self._draw_epilines(left_image,
                                                    epilines_left[:num_lines], points_left[:num_lines])
        right_image_with_lines = self._draw_epilines(right_image,
                                                     epilines_right[:num_lines], points_right[:num_lines])

        return left_image_with_lines, right_image_with_lines

    def compute_epilines_from_corners(self,
                                      left_image: np.ndarray,
                                      right_image: np.ndarray,
                                      corners_left: np.ndarray,
                                      corners_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute epipolar lines from the corner points of detected ArUco markers.

        Parameters
        ----------
        left_image : numpy.ndarray
            The left input image.
        right_image : numpy.ndarray
            The right input image.
        corners_left : numpy.ndarray
            Corner points from the left image.
        corners_right : numpy.ndarray
            Corner points from the right image.

        Returns
        -------
        left_image_with_lines : numpy.ndarray
            The left image with epipolar lines drawn.
        right_image_with_lines : numpy.ndarray
            The right image with epipolar lines drawn.
        """
        # Ensure points are in the correct shape and type
        points_left = np.asarray(corners_left, dtype=np.float32).reshape(-1, 2)
        points_right = np.asarray(corners_right, dtype=np.float32).reshape(-1, 2)

        if self.fundamental_matrix is None:
            self.fundamental_matrix, _ = cv2.findFundamentalMat(points_left, points_right,
                                                        method=cv2.FM_RANSAC,
                                                        ransacReprojThreshold=3,
                                                        confidence=0.99,
                                                        maxIters=100)

        points_left = corners_left.reshape(-1, 2)
        points_right = corners_right.reshape(-1, 2)

        epilines_left = cv2.computeCorrespondEpilines(points_right, 2, self.fundamental_matrix).reshape(-1, 3)
        epilines_right = cv2.computeCorrespondEpilines(points_left, 1, self.fundamental_matrix).reshape(-1, 3)

        left_image_with_lines = self._draw_epilines(left_image, epilines_left, points_left)
        right_image_with_lines = self._draw_epilines(right_image, epilines_right, points_right)

        return left_image_with_lines, right_image_with_lines

    def _draw_epilines(self, image: np.ndarray, epilines: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Draws the epipolar lines on the image.

        Parameters
        ----------
        image : numpy.ndarray
            The image on which to draw the epipolar lines.
        epilines : numpy.ndarray
            The epipolar lines to be drawn.
        points : numpy.ndarray
            Points corresponding to the epipolar lines.

        Returns
        -------
        image_with_lines : numpy.ndarray
            The image with epipolar lines drawn.
        """
        image_with_lines = image.copy()
        for r, pt in zip(epilines, points):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [image.shape[1], -(r[2] + r[0] * image.shape[1]) / r[1]])
            image_with_lines = cv2.line(image_with_lines, (x0, y0), (x1, y1), color, 1)
            image_with_lines = cv2.circle(image_with_lines, tuple(map(int, pt)), 5, color, -1)
        return image_with_lines
