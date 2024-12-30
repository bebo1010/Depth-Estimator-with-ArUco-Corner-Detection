"""
Module: epipolar_line_detector
Description: This module contains the EpipolarLineDetector class for detecting epipolar lines using OpenCV.
"""
from typing import List, Tuple, Optional

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
            ("SIFT", cv2.SIFT_create()),
            # ("SURF", cv2.xfeatures2d.SURF_create()),
            # ("FAST", cv2.FastFeatureDetector_create()),
            # ("BRIEF", cv2.xfeatures2d.BriefDescriptorExtractor_create()),
            ("ORB", cv2.ORB_create()),
            # ("KAZE", cv2.KAZE_create())
        ]
        self.set_feature_detector(self.detectors[self.detector_index][1])

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

    def compute_epilines(self,
                         left_image: np.ndarray,
                         right_image: np.ndarray,
                         fundamental_matrix: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects features and computes the epipolar lines for the detected points.

        Parameters
        ----------
        left_image : numpy.ndarray
            The left input image in which to detect features.
        right_image : numpy.ndarray
            The right input image in which to detect features.
        fundamental_matrix : numpy.ndarray, optional
            The fundamental matrix. If not provided, it will be computed from the feature points.

        Returns
        -------
        left_image_with_lines : numpy.ndarray
            The left image with epipolar lines drawn.
        right_image_with_lines : numpy.ndarray
            The right image with epipolar lines drawn.
        """
        if self.detector is None:
            raise ValueError("Feature detector is not set. Use set_feature_detector method to set it.")

        keypoints_left, descriptors_left = self.detector.detectAndCompute(left_image, None)
        keypoints_right, descriptors_right = self.detector.detectAndCompute(right_image, None)

        if fundamental_matrix is None:
            fundamental_matrix = self._compute_fundamental_matrix(keypoints_left, descriptors_left,
                                                                  keypoints_right, descriptors_right)

        points_left = cv2.KeyPoint.convert(keypoints_left)
        points_right = cv2.KeyPoint.convert(keypoints_right)

        epilines_left = cv2.computeCorrespondEpilines(points_right, 2, fundamental_matrix).reshape(-1, 3)
        epilines_right = cv2.computeCorrespondEpilines(points_left, 1, fundamental_matrix).reshape(-1, 3)

        # Limit the number of epipolar lines to 10
        num_lines = min(10, len(epilines_left), len(epilines_right))

        left_image_with_lines = self._draw_epilines(left_image,
                                                    epilines_left[:num_lines], keypoints_left[:num_lines])
        right_image_with_lines = self._draw_epilines(right_image,
                                                     epilines_right[:num_lines], keypoints_right[:num_lines])

        return left_image_with_lines, right_image_with_lines

    def _compute_fundamental_matrix(self,
                                    keypoints_left: List[cv2.KeyPoint],
                                    descriptors_left: np.ndarray,
                                    keypoints_right: List[cv2.KeyPoint],
                                    descriptors_right: np.ndarray) -> np.ndarray:
        """
        Computes the fundamental matrix from the matched feature points.

        Parameters
        ----------
        keypoints_left : list
            List of keypoints from the left image.
        descriptors_left : numpy.ndarray
            Descriptors from the left image.
        keypoints_right : list
            List of keypoints from the right image.
        descriptors_right : numpy.ndarray
            Descriptors from the right image.

        Returns
        -------
        fundamental_matrix : numpy.ndarray
            The computed fundamental matrix.
        """
        # Match features between the two images using BFMatcher for binary descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors_left, descriptors_right)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        points_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches])
        points_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches])

        # Compute the fundamental matrix
        fundamental_matrix, _ = cv2.findFundamentalMat(points_left, points_right, cv2.FM_LMEDS)

        return fundamental_matrix

    def _draw_epilines(self, image: np.ndarray, epilines: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """
        Draws the epipolar lines on the image.

        Parameters
        ----------
        image : numpy.ndarray
            The image on which to draw the epipolar lines.
        epilines : numpy.ndarray
            The epipolar lines to be drawn.
        keypoints : list
            List of keypoints corresponding to the epipolar lines.

        Returns
        -------
        image_with_lines : numpy.ndarray
            The image with epipolar lines drawn.
        """
        image_with_lines = image.copy()
        for r, kp in zip(epilines, keypoints):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [image.shape[1], -(r[2] + r[0] * image.shape[1]) / r[1]])
            image_with_lines = cv2.line(image_with_lines, (x0, y0), (x1, y1), color, 1)
            image_with_lines = cv2.circle(image_with_lines, tuple(map(int, kp.pt)), 5, color, -1)
        return image_with_lines
