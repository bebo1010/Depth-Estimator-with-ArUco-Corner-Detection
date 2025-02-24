"""
Abstract class for two cameras system.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import cv2

class TwoCamerasSystem(ABC):
    """
    Abstract class for two cameras system.
    """

    def __init__(self) -> None:
        """
        Initialize two cameras system.

        Returns
        -------
        None
        """

    @abstractmethod
    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get grayscale images for both cameras.
        """
        return

    def get_rgb_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get RGB images for both cameras.

        Returns
        -------
        Tuple[bool, np.ndarray, np.ndarray]
            - bool: Whether images grabbing is successful or not.
            - np.ndarray: Left RGB image (BGR in OpenCV format).
            - np.ndarray: Right RGB image (BGR in OpenCV format).
        """
        total_success, left_image1, left_image2 = self.get_grayscale_images()
        try:
            left_image1 = cv2.cvtColor(left_image1, cv2.COLOR_GRAY2BGR)
            left_image2 = cv2.cvtColor(left_image2, cv2.COLOR_GRAY2BGR)
        except cv2.error:
            total_success = False

        return total_success, left_image1, left_image2

    @abstractmethod
    def get_depth_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get depth images for the camera system.

        Returns
        -------
        Tuple[bool, np.ndarray, np.ndarray]
            - bool: Whether depth image grabbing is successful or not.
            - np.ndarray: First depth grayscale image.
            - np.ndarray: Second depth grayscale image.
        """
        return

    @abstractmethod
    def get_width(self) -> int:
        """
        Get width for the camera system.

        Returns
        -------
        int
            Width of the camera system.
        """
        return

    @abstractmethod
    def get_height(self) -> int:
        """
        Get height for the camera system.

        Returns
        -------
        int
            Height of the camera system.
        """
        return

    @abstractmethod
    def release(self) -> bool:
        """
        Release the camera system.

        Returns
        -------
        bool
            Whether releasing is successful or not.
        """
        return
