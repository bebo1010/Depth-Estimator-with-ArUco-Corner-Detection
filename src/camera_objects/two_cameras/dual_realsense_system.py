"""
Module for Realsense camera system.
"""
from typing import Tuple

import numpy as np

from .two_cameras_system import TwoCamerasSystem
from .realsense_camera_system import RealsenseCameraSystem

class DualRealsenseSystem(TwoCamerasSystem):
    """
    Realsense camera system, inherited from TwoCamerasSystem.
    """
    def __init__(self, camera1: RealsenseCameraSystem, camera2: RealsenseCameraSystem) -> None:
        """
        Initialize dual realsense camera system.

        Parameters
        ----------
        camera1 : RealsenseCameraSystem
            First realsense camera system.
        camera2 : RealsenseCameraSystem
            Second realsense camera system.

        Returns
        -------
        None
        """
        super().__init__()

        self.camera1 = camera1
        self.camera2 = camera2

        self.width = camera1.get_width()
        self.height = camera1.get_height()

    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get grayscale images for both cameras.

        Returns
        -------
        Tuple[bool, np.ndarray, np.ndarray]
            - bool: Whether images grabbing is successful or not.
            - np.ndarray: Grayscale image for left camera.
            - np.ndarray: Grayscale image for right camera.
        """
        success1, left_image1, _ = self.camera1.get_grayscale_images()
        success2, left_image2, _ = self.camera2.get_grayscale_images()

        total_success = success1 and success2
        return total_success, left_image1, left_image2

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
        success1, depth_image1, _ = self.camera1.get_depth_images()
        success2, depth_image2, _ = self.camera2.get_depth_images()

        total_success = success1 and success2
        return total_success, depth_image1, depth_image2

    def get_width(self) -> int:
        """
        Get width for the camera system.

        Returns
        -------
        int
            Width of the camera system.
        """
        return self.width

    def get_height(self) -> int:
        """
        Get height for the camera system.

        Returns
        -------
        int
            Height of the camera system.
        """
        return self.height

    def release(self) -> bool:
        """
        Release the camera system.

        Returns
        -------
        bool
            Whether releasing is successful or not.
        """
        self.camera1.release()
        self.camera2.release()
        return True
