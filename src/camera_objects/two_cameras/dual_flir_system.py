"""
Module for dual FLIR camera system.
"""
from typing import Tuple

import numpy as np

from src.camera_objects.single_camera import FlirCameraSystem
from .two_cameras_system import TwoCamerasSystem

class DualFlirSystem(TwoCamerasSystem):
    """
    Dual FLIR camera system, inherited from TwoCamerasSystem.
    """
    def __init__(self, camera1: FlirCameraSystem, camera2: FlirCameraSystem, synchronized: bool = True) -> None:
        """
        Initialize dual FLIR camera system.

        Parameters
        ----------
        camera1 : FlirCameraSystem
            First FLIR camera system.
        camera2 : FlirCameraSystem
            Second FLIR camera system.
        synchronized : bool, optional
            Should the cameras be synchronized or not, default is True

        Returns
        -------
        None
        """
        super().__init__()

        self.camera1 = camera1
        self.camera2 = camera2

        if synchronized:
            self.camera1.configure_gpio_primary()
            self.camera2.configure_gpio_secondary()

            self.camera1.disable_trigger_mode()

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
        success1, left_image1 = self.camera1.get_grayscale_image()
        success2, left_image2 = self.camera2.get_grayscale_image()

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
        success1, depth_image1 = self.camera1.get_depth_image()
        success2, depth_image2 = self.camera2.get_depth_image()

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
