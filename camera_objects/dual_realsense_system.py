"""
Module for Realsense camera system.
"""
from typing import Tuple

import numpy as np

from camera_objects.camera_abstract_class import TwoCamerasSystem
from camera_objects.realsense_camera_system import RealsenseCameraSystem

class DualRealsenseSystem(TwoCamerasSystem):
    """
    Realsense camera system, inherited from TwoCamerasSystem.

    Functions:
        __init__(RealsenseCameraSystem, RealsenseCameraSystem) -> None
        get_grayscale_images() -> Tuple[bool, np.ndarray, np.ndarray]
        get_depth_image() -> Tuple[bool, np.ndarray]
        get_width() -> int
        get_height() -> int
        release() -> bool
    """
    def __init__(self, camera1: RealsenseCameraSystem, camera2: RealsenseCameraSystem) -> None:
        """
        Initialize dual realsense camera system.

        args:
            camera1 (RealsenseCameraSystem): First realsense camera system.
            camera2 (RealsenseCameraSystem): Second realsense camera system.

        returns:
        No return.
        """
        super().__init__()

        self.camera1 = camera1
        self.camera2 = camera2

        self.width = camera1.get_width()
        self.height = camera1.get_height()

    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get grayscale images for both cameras.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray, np.ndarray]:
            - bool: Whether images grabbing is successful or not.
            - np.ndarray: grayscale image for left camera.
            - np.ndarray: grayscale image for right camera.
        """
        success1, left_image1, _ = self.camera1.get_grayscale_images()
        success2, left_image2, _ = self.camera2.get_grayscale_images()

        total_success = success1 and success2
        return total_success, left_image1, left_image2

    def get_depth_image(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get depth images for the camera system.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray]:
            - bool: Whether depth image grabbing is successful or not.
            - np.ndarray: first depth grayscale image.
            - np.ndarray: second depth grayscale image.
        """
        success1, depth_image1, _ = self.camera1.get_depth_image()
        success2, depth_image2, _ = self.camera2.get_depth_image()

        total_success = success1 and success2
        return total_success, depth_image1, depth_image2

    def get_width(self) -> int:
        """
        Get width for the camera system.

        args:
        No arguments.

        returns:
        int:
            - int: Width of the camera system.
        """
        return self.width

    def get_height(self) -> int:
        """
        Get height for the camera system.

        args:
        No arguments.

        returns:
        int:
            - int: Height of the camera system.
        """
        return self.height

    def release(self) -> bool:
        """
        Release the camera system.

        args:
        No arguments.

        returns:
        bool:
            - bool: Whether releasing is successful or not.
        """
        self.camera1.release()
        self.camera2.release()
        return True
