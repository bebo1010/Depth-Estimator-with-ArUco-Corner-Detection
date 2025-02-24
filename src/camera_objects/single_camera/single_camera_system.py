"""
Abstract class for single camera system.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

class SingleCameraSystem(ABC):
    """
    Abstract class for single camera system.
    """
    def __init__(self) -> None:
        """
        Initialize single camera system.

        Returns
        -------
        None
        """
    @abstractmethod
    def get_grayscale_image(self) -> Tuple[bool, np.ndarray]:
        """
        Get grayscale image for the camera.

        Returns
        -------
        Tuple[bool, np.ndarray]
            - bool: Whether image grabbing is successful or not.
            - np.ndarray: Grayscale image.
        """
        return
    @abstractmethod
    def get_depth_image(self) -> Tuple[bool, np.ndarray]:
        """
        Get depth images for the camera system.

        Returns
        -------
        Tuple[bool, np.ndarray]
            - bool: Whether depth image grabbing is successful or not.
            - np.ndarray: Depth grayscale image.
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
