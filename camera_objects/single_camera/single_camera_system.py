"""
    Abstract class for single camera system.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

class SingleCameraSystem(ABC):
    """
    Abstract class for single camera system.

    Functions:
        __init__() -> None
        get_grayscale_image() -> Tuple[bool, np.ndarray]
        get_depth_image() -> Tuple[bool, np.ndarray]
        get_width() -> int
        get_height() -> int
        release() -> bool
    """
    def __init__(self) -> None:
        """
        Initialize single camera system.

        args:
        No arguments.

        returns:
        No return.
        """
    @abstractmethod
    def get_grayscale_image(self) -> Tuple[bool, np.ndarray]:
        """
        Get grayscale image for the camera.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray]:
            - bool: Whether image grabbing is successful or not.
            - np.ndarray: grayscale image.
        """
        return
    @abstractmethod
    def get_depth_image(self) -> Tuple[bool, np.ndarray]:
        """
        Get depth images for the camera system.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray]:
            - bool: Whether depth image grabbing is successful or not.
            - np.ndarray: depth grayscale image.
        """
        return
    @abstractmethod
    def get_width(self) -> int:
        """
        Get width for the camera system.

        args:
        No arguments.

        returns:
        int:
            - int: Width of the camera system.
        """
        return
    @abstractmethod
    def get_height(self) -> int:
        """
        Get height for the camera system.

        args:
        No arguments.

        returns:
        int:
            - int: Height of the camera system.
        """
        return
    @abstractmethod
    def release(self) -> bool:
        """
        Release the camera system.

        args:
        No arguments.

        returns:
        bool:
            - bool: Whether releasing is successful or not.
        """
        return
