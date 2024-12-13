"""
Module for Realsense camera system.
"""
import logging
from typing import Tuple, List

import numpy as np
import pyrealsense2 as rs

from camera_objects.camera_abstract_class import TwoCamerasSystem
from camera_objects.realsense_camera_system import RealsenseCameraSystem

class DualRealsenseSystem(TwoCamerasSystem):
    """
    Realsense camera system, inherited from TwoCamerasSystem.

    Functions:
        __init__(int, int) -> None
        get_grayscale_images() -> Tuple[bool, np.ndarray, np.ndarray]
        get_depth_image() -> Tuple[bool, np.ndarray]
        get_width() -> int
        get_height() -> int
        release() -> bool
    """
    def __init__(self, width: int, height: int) -> None:
        """
        Initialize realsense camera system.

        args:
            width (int): width of realsense camera stream.
            height (int): height of realsense camera stream.

        returns:
        No return.
        """
        super().__init__()

        self.width = width
        self.height = height

        # Configure the RealSense pipeline
        # Create a context object. This object manages all connected devices
        context = rs.context()

        # Get a list of all connected devices
        connected_devices = context.query_devices()

        if len(connected_devices) < 2:
            logging.error("Not enough cameras, only detected %d cameras.", len(connected_devices))
            raise ValueError(f"Not enough cameras, only detected {len(connected_devices)} cameras.")

        self.realsense_system_list: List[RealsenseCameraSystem] = []

        # List all connected cameras with their serial numbers
        for i, device in enumerate(connected_devices):
            device_name = device.get_info(rs.camera_info.name)
            serial_number = device.get_info(rs.camera_info.serial_number)
            logging.info("Connected RealSense device %s: %s, Serial Number: %s", i, device_name, serial_number)

            realsense_camera = RealsenseCameraSystem(width, height, serial_number)
            self.realsense_system_list.append(realsense_camera)

    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get grayscale images for both camera.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray, np.ndarray]:
            - bool: Whether images grabbing is successful or not.
            - np.ndarray: grayscale image for left camera.
            - np.ndarray: grayscale image for right camera.
        """
        total_success = True
        left_images = []
        for realsense_camera in self.realsense_system_list:
            success, left_image, _ = realsense_camera.get_grayscale_images()
            total_success = total_success and success
            left_images.append(left_image)

        return total_success, left_images[0], left_images[1]

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
        # TODO: modify this function to return depth images for both cameras
        # FIXME: this change will require modification of entire API calls
        return self.realsense_system_list[0].get_depth_image()

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
        for realsense_camera in self.realsense_system_list:
            realsense_camera.release()
        return True
