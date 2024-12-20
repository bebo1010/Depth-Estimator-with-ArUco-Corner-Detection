"""
Module for Realsense camera system.
"""
import logging
from typing import Tuple, Optional

import numpy as np
import pyrealsense2 as rs

from camera_objects.two_cameras.two_cameras_system import TwoCamerasSystem

class RealsenseCameraSystem(TwoCamerasSystem):
    """
    Realsense camera system, inherited from TwoCamerasSystem.

    Functions:
        __init__(int, int, int) -> None
        get_grayscale_images() -> Tuple[bool, np.ndarray, np.ndarray]
        get_depth_images() -> Tuple[bool, np.ndarray, np.ndarray]
        get_width() -> int
        get_height() -> int
        release() -> bool
    """
    def __init__(self, width: int, height: int, serial_number: Optional[str] = None) -> None:
        """
        Initialize realsense camera system.

        args:
            width (int): width of realsense camera stream.
            height (int): height of realsense camera stream.
            serial_number (str, optional): serial number of the realsense camera. \
                Connect to the first realsense camera if not provided.

        returns:
        No return.
        """
        super().__init__()
        # Configure the RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        logging.info("Initializing Realsense camera system with width: %d, height: %d", width, height)
        if serial_number is not None:
            config.enable_device(serial_number)
            logging.info("Using Realsense camera with serial number: %s", serial_number)

        self.width = width
        self.height = height

        # Enable the depth and infrared streams
        config.enable_stream(rs.stream.depth, self.width, self.height,
                            rs.format.z16, 30)  # Depth
        config.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                            rs.format.y8, 30)  # Left IR (Y8)
        config.enable_stream(rs.stream.infrared, 2, self.width, self.height,
                            rs.format.y8, 30)  # Right IR (Y8)

        # Start the pipeline
        logging.info("Starting the Realsense pipeline")
        self.pipeline.start(config)
        logging.info("Realsense pipeline started successfully")

    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get grayscale images for both camera.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray, np.ndarray]:
            - bool: Whether images grabbing is successful or not.
            - np.ndarray: left grayscale image.
            - np.ndarray: right grayscale image.
        """
        logging.info("Grabbing grayscale images from Realsense camera")
        frames = self.pipeline.wait_for_frames()
        ir_frame_left = frames.get_infrared_frame(1)  # Left IR
        ir_frame_right = frames.get_infrared_frame(2)  # Right IR

        if not ir_frame_left:
            logging.error("Failed to get images from Realsense left IR stream")
            return [False, None, None]
        if not ir_frame_right:
            logging.error("Failed to get images from Realsense right IR stream")
            return [False, None, None]
        logging.info("Successfully grabbed grayscale images from Realsense camera")
        # Convert images to numpy arrays
        ir_image_left = np.asanyarray(ir_frame_left.get_data())
        ir_image_right = np.asanyarray(ir_frame_right.get_data())
        return [True, ir_image_left, ir_image_right]

    def get_depth_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get depth images for the camera system.

        args:
        No arguments.

        returns:
        Tuple[bool, np.ndarray, np.ndarray]:
            - bool: Whether depth image grabbing is successful or not.
            - np.ndarray: first depth grayscale image.
            - np.ndarray: second depth grayscale image.
        """
        logging.info("Grabbing depth image from Realsense camera")
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            logging.error("Failed to get images from Realsense depth stream")
            return False, None, None
        logging.info("Successfully grabbed depth image from Realsense camera")
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        return True, depth_image, None

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
        logging.info("Releasing the Realsense camera system")
        self.pipeline.stop()
        logging.info("Realsense camera system released successfully")
        return True
