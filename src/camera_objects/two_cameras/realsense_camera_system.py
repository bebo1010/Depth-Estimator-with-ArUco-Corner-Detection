"""
Module for Realsense camera system.
"""
import logging
from typing import Tuple, Optional

import numpy as np
import pyrealsense2 as rs

from .two_cameras_system import TwoCamerasSystem

class RealsenseCameraSystem(TwoCamerasSystem):
    """
    Realsense camera system, inherited from TwoCamerasSystem.
    """
    def __init__(self, width: int, height: int, serial_number: Optional[str] = None) -> None:
        """
        Initialize realsense camera system.

        Parameters
        ----------
        width : int
            Width of realsense camera stream.
        height : int
            Height of realsense camera stream.
        serial_number : str, optional
            Serial number of the realsense camera. Connect to the first realsense camera if not provided.

        Returns
        -------
        None
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
        pipeline_profile = self.pipeline.start(config)

        # Disable IR projector
        device = pipeline_profile.get_device()
        depth_sensor = device.query_sensors()[0]
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        logging.info("Realsense pipeline started successfully")

    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get grayscale images for both cameras.

        Returns
        -------
        Tuple[bool, np.ndarray, np.ndarray]
            - bool: Whether images grabbing is successful or not.
            - np.ndarray: Left grayscale image (or None if failed).
            - np.ndarray: Right grayscale image (or None if failed).
        """
        logging.info("Grabbing grayscale images from Realsense camera")
        frames = self.pipeline.wait_for_frames()
        ir_frame_left = frames.get_infrared_frame(1)  # Left IR
        ir_frame_right = frames.get_infrared_frame(2)  # Right IR

        if not ir_frame_left and not ir_frame_right:
            logging.error("Failed to get images from Realsense IR streams")
            return [False, None, None]

        ir_image_left = np.asanyarray(ir_frame_left.get_data()) if ir_frame_left else None
        ir_image_right = np.asanyarray(ir_frame_right.get_data()) if ir_frame_right else None

        success = ir_image_left is not None and ir_image_right is not None
        logging.info("Successfully grabbed grayscale images from Realsense camera" if success \
                     else "Failed to grab both grayscale images from Realsense camera")
        return [success, ir_image_left, ir_image_right]

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
        logging.info("Releasing the Realsense camera system")
        self.pipeline.stop()
        logging.info("Realsense camera system released successfully")
        return True
