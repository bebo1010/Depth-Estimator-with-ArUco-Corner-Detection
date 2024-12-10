import logging
from typing import Tuple

import numpy as np
import pyrealsense2 as rs

from .camera_abstract_class import TwoCamerasSystem

class RealsenseCameraSystem(TwoCamerasSystem):
    """
    Realsense camera system, inherited from TwoCamerasSystem.

    Functions:
        __init__() -> None
        get_grayscale_images() -> Tuple[bool, np.ndarray, np.ndarray]
        get_depth_image() -> Tuple[bool, np.ndarray]
        get_width() -> int
        get_height() -> int
        release() -> bool
    """
    def __init__(self) -> None:
        """
        Initialize realsense camera system.

        args:
        No arguments.

        returns:
        No return.
        """
        super().__init__()
        # Configure the RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        self.width = 848
        self.height = 480

        # Enable the depth and infrared streams
        config.enable_stream(rs.stream.depth, self.width, self.height,
                            rs.format.z16, 30)  # Depth
        config.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                            rs.format.y8, 30)  # Left IR (Y8)
        config.enable_stream(rs.stream.infrared, 2, self.width, self.height,
                            rs.format.y8, 30)  # Right IR (Y8)

        # Start the pipeline
        self.pipeline.start(config)
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
        frames = self.pipeline.wait_for_frames()
        ir_frame_left = frames.get_infrared_frame(1)  # Left IR
        ir_frame_right = frames.get_infrared_frame(2)  # Right IR

        if not ir_frame_left:
            logging.error("Failed to get images from Realsense left IR stream")
            return [False, None, None]
        if not ir_frame_right:
            logging.error("Failed to get images from Realsense right IR stream")
            return [False, None, None]
        # Convert images to numpy arrays
        ir_image_left = np.asanyarray(ir_frame_left.get_data())
        ir_image_right = np.asanyarray(ir_frame_right.get_data())
        return [True, ir_image_left, ir_image_right]       
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
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
    
        if not depth_frame:
            logging.error("Failed to get images from Realsense depth stream")
            return [False, None]
        else:
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            return [True, depth_image]
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
        self.pipeline.stop()
        return True
    