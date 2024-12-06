import logging
from typing import Tuple

import numpy as np
import pyrealsense2 as rs

from .camera_abstract_class import Two_Cameras_System

class Realsense_Camera_System(Two_Cameras_System):
    def __init__(self):
        super().__init__()
        # Configure the RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        self.width = 848
        self.height = 480

        # Enable the depth and infrared streams
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)  # Depth
        config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, 30)  # Left IR (Y8)
        config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, 30)  # Right IR (Y8)

        # Start the pipeline
        self.pipeline.start(config)

    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        ir_frame_left = frames.get_infrared_frame(1)  # Left IR
        ir_frame_right = frames.get_infrared_frame(2)  # Right IR

        if not ir_frame_left:
            logging.error("Failed to get images from Realsense left IR stream")
            return [False, None, None]
        elif not ir_frame_right:
            logging.error("Failed to get images from Realsense right IR stream")
            return [False, None, None]
        else:
            # Convert images to numpy arrays
            ir_image_left = np.asanyarray(ir_frame_left.get_data())
            ir_image_right = np.asanyarray(ir_frame_right.get_data())
            return [True, ir_image_left, ir_image_right]
        
    def get_depth_image(self) -> Tuple[bool, np.ndarray]:
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
        return self.width
    
    def get_height(self) -> int:
        return self.height