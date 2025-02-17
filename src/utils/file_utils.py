"""
This module provides utility functions for file operations,
such as determining the starting index for image files in a directory.
"""

import os
import logging
import json
from typing import Optional, Tuple
import time

import yaml
import cv2
import numpy as np

def get_starting_index(directory: str) -> int:
    """
    Get the starting index for image files in the given directory.

    args:
        directory (str): The directory to search for image files.

    return:
        int:
            - int: The starting index for image files in the given directory.
    """
    if not os.path.exists(directory):
        return 1
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    indices = [
        int(os.path.splitext(f)[0].split("image")[-1])
        for f in files
    ]
    return max(indices, default=0) + 1

def parse_yaml_config(config_yaml_path: str) -> dict:
    """
    Parse configuration file for flir camera system.

    args:
    config_yaml_path (str): path to config file.

    returns:
    dict:
        - dict: dictionary of full configs or None if an error occurs.
    """
    try:
        with open(config_yaml_path, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logging.info("Configuration file at %s successfully loaded", config_yaml_path)
            return config
    except (OSError, yaml.YAMLError) as e:
        logging.error("Error when loading or parsing configuration file at %s: %s", config_yaml_path, e)
        return None

def setup_directories(base_dir: str) -> None:
    """
    Make directories for storing images and logs.

    Args:
        base_dir (str): The base directory to create subdirectories in.

    Returns:
        None.
    """
    os.makedirs(base_dir, exist_ok=True)

    left_ir_dir = os.path.join(base_dir, "left_ArUco_images")
    right_ir_dir = os.path.join(base_dir, "right_ArUco_images")
    depth_dir = os.path.join(base_dir, "depth_images")
    left_chessboard_dir = os.path.join(base_dir, "left_chessboard_images")
    right_chessboard_dir = os.path.join(base_dir, "right_chessboard_images")

    os.makedirs(left_ir_dir, exist_ok=True)
    os.makedirs(right_ir_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(left_chessboard_dir, exist_ok=True)
    os.makedirs(right_chessboard_dir, exist_ok=True)

def setup_logging(base_dir: str) -> None:
    """
    Setup logging for the application.

    Args:
        base_dir (str): The base directory to save the log file in.

    Returns:
        None.
    """
    log_path = os.path.join(base_dir, "aruco_depth_log.txt")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )

def save_images(base_dir: str,
                left_gray_image: np.ndarray,
                right_gray_image: np.ndarray,
                image_index: int,
                first_depth_image: Optional[np.ndarray] = None,
                second_depth_image: Optional[np.ndarray] = None,
                prefix: str = ""
                ) -> None:
    """
    Save the images to disk.

    Args:
        base_dir (str): The base directory to save the images.
        left_gray_image (np.ndarray): Grayscale image of the left camera.
        right_gray_image (np.ndarray): Grayscale image of the right camera.
        image_index (int): The index for naming the saved images.
        first_depth_image (Optional[np.ndarray]): First depth image.
        second_depth_image (Optional[np.ndarray]): Second depth image.
        prefix (str): Prefix for the image directories.

    Returns:
        None.
    """
    # File paths
    left_gray_dir = os.path.join(base_dir, f"left_{prefix}_images")
    right_gray_dir = os.path.join(base_dir, f"right_{prefix}_images")
    depth_dir = os.path.join(base_dir, "depth_images")

    # Paths for left and right images
    left_gray_path = os.path.join(left_gray_dir, f"left_image{image_index}.png")
    right_gray_path = os.path.join(right_gray_dir, f"right_image{image_index}.png")

    # Save the left and right grayscale images
    start_time = time.perf_counter()
    cv2.imwrite(left_gray_path, left_gray_image)
    logging.info("Time to save left image: %s", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    cv2.imwrite(right_gray_path, right_gray_image)
    logging.info("Time to save right image: %s", time.perf_counter() - start_time)

    log_message = [
        f"Saved images - Left {prefix}: {left_gray_path}, Right {prefix}: {right_gray_path}"
    ]

    # Handle first depth image
    if first_depth_image is not None:
        depth_png_path_1 = os.path.join(depth_dir, f"depth_image1_{image_index}.png")
        depth_npy_path_1 = os.path.join(depth_dir, f"depth_image1_{image_index}.npy")
        cv2.imwrite(depth_png_path_1, first_depth_image)
        np.save(depth_npy_path_1, first_depth_image)

        log_message.extend([
            f"Depth PNG 1: {depth_png_path_1}",
            f"Depth NPY 1: {depth_npy_path_1}"
        ])

    # Handle second depth image
    if second_depth_image is not None:
        depth_png_path_2 = os.path.join(depth_dir, f"depth_image2_{image_index}.png")
        depth_npy_path_2 = os.path.join(depth_dir, f"depth_image2_{image_index}.npy")
        cv2.imwrite(depth_png_path_2, second_depth_image)
        np.save(depth_npy_path_2, second_depth_image)

        log_message.extend([
            f"Depth PNG 2: {depth_png_path_2}",
            f"Depth NPY 2: {depth_npy_path_2}"
        ])

    # Log all the saved paths
    logging.info(", ".join(log_message))

def load_images_from_directory(selected_dir: str) -> Tuple[Optional[list], Optional[str]]:
    """
    Load images from a selected directory.

    Args:
        selected_dir (str): The directory to load images from.

    Returns:
        Tuple[Optional[list], Optional[str]]:
            - List of tuples containing paths to left, right, and depth images.
            - Error message if any error occurs, otherwise None.
    """
    left_aruco_dir = os.path.join(selected_dir, "left_ArUco_images")
    right_aruco_dir = os.path.join(selected_dir, "right_ArUco_images")
    depth_dir = os.path.join(selected_dir, "depth_images")

    if not os.path.exists(left_aruco_dir) or not os.path.exists(right_aruco_dir):
        return None, "Invalid directory structure."

    left_images = sorted([os.path.join(left_aruco_dir, f) for f in os.listdir(left_aruco_dir) if f.endswith(".png")])
    right_images = sorted([os.path.join(right_aruco_dir, f) for f in os.listdir(right_aruco_dir) if f.endswith(".png")])
    left_depth_images = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) \
                                if f.startswith("depth_image1_") and f.endswith(".npy")])
    right_depth_images = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) \
                                 if f.startswith("depth_image2_") and f.endswith(".npy")])

    if not left_images or not right_images or len(left_images) != len(right_images):
        return None, "No images found or mismatched image counts."

    # Pad depth images if they do not exist
    if not left_depth_images:
        left_depth_images = [None] * len(left_images)
    if not right_depth_images:
        right_depth_images = [None] * len(right_images)

    loaded_images = list(zip(left_images, right_images, left_depth_images, right_depth_images))
    return loaded_images, None

def save_setup_info(base_dir: str, camera_params: dict) -> None:
    """
    Save the setup information to a JSON file.

    Args:
        base_dir (str): The base directory to save the setup information.
        camera_params (dict): The camera parameters.
    Returns:
        None.
    """
    setup_info = {
        "system_prefix": camera_params['system_prefix'],
        "focal_length": camera_params['focal_length'],
        "baseline": camera_params['baseline'],
        "width": camera_params['width'],
        "height": camera_params['height'],
        "principal_point": camera_params['principal_point']
    }
    setup_path = os.path.join(base_dir, "setup.json")
    with open(setup_path, 'w', encoding="utf-8") as f:
        json.dump(setup_info, f, indent=4)

def load_setup_info(directory: str) -> Optional[dict]:
    """
    Load the setup information from a JSON file.

    Args:
        directory (str): The directory to load the setup information from.

    Returns:
        Optional[dict]: The setup information if loaded successfully, otherwise None.
    """
    setup_path = os.path.join(directory, "setup.json")
    if not os.path.exists(setup_path):
        logging.warning("Setup file not found.")
        return None

    with open(setup_path, 'r', encoding="utf-8") as f:
        setup_info = json.load(f)

    return setup_info
