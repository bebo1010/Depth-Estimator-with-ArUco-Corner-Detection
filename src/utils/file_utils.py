"""
This module provides utility functions for file operations,
such as determining the starting index for image files in a directory.
"""

import os
import logging

import yaml

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
