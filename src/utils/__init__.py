"""
This module initializes utility functions for the Depth Estimator ArUco project.
Imports:
    get_starting_index (from .file_utils): Function to get the starting index.
    parse_yaml_config (from .file_utils): Function to parse YAML configuration files.
__all__:
    List of public objects of that module, as interpreted by import *.
    - 'get_starting_index'
    - 'parse_yaml_config'
"""

from .file_utils import get_starting_index, parse_yaml_config

__all__ = ['get_starting_index', 'parse_yaml_config']
