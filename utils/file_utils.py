"""
This module provides utility functions for file operations,
such as determining the starting index for image files in a directory.
"""

import os

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
