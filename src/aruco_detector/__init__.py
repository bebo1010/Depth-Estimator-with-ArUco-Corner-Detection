"""
This module initializes the ArUco detector package.
It imports the ArUcoDetector class from the aruco_detector module and
sets the __all__ variable to include only the ArUcoDetector class.
Classes:
    ArUcoDetector: A class for detecting ArUco markers.
__all__:
    List of public objects of that module, as interpreted by import *.
"""

from .aruco_detector import ArUcoDetector

__all__ = ["ArUcoDetector"]
