"""
This module initializes the OpenCV objects package.
It imports the ArUcoDetector and EpipolarLineDetector classes from their respective modules and
sets the __all__ variable to include both classes.
Classes:
    ArUcoDetector: A class for detecting ArUco markers.
    EpipolarLineDetector: A class for detecting epipolar lines.
__all__:
    List of public objects of that module, as interpreted by import *.
"""

from .aruco_detector import ArUcoDetector
from .epipolar_line_detector import EpipolarLineDetector

__all__ = ["ArUcoDetector", "EpipolarLineDetector"]
