"""
This module initializes the OpenCV objects package.

Classes:
    ArUcoDetector: A class for detecting ArUco markers.
    EpipolarLineDetector: A class for detecting epipolar lines.
    ChessboardCalibrator: A class for calibrating stereo cameras.

__all__:
    List of public objects of that module, as interpreted by import *.
"""

from .aruco_detector import ArUcoDetector
from .epipolar_line_detector import EpipolarLineDetector
from .chessboard_calibration import ChessboardCalibrator

__all__ = ["ArUcoDetector", "EpipolarLineDetector", "ChessboardCalibrator"]
