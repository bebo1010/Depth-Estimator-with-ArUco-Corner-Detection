"""
This module initializes the Epipolar Line Detector package.
It imports the EpipolarLineDetector class from the epipolar_line_detector module and
sets the __all__ variable to include only the EpipolarLineDetector class.

Classes:
    EpipolarLineDetector: A class for detecting epipolar lines.

__all__:
    List of public objects of that module, as interpreted by import *.
"""

from .epipolar_line_detector import EpipolarLineDetector

__all__ = ['EpipolarLineDetector']
