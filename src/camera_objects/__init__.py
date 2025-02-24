"""
This module initializes the camera objects by importing necessary classes from
single_camera and two_cameras modules.

Classes imported:
    SingleCameraSystem
    FlirCameraSystem
    TwoCamerasSystem
    RealsenseCameraSystem
    DualRealsenseSystem
    DualFlirSystem

__all__:
    These classes are made available for external use through the __all__ list.
"""

from .single_camera import *

from .two_cameras import *

__all__ = ['SingleCameraSystem',
           'TwoCamerasSystem', 'RealsenseCameraSystem', 'DualRealsenseSystem']
try:
    import PySpin
    __all__.append('FlirCameraSystem')
    __all__.append('DualFlirSystem')
except ModuleNotFoundError:
    pass
