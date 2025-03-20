"""
This module initializes the camera objects by importing necessary classes from
single_camera and two_cameras modules.

Classes:
    SingleCameraSystem
    FlirCameraSystem
    TwoCamerasSystem
    RealsenseCameraSystem
    DualRealsenseSystem
    DualFlirSystem

__all__:
    These classes are made available for external use through the __all__ list.
"""

from .single_camera import SingleCameraSystem
from .two_cameras import TwoCamerasSystem

__all__ = ['SingleCameraSystem', 'TwoCamerasSystem']

try:
    import PySpin

    from .single_camera import FlirCameraSystem
    from .two_cameras import DualFlirSystem

    __all__.append('FlirCameraSystem')
    __all__.append('DualFlirSystem')
except ModuleNotFoundError:
    pass

try:
    import pyrealsense2

    from .two_cameras import RealsenseCameraSystem, DualRealsenseSystem

    __all__.append('RealsenseCameraSystem')
    __all__.append('DualRealsenseSystem')
except ModuleNotFoundError:
    pass
