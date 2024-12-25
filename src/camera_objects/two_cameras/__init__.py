"""
This module initializes and exposes the camera systems available in the package.
Classes:
    TwoCamerasSystem: Handles operations for a system with two cameras.
    RealsenseCameraSystem: Manages a system using Intel RealSense cameras.
    DualRealsenseSystem: Manages a system with two Intel RealSense cameras.
    DualFlirSystem: Manages a system with two FLIR cameras.
__all__:
    List of classes that are available for import when the module is imported.
"""

from .two_cameras_system import TwoCamerasSystem
from .realsense_camera_system import RealsenseCameraSystem
from .dual_realsense_system import DualRealsenseSystem
__all__ = ['TwoCamerasSystem', 'RealsenseCameraSystem', 'DualRealsenseSystem']
try:
    import PySpin
    from .dual_flir_system import DualFlirSystem
    __all__.append('DualFlirSystem')
except ModuleNotFoundError:
    pass
