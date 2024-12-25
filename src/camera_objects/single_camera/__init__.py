"""
This module initializes the single camera systems by importing the necessary classes.
Classes:
    SingleCameraSystem: Handles operations related to a single camera system.
    FlirCameraSystem: Manages operations for FLIR camera systems.
__all__:
    List of public objects of this module, as interpreted by `import *`.
"""

from .single_camera_system import SingleCameraSystem
__all__ = ['SingleCameraSystem']
try:
    import PySpin
    from .flir_camera_system import FlirCameraSystem
    __all__.append('FlirCameraSystem')
except ModuleNotFoundError:
    pass
