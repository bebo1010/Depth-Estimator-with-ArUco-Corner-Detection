"""
Main funtion to start the application with FLIR camera system.
"""
from ui_objects.opencv_ui_controller import OpencvUIController
from camera_objects.flir_camera_system import FlirCameraSystem

if __name__ == "__main__":
    FOCAL_LENGTH = 1060  # in pixels
    BASELINE = 80  # in mm

    UI = OpencvUIController(system_prefix="GH3", focal_length=FOCAL_LENGTH, baseline=BASELINE)

    cameras = FlirCameraSystem("./camera_config/GH3_camera_config.yaml")
    UI.set_camera_system(cameras)

    UI.start()
    