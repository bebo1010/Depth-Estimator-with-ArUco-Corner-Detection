"""
Main funtion to start the application with Realsense camera system.
"""
from ui_objects.opencv_ui_controller import OpencvUIController
from camera_objects.two_cameras.realsense_camera_system import RealsenseCameraSystem

if __name__ == "__main__":
    # D415
    FOCAL_LENGTH = 908.36  # in pixels
    BASELINE = 55  # in mm
    WIDTH = 1280
    HEIGHT = 720

    # D435
    # FOCAL_LENGTH = 425.203  # in pixels
    # BASELINE = 50  # in mm
    # WIDTH = 848
    # HEIGHT = 480

    UI = OpencvUIController(system_prefix="Realsense", focal_length=FOCAL_LENGTH, baseline=BASELINE)

    cameras = RealsenseCameraSystem(width = WIDTH, height = HEIGHT)
    UI.set_camera_system(cameras)

    UI.start()
