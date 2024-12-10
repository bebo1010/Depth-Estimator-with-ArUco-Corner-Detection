"""
Main funtion to start the application with Realsense camera system.
"""
from ui_objects.opencv_ui_controller import OpencvUIController
from camera_objects.realsense_camera_system import RealsenseCameraSystem

if __name__ == "__main__":
    # Camera intrinsic parameters, should be loaded somewhere else?
    # D415
    FOCAL_LENGTH = 908.36  # in pixels
    BASELINE = 55  # in mm

    # Camera intrinsic parameters, should be loaded somewhere else?
    # D435
    # FOCAL_LENGTH = 425.203  # in pixels
    # BASELINE = 50  # in mm

    UI = OpencvUIController(system_prefix="Realsense", focal_length=FOCAL_LENGTH, baseline=BASELINE)

    cameras = RealsenseCameraSystem()
    UI.set_camera_system(cameras)

    UI.start()
