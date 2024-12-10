
from ui_objects.opencv_ui_controller import OpencvUIController
from camera_objects.realsense_camera_system import RealsenseCameraSystem

if __name__ == "__main__":
    # Camera intrinsic parameters, should be loaded somewhere else?
    # D415
    focal_length = 908.36  # in pixels
    baseline = 55  # in mm

    # Camera intrinsic parameters, should be loaded somewhere else?
    # D435
    # focal_length = 425.203  # in pixels
    # baseline = 50  # in mm

    UI = OpencvUIController(system_prefix="Realsense", focal_length=focal_length, baseline=baseline)

    cameras = RealsenseCameraSystem()
    UI.set_camera_system(cameras)

    UI.start()