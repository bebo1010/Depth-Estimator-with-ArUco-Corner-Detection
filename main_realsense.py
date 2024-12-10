
from opencv_ui_controller import OpenCV_UI_Controller
from camera_objects.realsense_camera_system import Realsense_Camera_System

if __name__ == "__main__":
    # Camera intrinsic parameters, should be loaded somewhere else?
    # D415
    focal_length = 908.36  # in pixels
    baseline = 55  # in mm

    # Camera intrinsic parameters, should be loaded somewhere else?
    # D435
    # focal_length = 425.203  # in pixels
    # baseline = 50  # in mm

    UI = OpenCV_UI_Controller(system_prefix="Realsense", focal_length=focal_length, baseline=baseline)

    cameras = Realsense_Camera_System()
    UI.set_camera_system(cameras)

    UI.start()