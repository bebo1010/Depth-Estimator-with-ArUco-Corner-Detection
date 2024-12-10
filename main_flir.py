
from ui_objects.opencv_ui_controller import OpenCV_UI_Controller
from camera_objects.flir_camera_system import Flir_Camera_System

if __name__ == "__main__":
    focal_length = 1060  # in pixels
    baseline = 80  # in mm

    UI = OpenCV_UI_Controller(system_prefix="GH3", focal_length=focal_length, baseline=baseline)

    cameras = Flir_Camera_System("./camera_config/GH3_camera_config.yaml")
    UI.set_camera_system(cameras)

    UI.start()